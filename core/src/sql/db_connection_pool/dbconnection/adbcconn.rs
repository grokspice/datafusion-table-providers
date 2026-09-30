// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use adbc_core::Statement;
use adbc_core::{Connection, Database};
use async_stream::stream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use std::any::Any;
use std::cell::RefCell;

use adbc_core::options::ObjectDepth;
use arrow::array::{AsArray, RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::SchemaRef;
use datafusion::common::TableReference;
use datafusion::error::DataFusionError;
use datafusion::execution::SendableRecordBatchStream;
use r2d2_adbc::AdbcConnectionManager;
use snafu::{prelude::*, ResultExt};
use std::marker::Send;
use std::marker::Sync;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::Sender;

use crate::sql::db_connection_pool::runtime::run_sync_with_tokio;

use super::DbConnection;
use super::Result;
use super::SyncDbConnection;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("ADBC Error: {source}"))]
    AdbcError { source: adbc_core::error::Error },

    #[snafu(display(
        "An unexpected error occurred.\n{message}\nVerify the configuration and try again"
    ))]
    ChannelError { message: String },
}

pub struct AdbcDbConnection<D>
where
    D: Database + Send + 'static,
    D::ConnectionType: Send + Sync,
    <D::ConnectionType as Connection>::StatementType: CancellableStatement,
{
    pub conn: Arc<Mutex<RefCell<r2d2::PooledConnection<AdbcConnectionManager<D>>>>>,
}

impl<D> DbConnection<r2d2::PooledConnection<AdbcConnectionManager<D>>, RecordBatch>
    for AdbcDbConnection<D>
where
    D: Database + Send + 'static,
    D::ConnectionType: Send + Sync,
    <D::ConnectionType as Connection>::StatementType: CancellableStatement,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn as_sync(
        &self,
    ) -> Option<&dyn SyncDbConnection<r2d2::PooledConnection<AdbcConnectionManager<D>>, RecordBatch>>
    {
        Some(self)
    }
}

/// A handle that cancels the query its statement is running.
pub trait StatementCancelHandle: Send + 'static {
    /// Cancels the in-flight query on the statement this handle came from.
    ///
    /// Called while that statement is inside [`Statement::execute`], which is
    /// the one moment ADBC defines `AdbcStatementCancel` for, and must not wait
    /// for the query to unwind.
    fn cancel(&mut self) -> adbc_core::error::Result<()>;
}

/// A statement that can hand out a handle for cancelling its in-flight query.
///
/// The thread inside [`Statement::execute`] holds the only `&mut` to the
/// statement for as long as the query runs, so cancelling it needs a second
/// handle — and that handle has to address *the same* driver statement.
/// `Clone` alone does not promise this: a `Clone` implementation is free to
/// produce independent cancellation state, and cancelling through such a clone
/// would return success while the query kept running. Implement this only where
/// the handle genuinely aliases the statement it came from.
pub trait CancellableStatement: Statement {
    /// The handle type. Cancelling through it must interrupt a call already
    /// running on the statement that produced it.
    type CancelHandle: StatementCancelHandle;

    /// Returns a handle for cancelling this statement's in-flight query.
    fn cancel_handle(&self) -> Self::CancelHandle;
}

// A statement type defined outside this crate and outside the implementing
// crate cannot be opted in downstream, because neither the trait nor the type
// would be local there. That is the cost of naming the requirement instead of
// taking `Clone`, which any type can satisfy without aliasing the statement it
// came from; an implementation for another statement type belongs here.

/// `ManagedStatement` clones share one `Arc`'d FFI statement, so a clone
/// addresses the same driver statement and `AdbcStatementCancel` through it
/// reaches the running call.
///
/// This needs an `adbc_driver_manager` that issues `AdbcStatementCancel` without
/// taking the lock its other statement functions use. Published 0.23 and 0.24
/// serialize the two, so `cancel` there waits for the `execute` it is meant to
/// interrupt and the query is not stopped — the call still returns `Ok`, so a
/// consumer sees a cancellation that did nothing.
impl StatementCancelHandle for adbc_driver_manager::ManagedStatement {
    fn cancel(&mut self) -> adbc_core::error::Result<()> {
        Statement::cancel(self)
    }
}

impl CancellableStatement for adbc_driver_manager::ManagedStatement {
    type CancelHandle = Self;

    fn cancel_handle(&self) -> Self::CancelHandle {
        self.clone()
    }
}

/// Shared between the stream handed to the caller and the thread running the
/// query, so a caller that goes away can stop a query that has already started.
enum QueryCancellation<S> {
    /// The statement has not been created yet.
    Preparing,
    /// The query is running and can be cancelled through this handle.
    Running(S), // handle, not the statement itself
    /// The caller went away. If the query has not started, it must not start.
    Abandoned,
    /// The query ended on its own; there is nothing to cancel.
    Finished,
}

/// A record-batch stream that cancels its query when it is dropped.
///
/// Dropping the stream is how a caller that has gone away — a client deadline,
/// a disconnect, a cancelled plan — reaches this layer. Without this, the
/// blocking thread stays inside `Statement::execute` until the remote query
/// finishes on its own, holding its pooled connection for that whole time, and
/// the remote database keeps doing the work nobody is waiting for.
struct CancelOnDrop<S: StatementCancelHandle> {
    inner: SendableRecordBatchStream,
    cancellation: Arc<Mutex<QueryCancellation<S>>>,
}

impl<S> futures::Stream for CancelOnDrop<S>
where
    S: StatementCancelHandle,
{
    type Item = datafusion::common::Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<S> datafusion::execution::RecordBatchStream for CancelOnDrop<S>
where
    S: StatementCancelHandle,
{
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
}

impl<S: StatementCancelHandle> Drop for CancelOnDrop<S> {
    fn drop(&mut self) {
        let mut cancellation = match self.cancellation.lock() {
            Ok(cancellation) => cancellation,
            Err(poisoned) => poisoned.into_inner(),
        };
        match std::mem::replace(&mut *cancellation, QueryCancellation::Abandoned) {
            QueryCancellation::Running(mut statement) => {
                // `AdbcStatementCancel` is the one statement call a driver must
                // accept while another is in flight, and it only signals; it does
                // not wait for the query to unwind.
                if let Err(error) = statement.cancel() {
                    tracing::debug!("Failed to cancel abandoned ADBC query: {error}");
                }
            }
            QueryCancellation::Finished => {
                *cancellation = QueryCancellation::Finished;
            }
            QueryCancellation::Preparing | QueryCancellation::Abandoned => {}
        }
    }
}

fn lock_cancellation<S>(
    cancellation: &Arc<Mutex<QueryCancellation<S>>>,
) -> std::sync::MutexGuard<'_, QueryCancellation<S>> {
    match cancellation.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn blocking_channel_send<T>(channel: &Sender<T>, item: T) -> Result<()> {
    match channel.blocking_send(item) {
        Ok(()) => Ok(()),
        Err(e) => Err(Error::ChannelError {
            message: format!("{e}"),
        }
        .into()),
    }
}

impl<D> SyncDbConnection<r2d2::PooledConnection<AdbcConnectionManager<D>>, RecordBatch>
    for AdbcDbConnection<D>
where
    D: Database + Send + 'static,
    D::ConnectionType: Send + Sync,
    <D::ConnectionType as Connection>::StatementType: CancellableStatement,
{
    fn new(conn: r2d2::PooledConnection<AdbcConnectionManager<D>>) -> Self {
        AdbcDbConnection {
            conn: Arc::new(Mutex::new(RefCell::new(conn))),
        }
    }

    fn tables(&self, schema: &str) -> Result<Vec<String>, super::Error> {
        let conn_mx = self.conn.lock().unwrap();
        let conn = conn_mx.borrow();
        let result = conn
            .get_objects(ObjectDepth::Tables, None, Some(schema), None, None, None)
            .boxed()
            .context(super::UnableToGetTablesSnafu)?;

        let mut tables = vec![];
        for batch in result {
            // Process each batch to extract table names
            //
            // Schema is as follows:
            // 0: CATALOG_NAME
            // 1: list<DB_SCHEMA_SCHEMA>
            //
            // DB_SCHEMA_SCHEMA is as follows:
            // 0: SCHEMA_NAME
            // 1: list<TABLE_INFO>
            //
            // TABLE_INFO is as follows:
            // 0: TABLE_NAME
            // 1: TABLE_TYPE
            // 2: list<COLUMN_SCHEMA>
            // 3: list<CONSTRAINT_SCHEMA>
            //
            // so we need to drill down to the table names
            let b = batch.boxed().context(super::UnableToGetTablesSnafu)?;
            b.column(1).as_list::<i32>().iter().for_each(|value| {
                if let Some(db_schema_schema) = value {
                    db_schema_schema
                        .as_struct()
                        .column(1)
                        .as_list::<i32>()
                        .iter()
                        .for_each(|table_info| {
                            if let Some(table_struct) = table_info {
                                tables.extend(
                                    table_struct
                                        .as_struct()
                                        .column(0)
                                        .as_string::<i32>()
                                        .iter()
                                        .flatten()
                                        .map(|name| name.to_string()),
                                );
                            }
                        })
                }
            })
        }

        Ok(tables)
    }

    fn schemas(&self) -> Result<Vec<String>, super::Error> {
        let conn_mx = self.conn.lock().unwrap();
        let conn = conn_mx.borrow();

        let result = conn
            .get_objects(ObjectDepth::Schemas, None, None, None, None, None)
            .boxed()
            .context(super::UnableToGetSchemaSnafu)?;

        let mut schemas = vec![];
        for batch in result {
            // Process each batch to extract schema names
            //
            // Schema is as follows:
            // 0: CATALOG_NAME
            // 1: list<DB_SCHEMA_SCHEMA>
            //
            // DB_SCHEMA_SCHEMA is as follows:
            // 0: SCHEMA_NAME
            // 1: list<TABLE_INFO>
            //
            // so we need to drill down to the schema names
            let b = batch.boxed().context(super::UnableToGetSchemaSnafu)?;
            b.column(1).as_list::<i32>().iter().for_each(|value| {
                if let Some(db_schema_schema) = value {
                    db_schema_schema
                        .as_struct()
                        .column(0)
                        .as_string::<i32>()
                        .iter()
                        .flatten()
                        .for_each(|name| schemas.push(name.to_string()));
                }
            });
        }
        Ok(schemas)
    }

    fn get_schema(&self, table_reference: &TableReference) -> Result<SchemaRef, super::Error> {
        let conn_mx = self.conn.lock().unwrap();
        let conn = conn_mx.borrow();

        let schema = conn
            .get_table_schema(
                table_reference.catalog(),
                table_reference.schema(),
                table_reference.table(),
            )
            .boxed()
            .context(super::UnableToGetSchemaSnafu)?;

        Ok(Arc::new(schema))
    }

    fn query_arrow(
        &self,
        sql: &str,
        params: &[RecordBatch],
        _projected_schema: Option<SchemaRef>,
    ) -> Result<SendableRecordBatchStream> {
        let (batch_tx, mut batch_rx) = tokio::sync::mpsc::channel::<RecordBatch>(4);

        // Schema discovery below runs before the stream exists, so there is
        // nothing for a caller to drop yet and this phase cannot be cancelled.
        // For a driver that answers `execute_schema` without running the query —
        // a dry run, say — that is a metadata round trip. For one that falls
        // back to the `LIMIT 0` wrapper, the remote database executes it, and a
        // caller that goes away during it waits for that to finish.
        let create_stream = || -> Result<SendableRecordBatchStream> {
            let schema: SchemaRef;
            {
                let conn_mx = self.conn.lock().unwrap();
                let mut conn = conn_mx.borrow_mut();
                let mut stmt = conn
                    .new_statement()
                    .boxed()
                    .context(super::UnableToQueryArrowSnafu)?;
                stmt.set_sql_query(sql)?;

                match stmt.execute_schema() {
                    Ok(s) => schema = s.into(),
                    // not all drivers implement execute_schema, so fall back to executing
                    // a row-free form of the query to get the schema.
                    Err(_) => {
                        stmt.set_sql_query(schema_probe_query(sql))?;
                        let result = stmt
                            .execute()
                            .boxed()
                            .context(super::UnableToQueryArrowSnafu)?;
                        schema = result.schema();
                    }
                }
            }

            let cloned_conn = Arc::clone(&self.conn);

            let sql_owned = sql.to_string();
            let params_owned = params.to_vec();

            let cancellation = Arc::new(Mutex::new(QueryCancellation::Preparing));
            let task_cancellation = Arc::clone(&cancellation);

            let join_handle = tokio::task::spawn_blocking(move || {
                let conn_mx = cloned_conn.lock().unwrap();
                let mut conn = conn_mx.borrow_mut();
                let mut stmt = conn
                    .new_statement()
                    .boxed()
                    .context(super::UnableToQueryArrowSnafu)?;
                stmt.set_sql_query(&sql_owned)?;

                match params_owned.len() {
                    0 => {}
                    1 => stmt.bind(params_owned[0].clone())?,
                    _ => {
                        let param_schema = params_owned[0].schema();
                        let reader = RecordBatchIterator::new(
                            params_owned.into_iter().map(Ok),
                            param_schema,
                        );

                        stmt.bind_stream(Box::new(reader))?;
                    }
                }

                {
                    let mut state = lock_cancellation(&task_cancellation);
                    if matches!(*state, QueryCancellation::Abandoned) {
                        // The caller went away while the statement was being
                        // prepared. Starting the query now would run it for
                        // nobody, and hold this pooled connection while it did.
                        return Ok(());
                    }
                    *state = QueryCancellation::Running(stmt.cancel_handle());
                }
                // Publishing the handle and entering `execute` cannot be made
                // one step: holding the lock across the call would make `Drop`
                // wait for the whole query. A caller that goes away inside that
                // window gets a cancel the driver may have nothing to apply it
                // to yet, so the query can still start; the check after
                // `execute` is what stops it being streamed and read for nobody.

                // Every non-panicking way out of the query — success, a failed
                // execute, a bad batch, a receiver that has gone — leaves the
                // query over. Record that before returning, so a consumer that
                // drops the stream on the error does not then cancel an
                // operation that has already ended.
                let outcome =
                    (|| -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        let results = stmt
                            .execute()
                            .boxed()
                            .context(super::UnableToQueryArrowSnafu)?;
                        if matches!(
                            *lock_cancellation(&task_cancellation),
                            QueryCancellation::Abandoned
                        ) {
                            return Ok(());
                        }
                        for batch in results {
                            let b = batch.boxed().context(super::UnableToQueryArrowSnafu)?;
                            blocking_channel_send(&batch_tx, b)?;
                        }
                        Ok(())
                    })();
                *lock_cancellation(&task_cancellation) = QueryCancellation::Finished;
                outcome
            });

            let output_stream = stream! {
                while let Some(batch) = batch_rx.recv().await {
                    yield Ok(batch);
                }

                match join_handle.await {
                    Ok(Ok(())) => {},
                    Ok(Err(task_error)) => {
                        yield Err(DataFusionError::Execution(format!(
                            "Failed to execute ADBC query: {task_error}"
                        )))
                    },
                    Err(join_error) => {
                        yield Err(DataFusionError::Execution(format!(
                            "Failed to execute ADBC query: {join_error}"
                        )))
                    },
                }
            };

            Ok(Box::pin(CancelOnDrop {
                inner: Box::pin(RecordBatchStreamAdapter::new(schema, output_stream)),
                cancellation,
            }))
        };

        run_sync_with_tokio(create_stream)
    }

    fn execute(&self, sql: &str, params: &[RecordBatch]) -> Result<u64> {
        let conn_mx = self.conn.lock().unwrap();
        let mut conn = conn_mx.borrow_mut();
        let mut stmt = conn.new_statement().context(AdbcSnafu)?;
        stmt.set_sql_query(sql)?;

        let params_owned = params.to_vec();
        match params.len() {
            0 => {}
            1 => stmt.bind(params_owned[0].clone())?,
            _ => {
                let param_schema = params_owned[0].schema();
                let reader =
                    RecordBatchIterator::new(params_owned.into_iter().map(Ok), param_schema);

                stmt.bind_stream(Box::new(reader))?;
            }
        }

        let count: Option<i64> = stmt.execute_update().context(AdbcSnafu)?;

        Ok(count.unwrap_or(-1) as u64)
    }
}

/// How a query whose schema is wanted can be made to return no rows.
#[derive(Debug, PartialEq, Eq)]
enum SchemaProbe {
    /// Wrap the query in a CTE that is then selected from with `LIMIT 0`.
    Wrap,
    /// Append `LIMIT 0` to the query itself.
    AppendLimit,
    /// Run the query unchanged; it already bounds its own row count.
    AsIs,
}

/// Picks how to ask a driver for a query's schema without materialising it.
///
/// Wrapping is the general form, but it is not always legal: `WITH RECURSIVE`
/// is only allowed at the top level of a `BigQuery` statement, so wrapping a
/// recursive query turns valid SQL into a syntax error. A query that opens its
/// own `WITH` therefore keeps its text — and its CTE's position — and takes the
/// row limit at the end instead, or runs unchanged when it already has one.
///
/// Decided on the text rather than by parsing it. A leading `WITH` is
/// unambiguous — nothing may precede it, so no literal or comment can spell one
/// — and parsing is both a needless second pass over every federated statement
/// and, measured on a 6.8 KB generated one, deep enough to overflow the stack
/// of the thread this runs on.
fn schema_probe_for(sql: &str) -> SchemaProbe {
    let sql = sql.trim_start();
    if !starts_with_keyword(sql, "WITH") {
        return SchemaProbe::Wrap;
    }
    if ends_with_row_limit(sql) {
        SchemaProbe::AsIs
    } else {
        SchemaProbe::AppendLimit
    }
}

/// Whether `sql` opens with `keyword` as a word of its own.
fn starts_with_keyword(sql: &str, keyword: &str) -> bool {
    let Some(rest) = sql.get(..keyword.len()) else {
        return false;
    };
    if !rest.eq_ignore_ascii_case(keyword) {
        return false;
    }
    // A statement that is only the keyword is not one we can reason about.
    sql[keyword.len()..]
        .chars()
        .next()
        .is_some_and(|c| c.is_whitespace() || c == '(')
}

/// Whether `sql` already bounds its own rows, so a `LIMIT` cannot be appended.
///
/// Conservative: anything that is not plainly free of a trailing limit is
/// treated as having one, which costs an unbounded schema fetch rather than a
/// statement the remote refuses to parse.
fn ends_with_row_limit(sql: &str) -> bool {
    let tail = sql.trim_end().to_ascii_uppercase();
    let tail = tail
        .rsplit_once(')')
        .map_or(tail.as_str(), |(_, after)| after);
    ["LIMIT", "OFFSET", "FETCH"]
        .iter()
        .any(|keyword| tail.contains(keyword))
}

/// Builds the query to execute to learn the schema of `sql` without returning
/// its rows. See [`schema_probe_for`].
fn schema_probe_query(sql: &str) -> String {
    match schema_probe_for(sql) {
        SchemaProbe::Wrap => {
            format!("WITH fetch_schema AS ({sql}) SELECT * FROM fetch_schema LIMIT 0")
        }
        SchemaProbe::AppendLimit => format!("{sql}\nLIMIT 0"),
        SchemaProbe::AsIs => sql.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adbc_core::error::{Error as AdbcError, Result as AdbcResult, Status};
    use adbc_core::options::{
        InfoCode, ObjectDepth, OptionConnection, OptionDatabase, OptionStatement, OptionValue,
    };
    use adbc_core::{Optionable, PartitionedResult};
    use arrow_schema::Schema;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Condvar;
    use std::time::{Duration, Instant};

    /// A query with no CTE of its own is wrapped, as it always was.
    #[test]
    fn a_plain_query_is_wrapped_to_fetch_its_schema() {
        assert_eq!(
            schema_probe_query("SELECT `a` FROM `t`"),
            "WITH fetch_schema AS (SELECT `a` FROM `t`) SELECT * FROM fetch_schema LIMIT 0"
        );
    }

    /// Wrapping a recursive query is a syntax error on `BigQuery`, which allows
    /// `WITH RECURSIVE` only at the top level, so the limit goes at the end and
    /// the query text is left alone.
    #[test]
    fn a_query_with_its_own_cte_keeps_its_text_and_takes_a_trailing_limit() {
        let sql = "WITH RECURSIVE `grid` AS ((SELECT 0 AS `n`) UNION ALL \
                   (SELECT `n` + 1 FROM `grid` WHERE `n` < 9)) SELECT * FROM `grid`";

        let probe = schema_probe_query(sql);

        assert_eq!(probe, format!("{sql}\nLIMIT 0"));
        assert!(
            probe.starts_with("WITH RECURSIVE"),
            "the recursive CTE left the top level: {probe}"
        );
    }

    /// A non-recursive `WITH` takes the same path: the parser cannot tell what
    /// the remote will accept nested, so no query that opens a `WITH` is wrapped.
    #[test]
    fn a_non_recursive_cte_is_also_not_wrapped() {
        assert_eq!(
            schema_probe_for("WITH `x` AS (SELECT 1 AS `n`) SELECT * FROM `x`"),
            SchemaProbe::AppendLimit
        );
    }

    /// A second `LIMIT` would not parse, so a query that already limits itself
    /// runs unchanged — it is bounded already.
    #[test]
    fn a_cte_query_that_already_limits_itself_runs_unchanged() {
        let sql = "WITH `x` AS (SELECT 1 AS `n`) SELECT * FROM `x` LIMIT 5";
        assert_eq!(schema_probe_for(sql), SchemaProbe::AsIs);
        assert_eq!(schema_probe_query(sql), sql);
    }

    /// A trailing `ORDER BY` still leaves room for the limit.
    #[test]
    fn an_ordered_cte_query_takes_a_trailing_limit() {
        assert_eq!(
            schema_probe_for("WITH `x` AS (SELECT 1 AS `n`) SELECT * FROM `x` ORDER BY `n`"),
            SchemaProbe::AppendLimit
        );
    }

    /// A `WITH` that is only the tail of a longer word is not a CTE.
    #[test]
    fn a_word_beginning_with_with_is_not_a_cte() {
        assert_eq!(
            schema_probe_for("WITHDRAWN SELECT 1"),
            SchemaProbe::Wrap,
            "`WITHDRAWN` is not a `WITH` clause"
        );
        assert_eq!(schema_probe_for("WITH"), SchemaProbe::Wrap);
    }

    /// The real statement that motivated this: a 6.8 KB generated recursive
    /// query. The first version of this check parsed the SQL to make the same
    /// decision and overflowed the stack of the thread it ran on, so the wrapper
    /// went back on and the statement was refused by the remote as before.
    #[test]
    fn a_large_generated_recursive_statement_is_not_wrapped() {
        let sql = format!(
            "WITH RECURSIVE `g` AS ((SELECT 0 AS `n`) UNION ALL \
             (SELECT `g`.`n` + 1 FROM `g` WHERE `g`.`n` < 720)) SELECT {} FROM `g`",
            (0..400)
                .map(|i| format!("CASE WHEN `g`.`n` = {i} THEN {i} ELSE 0 END AS `c{i}`"))
                .collect::<Vec<_>>()
                .join(", ")
        );

        assert!(sql.len() > 6_000, "the statement under test got smaller");
        assert_eq!(schema_probe_for(&sql), SchemaProbe::AppendLimit);
        assert!(schema_probe_query(&sql).starts_with("WITH RECURSIVE"));
    }

    /// Leading whitespace does not hide the `WITH`.
    #[test]
    fn leading_whitespace_does_not_hide_the_cte() {
        assert_eq!(
            schema_probe_for("\n  WITH `x` AS (SELECT 1 AS `n`) SELECT * FROM `x`"),
            SchemaProbe::AppendLimit
        );
    }

    /// What the fake driver did, so a test can assert on it rather than on
    /// timing alone.
    #[derive(Default)]
    struct DriverActivity {
        executing: Mutex<bool>,
        started: Condvar,
        cancels: AtomicUsize,
        executes: AtomicUsize,
        /// How many times the result reader was read from.
        reads: Arc<AtomicUsize>,
        connections_open: AtomicUsize,
    }

    impl DriverActivity {
        fn wait_until_executing(&self, timeout: Duration) -> bool {
            let deadline = Instant::now() + timeout;
            let mut executing = self.executing.lock().unwrap_or_else(|e| e.into_inner());
            while !*executing {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return false;
                }
                let (guard, _) = self
                    .started
                    .wait_timeout(executing, remaining)
                    .unwrap_or_else(|e| e.into_inner());
                executing = guard;
            }
            true
        }
    }

    /// A statement whose `execute` blocks like a driver waiting on a remote
    /// query, and returns only when cancelled.
    #[derive(Clone)]
    struct FakeStatement {
        activity: Arc<DriverActivity>,
        cancelled: Arc<(Mutex<bool>, Condvar)>,
        /// Makes `execute` fail at once instead of blocking, so a test can take
        /// the error path out of the query.
        fail_fast: Arc<AtomicBool>,
        /// Makes `execute` return rows after being cancelled instead of an
        /// error, standing in for a driver that got the cancel too early to
        /// apply it and ran the query anyway.
        ignore_cancel: Arc<AtomicBool>,
    }

    /// A reader that records whether anything read from it.
    struct CountingReader {
        schema: Arc<Schema>,
        reads: Arc<AtomicUsize>,
    }

    impl Iterator for CountingReader {
        type Item = std::result::Result<RecordBatch, arrow_schema::ArrowError>;
        fn next(&mut self) -> Option<Self::Item> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            None
        }
    }

    impl RecordBatchReader for CountingReader {
        fn schema(&self) -> Arc<Schema> {
            Arc::clone(&self.schema)
        }
    }

    /// Long enough that a test failure is a failure rather than a flake, short
    /// enough that a broken cancel does not hang a suite.
    const EXECUTE_GIVE_UP: Duration = Duration::from_secs(60);

    impl Optionable for FakeStatement {
        type Option = OptionStatement;
        fn set_option(&mut self, _key: Self::Option, _value: OptionValue) -> AdbcResult<()> {
            Ok(())
        }
        fn get_option_string(&self, _key: Self::Option) -> AdbcResult<String> {
            Err(unsupported())
        }
        fn get_option_bytes(&self, _key: Self::Option) -> AdbcResult<Vec<u8>> {
            Err(unsupported())
        }
        fn get_option_int(&self, _key: Self::Option) -> AdbcResult<i64> {
            Err(unsupported())
        }
        fn get_option_double(&self, _key: Self::Option) -> AdbcResult<f64> {
            Err(unsupported())
        }
    }

    fn unsupported() -> AdbcError {
        AdbcError::with_message_and_status(
            "not supported by the fake driver",
            Status::NotImplemented,
        )
    }

    // The fake shares its cancellation state through `Arc`s, so a handle taken
    // from a statement really does cancel that statement's query.
    impl StatementCancelHandle for FakeStatement {
        fn cancel(&mut self) -> AdbcResult<()> {
            Statement::cancel(self)
        }
    }

    impl CancellableStatement for FakeStatement {
        type CancelHandle = Self;

        fn cancel_handle(&self) -> Self::CancelHandle {
            self.clone()
        }
    }

    impl Statement for FakeStatement {
        fn bind(&mut self, _batch: RecordBatch) -> AdbcResult<()> {
            Ok(())
        }
        fn bind_stream(&mut self, _reader: Box<dyn RecordBatchReader + Send>) -> AdbcResult<()> {
            Ok(())
        }
        fn cancel(&mut self) -> AdbcResult<()> {
            self.activity.cancels.fetch_add(1, Ordering::SeqCst);
            let (lock, signal) = &*self.cancelled;
            let mut cancelled = lock.lock().unwrap_or_else(|e| e.into_inner());
            *cancelled = true;
            signal.notify_all();
            Ok(())
        }
        fn execute(&mut self) -> AdbcResult<Box<dyn RecordBatchReader + Send + 'static>> {
            self.activity.executes.fetch_add(1, Ordering::SeqCst);
            if self.fail_fast.load(Ordering::SeqCst) {
                return Err(AdbcError::with_message_and_status(
                    "query failed",
                    Status::Internal,
                ));
            }
            {
                let mut executing = self
                    .activity
                    .executing
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                *executing = true;
                self.activity.started.notify_all();
            }

            let (lock, signal) = &*self.cancelled;
            let mut cancelled = lock.lock().unwrap_or_else(|e| e.into_inner());
            let deadline = Instant::now() + EXECUTE_GIVE_UP;
            while !*cancelled {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }
                let (guard, _) = signal
                    .wait_timeout(cancelled, remaining)
                    .unwrap_or_else(|e| e.into_inner());
                cancelled = guard;
            }
            if self.ignore_cancel.load(Ordering::SeqCst) {
                return Ok(Box::new(CountingReader {
                    schema: Arc::new(Schema::empty()),
                    reads: Arc::clone(&self.activity.reads),
                }));
            }
            Err(AdbcError::with_message_and_status(
                "query cancelled",
                Status::Cancelled,
            ))
        }
        fn execute_update(&mut self) -> AdbcResult<Option<i64>> {
            Err(unsupported())
        }
        fn execute_schema(&mut self) -> AdbcResult<Schema> {
            Ok(Schema::empty())
        }
        fn execute_partitions(&mut self) -> AdbcResult<PartitionedResult> {
            Err(unsupported())
        }
        fn get_parameter_schema(&self) -> AdbcResult<Schema> {
            Err(unsupported())
        }
        fn prepare(&mut self) -> AdbcResult<()> {
            Ok(())
        }
        fn set_sql_query(&mut self, _query: impl AsRef<str>) -> AdbcResult<()> {
            Ok(())
        }
        fn set_substrait_plan(&mut self, _plan: impl AsRef<[u8]>) -> AdbcResult<()> {
            Err(unsupported())
        }
    }

    struct FakeConnection {
        activity: Arc<DriverActivity>,
        cancelled: Arc<(Mutex<bool>, Condvar)>,
        fail_fast: Arc<AtomicBool>,
        ignore_cancel: Arc<AtomicBool>,
    }

    impl Drop for FakeConnection {
        fn drop(&mut self) {
            self.activity
                .connections_open
                .fetch_sub(1, Ordering::SeqCst);
        }
    }

    impl Optionable for FakeConnection {
        type Option = OptionConnection;
        fn set_option(&mut self, _key: Self::Option, _value: OptionValue) -> AdbcResult<()> {
            Ok(())
        }
        fn get_option_string(&self, _key: Self::Option) -> AdbcResult<String> {
            Err(unsupported())
        }
        fn get_option_bytes(&self, _key: Self::Option) -> AdbcResult<Vec<u8>> {
            Err(unsupported())
        }
        fn get_option_int(&self, _key: Self::Option) -> AdbcResult<i64> {
            Err(unsupported())
        }
        fn get_option_double(&self, _key: Self::Option) -> AdbcResult<f64> {
            Err(unsupported())
        }
    }

    impl Connection for FakeConnection {
        type StatementType = FakeStatement;

        fn new_statement(&mut self) -> AdbcResult<Self::StatementType> {
            Ok(FakeStatement {
                activity: Arc::clone(&self.activity),
                cancelled: Arc::clone(&self.cancelled),
                fail_fast: Arc::clone(&self.fail_fast),
                ignore_cancel: Arc::clone(&self.ignore_cancel),
            })
        }
        fn cancel(&mut self) -> AdbcResult<()> {
            Err(unsupported())
        }
        fn get_info(
            &self,
            _codes: Option<std::collections::HashSet<InfoCode>>,
        ) -> AdbcResult<Box<dyn RecordBatchReader + Send + 'static>> {
            Err(unsupported())
        }
        fn get_objects(
            &self,
            _depth: ObjectDepth,
            _catalog: Option<&str>,
            _db_schema: Option<&str>,
            _table_name: Option<&str>,
            _table_type: Option<Vec<&str>>,
            _column_name: Option<&str>,
        ) -> AdbcResult<Box<dyn RecordBatchReader + Send + 'static>> {
            Err(unsupported())
        }
        fn get_table_schema(
            &self,
            _catalog: Option<&str>,
            _db_schema: Option<&str>,
            _table_name: &str,
        ) -> AdbcResult<Schema> {
            Err(unsupported())
        }
        fn get_table_types(&self) -> AdbcResult<Box<dyn RecordBatchReader + Send + 'static>> {
            Err(unsupported())
        }
        fn get_statistic_names(&self) -> AdbcResult<Box<dyn RecordBatchReader + Send + 'static>> {
            Err(unsupported())
        }
        fn get_statistics(
            &self,
            _catalog: Option<&str>,
            _db_schema: Option<&str>,
            _table_name: Option<&str>,
            _approximate: bool,
        ) -> AdbcResult<Box<dyn RecordBatchReader + Send + 'static>> {
            Err(unsupported())
        }
        fn commit(&mut self) -> AdbcResult<()> {
            Err(unsupported())
        }
        fn rollback(&mut self) -> AdbcResult<()> {
            Err(unsupported())
        }
        fn read_partition(
            &self,
            _partition: impl AsRef<[u8]>,
        ) -> AdbcResult<Box<dyn RecordBatchReader + Send + 'static>> {
            Err(unsupported())
        }
    }

    #[derive(Clone)]
    struct FakeDatabase {
        activity: Arc<DriverActivity>,
        cancelled: Arc<(Mutex<bool>, Condvar)>,
        fail_fast: Arc<AtomicBool>,
        ignore_cancel: Arc<AtomicBool>,
    }

    impl Optionable for FakeDatabase {
        type Option = OptionDatabase;
        fn set_option(&mut self, _key: Self::Option, _value: OptionValue) -> AdbcResult<()> {
            Ok(())
        }
        fn get_option_string(&self, _key: Self::Option) -> AdbcResult<String> {
            Err(unsupported())
        }
        fn get_option_bytes(&self, _key: Self::Option) -> AdbcResult<Vec<u8>> {
            Err(unsupported())
        }
        fn get_option_int(&self, _key: Self::Option) -> AdbcResult<i64> {
            Err(unsupported())
        }
        fn get_option_double(&self, _key: Self::Option) -> AdbcResult<f64> {
            Err(unsupported())
        }
    }

    impl Database for FakeDatabase {
        type ConnectionType = FakeConnection;

        fn new_connection(&self) -> AdbcResult<Self::ConnectionType> {
            self.activity
                .connections_open
                .fetch_add(1, Ordering::SeqCst);
            Ok(FakeConnection {
                activity: Arc::clone(&self.activity),
                cancelled: Arc::clone(&self.cancelled),
                fail_fast: Arc::clone(&self.fail_fast),
                ignore_cancel: Arc::clone(&self.ignore_cancel),
            })
        }

        fn new_connection_with_opts(
            &self,
            _opts: impl IntoIterator<Item = (OptionConnection, OptionValue)>,
        ) -> AdbcResult<Self::ConnectionType> {
            self.new_connection()
        }
    }

    fn fake_pool(
        activity: &Arc<DriverActivity>,
        fail_fast: &Arc<AtomicBool>,
        ignore_cancel: &Arc<AtomicBool>,
    ) -> Arc<crate::sql::db_connection_pool::adbcpool::ADBCPool<FakeDatabase>> {
        let database = FakeDatabase {
            activity: Arc::clone(activity),
            cancelled: Arc::new((Mutex::new(false), Condvar::new())),
            fail_fast: Arc::clone(fail_fast),
            ignore_cancel: Arc::clone(ignore_cancel),
        };
        let pool =
            crate::sql::db_connection_pool::adbcpool::AdbcConnectionPoolBuilder::new(database)
                .with_max_size(Some(1))
                .build()
                .expect("the pool should build");
        Arc::new(pool)
    }

    /// Dropping the stream must cancel the running query and give the pooled
    /// connection back, which is what a client that goes away needs to happen.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dropping_the_stream_cancels_the_query_and_frees_the_connection() {
        use crate::sql::db_connection_pool::DbConnectionPool;

        let activity = Arc::new(DriverActivity::default());
        let pool = fake_pool(
            &activity,
            &Arc::new(AtomicBool::new(false)),
            &Arc::new(AtomicBool::new(false)),
        );

        let conn = pool
            .connect()
            .await
            .expect("a connection should be available");
        let stream = super::super::query_arrow(conn, "SELECT 1".to_string(), None)
            .await
            .expect("the query should start");

        assert!(
            activity.wait_until_executing(Duration::from_secs(10)),
            "the driver never started executing"
        );

        drop(stream);

        // The pool holds one connection: it can only be handed out again once
        // the cancelled query has released it.
        let waited = Instant::now();
        let second = tokio::time::timeout(Duration::from_secs(20), pool.connect())
            .await
            .expect("the pool connection should come back after cancellation")
            .expect("a connection should be available");
        drop(second);

        assert_eq!(
            activity.cancels.load(Ordering::SeqCst),
            1,
            "the abandoned query was not cancelled"
        );
        assert!(
            waited.elapsed() < Duration::from_secs(20),
            "the pool connection took {:?} to come back",
            waited.elapsed()
        );
    }

    /// A caller that goes away while the statement is still being prepared must
    /// stop the query from starting at all.
    #[test]
    fn abandoning_before_the_query_starts_stops_it_starting() {
        let cancellation: Arc<Mutex<QueryCancellation<FakeStatement>>> =
            Arc::new(Mutex::new(QueryCancellation::Preparing));

        let guard = CancelOnDrop {
            inner: Box::pin(RecordBatchStreamAdapter::new(
                Arc::new(Schema::empty()),
                futures::stream::empty(),
            )),
            cancellation: Arc::clone(&cancellation),
        };
        drop(guard);

        assert!(
            matches!(
                *lock_cancellation(&cancellation),
                QueryCancellation::Abandoned
            ),
            "the query was not marked abandoned, so it would still be started"
        );
    }

    /// A cancel that reaches the driver too early to stop the query must still
    /// stop the result being read for a caller that has gone.
    ///
    /// The handle is published just before `execute` is entered, and the lock
    /// cannot be held across that call, so a caller dropping the stream in that
    /// window may cancel a query the driver has not started. The check after
    /// `execute` is what keeps the runtime from then draining a result set
    /// nobody is waiting for.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_result_is_not_read_for_a_caller_that_has_gone() {
        use crate::sql::db_connection_pool::DbConnectionPool;

        let activity = Arc::new(DriverActivity::default());
        let pool = fake_pool(
            &activity,
            &Arc::new(AtomicBool::new(false)),
            &Arc::new(AtomicBool::new(true)),
        );

        let conn = pool
            .connect()
            .await
            .expect("a connection should be available");
        let stream = super::super::query_arrow(conn, "SELECT 1".to_string(), None)
            .await
            .expect("the query should start");

        assert!(
            activity.wait_until_executing(Duration::from_secs(10)),
            "the driver never started executing"
        );
        drop(stream);

        // Waiting for the connection back is waiting for the worker to finish.
        let second = tokio::time::timeout(Duration::from_secs(20), pool.connect())
            .await
            .expect("the pool connection should come back")
            .expect("a connection should be available");
        drop(second);

        assert_eq!(
            activity.reads.load(Ordering::SeqCst),
            0,
            "the result was read for a caller that had already gone"
        );
    }

    /// A query that ends in an error must not be cancelled when its stream is
    /// dropped: it is over either way, and cancelling reaches a statement that
    /// has already been released.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_failed_query_is_finalized_and_not_cancelled() {
        use crate::sql::db_connection_pool::DbConnectionPool;
        use futures::StreamExt;

        let activity = Arc::new(DriverActivity::default());
        let pool = fake_pool(
            &activity,
            &Arc::new(AtomicBool::new(true)),
            &Arc::new(AtomicBool::new(false)),
        );

        let conn = pool
            .connect()
            .await
            .expect("a connection should be available");
        let mut stream = super::super::query_arrow(conn, "SELECT 1".to_string(), None)
            .await
            .expect("the stream should be created");

        let first = stream.next().await;
        assert!(
            matches!(first, Some(Err(_))),
            "the failed query should surface its error, got {first:?}"
        );
        drop(stream);

        assert_eq!(
            activity.cancels.load(Ordering::SeqCst),
            0,
            "a query that had already failed was cancelled"
        );
    }

    /// A query that finishes normally must not be cancelled when its stream is
    /// dropped.
    #[test]
    fn a_finished_query_is_not_cancelled() {
        let activity = Arc::new(DriverActivity::default());
        let cancellation: Arc<Mutex<QueryCancellation<FakeStatement>>> =
            Arc::new(Mutex::new(QueryCancellation::Finished));

        let guard = CancelOnDrop {
            inner: Box::pin(RecordBatchStreamAdapter::new(
                Arc::new(Schema::empty()),
                futures::stream::empty(),
            )),
            cancellation: Arc::clone(&cancellation),
        };
        drop(guard);

        assert_eq!(
            activity.cancels.load(Ordering::SeqCst),
            0,
            "a completed query was cancelled"
        );
        assert!(matches!(
            *lock_cancellation(&cancellation),
            QueryCancellation::Finished
        ));
    }
}
