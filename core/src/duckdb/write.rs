use std::time::{SystemTime, UNIX_EPOCH};
use std::{fmt, sync::Arc};

use crate::duckdb::DuckDB;
use crate::sql::db_connection_pool::dbconnection::duckdbconn::DuckDbConnection;
use crate::sql::db_connection_pool::duckdbpool::DuckDbConnectionPool;
use crate::sql::sql_provider_datafusion::expr;
use crate::util::{
    constraints,
    count_exec::make_count_exec,
    dml::{
        assignments_to_sql_with_schema, delete_statement, filters_to_sql_with_schema,
        update_statement, DeletionExec, DeletionSink, UpdateExec, UpdateSink,
    },
    on_conflict::OnConflict,
    retriable_error::{check_and_mark_retriable_error, to_retriable_data_write_error},
};
use arrow::array::RecordBatchReader;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::{array::RecordBatch, datatypes::SchemaRef};
use arrow_schema::ArrowError;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::TableReference;
use datafusion::common::{not_impl_err, Constraints, SchemaExt};
use datafusion::datasource::sink::{DataSink, DataSinkExec};
use datafusion::logical_expr::dml::InsertOp;
use datafusion::{
    datasource::{TableProvider, TableType},
    error::DataFusionError,
    execution::{SendableRecordBatchStream, TaskContext},
    logical_expr::Expr,
    physical_plan::{metrics::MetricsSet, DisplayAs, DisplayFormatType, ExecutionPlan},
};
use duckdb::Transaction;
use futures::StreamExt;
use snafu::prelude::*;
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::task::JoinHandle;

use super::creator::{TableDefinition, TableManager, ViewCreator};
use super::file_swap;
use super::upsert_groups;
use super::write_settings::DuckDBWriteSettings;
use super::{to_datafusion_error, RelationName};
use crate::sql::db_connection_pool::Mode;

/// A callback handler that is invoked after data has been successfully written to a DuckDB table
/// but before the transaction is committed.
///
/// The handler executes within the same transaction as the data write operation. This means:
/// - Any operations performed by the handler are atomic with the data write
/// - If the handler returns an error, both the handler's operations and the data write are rolled back
/// - The handler has access to the newly written data for queries and validations
pub type WriteCompletionHandler = Arc<
    dyn Fn(&Transaction<'_>, &TableManager, &SchemaRef, u64) -> datafusion::common::Result<()>
        + Send
        + Sync
        + 'static,
>;

// checking schemas are equivalent is disabled because it incorrectly marks single-level list fields are different when the name of the field is different
// e.g. List(Field { name: 'a', data_type: Int32 }) != List(Field { name: 'b', data_type: Int32 })
// but, in this case, they are actually equivalent because the field name does not matter for the schema.
// related: https://github.com/apache/arrow-rs/issues/6733#issuecomment-2482582556
const SCHEMA_EQUIVALENCE_ENABLED: bool = false;

#[derive(Default)]
pub struct DuckDBTableWriterBuilder {
    read_provider: Option<Arc<dyn TableProvider>>,
    pool: Option<Arc<DuckDbConnectionPool>>,
    on_conflict: Option<OnConflict>,
    table_definition: Option<Arc<TableDefinition>>,
    on_data_written: Option<WriteCompletionHandler>,
    write_settings: Option<DuckDBWriteSettings>,
}

impl DuckDBTableWriterBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_read_provider(mut self, read_provider: Arc<dyn TableProvider>) -> Self {
        self.read_provider = Some(read_provider);
        self
    }

    #[must_use]
    pub fn with_pool(mut self, pool: Arc<DuckDbConnectionPool>) -> Self {
        self.pool = Some(pool);
        self
    }

    #[must_use]
    pub fn set_on_conflict(mut self, on_conflict: Option<OnConflict>) -> Self {
        self.on_conflict = on_conflict;
        self
    }

    #[must_use]
    pub fn with_table_definition(mut self, table_definition: TableDefinition) -> Self {
        self.table_definition = Some(Arc::new(table_definition));
        self
    }

    #[must_use]
    pub fn with_on_data_written(mut self, on_data_written: WriteCompletionHandler) -> Self {
        self.on_data_written = Some(on_data_written);
        self
    }

    #[must_use]
    pub fn with_write_settings(mut self, write_settings: DuckDBWriteSettings) -> Self {
        self.write_settings = Some(write_settings);
        self
    }

    /// Builds a `DuckDBTableWriter` from the provided configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the required fields are missing:
    /// - `read_provider`
    /// - `pool`
    /// - `table_definition`
    pub fn build(self) -> super::Result<DuckDBTableWriter> {
        let Some(read_provider) = self.read_provider else {
            return Err(super::Error::MissingReadProvider);
        };

        let Some(pool) = self.pool else {
            return Err(super::Error::MissingPool);
        };

        let Some(table_definition) = self.table_definition else {
            return Err(super::Error::MissingTableDefinition);
        };

        Ok(DuckDBTableWriter {
            read_provider,
            on_conflict: self.on_conflict,
            table_definition,
            pool,
            on_data_written: self.on_data_written,
            write_settings: self.write_settings.unwrap_or_default(),
        })
    }
}

#[derive(Clone)]
pub struct DuckDBTableWriter {
    pub read_provider: Arc<dyn TableProvider>,
    pool: Arc<DuckDbConnectionPool>,
    table_definition: Arc<TableDefinition>,
    on_conflict: Option<OnConflict>,
    on_data_written: Option<WriteCompletionHandler>,
    write_settings: DuckDBWriteSettings,
}

impl std::fmt::Debug for DuckDBTableWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuckDBTableWriter")
            .field("read_provider", &self.read_provider)
            .field("pool", &self.pool)
            .field("table_definition", &self.table_definition)
            .field("on_conflict", &self.on_conflict)
            .field(
                "on_data_written",
                &self
                    .on_data_written
                    .as_ref()
                    .map_or("None", |_| "Some(callback)"),
            )
            .field("write_settings", &self.write_settings)
            .finish()
    }
}

impl DuckDBTableWriter {
    #[must_use]
    pub fn pool(&self) -> Arc<DuckDbConnectionPool> {
        Arc::clone(&self.pool)
    }

    #[must_use]
    pub fn table_definition(&self) -> Arc<TableDefinition> {
        Arc::clone(&self.table_definition)
    }

    #[must_use]
    pub fn on_conflict(&self) -> Option<&OnConflict> {
        self.on_conflict.as_ref()
    }

    #[must_use]
    pub fn write_settings(&self) -> &DuckDBWriteSettings {
        &self.write_settings
    }

    #[must_use]
    pub fn with_on_data_written_handler(mut self, on_data_written: WriteCompletionHandler) -> Self {
        self.on_data_written = Some(on_data_written);
        self
    }
}

#[async_trait]
impl TableProvider for DuckDBTableWriter {
    fn schema(&self) -> SchemaRef {
        self.read_provider.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn constraints(&self) -> Option<&Constraints> {
        self.table_definition.constraints()
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        self.read_provider
            .scan(state, projection, filters, limit)
            .await
    }

    async fn insert_into(
        &self,
        _state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        op: InsertOp,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        // `InsertOp::Replace` promises an atomic per-row upsert. DuckDB delivers
        // that only with an `ON CONFLICT (...) DO UPDATE`, i.e.
        // `OnConflict::Upsert`. A missing `on_conflict` (plain append) or a
        // `DoNothing`/`DoNothingAll` (leaves the existing row unchanged) does
        // NOT replace, so refuse rather than silently accept a non-replacing
        // write.
        if matches!(op, InsertOp::Replace)
            && !matches!(self.on_conflict, Some(OnConflict::Upsert(_)))
        {
            return not_impl_err!(
                "InsertOp::Replace requires an on_conflict upsert target on the DuckDB writer"
            );
        }
        let mut sink = DuckDBDataSink::new(
            Arc::clone(&self.pool),
            Arc::clone(&self.table_definition),
            op,
            self.on_conflict.clone(),
            self.schema(),
        )
        .with_write_settings(self.write_settings.clone());

        if let Some(handler) = &self.on_data_written {
            sink = sink.with_on_data_written_handler(Arc::clone(handler));
        }

        Ok(Arc::new(DataSinkExec::new(input, Arc::new(sink), None)) as _)
    }

    async fn delete_from(
        &self,
        _state: &dyn Session,
        filters: Vec<Expr>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        let sql_where = if filters.is_empty() {
            None
        } else {
            Some(filters_to_sql_with_schema(
                &filters,
                Some(expr::Engine::DuckDB),
                Some(&self.schema()),
            )?)
        };
        let table_definition = Arc::clone(&self.table_definition);
        let pool = Arc::clone(&self.pool);

        Ok(Arc::new(DeletionExec::new(Arc::new(DuckDBDeletionSink {
            pool,
            table_definition,
            sql_where,
        }))))
    }

    async fn update(
        &self,
        _state: &dyn Session,
        assignments: Vec<(String, Expr)>,
        filters: Vec<Expr>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        if assignments.is_empty() {
            return make_count_exec(0);
        }

        // Bound once so the SET clause and the WHERE clause are visibly rendered against the
        // same schema.
        let schema = self.schema();
        let set_clause = assignments_to_sql_with_schema(
            &assignments,
            Some(expr::Engine::DuckDB),
            Some(&schema),
        )?;
        let sql_where = if filters.is_empty() {
            None
        } else {
            Some(filters_to_sql_with_schema(
                &filters,
                Some(expr::Engine::DuckDB),
                Some(&schema),
            )?)
        };
        let table_definition = Arc::clone(&self.table_definition);
        let pool = Arc::clone(&self.pool);

        Ok(Arc::new(UpdateExec::new(Arc::new(DuckDBUpdateSink {
            pool,
            table_definition,
            set_clause,
            sql_where,
        }))))
    }
}

struct DuckDBDeletionSink {
    pool: Arc<DuckDbConnectionPool>,
    table_definition: Arc<TableDefinition>,
    sql_where: Option<String>,
}

#[async_trait]
impl DeletionSink for DuckDBDeletionSink {
    async fn delete_from(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let pool = Arc::clone(&self.pool);
        let table_definition = Arc::clone(&self.table_definition);
        let sql_where = self.sql_where.clone();

        tokio::task::spawn_blocking(
            move || -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
                let write_gate = pool.write_gate();
                let _write_guard = write_gate
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);

                let mut db_conn = pool.connect_sync()?;
                let duckdb_conn = DuckDB::duckdb_conn(&mut db_conn)?;
                let tx = duckdb_conn.conn.transaction()?;

                let table_name = table_definition.resolve_dml_table_name(&tx)?;

                // `resolve_dml_table_name` returns the internal relation the accelerator writes
                // to, which is unqualified.
                let delete_sql =
                    delete_statement(&TableReference::bare(table_name), sql_where.as_deref());
                let count = tx.execute(&delete_sql, [])?;

                tx.commit()?;

                Ok(count as u64)
            },
        )
        .await?
    }
}

struct DuckDBUpdateSink {
    pool: Arc<DuckDbConnectionPool>,
    table_definition: Arc<TableDefinition>,
    set_clause: String,
    sql_where: Option<String>,
}

#[async_trait]
impl UpdateSink for DuckDBUpdateSink {
    async fn execute_update(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let pool = Arc::clone(&self.pool);
        let table_definition = Arc::clone(&self.table_definition);
        let set_clause = self.set_clause.clone();
        let sql_where = self.sql_where.clone();

        tokio::task::spawn_blocking(
            move || -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
                let write_gate = pool.write_gate();
                let _write_guard = write_gate
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);

                let mut db_conn = pool.connect_sync()?;
                let duckdb_conn = DuckDB::duckdb_conn(&mut db_conn)?;
                let tx = duckdb_conn.conn.transaction()?;

                let table_name = table_definition.resolve_dml_table_name(&tx)?;

                // As above: the resolved internal relation carries no qualifier.
                let sql = update_statement(
                    &TableReference::bare(table_name),
                    &set_clause,
                    sql_where.as_deref(),
                );
                let count = tx.execute(&sql, [])?;

                tx.commit()?;

                Ok(count as u64)
            },
        )
        .await?
    }
}

pub(crate) struct DuckDBDataSink {
    pool: Arc<DuckDbConnectionPool>,
    table_definition: Arc<TableDefinition>,
    overwrite: InsertOp,
    on_conflict: Option<OnConflict>,
    schema: SchemaRef,
    on_data_written: Option<WriteCompletionHandler>,
    write_settings: DuckDBWriteSettings,
}

#[async_trait]
impl DataSink for DuckDBDataSink {
    fn metrics(&self) -> Option<MetricsSet> {
        None
    }

    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    async fn write_all(
        &self,
        mut data: SendableRecordBatchStream,
        _context: &Arc<TaskContext>,
    ) -> datafusion::common::Result<u64> {
        let pool = Arc::clone(&self.pool);
        let table_definition = Arc::clone(&self.table_definition);
        let overwrite = self.overwrite;
        let on_conflict = self.on_conflict.clone();
        let on_data_written = self.on_data_written.clone();
        let write_settings = self.write_settings.clone();

        // Limit channel size to a maximum of 100 RecordBatches queued for cases when DuckDB is slower than the writer stream,
        // so that we don't significantly increase memory usage. After the maximum RecordBatches are queued, the writer stream will wait
        // until DuckDB is able to process more data.
        let (batch_tx, batch_rx): (Sender<RecordBatch>, Receiver<RecordBatch>) = mpsc::channel(100);

        // Since the main task/stream can be dropped or fail, we use a oneshot channel to signal that all data is received and we should commit the transaction
        let (notify_commit_transaction, on_commit_transaction) = tokio::sync::oneshot::channel();

        let schema = data.schema();

        let duckdb_write_handle: JoinHandle<datafusion::common::Result<u64>> =
            tokio::task::spawn_blocking(move || {
                let num_rows = match overwrite {
                    InsertOp::Overwrite => {
                        if write_settings.overwrite_file_swap && pool.mode() == Mode::File {
                            file_swap::insert_overwrite_swap(
                                pool,
                                &table_definition,
                                batch_rx,
                                on_conflict.as_ref(),
                                on_data_written.as_ref(),
                                on_commit_transaction,
                                schema,
                                &write_settings,
                            )?
                        } else {
                            if write_settings.overwrite_file_swap {
                                tracing::warn!(
                                    "The DuckDB overwrite file swap requires a file-backed instance; overwriting in place"
                                );
                            }
                            insert_overwrite(
                                pool,
                                &table_definition,
                                batch_rx,
                                on_conflict.as_ref(),
                                on_data_written.as_ref(),
                                on_commit_transaction,
                                schema,
                                &write_settings,
                            )?
                        }
                    }
                    InsertOp::Append | InsertOp::Replace => insert_append(
                        pool,
                        &table_definition,
                        batch_rx,
                        on_conflict.as_ref(),
                        on_data_written.as_ref(),
                        on_commit_transaction,
                        schema,
                        &write_settings,
                    )?,
                };

                Ok(num_rows)
            });

        while let Some(batch) = data.next().await {
            let batch = batch.map_err(check_and_mark_retriable_error)?;

            if let Some(constraints) = self.table_definition.constraints() {
                constraints::validate_batch_with_constraints(
                    vec![batch.clone()],
                    constraints,
                    &crate::util::constraints::UpsertOptions::default(),
                )
                .await
                .context(super::ConstraintViolationSnafu)
                .map_err(to_datafusion_error)?;
            }

            if let Err(send_error) = batch_tx.send(batch).await {
                match duckdb_write_handle.await {
                    Err(join_error) => {
                        return Err(DataFusionError::Execution(format!(
                            "Error writing to DuckDB: {join_error}"
                        )));
                    }
                    Ok(Err(datafusion_error)) => {
                        return Err(datafusion_error);
                    }
                    _ => {
                        return Err(DataFusionError::Execution(format!(
                            "Unable to send RecordBatch to DuckDB writer: {send_error}"
                        )))
                    }
                };
            }
        }

        if notify_commit_transaction.send(()).is_err() {
            return Err(DataFusionError::Execution(
                "Unable to send message to commit transaction to DuckDB writer.".to_string(),
            ));
        };

        // Drop the sender to signal the receiver that no more data is coming
        drop(batch_tx);

        match duckdb_write_handle.await {
            Ok(result) => result,
            Err(e) => Err(DataFusionError::Execution(format!(
                "Error writing to DuckDB: {e}"
            ))),
        }
    }
}

impl DuckDBDataSink {
    pub(crate) fn new(
        pool: Arc<DuckDbConnectionPool>,
        table_definition: Arc<TableDefinition>,
        overwrite: InsertOp,
        on_conflict: Option<OnConflict>,
        schema: SchemaRef,
    ) -> Self {
        Self {
            pool,
            table_definition,
            overwrite,
            on_conflict,
            schema,
            on_data_written: None,
            write_settings: DuckDBWriteSettings::default(),
        }
    }

    #[must_use]
    pub fn with_on_data_written_handler(mut self, handler: WriteCompletionHandler) -> Self {
        self.on_data_written = Some(handler);
        self
    }

    #[must_use]
    pub fn with_write_settings(mut self, write_settings: DuckDBWriteSettings) -> Self {
        self.write_settings = write_settings;
        self
    }
}

impl std::fmt::Debug for DuckDBDataSink {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "DuckDBDataSink")
    }
}

impl DisplayAs for DuckDBDataSink {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> std::fmt::Result {
        write!(f, "DuckDBDataSink")
    }
}

#[allow(clippy::too_many_arguments)]
fn insert_append(
    pool: Arc<DuckDbConnectionPool>,
    table_definition: &Arc<TableDefinition>,
    batch_rx: Receiver<RecordBatch>,
    on_conflict: Option<&OnConflict>,
    on_data_written: Option<&WriteCompletionHandler>,
    mut on_commit_transaction: tokio::sync::oneshot::Receiver<()>,
    schema: SchemaRef,
    write_settings: &DuckDBWriteSettings,
) -> datafusion::common::Result<u64> {
    // Writers hold the write gate in shared mode (acquired before the
    // connection checkout) so a database file swap can never interleave with
    // this write; see `DuckDbConnectionPool::write_gate`.
    let write_gate = pool.write_gate();
    let _write_guard = write_gate
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let mut db_conn = pool
        .connect_sync()
        .context(super::DbConnectionPoolSnafu)
        .map_err(to_retriable_data_write_error)?;

    let duckdb_conn = DuckDB::duckdb_conn(&mut db_conn).map_err(to_retriable_data_write_error)?;

    let tx = duckdb_conn
        .conn
        .transaction()
        .context(super::UnableToBeginTransactionSnafu)
        .map_err(to_retriable_data_write_error)?;

    // Check if there's an internal table from a previous Overwrite operation.
    // If so, we need to append to that internal table instead of the base table name
    // (which would be a view after Overwrite).
    let append_table = {
        let base_table_manager = TableManager::new(Arc::clone(table_definition))
            .with_internal(false)
            .map_err(to_retriable_data_write_error)?;

        // List internal tables to see if a previous Overwrite created a view structure
        let internal_tables = table_definition
            .list_internal_tables(&tx)
            .map_err(to_retriable_data_write_error)?;

        if let Some((latest_internal_table_name, _)) = internal_tables.last() {
            // There's an internal table from a previous Overwrite - use it for appending
            tracing::debug!(
                "Found internal table {} from previous overwrite, using it for append instead of base table {}",
                latest_internal_table_name,
                base_table_manager.table_name()
            );
            TableManager::from_table_name(
                Arc::clone(table_definition),
                latest_internal_table_name.clone(),
            )
        } else {
            // No internal tables - use the base table as normal
            base_table_manager
        }
    };

    let should_have_indexes = !append_table.indexes_vec().is_empty();
    let has_indexes = !append_table
        .current_indexes(&tx)
        .map_err(to_retriable_data_write_error)?
        .is_empty();
    let is_empty_table = append_table
        .get_row_count(&tx)
        .map_err(to_retriable_data_write_error)?
        == 0;
    let should_apply_indexes = should_have_indexes && !has_indexes && is_empty_table;

    let append_table_schema = append_table
        .current_schema(&tx)
        .map_err(to_retriable_data_write_error)?;

    if SCHEMA_EQUIVALENCE_ENABLED && !schema.equivalent_names_and_types(&append_table_schema) {
        return Err(DataFusionError::Execution(
            "Schema of the append table does not match the schema of the new append data."
                .to_string(),
        ));
    }

    tracing::debug!(
        "Append load for {table_name}",
        table_name = append_table.table_name()
    );
    let num_rows = write_to_table(
        &append_table,
        &tx,
        Arc::clone(&schema),
        batch_rx,
        on_conflict,
    )
    .map_err(to_retriable_data_write_error)?;

    if let Some(callback) = on_data_written {
        callback(&tx, &append_table, &schema, num_rows)?;
    }

    if write_settings.recompute_statistics_on_write {
        execute_analyze_sql(&tx, &append_table.table_name().to_string());
    }

    on_commit_transaction
        .try_recv()
        .map_err(to_retriable_data_write_error)?;

    tx.commit()
        .context(super::UnableToCommitTransactionSnafu)
        .map_err(to_retriable_data_write_error)?;

    let tx = duckdb_conn
        .conn
        .transaction()
        .context(super::UnableToBeginTransactionSnafu)
        .map_err(to_datafusion_error)?;

    // apply indexes if new table
    if should_apply_indexes {
        tracing::debug!(
            "Load for table {table_name} complete, applying constraints and indexes.",
            table_name = append_table.table_name()
        );

        append_table
            .create_indexes(&tx)
            .map_err(to_retriable_data_write_error)?;
    }

    let primary_keys_match = append_table
        .verify_primary_keys_match(&append_table, &tx)
        .map_err(to_retriable_data_write_error)?;
    let indexes_match = append_table
        .verify_indexes_match(&append_table, &tx)
        .map_err(to_retriable_data_write_error)?;

    if !primary_keys_match {
        return Err(DataFusionError::Execution(
            "Primary keys do not match between the new table and the existing table.\nEnsure primary key configuration is the same as the existing table, or manually migrate the table.".to_string(),
        ));
    }

    if !indexes_match {
        return Err(DataFusionError::Execution(
            "Indexes do not match between the new table and the existing table.\nEnsure index configuration is the same as the existing table, or manually migrate the table.".to_string(),
        ));
    }

    tx.commit()
        .context(super::UnableToCommitTransactionSnafu)
        .map_err(to_retriable_data_write_error)?;

    Ok(num_rows)
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
fn insert_overwrite(
    pool: Arc<DuckDbConnectionPool>,
    table_definition: &Arc<TableDefinition>,
    batch_rx: Receiver<RecordBatch>,
    on_conflict: Option<&OnConflict>,
    on_data_written: Option<&WriteCompletionHandler>,
    mut on_commit_transaction: tokio::sync::oneshot::Receiver<()>,
    schema: SchemaRef,
    write_settings: &DuckDBWriteSettings,
) -> datafusion::common::Result<u64> {
    // Writers hold the write gate in shared mode (acquired before the
    // connection checkout) so a database file swap can never interleave with
    // this write; see `DuckDbConnectionPool::write_gate`.
    let write_gate = pool.write_gate();
    let _write_guard = write_gate
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let cloned_pool = Arc::clone(&pool);
    let mut db_conn = pool
        .connect_sync()
        .context(super::DbConnectionPoolSnafu)
        .map_err(to_retriable_data_write_error)?;

    let duckdb_conn = DuckDB::duckdb_conn(&mut db_conn).map_err(to_retriable_data_write_error)?;

    let tx = duckdb_conn
        .conn
        .transaction()
        .context(super::UnableToBeginTransactionSnafu)
        .map_err(to_retriable_data_write_error)?;

    let new_table = TableManager::new(Arc::clone(table_definition))
        .with_internal(true)
        .map_err(to_retriable_data_write_error)?;

    new_table
        .create_table(cloned_pool, &tx)
        .map_err(to_retriable_data_write_error)?;

    let existing_tables = new_table
        .list_other_internal_tables(&tx)
        .map_err(to_retriable_data_write_error)?;
    let base_table = new_table
        .base_table(&tx)
        .map_err(to_retriable_data_write_error)?;
    let last_table = match (existing_tables.last(), base_table.as_ref()) {
        (Some(internal_table), Some(base_table)) => {
            return Err(DataFusionError::Execution(
                format!("Failed to insert data for DuckDB - both an internal table and definition base table were found.\nManual table migration is required - delete the table '{internal_table}' or '{base_table}' and try again.",
                internal_table = internal_table.0.table_name(),
                base_table = base_table.table_name())));
        }
        (Some((table, _)), None) | (None, Some(table)) => Some(table),
        (None, None) => None,
    };

    if let Some(last_table) = last_table {
        let should_have_indexes = !last_table.indexes_vec().is_empty();
        let has_indexes = !last_table
            .current_indexes(&tx)
            .map_err(to_retriable_data_write_error)?
            .is_empty();
        let is_empty_table = last_table
            .get_row_count(&tx)
            .map_err(to_retriable_data_write_error)?
            == 0;
        let should_apply_indexes = should_have_indexes && !has_indexes && is_empty_table;

        let last_table_schema = last_table
            .current_schema(&tx)
            .map_err(to_retriable_data_write_error)?;
        let new_table_schema = new_table
            .current_schema(&tx)
            .map_err(to_retriable_data_write_error)?;

        if SCHEMA_EQUIVALENCE_ENABLED
            && !new_table_schema.equivalent_names_and_types(&last_table_schema)
        {
            return Err(DataFusionError::Execution(
                "Schema does not match between the new table and the existing table.".to_string(),
            ));
        }

        if !should_apply_indexes {
            // compare indexes and primary keys
            let primary_keys_match = new_table
                .verify_primary_keys_match(last_table, &tx)
                .map_err(to_retriable_data_write_error)?;
            let indexes_match = new_table
                .verify_indexes_match(last_table, &tx)
                .map_err(to_retriable_data_write_error)?;

            if !primary_keys_match {
                return Err(DataFusionError::Execution(
                    "Primary keys do not match between the new table and the existing table.\nEnsure primary key configuration is the same as the existing table, or manually migrate the table."
                        .to_string(),
                ));
            }

            if !indexes_match {
                return Err(DataFusionError::Execution(
                    "Indexes do not match between the new table and the existing table.\nEnsure index configuration is the same as the existing table, or manually migrate the table.".to_string(),
                ));
            }
        }
    }

    tracing::debug!("Initial load for {}", new_table.table_name());
    let num_rows = write_to_table(&new_table, &tx, Arc::clone(&schema), batch_rx, on_conflict)
        .map_err(to_retriable_data_write_error)?;

    on_commit_transaction
        .try_recv()
        .map_err(to_retriable_data_write_error)?;

    if let Some(base_table) = base_table {
        base_table
            .delete_table(&tx)
            .map_err(to_retriable_data_write_error)?;
    }

    new_table
        .create_view(&tx)
        .map_err(to_retriable_data_write_error)?;

    if let Some(callback) = on_data_written {
        callback(&tx, &new_table, &schema, num_rows)?;
    }

    if write_settings.recompute_statistics_on_write {
        execute_analyze_sql(&tx, &new_table.table_name().to_string());
    }

    tx.commit()
        .context(super::UnableToCommitTransactionSnafu)
        .map_err(to_retriable_data_write_error)?;

    tracing::debug!(
        "Load for table {table_name} complete, applying constraints and indexes.",
        table_name = new_table.table_name()
    );

    let tx = duckdb_conn
        .conn
        .transaction()
        .context(super::UnableToBeginTransactionSnafu)
        .map_err(to_datafusion_error)?;

    for (table, _) in existing_tables {
        table
            .delete_table(&tx)
            .map_err(to_retriable_data_write_error)?;
    }

    // Apply constraints and indexes.
    new_table
        .create_indexes(&tx)
        .map_err(to_retriable_data_write_error)?;

    tx.commit()
        .context(super::UnableToCommitTransactionSnafu)
        .map_err(to_retriable_data_write_error)?;

    if write_settings.checkpoint_on_write {
        checkpoint_after_write(duckdb_conn, table_definition.name());
    }

    Ok(num_rows)
}

/// Checkpoints the database after an overwrite completes.
///
/// A plain `CHECKPOINT` is tried first: it fails fast if other transactions
/// hold the shared checkpoint lock and is free of side effects when the
/// database is quiet. Only on failure does it escalate to `FORCE CHECKPOINT`,
/// which waits for active transactions to finish while blocking new ones from
/// starting — a stall bounded by the slowest in-flight query plus the
/// checkpoint write itself, paid only when
/// actually needed. Failures never fail the write: the data is already
/// durably committed.
fn checkpoint_after_write(duckdb_conn: &mut DuckDbConnection, table_name: &RelationName) {
    // Error messages can span lines; keep log records single-line.
    fn single_line(e: &duckdb::Error) -> String {
        e.to_string().replace('\n', " ")
    }

    let start = std::time::Instant::now();
    let force = match duckdb_conn.conn.execute_batch("CHECKPOINT") {
        Ok(()) => false,
        Err(checkpoint_err) => {
            tracing::debug!(
                "CHECKPOINT after overwrite of {table_name} could not run ({}); escalating to FORCE CHECKPOINT",
                single_line(&checkpoint_err)
            );
            // A forced checkpoint waits for in-flight transactions, so its
            // duration approximates how long other queries on this database
            // were stalled behind the checkpoint.
            match duckdb_conn.conn.execute_batch("FORCE CHECKPOINT") {
                Ok(()) => true,
                Err(force_err) => {
                    tracing::warn!(
                        duration_ms = start.elapsed().as_millis() as u64,
                        "Failed to checkpoint DuckDB after overwrite of {table_name}: {}",
                        single_line(&force_err),
                    );
                    return;
                }
            }
        }
    };
    tracing::debug!(
        force,
        duration_ms = start.elapsed().as_millis() as u64,
        "Checkpoint after overwrite of {table_name} complete"
    );
}

/// Writes every batch of `data_batches` into `table` on `tx`.
///
/// An upsert (`ON CONFLICT … DO UPDATE`) is written as one `INSERT` statement
/// per run of rows that repeats no key, so that the last copy of a repeated key
/// is the one kept; see [`upsert_groups`]. Every other write is one statement.
pub(super) fn write_to_table(
    table: &TableManager,
    tx: &Transaction<'_>,
    schema: SchemaRef,
    data_batches: Receiver<RecordBatch>,
    on_conflict: Option<&OnConflict>,
) -> datafusion::common::Result<u64> {
    if let Some(OnConflict::Upsert(target)) = on_conflict {
        return upsert_groups::write_statements(data_batches, &schema, target.iter(), |reader| {
            insert_from_reader(table, tx, reader, on_conflict)
        });
    }
    insert_from_reader(
        table,
        tx,
        Box::new(RecordBatchReaderFromStream::new(data_batches, schema)),
        on_conflict,
    )
}

/// One `INSERT INTO table SELECT * FROM <arrow scan of reader>` statement.
fn insert_from_reader(
    table: &TableManager,
    tx: &Transaction<'_>,
    reader: Box<dyn RecordBatchReader + Send>,
    on_conflict: Option<&OnConflict>,
) -> datafusion::common::Result<u64> {
    let stream = FFI_ArrowArrayStream::new(reader);

    let current_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context(super::UnableToGetSystemTimeSnafu)
        .map_err(to_datafusion_error)?
        .as_millis();

    let view_name = format!("__scan_{}_{current_ts}", table.table_name());
    tx.register_arrow_scan_view(&view_name, &stream)
        .context(super::UnableToRegisterArrowScanViewSnafu)
        .map_err(to_datafusion_error)?;

    let view = ViewCreator::from_name(RelationName::new(view_name));
    let rows = view
        .insert_into(table, tx, on_conflict)
        .map_err(to_datafusion_error)?;
    view.drop(tx).map_err(to_datafusion_error)?;

    Ok(rows as u64)
}

/// Executes an ANALYZE statement to update query optimizer statistics.
/// This helps DuckDB's query planner generate better execution plans, especially after large data changes.
/// https://duckdb.org/docs/stable/sql/statements/analyze
///
/// Errors are logged but do not fail the operation since statistics updates are non-critical.
pub fn execute_analyze_sql(tx: &Transaction, table_name: &str) {
    // DuckDB doesn't support parameterized table names in the ANALYZE statement, so the name is
    // quoted as an identifier instead.
    let analyze_sql = format!("ANALYZE {}", expr::quoted_identifier(table_name));
    tracing::debug!("Executing analyze SQL: {analyze_sql}");
    match tx.prepare(&analyze_sql) {
        Ok(mut stmt) => match stmt.execute([]) {
            Ok(_) => {
                tracing::debug!("Analyze SQL executed for table '{table_name}'");
            }
            Err(e) => {
                tracing::error!("Failed to execute analyze SQL for table '{table_name}': {e}");
            }
        },
        Err(e) => {
            tracing::error!("Failed to prepare analyze SQL for table '{table_name}': {e}");
        }
    }
}

struct RecordBatchReaderFromStream {
    stream: Receiver<RecordBatch>,
    schema: SchemaRef,
}

impl RecordBatchReaderFromStream {
    fn new(stream: Receiver<RecordBatch>, schema: SchemaRef) -> Self {
        Self { stream, schema }
    }
}

impl Iterator for RecordBatchReaderFromStream {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.stream.blocking_recv().map(Ok)
    }
}

impl RecordBatchReader for RecordBatchReaderFromStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod test {
    use arrow::array::{Int64Array, StringArray};
    use datafusion::physical_plan::memory::MemoryStream;

    use super::*;
    use crate::{
        duckdb::creator::tests::{get_basic_table_definition, get_mem_duckdb, init_tracing},
        util::{column_reference::ColumnReference, indexes::IndexType},
    };

    #[tokio::test]
    async fn test_write_to_table_overwrite_without_previous_table() {
        // Test scenario: Write to a table with overwrite mode without a previous table
        // Expected behavior: Data sink creates a new internal table, writes data to it, and creates a view with the table definition name

        let _guard = init_tracing(None);
        let pool = get_mem_duckdb();

        let table_definition = get_basic_table_definition();

        let duckdb_sink = DuckDBDataSink::new(
            Arc::clone(&pool),
            Arc::clone(&table_definition),
            InsertOp::Overwrite,
            None,
            table_definition.schema(),
        );
        let data_sink: Arc<dyn DataSink> = Arc::new(duckdb_sink);

        // id, name
        // 1, "a"
        // 2, "b"
        let batches = vec![RecordBatch::try_new(
            Arc::clone(&table_definition.schema()),
            vec![
                Arc::new(Int64Array::from(vec![Some(1), Some(2)])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
            ],
        )
        .expect("should create a record batch")];

        let stream = Box::pin(
            MemoryStream::try_new(batches, table_definition.schema(), None).expect("to get stream"),
        );

        data_sink
            .write_all(stream, &Arc::new(TaskContext::default()))
            .await
            .expect("to write all");

        let mut conn = pool.connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        let tx = duckdb.conn.transaction().expect("to begin transaction");
        let mut internal_tables = table_definition
            .list_internal_tables(&tx)
            .expect("to list internal tables");
        assert_eq!(internal_tables.len(), 1);

        let table_name = internal_tables.pop().expect("should have a table").0;

        let rows = tx
            .query_row(&format!("SELECT COUNT(1) FROM {table_name}"), [], |row| {
                row.get::<_, i64>(0)
            })
            .expect("to get count");
        assert_eq!(rows, 2);

        // expect a view to be created with the table definition name
        let view_rows = tx
            .query_row(
                &format!(
                    "SELECT COUNT(1) FROM {view_name}",
                    view_name = table_definition.name()
                ),
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("to get count");

        assert_eq!(view_rows, 2);

        tx.rollback().expect("to rollback");
    }

    #[tokio::test]
    async fn test_write_to_table_overwrite_with_previous_base_table() {
        // Test scenario: Write to a table with overwrite mode with a previous base table
        // Expected behavior: Data sink creates a new internal table, writes data to it.
        // Before creating the view, the base table needs to get dropped as we need to create a view with the same name.

        let _guard = init_tracing(None);
        let pool = get_mem_duckdb();

        let table_definition = get_basic_table_definition();

        let cloned_pool = Arc::clone(&pool);
        let mut conn = cloned_pool.connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        let tx = duckdb.conn.transaction().expect("to begin transaction");

        // make an existing table to overwrite
        let overwrite_table = TableManager::new(Arc::clone(&table_definition))
            .with_internal(false)
            .expect("to create table");

        overwrite_table
            .create_table(Arc::clone(&pool), &tx)
            .expect("to create table");

        tx.execute(
            &format!(
                "INSERT INTO {table_name} VALUES (3, 'c')",
                table_name = overwrite_table.table_name()
            ),
            [],
        )
        .expect("to insert");

        tx.commit().expect("to commit");

        let duckdb_sink = DuckDBDataSink::new(
            Arc::clone(&pool),
            Arc::clone(&table_definition),
            InsertOp::Overwrite,
            None,
            table_definition.schema(),
        );
        let data_sink: Arc<dyn DataSink> = Arc::new(duckdb_sink);

        // id, name
        // 1, "a"
        // 2, "b"
        let batches = vec![RecordBatch::try_new(
            Arc::clone(&table_definition.schema()),
            vec![
                Arc::new(Int64Array::from(vec![Some(1), Some(2)])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
            ],
        )
        .expect("should create a record batch")];

        let stream = Box::pin(
            MemoryStream::try_new(batches, table_definition.schema(), None).expect("to get stream"),
        );

        data_sink
            .write_all(stream, &Arc::new(TaskContext::default()))
            .await
            .expect("to write all");

        let mut conn = pool.connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        let tx = duckdb.conn.transaction().expect("to begin transaction");
        let mut internal_tables = table_definition
            .list_internal_tables(&tx)
            .expect("to list internal tables");
        assert_eq!(internal_tables.len(), 1);

        let table_name = internal_tables.pop().expect("should have a table").0;

        let rows = tx
            .query_row(&format!("SELECT COUNT(1) FROM {table_name}"), [], |row| {
                row.get::<_, i64>(0)
            })
            .expect("to get count");
        assert_eq!(rows, 2);

        let table_creator =
            TableManager::from_table_name(Arc::clone(&table_definition), table_name);
        let base_table = table_creator.base_table(&tx).expect("to get base table");

        assert!(base_table.is_none()); // base table should get deleted

        // expect a view to be created with the table definition name
        let view_rows = tx
            .query_row(
                &format!(
                    "SELECT COUNT(1) FROM {view_name}",
                    view_name = table_definition.name()
                ),
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("to get count");

        assert_eq!(view_rows, 2);

        tx.rollback().expect("to rollback");
    }

    #[tokio::test]
    async fn test_write_to_table_overwrite_with_previous_internal_table() {
        // Test scenario: Write to a table with overwrite mode with a previous base table
        // Expected behavior: Data sink creates a new internal table, writes data to it.
        // Before creating the view, the base table needs to get dropped as we need to create a view with the same name.

        let _guard = init_tracing(None);
        let pool = get_mem_duckdb();

        let table_definition = get_basic_table_definition();

        let cloned_pool = Arc::clone(&pool);
        let mut conn = cloned_pool.connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        let tx = duckdb.conn.transaction().expect("to begin transaction");

        // make an existing table to overwrite
        let overwrite_table = TableManager::new(Arc::clone(&table_definition))
            .with_internal(true)
            .expect("to create table");

        overwrite_table
            .create_table(Arc::clone(&pool), &tx)
            .expect("to create table");

        tx.execute(
            &format!(
                "INSERT INTO {table_name} VALUES (3, 'c')",
                table_name = overwrite_table.table_name()
            ),
            [],
        )
        .expect("to insert");

        tx.commit().expect("to commit");

        let duckdb_sink = DuckDBDataSink::new(
            Arc::clone(&pool),
            Arc::clone(&table_definition),
            InsertOp::Overwrite,
            None,
            table_definition.schema(),
        );
        let data_sink: Arc<dyn DataSink> = Arc::new(duckdb_sink);

        // id, name
        // 1, "a"
        // 2, "b"
        let batches = vec![RecordBatch::try_new(
            Arc::clone(&table_definition.schema()),
            vec![
                Arc::new(Int64Array::from(vec![Some(1), Some(2)])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
            ],
        )
        .expect("should create a record batch")];

        let stream = Box::pin(
            MemoryStream::try_new(batches, table_definition.schema(), None).expect("to get stream"),
        );

        data_sink
            .write_all(stream, &Arc::new(TaskContext::default()))
            .await
            .expect("to write all");

        let mut conn = pool.connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        let tx = duckdb.conn.transaction().expect("to begin transaction");
        let mut internal_tables = table_definition
            .list_internal_tables(&tx)
            .expect("to list internal tables");
        assert_eq!(internal_tables.len(), 1);

        let table_name = internal_tables.pop().expect("should have a table").0;

        let rows = tx
            .query_row(&format!("SELECT COUNT(1) FROM {table_name}"), [], |row| {
                row.get::<_, i64>(0)
            })
            .expect("to get count");
        assert_eq!(rows, 2);

        // expect a view to be created with the table definition name
        let view_rows = tx
            .query_row(
                &format!(
                    "SELECT COUNT(1) FROM {view_name}",
                    view_name = table_definition.name()
                ),
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("to get count");

        assert_eq!(view_rows, 2);

        tx.rollback().expect("to rollback");
    }

    #[tokio::test]
    async fn test_write_to_table_append_after_overwrite() {
        // Test scenario: Write with Overwrite, then write with Append
        // This tests the fix for the issue where Append fails after Overwrite because
        // Overwrite creates a view pointing to an internal table, and Append was trying
        // to insert into the view instead of the internal table.
        // Expected behavior: Append should detect the internal table and insert into it.

        let _guard = init_tracing(None);
        let pool = get_mem_duckdb();

        let table_definition = get_basic_table_definition();

        // Step 1: Do an Overwrite operation (creates internal table + view)
        let duckdb_sink = DuckDBDataSink::new(
            Arc::clone(&pool),
            Arc::clone(&table_definition),
            InsertOp::Overwrite,
            None,
            table_definition.schema(),
        );
        let data_sink: Arc<dyn DataSink> = Arc::new(duckdb_sink);

        let batches = vec![RecordBatch::try_new(
            Arc::clone(&table_definition.schema()),
            vec![
                Arc::new(Int64Array::from(vec![Some(1), Some(2)])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
            ],
        )
        .expect("should create a record batch")];

        let stream = Box::pin(
            MemoryStream::try_new(batches, table_definition.schema(), None).expect("to get stream"),
        );

        data_sink
            .write_all(stream, &Arc::new(TaskContext::default()))
            .await
            .expect("to write all with overwrite");

        // Verify Overwrite created an internal table and view
        {
            let mut conn = Arc::clone(&pool).connect_sync().expect("to connect");
            let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
            let tx = duckdb.conn.transaction().expect("to begin transaction");
            let internal_tables = table_definition
                .list_internal_tables(&tx)
                .expect("to list internal tables");
            assert_eq!(
                internal_tables.len(),
                1,
                "Overwrite should create 1 internal table"
            );

            let view_rows: i64 = tx
                .query_row(
                    &format!("SELECT COUNT(1) FROM {}", table_definition.name()),
                    [],
                    |row| row.get(0),
                )
                .expect("to query view");
            assert_eq!(view_rows, 2, "View should have 2 rows after overwrite");
            tx.rollback().expect("to rollback");
        }

        // Step 2: Do an Append operation (should append to the internal table, not fail)
        let duckdb_sink_append = DuckDBDataSink::new(
            Arc::clone(&pool),
            Arc::clone(&table_definition),
            InsertOp::Append,
            None,
            table_definition.schema(),
        );
        let data_sink_append: Arc<dyn DataSink> = Arc::new(duckdb_sink_append);

        let batches_append = vec![RecordBatch::try_new(
            Arc::clone(&table_definition.schema()),
            vec![
                Arc::new(Int64Array::from(vec![Some(3), Some(4)])),
                Arc::new(StringArray::from(vec![Some("c"), Some("d")])),
            ],
        )
        .expect("should create a record batch")];

        let stream_append = Box::pin(
            MemoryStream::try_new(batches_append, table_definition.schema(), None)
                .expect("to get stream"),
        );

        // This should NOT fail with "is not a table" error
        data_sink_append
            .write_all(stream_append, &Arc::new(TaskContext::default()))
            .await
            .expect("to write all with append after overwrite");

        // Verify the append worked - should now have 4 rows total
        let mut conn = pool.connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        let tx = duckdb.conn.transaction().expect("to begin transaction");

        // Still should have just 1 internal table
        let internal_tables = table_definition
            .list_internal_tables(&tx)
            .expect("to list internal tables");
        assert_eq!(
            internal_tables.len(),
            1,
            "Should still have 1 internal table after append"
        );

        // Query through the view - should see all 4 rows
        let total_rows: i64 = tx
            .query_row(
                &format!("SELECT COUNT(1) FROM {}", table_definition.name()),
                [],
                |row| row.get(0),
            )
            .expect("to query view");
        assert_eq!(total_rows, 4, "View should have 4 rows after append");

        // Query the internal table directly - should also have 4 rows
        let (internal_table_name, _) = internal_tables.first().expect("should have internal table");
        let internal_rows: i64 = tx
            .query_row(
                &format!("SELECT COUNT(1) FROM {internal_table_name}"),
                [],
                |row| row.get(0),
            )
            .expect("to query internal table");
        assert_eq!(internal_rows, 4, "Internal table should have 4 rows");

        tx.rollback().expect("to rollback");
    }

    #[tokio::test]
    async fn test_write_to_table_append_with_previous_table() {
        // Test scenario: Write to a table with append mode with a previous table
        // Expected behavior: Data sink appends data to the existing table. No new internal table should be created.
        // The existing table is re-used.

        let _guard = init_tracing(None);
        let pool = get_mem_duckdb();

        let cloned_pool = Arc::clone(&pool);
        let mut conn = cloned_pool.connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        let tx = duckdb.conn.transaction().expect("to begin transaction");

        let table_definition = get_basic_table_definition();

        // make an existing table to append from
        let append_table = TableManager::new(Arc::clone(&table_definition))
            .with_internal(false)
            .expect("to create table");

        append_table
            .create_table(Arc::clone(&pool), &tx)
            .expect("to create table");

        tx.execute(
            &format!(
                "INSERT INTO {table_name} VALUES (3, 'c')",
                table_name = append_table.table_name()
            ),
            [],
        )
        .expect("to insert");

        tx.commit().expect("to commit");

        let duckdb_sink = DuckDBDataSink::new(
            Arc::clone(&pool),
            Arc::clone(&table_definition),
            InsertOp::Append,
            None,
            table_definition.schema(),
        );
        let data_sink: Arc<dyn DataSink> = Arc::new(duckdb_sink);

        // id, name
        // 1, "a"
        // 2, "b"
        let batches = vec![RecordBatch::try_new(
            Arc::clone(&table_definition.schema()),
            vec![
                Arc::new(Int64Array::from(vec![Some(1), Some(2)])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
            ],
        )
        .expect("should create a record batch")];

        let stream = Box::pin(
            MemoryStream::try_new(batches, table_definition.schema(), None).expect("to get stream"),
        );

        data_sink
            .write_all(stream, &Arc::new(TaskContext::default()))
            .await
            .expect("to write all");

        let tx = duckdb.conn.transaction().expect("to begin transaction");

        let internal_tables = table_definition
            .list_internal_tables(&tx)
            .expect("to list internal tables");
        assert_eq!(internal_tables.len(), 0);

        let base_table = append_table
            .base_table(&tx)
            .expect("to get base table")
            .expect("should have a base table");

        let rows = tx
            .query_row(
                &format!(
                    "SELECT COUNT(1) FROM {table_name}",
                    table_name = base_table.table_name()
                ),
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("to get count");
        assert_eq!(rows, 3);

        tx.rollback().expect("to rollback");
    }

    #[tokio::test]
    async fn test_write_to_table_append_with_previous_table_needs_indexes() {
        // Test scenario: Write to a table with append mode with a previous table
        // Expected behavior: Data sink appends data to the existing table. No new internal table should be created.
        // The existing table is re-used.

        let _guard = init_tracing(None);
        let pool = get_mem_duckdb();

        let cloned_pool = Arc::clone(&pool);
        let mut conn = cloned_pool.connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        let tx = duckdb.conn.transaction().expect("to begin transaction");

        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
        ]));

        let table_definition = Arc::new(
            TableDefinition::new(RelationName::new("test_table"), Arc::clone(&schema))
                .with_indexes(
                    vec![(
                        ColumnReference::try_from("id").expect("valid column ref"),
                        IndexType::Enabled,
                    )]
                    .into_iter()
                    .collect(),
                ),
        );

        // make an existing table to append from
        let append_table = TableManager::new(Arc::clone(&table_definition))
            .with_internal(false)
            .expect("to create table");

        append_table
            .create_table(Arc::clone(&pool), &tx)
            .expect("to create table");

        // don't apply indexes, and leave the table empty to simulate a new table from TableProviderFactory::create()

        tx.commit().expect("to commit");

        let duckdb_sink = DuckDBDataSink::new(
            Arc::clone(&pool),
            Arc::clone(&table_definition),
            InsertOp::Append,
            None,
            table_definition.schema(),
        );
        let data_sink: Arc<dyn DataSink> = Arc::new(duckdb_sink);

        // id, name
        // 1, "a"
        // 2, "b"
        let batches = vec![RecordBatch::try_new(
            Arc::clone(&table_definition.schema()),
            vec![
                Arc::new(Int64Array::from(vec![Some(1), Some(2)])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
            ],
        )
        .expect("should create a record batch")];

        let stream = Box::pin(
            MemoryStream::try_new(batches, table_definition.schema(), None).expect("to get stream"),
        );

        data_sink
            .write_all(stream, &Arc::new(TaskContext::default()))
            .await
            .expect("to write all");

        let tx = duckdb.conn.transaction().expect("to begin transaction");

        let internal_tables = table_definition
            .list_internal_tables(&tx)
            .expect("to list internal tables");
        assert_eq!(internal_tables.len(), 0);

        let base_table = append_table
            .base_table(&tx)
            .expect("to get base table")
            .expect("should have a base table");

        let rows = tx
            .query_row(
                &format!(
                    "SELECT COUNT(1) FROM {table_name}",
                    table_name = base_table.table_name()
                ),
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("to get count");
        assert_eq!(rows, 2);

        // at this point, indexes should be applied
        let indexes = append_table.current_indexes(&tx).expect("to get indexes");
        assert_eq!(indexes.len(), 1);

        tx.rollback().expect("to rollback");
    }

    #[tokio::test]
    async fn test_write_completion_custom_handler() {
        // Test scenario: Verify that the on_data_written callback signature is correct and callback is called during write_all
        // Expected behavior: The callback is executed, callback can delete data as part of append operation

        //Custom callback that removes records with id > 1
        let callback: WriteCompletionHandler = Arc::new(
            |tx: &Transaction<'_>,
             table_manager: &TableManager,
             schema: &SchemaRef,
             num_rows: u64| {
                assert!(!schema.fields().is_empty(), "Schema should have fields");
                assert_eq!(table_manager.table_name().to_string(), "test_table");
                assert_eq!(num_rows, 2, "Should have written 2 rows");
                let table_name = table_manager.table_name();
                tx.execute(&format!("DELETE FROM {table_name} WHERE id > 1"), [])
                    .map_err(|e| {
                        DataFusionError::Execution(format!(
                            "Failed to delete rows with id > 1: {e}"
                        ))
                    })?;
                Ok(())
            },
        );

        let _guard = init_tracing(None);
        let pool = get_mem_duckdb();

        let cloned_pool = Arc::clone(&pool);
        let mut conn = cloned_pool.connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        let tx = duckdb.conn.transaction().expect("to begin transaction");

        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
        ]));

        let table_definition = Arc::new(
            TableDefinition::new(RelationName::new("test_table"), Arc::clone(&schema))
                .with_indexes(
                    vec![(
                        ColumnReference::try_from("id").expect("valid column ref"),
                        IndexType::Enabled,
                    )]
                    .into_iter()
                    .collect(),
                ),
        );

        // make an existing table to append from
        let append_table = TableManager::new(Arc::clone(&table_definition))
            .with_internal(false)
            .expect("to create table");

        append_table
            .create_table(Arc::clone(&pool), &tx)
            .expect("to create table");

        // don't apply indexes, and leave the table empty to simulate a new table from TableProviderFactory::create()

        tx.commit().expect("to commit");

        let duckdb_sink = DuckDBDataSink::new(
            Arc::clone(&pool),
            Arc::clone(&table_definition),
            InsertOp::Append,
            None,
            table_definition.schema(),
        )
        .with_on_data_written_handler(callback);
        let data_sink: Arc<dyn DataSink> = Arc::new(duckdb_sink);

        // id, name
        // 1, "a"
        // 2, "b"
        let batches = vec![RecordBatch::try_new(
            Arc::clone(&table_definition.schema()),
            vec![
                Arc::new(Int64Array::from(vec![Some(1), Some(2)])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
            ],
        )
        .expect("should create a record batch")];

        let stream = Box::pin(
            MemoryStream::try_new(batches, table_definition.schema(), None).expect("to get stream"),
        );

        data_sink
            .write_all(stream, &Arc::new(TaskContext::default()))
            .await
            .expect("to write all");

        let tx = duckdb.conn.transaction().expect("to begin transaction");

        let base_table = append_table
            .base_table(&tx)
            .expect("to get base table")
            .expect("should have a base table");

        let rows = tx
            .query_row(
                &format!(
                    "SELECT COUNT(1) FROM {table_name}",
                    table_name = base_table.table_name()
                ),
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("to get count");

        // Must be single row with id = 1 only
        assert_eq!(rows, 1);

        tx.rollback().expect("to rollback");
    }

    /// Helper: set up a DuckDB base table with columns (id: Int64, name: Utf8),
    /// insert the given rows, and return `(DuckDBTableWriter, pool)`.
    async fn setup_writer_with_data(
        ids: Vec<i64>,
        names: Vec<&str>,
    ) -> (DuckDBTableWriter, Arc<DuckDbConnectionPool>) {
        let pool = get_mem_duckdb();
        let table_definition = get_basic_table_definition();

        // Create a base (non-internal) table so DELETE/UPDATE can target it by name.
        {
            let mut conn = Arc::clone(&pool).connect_sync().expect("to connect");
            let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
            let tx = duckdb.conn.transaction().expect("to begin transaction");

            let table_manager = TableManager::new(Arc::clone(&table_definition))
                .with_internal(false)
                .expect("to create table manager");

            table_manager
                .create_table(Arc::clone(&pool), &tx)
                .expect("to create table");

            tx.commit().expect("to commit");
        }

        // Insert seed data via DuckDBDataSink (Append mode reuses the base table).
        let schema = table_definition.schema();
        let duckdb_sink = DuckDBDataSink::new(
            Arc::clone(&pool),
            Arc::clone(&table_definition),
            InsertOp::Append,
            None,
            Arc::clone(&schema),
        );
        let batches = vec![RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .expect("should create a record batch")];
        let stream = Box::pin(
            MemoryStream::try_new(batches, Arc::clone(&schema), None).expect("to get stream"),
        );
        Arc::new(duckdb_sink)
            .write_all(stream, &Arc::new(TaskContext::default()))
            .await
            .expect("to write all");

        // Build a DuckDBTableWriter. delete_from / update only need pool +
        // table_definition; the read_provider is unused for DML but required
        // by the builder. Use a datafusion MemTable as a lightweight stand-in.
        let mem_table: Arc<dyn TableProvider> = Arc::new(
            datafusion::datasource::MemTable::try_new(schema, vec![vec![]])
                .expect("to create mem table"),
        );
        let writer = DuckDBTableWriterBuilder::new()
            .with_read_provider(mem_table)
            .with_pool(Arc::clone(&pool))
            .with_table_definition((*table_definition).clone())
            .build()
            .expect("to build writer");

        (writer, pool)
    }

    /// Helper: execute a DML plan and return the count.
    async fn extract_count(plan: Arc<dyn ExecutionPlan>) -> u64 {
        let batches = datafusion::physical_plan::collect(plan, Arc::new(TaskContext::default()))
            .await
            .expect("collect should succeed");
        assert_eq!(batches.len(), 1, "expected exactly one batch");
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .expect("count column should be UInt64Array")
            .value(0)
    }

    /// Helper: query all rows from the test_table ordered by id and return
    /// `(ids, names)`.
    fn query_all_rows(pool: &Arc<DuckDbConnectionPool>) -> (Vec<i64>, Vec<String>) {
        let mut conn = Arc::clone(pool).connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        let mut stmt = duckdb
            .conn
            .prepare("SELECT id, name FROM test_table ORDER BY id")
            .expect("to prepare statement");
        let mut rows = stmt.query([]).expect("to query");
        let mut ids = Vec::new();
        let mut names = Vec::new();
        while let Some(row) = rows.next().expect("to get next row") {
            ids.push(row.get::<_, i64>(0).expect("to get id"));
            names.push(row.get::<_, String>(1).expect("to get name"));
        }
        (ids, names)
    }

    #[tokio::test]
    async fn test_delete_from_with_filter() {
        let _guard = init_tracing(None);
        let (writer, pool) = setup_writer_with_data(vec![1, 2, 3], vec!["a", "b", "c"]).await;

        let ctx = datafusion::prelude::SessionContext::new();

        // DELETE WHERE id = 2
        let filters =
            vec![datafusion::logical_expr::col("id").eq(datafusion::logical_expr::lit(2i64))];
        let plan = writer
            .delete_from(&ctx.state(), filters)
            .await
            .expect("delete_from should succeed");

        let count = extract_count(plan).await;
        assert_eq!(count, 1, "should have deleted exactly 1 row");

        // Verify remaining rows
        let (ids, names) = query_all_rows(&pool);
        assert_eq!(ids.len(), 2, "should have 2 rows remaining");
        assert_eq!(ids, vec![1, 3]);
        assert_eq!(names, vec!["a", "c"]);
    }

    #[tokio::test]
    async fn test_delete_from_empty_filters() {
        let _guard = init_tracing(None);
        let (writer, pool) = setup_writer_with_data(vec![1, 2, 3], vec!["a", "b", "c"]).await;

        let ctx = datafusion::prelude::SessionContext::new();

        // DELETE with empty filters should delete all rows
        let plan = writer
            .delete_from(&ctx.state(), vec![])
            .await
            .expect("delete_from should succeed");

        let count = extract_count(plan).await;
        assert_eq!(
            count, 3,
            "should have deleted all 3 rows with empty filters"
        );

        // Verify no rows remain
        let (ids, _names) = query_all_rows(&pool);
        assert_eq!(ids.len(), 0, "no rows should remain");
    }

    #[tokio::test]
    async fn test_update_with_filter() {
        let _guard = init_tracing(None);
        let (writer, pool) = setup_writer_with_data(vec![1, 2, 3], vec!["a", "b", "c"]).await;

        let ctx = datafusion::prelude::SessionContext::new();

        // UPDATE SET name = 'updated' WHERE id = 2
        let assignments = vec![("name".to_string(), datafusion::logical_expr::lit("updated"))];
        let filters =
            vec![datafusion::logical_expr::col("id").eq(datafusion::logical_expr::lit(2i64))];
        let plan = writer
            .update(&ctx.state(), assignments, filters)
            .await
            .expect("update should succeed");

        let count = extract_count(plan).await;
        assert_eq!(count, 1, "should have updated exactly 1 row");

        // Verify data
        let (ids, names) = query_all_rows(&pool);
        assert_eq!(ids.len(), 3, "should still have 3 rows");
        for (id, name) in ids.iter().zip(names.iter()) {
            match *id {
                1 => assert_eq!(name, "a"),
                2 => assert_eq!(name, "updated"),
                3 => assert_eq!(name, "c"),
                other => panic!("unexpected id {other}"),
            }
        }
    }

    #[tokio::test]
    async fn test_update_without_filter() {
        let _guard = init_tracing(None);
        let (writer, pool) = setup_writer_with_data(vec![1, 2], vec!["a", "b"]).await;

        let ctx = datafusion::prelude::SessionContext::new();

        // UPDATE SET name = 'all' (no filter -> all rows)
        let assignments = vec![("name".to_string(), datafusion::logical_expr::lit("all"))];
        let plan = writer
            .update(&ctx.state(), assignments, vec![])
            .await
            .expect("update should succeed");

        let count = extract_count(plan).await;
        assert_eq!(count, 2, "should have updated 2 rows");

        // Verify all rows have name = "all"
        let (ids, names) = query_all_rows(&pool);
        assert_eq!(ids.len(), 2, "should still have 2 rows");
        for name in &names {
            assert_eq!(name, "all");
        }
    }

    /// Helper: set up a DuckDB table via Overwrite (which creates a view over
    /// an internal `__data_*` table), then return `(DuckDBTableWriter, pool)`.
    async fn setup_writer_with_overwrite_data(
        ids: Vec<i64>,
        names: Vec<&str>,
    ) -> (DuckDBTableWriter, Arc<DuckDbConnectionPool>) {
        let pool = get_mem_duckdb();
        let table_definition = get_basic_table_definition();

        // Insert seed data via DuckDBDataSink with Overwrite mode.
        // This creates an internal __data_* table and a view with the table definition name.
        let schema = table_definition.schema();
        let duckdb_sink = DuckDBDataSink::new(
            Arc::clone(&pool),
            Arc::clone(&table_definition),
            InsertOp::Overwrite,
            None,
            Arc::clone(&schema),
        );
        let batches = vec![RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .expect("should create a record batch")];
        let stream = Box::pin(
            MemoryStream::try_new(batches, Arc::clone(&schema), None).expect("to get stream"),
        );
        Arc::new(duckdb_sink)
            .write_all(stream, &Arc::new(TaskContext::default()))
            .await
            .expect("to write all");

        let mem_table: Arc<dyn TableProvider> = Arc::new(
            datafusion::datasource::MemTable::try_new(schema, vec![vec![]])
                .expect("to create mem table"),
        );
        let writer = DuckDBTableWriterBuilder::new()
            .with_read_provider(mem_table)
            .with_pool(Arc::clone(&pool))
            .with_table_definition((*table_definition).clone())
            .build()
            .expect("to build writer");

        (writer, pool)
    }

    #[tokio::test]
    async fn test_delete_from_view_backed_table_with_filter() {
        let _guard = init_tracing(None);
        let (writer, pool) =
            setup_writer_with_overwrite_data(vec![1, 2, 3], vec!["a", "b", "c"]).await;

        let ctx = datafusion::prelude::SessionContext::new();

        // DELETE WHERE id = 2
        let filters =
            vec![datafusion::logical_expr::col("id").eq(datafusion::logical_expr::lit(2i64))];
        let plan = writer
            .delete_from(&ctx.state(), filters)
            .await
            .expect("delete_from should succeed");

        let count = extract_count(plan).await;
        assert_eq!(count, 1, "should have deleted exactly 1 row");

        // Verify remaining rows via the view
        let (ids, names) = query_all_rows(&pool);
        assert_eq!(ids.len(), 2, "should have 2 rows remaining");
        assert_eq!(ids, vec![1, 3]);
        assert_eq!(names, vec!["a", "c"]);
    }

    #[tokio::test]
    async fn test_delete_from_view_backed_table_empty_filters() {
        let _guard = init_tracing(None);
        let (writer, pool) =
            setup_writer_with_overwrite_data(vec![1, 2, 3], vec!["a", "b", "c"]).await;

        let ctx = datafusion::prelude::SessionContext::new();
        let plan = writer
            .delete_from(&ctx.state(), vec![])
            .await
            .expect("delete_from should succeed");

        let count = extract_count(plan).await;
        assert_eq!(count, 3, "should have deleted all 3 rows");

        let (ids, _) = query_all_rows(&pool);
        assert!(ids.is_empty(), "table should be empty after delete-all");
    }

    #[tokio::test]
    async fn test_update_view_backed_table_with_filter() {
        let _guard = init_tracing(None);
        let (writer, pool) =
            setup_writer_with_overwrite_data(vec![1, 2, 3], vec!["a", "b", "c"]).await;

        let ctx = datafusion::prelude::SessionContext::new();

        let assignments = vec![("name".to_string(), datafusion::logical_expr::lit("updated"))];
        let filters =
            vec![datafusion::logical_expr::col("id").eq(datafusion::logical_expr::lit(2i64))];
        let plan = writer
            .update(&ctx.state(), assignments, filters)
            .await
            .expect("update should succeed");

        let count = extract_count(plan).await;
        assert_eq!(count, 1, "should have updated exactly 1 row");

        let (ids, names) = query_all_rows(&pool);
        assert_eq!(ids.len(), 3, "should still have 3 rows");
        for (id, name) in ids.iter().zip(names.iter()) {
            match *id {
                1 => assert_eq!(name, "a"),
                2 => assert_eq!(name, "updated"),
                3 => assert_eq!(name, "c"),
                other => panic!("unexpected id {other}"),
            }
        }
    }

    /// Rows per overwrite generation. Must exceed DuckDB's row group size
    /// (122,880) so the load writes row groups directly to the database file
    /// and the WAL stays below the automatic-checkpoint threshold — the setup
    /// where dropped generations leak without `checkpoint_on_write`.
    const GROWTH_ROWS_PER_GEN: usize = 130_000;
    const GROWTH_TABLES: usize = 3;
    const GROWTH_CYCLES: usize = 8;

    fn growth_schema() -> SchemaRef {
        Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("payload", arrow::datatypes::DataType::Utf8, false),
        ]))
    }

    fn growth_batch(generation: usize) -> RecordBatch {
        let ids = Int64Array::from_iter_values(0..GROWTH_ROWS_PER_GEN as i64);
        let payloads = StringArray::from_iter_values(
            (0..GROWTH_ROWS_PER_GEN)
                .map(|i| format!("gen{generation:04}-row{i:07}-{:>24}", i % 977)),
        );
        RecordBatch::try_new(growth_schema(), vec![Arc::new(ids), Arc::new(payloads)])
            .expect("should create growth record batch")
    }

    /// Runs a refresh-style workload: several logical tables backed by one
    /// file-based pool, writes serialized by a single mutex, one overwrite
    /// swap per table per cycle, with concurrent reader threads querying the
    /// views throughout. Returns the database file size after the last cycle.
    async fn run_overwrite_growth_workload(
        db_path: &std::path::Path,
        checkpoint_on_write: bool,
    ) -> u64 {
        let _ = std::fs::remove_file(db_path);
        let _ = std::fs::remove_file(db_path.with_extension("duckdb.wal"));

        let pool = Arc::new(
            DuckDbConnectionPool::new_file(
                db_path.to_str().expect("db path should be valid utf-8"),
                &duckdb::AccessMode::ReadWrite,
            )
            .expect("should create file-backed pool"),
        );

        let table_definitions: Vec<Arc<TableDefinition>> = (0..GROWTH_TABLES)
            .map(|i| {
                Arc::new(TableDefinition::new(
                    RelationName::new(format!("growth_t{i}")),
                    growth_schema(),
                ))
            })
            .collect();

        let write_settings =
            DuckDBWriteSettings::default().with_checkpoint_on_write(checkpoint_on_write);

        let write_one = |table_definition: Arc<TableDefinition>, generation: usize| {
            let pool = Arc::clone(&pool);
            let write_settings = write_settings.clone();
            async move {
                let sink = DuckDBDataSink::new(
                    pool,
                    table_definition,
                    InsertOp::Overwrite,
                    None,
                    growth_schema(),
                )
                .with_write_settings(write_settings);
                let stream = Box::pin(
                    MemoryStream::try_new(vec![growth_batch(generation)], growth_schema(), None)
                        .expect("to get stream"),
                );
                let data_sink: Arc<dyn DataSink> = Arc::new(sink);
                data_sink
                    .write_all(stream, &Arc::new(TaskContext::default()))
                    .await
                    .expect("overwrite should succeed");
            }
        };

        // Initial generation for every table so the reader views exist.
        for table_definition in &table_definitions {
            write_one(Arc::clone(table_definition), 0).await;
        }

        // Concurrent readers: dedicated connections from the same pool query
        // the views continuously. Errors are ignored — a query racing the
        // view swap is fine.
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader_handles: Vec<std::thread::JoinHandle<()>> = (0..2)
            .map(|reader_idx| {
                let pool = Arc::clone(&pool);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut conn = pool.connect_sync().expect("reader should connect");
                    let duckdb = DuckDB::duckdb_conn(&mut conn).expect("reader duckdb conn");
                    let mut i = reader_idx;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        let table = format!("growth_t{}", i % GROWTH_TABLES);
                        let _ = duckdb.conn.query_row(
                            &format!("SELECT count(*), max(payload) FROM {table}"),
                            [],
                            |row| row.get::<_, i64>(0),
                        );
                        i += 1;
                    }
                })
            })
            .collect();

        // Single mutex serializing all writes to the shared file, mirroring
        // the accelerator's per-file write serialization.
        let write_mutex = Arc::new(tokio::sync::Mutex::new(()));

        for cycle in 1..=GROWTH_CYCLES {
            for table_definition in &table_definitions {
                let _guard = write_mutex.lock().await;
                write_one(Arc::clone(table_definition), cycle).await;
            }
            let size = std::fs::metadata(db_path).map(|m| m.len()).unwrap_or(0);
            tracing::info!(
                "growth workload (checkpoint_on_write={checkpoint_on_write}) cycle {cycle}: file={:.1}MB",
                size as f64 / 1e6
            );
        }

        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for handle in reader_handles {
            handle.join().expect("reader thread should join");
        }

        // Correctness: every view must hold exactly the last generation.
        let mut conn = Arc::clone(&pool).connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        for (i, _) in table_definitions.iter().enumerate() {
            let (count, latest_gen_rows) = duckdb
                .conn
                .query_row(
                    &format!(
                        "SELECT count(*), sum(CASE WHEN payload LIKE 'gen{GROWTH_CYCLES:04}-%' THEN 1 ELSE 0 END) FROM growth_t{i}"
                    ),
                    [],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
                )
                .expect("to count view rows");
            assert_eq!(
                count, GROWTH_ROWS_PER_GEN as i64,
                "growth_t{i} should hold exactly one generation of rows"
            );
            assert_eq!(
                latest_gen_rows, GROWTH_ROWS_PER_GEN as i64,
                "growth_t{i} should hold the latest generation"
            );
        }

        std::fs::metadata(db_path).map(|m| m.len()).unwrap_or(0)
    }

    /// Upper bound for the database file after the checkpointed workload.
    /// The live working set is ~8MB (one ~2.5MB generation per table); without
    /// `checkpoint_on_write` each cycle leaks ~one generation per table and the
    /// file reaches ~48MB after [`GROWTH_CYCLES`] cycles, crossing this bound
    /// by the second cycle.
    const GROWTH_MAX_CHECKPOINTED_FILE_SIZE: u64 = 16_000_000;

    /// Repeated overwrite swaps (create new generation → repoint view → drop
    /// old generation) leak dropped generations unless the database is
    /// checkpointed; verifies `checkpoint_on_write` keeps the file bounded
    /// near the live working set.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_checkpoint_on_write_bounds_overwrite_growth() {
        let _guard = init_tracing(None);
        let db_path = std::env::temp_dir().join(format!(
            "dftp_growth_{}_{:x}.duckdb",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time should advance")
                .as_nanos()
        ));

        let final_size = run_overwrite_growth_workload(&db_path, true).await;
        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_file(db_path.with_extension("duckdb.wal"));

        tracing::info!(
            "growth workload with checkpoint_on_write finished at {:.1}MB",
            final_size as f64 / 1e6,
        );

        assert!(
            final_size < GROWTH_MAX_CHECKPOINTED_FILE_SIZE,
            "expected checkpointed file ({final_size} bytes) to stay under \
             {GROWTH_MAX_CHECKPOINTED_FILE_SIZE} bytes"
        );
    }

    fn upsert_schema() -> SchemaRef {
        Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
        ]))
    }

    fn upsert_batch(schema: &SchemaRef, rows: &[(i64, &str)]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::clone(schema),
            vec![
                Arc::new(Int64Array::from(
                    rows.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
                )),
            ],
        )
        .expect("should create a record batch")
    }

    /// `id` is the primary key, and the conflict target.
    fn upsert_table_definition(name: &str, schema: &SchemaRef) -> Arc<TableDefinition> {
        Arc::new(
            TableDefinition::new(RelationName::new(name), Arc::clone(schema)).with_constraints(
                crate::util::constraints::tests::get_pk_constraints(&["id"], Arc::clone(schema)),
            ),
        )
    }

    fn upsert_on_id() -> OnConflict {
        OnConflict::Upsert(ColumnReference::try_from("id").expect("valid column ref"))
    }

    /// Gives the test database one thread, so one `INSERT` statement processes
    /// its rows in arrival order. DuckDB keeps the copy it processes first, so a
    /// write that is one statement keeps the first copy of a repeated key on
    /// every run, where the parallel scan a multi-threaded database uses keeps
    /// whichever copy it reaches first.
    fn single_threaded(pool: &Arc<DuckDbConnectionPool>) {
        let mut conn = pool.connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        duckdb
            .conn
            .execute("SET threads = 1", [])
            .expect("to set threads");
    }

    /// The row count of `table`, and the `name` of the rows `key` selects.
    fn count_and_name(tx: &Transaction<'_>, table: &str, key: &str) -> (i64, Option<String>) {
        let count = tx
            .query_row(&format!("SELECT COUNT(1) FROM {table}"), [], |row| {
                row.get::<_, i64>(0)
            })
            .expect("to count rows");
        let name = tx
            .query_row(
                &format!("SELECT string_agg(name, ',' ORDER BY name) FROM {table} WHERE {key}"),
                [],
                |row| row.get::<_, Option<String>>(0),
            )
            .expect("to read the key");
        (count, name)
    }

    async fn upsert_write(
        pool: &Arc<DuckDbConnectionPool>,
        table_definition: &Arc<TableDefinition>,
        op: InsertOp,
        batches: Vec<RecordBatch>,
    ) {
        let sink = DuckDBDataSink::new(
            Arc::clone(pool),
            Arc::clone(table_definition),
            op,
            Some(upsert_on_id()),
            table_definition.schema(),
        );
        let data_sink: Arc<dyn DataSink> = Arc::new(sink);
        let stream = Box::pin(
            MemoryStream::try_new(batches, table_definition.schema(), None).expect("to get stream"),
        );
        data_sink
            .write_all(stream, &Arc::new(TaskContext::default()))
            .await
            .expect("to write all");
    }

    /// Regression test for spiceai/spiceai#14643: a full refresh that repeats a
    /// key in a later record batch keeps the last copy under `on_conflict: upsert`.
    #[tokio::test]
    async fn upsert_overwrite_keeps_the_last_copy_of_a_key_repeated_across_batches() {
        let _guard = init_tracing(None);
        let pool = get_mem_duckdb();
        single_threaded(&pool);
        let schema = upsert_schema();
        let table_definition = upsert_table_definition("upsert_across_batches", &schema);

        let first: Vec<(i64, &str)> = (0..8192).map(|id| (id, "first")).collect();
        let batches = vec![
            upsert_batch(&schema, &first),
            upsert_batch(&schema, &[(0, "last")]),
        ];
        upsert_write(&pool, &table_definition, InsertOp::Overwrite, batches).await;

        let mut conn = pool.connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        let tx = duckdb.conn.transaction().expect("to begin transaction");
        assert_eq!(
            count_and_name(&tx, &table_definition.name().to_string(), "id = 0"),
            (8192, Some("last".to_string()))
        );
    }

    /// An append resolves a repeat against a stored row and against an earlier
    /// row of the same write the same way: the last copy wins.
    #[tokio::test]
    async fn upsert_append_keeps_the_last_copy_over_stored_and_earlier_rows() {
        let _guard = init_tracing(None);
        let pool = get_mem_duckdb();
        single_threaded(&pool);
        let schema = upsert_schema();
        let table_definition = upsert_table_definition("upsert_append", &schema);

        let mut conn = pool.connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        let tx = duckdb.conn.transaction().expect("to begin transaction");
        let table = TableManager::new(Arc::clone(&table_definition))
            .with_internal(false)
            .expect("to create table");
        table
            .create_table(Arc::clone(&pool), &tx)
            .expect("to create table");
        tx.execute(
            &format!(
                "INSERT INTO {} VALUES (0, 'stored'), (1, 'stored')",
                table.table_name()
            ),
            [],
        )
        .expect("to seed");
        tx.commit().expect("to commit");

        let batches = vec![
            upsert_batch(&schema, &[(0, "first"), (2, "first")]),
            upsert_batch(&schema, &[(0, "last"), (2, "last")]),
        ];
        upsert_write(&pool, &table_definition, InsertOp::Append, batches).await;

        let tx = duckdb.conn.transaction().expect("to begin transaction");
        let name = table.table_name().to_string();
        assert_eq!(
            count_and_name(&tx, &name, "id = 0"),
            (3, Some("last".to_string()))
        );
        assert_eq!(
            count_and_name(&tx, &name, "id = 1"),
            (3, Some("stored".to_string()))
        );
        assert_eq!(
            count_and_name(&tx, &name, "id = 2"),
            (3, Some("last".to_string()))
        );
    }

    /// With the database's default thread count, which copy a single statement
    /// keeps is whichever its parallel scan reaches first; fifty copies of one
    /// key, one per batch, still end on the last.
    #[tokio::test]
    async fn upsert_overwrite_keeps_the_last_of_many_copies_under_a_parallel_scan() {
        let _guard = init_tracing(None);
        let pool = get_mem_duckdb();
        let schema = upsert_schema();
        let table_definition = upsert_table_definition("upsert_many_copies", &schema);

        let copies: Vec<String> = (0..50).map(|i| format!("copy {i}")).collect();
        let batches: Vec<RecordBatch> = copies
            .iter()
            .enumerate()
            .map(|(i, copy)| {
                let keep: i64 = i64::try_from(i).expect("fits") + 1;
                upsert_batch(&schema, &[(0, copy.as_str()), (keep, "kept")])
            })
            .collect();
        upsert_write(&pool, &table_definition, InsertOp::Overwrite, batches).await;

        let mut conn = pool.connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        let tx = duckdb.conn.transaction().expect("to begin transaction");
        assert_eq!(
            count_and_name(&tx, &table_definition.name().to_string(), "id = 0"),
            (51, Some("copy 49".to_string()))
        );
    }

    /// Writes `batches` straight into a fresh primary-key table with
    /// `write_to_table`, below the sink's per-batch constraint check, and
    /// returns `(count, name of key 0)`.
    fn write_to_table_upsert(
        name: &str,
        schema: &SchemaRef,
        batches: Vec<RecordBatch>,
        key: &str,
    ) -> (i64, Option<String>) {
        let pool = get_mem_duckdb();
        single_threaded(&pool);
        let table_definition = upsert_table_definition(name, schema);
        let mut conn = pool.connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        let tx = duckdb.conn.transaction().expect("to begin transaction");
        let table = TableManager::new(Arc::clone(&table_definition))
            .with_internal(false)
            .expect("to create table");
        table
            .create_table(Arc::clone(&pool), &tx)
            .expect("to create table");

        let (sender, receiver) = mpsc::channel(batches.len().max(1));
        for batch in batches {
            sender.try_send(batch).expect("to queue a batch");
        }
        drop(sender);
        let on_conflict = upsert_on_id();
        write_to_table(
            &table,
            &tx,
            Arc::clone(schema),
            receiver,
            Some(&on_conflict),
        )
        .expect("to write");

        count_and_name(&tx, &table.table_name().to_string(), key)
    }

    /// A key repeated inside one record batch keeps its last copy too.
    #[test]
    fn upsert_keeps_the_last_copy_of_a_key_repeated_within_a_batch() {
        let _guard = init_tracing(None);
        let schema = upsert_schema();
        let batches = vec![upsert_batch(&schema, &[(0, "a"), (1, "b"), (0, "c")])];
        assert_eq!(
            write_to_table_upsert("upsert_within_batch", &schema, batches, "id = 0"),
            (2, Some("c".to_string()))
        );
    }

    /// `-0.0` repeats a stored `0.0` and every NaN repeats a stored NaN in
    /// DuckDB, so a later copy under either spelling is the one kept.
    #[test]
    fn upsert_keeps_the_last_copy_of_a_float_key_duckdb_holds_as_one() {
        let _guard = init_tracing(None);
        let schema: SchemaRef = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
        ]));
        let float_batch = |rows: &[(f64, &str)]| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(arrow::array::Float64Array::from(
                        rows.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                    )),
                    Arc::new(StringArray::from(
                        rows.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
                    )),
                ],
            )
            .expect("should create a record batch")
        };
        let batches = vec![
            float_batch(&[(0.0, "zero"), (f64::NAN, "nan")]),
            float_batch(&[(-0.0, "negative zero"), (-f64::NAN, "another nan")]),
        ];
        assert_eq!(
            write_to_table_upsert("upsert_float_zero", &schema, batches.clone(), "id = 0"),
            (2, Some("negative zero".to_string()))
        );
        assert_eq!(
            write_to_table_upsert("upsert_float_nan", &schema, batches, "isnan(id)"),
            (2, Some("another nan".to_string()))
        );
    }

    /// A `Date64` key is stored as a `DATE`, so two values on one day are one
    /// key to `DuckDB`, and the later copy is the one kept.
    #[test]
    fn upsert_keeps_the_last_copy_of_a_date64_key_duckdb_stores_as_a_date() {
        let _guard = init_tracing(None);
        let schema: SchemaRef = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Date64, false),
            arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
        ]));
        let date_batch = |rows: &[(i64, &str)]| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(arrow::array::Date64Array::from(
                        rows.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                    )),
                    Arc::new(StringArray::from(
                        rows.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
                    )),
                ],
            )
            .expect("should create a record batch")
        };
        let batches = vec![
            date_batch(&[(1_000, "first second"), (86_400_000, "next day")]),
            date_batch(&[(2_000, "second second")]),
        ];
        assert_eq!(
            write_to_table_upsert("upsert_date64", &schema, batches, "id = DATE '1970-01-01'"),
            (2, Some("second second".to_string()))
        );
    }

    /// `DuckDB` resolves two NULL unique keys of one statement as one conflict
    /// and drops the later row, where across statements it keeps both, so a
    /// repeated NULL key reaches `DuckDB` in a statement of its own.
    #[test]
    fn upsert_keeps_every_row_whose_unique_key_is_null() {
        let _guard = init_tracing(None);
        let pool = get_mem_duckdb();
        single_threaded(&pool);
        let schema: SchemaRef = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, true),
            arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
        ]));
        let table_definition = Arc::new(TableDefinition::new(
            RelationName::new("upsert_null_unique"),
            Arc::clone(&schema),
        ));
        let mut conn = pool.connect_sync().expect("to connect");
        let duckdb = DuckDB::duckdb_conn(&mut conn).expect("to get duckdb conn");
        let tx = duckdb.conn.transaction().expect("to begin transaction");
        let table = TableManager::new(Arc::clone(&table_definition))
            .with_internal(false)
            .expect("to create table");
        table
            .create_table(Arc::clone(&pool), &tx)
            .expect("to create table");
        tx.execute(
            &format!(
                "CREATE UNIQUE INDEX upsert_null_unique_id ON {} (id)",
                table.table_name()
            ),
            [],
        )
        .expect("to create the unique index");

        let (sender, receiver) = mpsc::channel(1);
        sender
            .try_send(
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int64Array::from(vec![None, None, Some(1)])),
                        Arc::new(StringArray::from(vec!["a", "b", "c"])),
                    ],
                )
                .expect("should create a record batch"),
            )
            .expect("to queue the batch");
        drop(sender);
        let on_conflict = upsert_on_id();
        write_to_table(
            &table,
            &tx,
            Arc::clone(&schema),
            receiver,
            Some(&on_conflict),
        )
        .expect("to write");

        assert_eq!(
            count_and_name(&tx, &table.table_name().to_string(), "id IS NULL"),
            (3, Some("a,b".to_string()))
        );
    }
}
