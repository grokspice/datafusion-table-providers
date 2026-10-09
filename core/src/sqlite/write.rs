use std::{fmt, sync::Arc};

use arrow::array::RecordBatch;
use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::datasource::sink::{DataSink, DataSinkExec};
use datafusion::{
    catalog::Session,
    common::TableReference,
    common::{not_impl_err, Constraints},
    datasource::{TableProvider, TableType},
    error::DataFusionError,
    execution::{SendableRecordBatchStream, TaskContext},
    logical_expr::{dml::InsertOp, Expr},
    physical_plan::{metrics::MetricsSet, DisplayAs, DisplayFormatType, ExecutionPlan},
};
use futures::StreamExt;
use snafu::prelude::*;

use crate::sql::sql_provider_datafusion::expr;

use crate::util::{
    constraints,
    count_exec::make_count_exec,
    dml::{
        assignments_to_sql, delete_statement, filters_to_sql, update_statement, DeletionExec,
        DeletionSink, UpdateExec, UpdateSink,
    },
    on_conflict::OnConflict,
    retriable_error::{check_and_mark_retriable_error, to_retriable_data_write_error},
};

use super::{to_datafusion_error, Sqlite};

#[derive(Debug, Clone)]
pub struct SqliteTableWriter {
    pub read_provider: Arc<dyn TableProvider>,
    sqlite: Arc<Sqlite>,
    on_conflict: Option<OnConflict>,
}

impl SqliteTableWriter {
    pub fn create(
        read_provider: Arc<dyn TableProvider>,
        sqlite: Sqlite,
        on_conflict: Option<OnConflict>,
    ) -> Arc<Self> {
        Arc::new(Self {
            read_provider,
            sqlite: Arc::new(sqlite),
            on_conflict,
        })
    }

    pub fn sqlite(&self) -> Arc<Sqlite> {
        Arc::clone(&self.sqlite)
    }
}

#[async_trait]
impl TableProvider for SqliteTableWriter {
    fn schema(&self) -> SchemaRef {
        self.read_provider.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn constraints(&self) -> Option<&Constraints> {
        Some(self.sqlite.constraints())
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::error::Result<Vec<datafusion::logical_expr::TableProviderFilterPushDown>> {
        // Verify schema consistency before delegating
        // This is a cheap check since it's just comparing Arc<Schema> pointers
        if self.read_provider.schema() != self.schema() {
            tracing::warn!(
                "Schema mismatch detected in SqliteTableWriter for table {}",
                self.sqlite.table_name()
            );
        }

        self.read_provider.supports_filters_pushdown(filters)
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
        // `InsertOp::Replace` promises an atomic per-row upsert. SQLite delivers
        // that only with an `ON CONFLICT (...) DO UPDATE`, i.e.
        // `OnConflict::Upsert`. A missing `on_conflict` (plain append) or a
        // `DoNothing`/`DoNothingAll` (leaves the existing row unchanged) does
        // NOT replace, so refuse rather than silently accept a non-replacing
        // write.
        if matches!(op, InsertOp::Replace)
            && !matches!(self.on_conflict, Some(OnConflict::Upsert(_)))
        {
            return not_impl_err!(
                "InsertOp::Replace requires an on_conflict upsert target on the SQLite writer"
            );
        }
        Ok(Arc::new(DataSinkExec::new(
            input,
            Arc::new(SqliteDataSink::new(
                Arc::clone(&self.sqlite),
                op,
                self.on_conflict.clone(),
                self.schema(),
                self.sqlite.batch_insert_use_prepared_statements,
            )),
            None,
        )) as _)
    }

    async fn delete_from(
        &self,
        _state: &dyn Session,
        filters: Vec<Expr>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        let sql_where = if filters.is_empty() {
            None
        } else {
            Some(filters_to_sql(&filters, Some(expr::Engine::SQLite))?)
        };
        let table = self.sqlite().table_reference().clone();
        let sqlite = self.sqlite();

        Ok(Arc::new(DeletionExec::new(Arc::new(SqliteDeletionSink {
            sqlite,
            table,
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

        let set_clause = assignments_to_sql(&assignments, Some(expr::Engine::SQLite))?;
        let sqlite = self.sqlite();

        let sql_where = if filters.is_empty() {
            None
        } else {
            Some(filters_to_sql(&filters, Some(expr::Engine::SQLite))?)
        };
        let sql = update_statement(
            self.sqlite().table_reference(),
            &set_clause,
            sql_where.as_deref(),
        );

        Ok(Arc::new(UpdateExec::new(Arc::new(SqliteUpdateSink {
            sqlite,
            sql,
        }))))
    }
}

struct SqliteDeletionSink {
    sqlite: Arc<Sqlite>,
    table: TableReference,
    sql_where: Option<String>,
}

#[async_trait]
impl DeletionSink for SqliteDeletionSink {
    async fn delete_from(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let mut db_conn = self.sqlite.connect().await?;
        let sqlite_conn = Sqlite::sqlite_conn(&mut db_conn)?;
        let table = self.table.clone();
        let sql_where = self.sql_where.clone();

        let count = sqlite_conn
            .conn
            .call(move |conn| -> Result<u64, rusqlite::Error> {
                let tx = conn.transaction()?;
                let delete_sql = delete_statement(&table, sql_where.as_deref());
                tx.execute(&delete_sql, [])?;
                // rusqlite 0.40 removed FromSql for u64; changes() is always non-negative.
                let count: i64 = tx.query_row("SELECT changes()", [], |row| row.get(0))?;
                tx.commit()?;
                Ok(count as u64)
            })
            .await?;

        Ok(count)
    }
}

struct SqliteUpdateSink {
    sqlite: Arc<Sqlite>,
    sql: String,
}

#[async_trait]
impl UpdateSink for SqliteUpdateSink {
    async fn execute_update(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let mut db_conn = self.sqlite.connect().await?;
        let sqlite_conn = Sqlite::sqlite_conn(&mut db_conn)?;
        let sql = self.sql.clone();

        let count = sqlite_conn
            .conn
            .call(move |conn| -> Result<u64, rusqlite::Error> {
                let tx = conn.transaction()?;
                tx.execute(&sql, [])?;
                // rusqlite 0.40 removed FromSql for u64; changes() is always non-negative.
                let count: i64 = tx.query_row("SELECT changes()", [], |row| row.get(0))?;
                tx.commit()?;
                Ok(count as u64)
            })
            .await?;

        Ok(count)
    }
}

#[derive(Clone)]
struct SqliteDataSink {
    sqlite: Arc<Sqlite>,
    overwrite: InsertOp,
    on_conflict: Option<OnConflict>,
    schema: SchemaRef,
    use_prepared_statements: bool,
}

#[async_trait]
impl DataSink for SqliteDataSink {
    fn metrics(&self) -> Option<MetricsSet> {
        None
    }

    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    async fn write_all(
        &self,
        data: SendableRecordBatchStream,
        _context: &Arc<TaskContext>,
    ) -> datafusion::common::Result<u64> {
        let (batch_tx, mut batch_rx) = tokio::sync::mpsc::channel::<RecordBatch>(1);

        // Since the main task/stream can be dropped or fail, we use a oneshot channel to signal that all data is received and we should commit the transaction
        let (notify_commit_transaction, mut on_commit_transaction) =
            tokio::sync::oneshot::channel();

        let mut db_conn = self
            .sqlite
            .connect()
            .await
            .map_err(to_retriable_data_write_error)?;
        let sqlite_conn =
            Sqlite::sqlite_conn(&mut db_conn).map_err(to_retriable_data_write_error)?;

        let constraints = self.sqlite.constraints().clone();
        let mut data = data;
        let task = tokio::spawn(async move {
            let mut num_rows: u64 = 0;
            while let Some(data_batch) = data.next().await {
                let data_batch = data_batch.map_err(check_and_mark_retriable_error)?;
                num_rows += u64::try_from(data_batch.num_rows()).map_err(|e| {
                    DataFusionError::Execution(format!("Unable to convert num_rows() to u64: {e}"))
                })?;

                constraints::validate_batch_with_constraints(
                    vec![data_batch.clone()],
                    &constraints,
                    &crate::util::constraints::UpsertOptions::default(),
                )
                .await
                .context(super::ConstraintViolationSnafu)
                .map_err(to_datafusion_error)?;

                batch_tx.send(data_batch).await.map_err(|err| {
                    DataFusionError::Execution(format!("Error sending data batch: {err}"))
                })?;
            }

            if notify_commit_transaction.send(()).is_err() {
                return Err(DataFusionError::Execution(
                    "Unable to send message to commit transaction to SQLite writer.".to_string(),
                ));
            };

            // Drop the sender to signal the receiver that no more data is coming
            drop(batch_tx);

            Ok::<_, DataFusionError>(num_rows)
        });

        let overwrite = self.overwrite;
        let sqlite = Arc::clone(&self.sqlite);
        let on_conflict = self.on_conflict.clone();
        let use_prepared_statements = self.use_prepared_statements;

        sqlite_conn
            .conn
            .call(move |conn| {
                let transaction = conn.transaction()?;

                if matches!(overwrite, InsertOp::Overwrite) {
                    sqlite.delete_all_table_data(&transaction)?;
                }

                while let Some(data_batch) = batch_rx.blocking_recv() {
                    if data_batch.num_rows() > 0 {
                        if use_prepared_statements {
                            sqlite.insert_batch_prepared(
                                &transaction,
                                data_batch,
                                on_conflict.as_ref(),
                            )?;
                        } else {
                            #[allow(deprecated)]
                            sqlite.insert_batch(&transaction, data_batch, on_conflict.as_ref())?;
                        }
                    }
                }

                if on_commit_transaction.try_recv().is_err() {
                    return Err(rusqlite::Error::InvalidQuery);
                }

                transaction.commit()?;

                Ok(())
            })
            .await
            .context(super::UnableToInsertIntoTableAsyncSnafu)
            .map_err(|e| {
                if let super::Error::UnableToInsertIntoTableAsync {
                    source:
                        tokio_rusqlite::Error::Error(rusqlite::Error::SqliteFailure(
                            rusqlite::ffi::Error {
                                code: rusqlite::ffi::ErrorCode::DiskFull,
                                ..
                            },
                            _,
                        )),
                } = e
                {
                    DataFusionError::External(super::Error::DiskFull {}.into())
                } else {
                    to_retriable_data_write_error(e)
                }
            })?;

        let num_rows = task.await.map_err(to_retriable_data_write_error)??;

        Ok(num_rows)
    }
}

impl SqliteDataSink {
    fn new(
        sqlite: Arc<Sqlite>,
        overwrite: InsertOp,
        on_conflict: Option<OnConflict>,
        schema: SchemaRef,
        use_prepared_statements: bool,
    ) -> Self {
        Self {
            sqlite,
            overwrite,
            on_conflict,
            schema,
            use_prepared_statements,
        }
    }
}

impl std::fmt::Debug for SqliteDataSink {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "SqliteDataSink")
    }
}

impl DisplayAs for SqliteDataSink {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> std::fmt::Result {
        write!(f, "SqliteDataSink")
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use datafusion::arrow::{
        array::{Int64Array, RecordBatch, StringArray},
        datatypes::{DataType, Schema},
    };
    use datafusion::{
        catalog::TableProviderFactory,
        common::{Constraints, TableReference, ToDFSchema},
        execution::context::SessionContext,
        logical_expr::{dml::InsertOp, CreateExternalTable},
        physical_plan::collect,
    };

    use datafusion::arrow::array::UInt64Array;
    use datafusion::logical_expr::{col, lit};

    use crate::sqlite::SqliteTableProviderFactory;
    use crate::util::test::MockExec;

    #[tokio::test]
    #[allow(clippy::unreadable_literal)]
    async fn test_round_trip_sqlite() {
        let schema = Arc::new(Schema::new(vec![
            datafusion::arrow::datatypes::Field::new("time_in_string", DataType::Utf8, false),
            datafusion::arrow::datatypes::Field::new("time_int", DataType::Int64, false),
        ]));
        let df_schema = ToDFSchema::to_dfschema_ref(Arc::clone(&schema)).expect("df schema");
        let external_table = CreateExternalTable {
            schema: df_schema,
            name: TableReference::bare("test_table"),
            locations: vec![],
            file_type: String::new(),
            table_partition_cols: vec![],
            if_not_exists: true,
            definition: None,
            order_exprs: vec![],
            unbounded: false,
            options: HashMap::new(),
            constraints: Constraints::default(),
            column_defaults: HashMap::default(),
            temporary: false,
            or_replace: false,
        };
        let ctx = SessionContext::new();
        let table = SqliteTableProviderFactory::default()
            .create(&ctx.state(), &external_table)
            .await
            .expect("table should be created");

        let arr1 = StringArray::from(vec![
            "1970-01-01",
            "2012-12-01T11:11:11Z",
            "2012-12-01T11:11:12Z",
        ]);
        let arr3 = Int64Array::from(vec![0, 1354360271, 1354360272]);
        let data = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(arr1), Arc::new(arr3)])
            .expect("data should be created");

        let exec = MockExec::new(vec![Ok(data)], schema);

        let insertion = table
            .insert_into(&ctx.state(), Arc::new(exec), InsertOp::Append)
            .await
            .expect("insertion should be successful");

        collect(insertion, ctx.task_ctx())
            .await
            .expect("insert successful");
    }

    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::unreadable_literal)]
    async fn test_all_arrow_types_to_sqlite() {
        use arrow::{
            array::*,
            datatypes::{DataType, Field, TimeUnit},
        };

        let num_rows = 10;
        // Create a comprehensive schema with all supported Arrow types
        let schema = Arc::new(Schema::new(vec![
            // Integer types
            Field::new("col_int8", DataType::Int8, true),
            Field::new("col_int16", DataType::Int16, true),
            Field::new("col_int32", DataType::Int32, true),
            Field::new("col_int64", DataType::Int64, true),
            Field::new("col_uint8", DataType::UInt8, true),
            Field::new("col_uint16", DataType::UInt16, true),
            Field::new("col_uint32", DataType::UInt32, true),
            Field::new("col_uint64", DataType::UInt64, true),
            // Float types (Float16 requires half crate - skip for now)
            Field::new("col_float32", DataType::Float32, true),
            Field::new("col_float64", DataType::Float64, true),
            // String types
            Field::new("col_utf8", DataType::Utf8, true),
            Field::new("col_large_utf8", DataType::LargeUtf8, true),
            Field::new("col_utf8_view", DataType::Utf8View, true),
            // Boolean
            Field::new("col_bool", DataType::Boolean, true),
            // Binary types
            Field::new("col_binary", DataType::Binary, true),
            Field::new("col_large_binary", DataType::LargeBinary, true),
            Field::new("col_binary_view", DataType::BinaryView, true),
            Field::new("col_fixed_binary", DataType::FixedSizeBinary(4), true),
            // Date types
            Field::new("col_date32", DataType::Date32, true),
            Field::new("col_date64", DataType::Date64, true),
            // Timestamp types
            Field::new(
                "col_ts_second",
                DataType::Timestamp(TimeUnit::Second, None),
                true,
            ),
            Field::new(
                "col_ts_milli",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                true,
            ),
            Field::new(
                "col_ts_micro",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                true,
            ),
            Field::new(
                "col_ts_nano",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                true,
            ),
            // Time types
            Field::new(
                "col_time32_second",
                DataType::Time32(TimeUnit::Second),
                true,
            ),
            Field::new(
                "col_time32_milli",
                DataType::Time32(TimeUnit::Millisecond),
                true,
            ),
            Field::new(
                "col_time64_micro",
                DataType::Time64(TimeUnit::Microsecond),
                true,
            ),
            Field::new(
                "col_time64_nano",
                DataType::Time64(TimeUnit::Nanosecond),
                true,
            ),
            // Duration types
            Field::new("col_dur_second", DataType::Duration(TimeUnit::Second), true),
            Field::new(
                "col_dur_milli",
                DataType::Duration(TimeUnit::Millisecond),
                true,
            ),
            Field::new(
                "col_dur_micro",
                DataType::Duration(TimeUnit::Microsecond),
                true,
            ),
            Field::new(
                "col_dur_nano",
                DataType::Duration(TimeUnit::Nanosecond),
                true,
            ),
        ]));

        let df_schema = ToDFSchema::to_dfschema_ref(Arc::clone(&schema)).expect("df schema");
        let external_table = CreateExternalTable {
            schema: df_schema,
            name: TableReference::bare(format!("test_all_types_{}", num_rows)),
            locations: vec![],
            file_type: String::new(),
            table_partition_cols: vec![],
            if_not_exists: true,
            definition: None,
            order_exprs: vec![],
            unbounded: false,
            options: HashMap::new(),
            constraints: Constraints::default(),
            column_defaults: HashMap::default(),
            temporary: false,
            or_replace: false,
        };

        let ctx = SessionContext::new();
        let table = SqliteTableProviderFactory::default()
            .create(&ctx.state(), &external_table)
            .await
            .expect("table should be created");

        // Generate test data dynamically based on num_rows
        let mut int8_values = Vec::with_capacity(num_rows);
        let mut int16_values = Vec::with_capacity(num_rows);
        let mut int32_values = Vec::with_capacity(num_rows);
        let mut int64_values = Vec::with_capacity(num_rows);
        let mut uint8_values = Vec::with_capacity(num_rows);
        let mut uint16_values = Vec::with_capacity(num_rows);
        let mut uint32_values = Vec::with_capacity(num_rows);
        let mut uint64_values = Vec::with_capacity(num_rows);
        let mut float32_values = Vec::with_capacity(num_rows);
        let mut float64_values = Vec::with_capacity(num_rows);
        let mut string_values = Vec::with_capacity(num_rows);
        let mut large_string_values = Vec::with_capacity(num_rows);
        let mut string_view_values = Vec::with_capacity(num_rows);
        let mut bool_values = Vec::with_capacity(num_rows);
        let mut binary_values: Vec<Option<Vec<u8>>> = Vec::with_capacity(num_rows);
        let mut large_binary_values: Vec<Option<Vec<u8>>> = Vec::with_capacity(num_rows);
        let mut binary_view_values: Vec<Option<Vec<u8>>> = Vec::with_capacity(num_rows);
        let mut fixed_binary_values: Vec<Option<Vec<u8>>> = Vec::with_capacity(num_rows);
        let mut date32_values = Vec::with_capacity(num_rows);
        let mut date64_values = Vec::with_capacity(num_rows);
        let mut ts_sec_values = Vec::with_capacity(num_rows);
        let mut ts_milli_values = Vec::with_capacity(num_rows);
        let mut ts_micro_values = Vec::with_capacity(num_rows);
        let mut ts_nano_values = Vec::with_capacity(num_rows);
        let mut time32_sec_values = Vec::with_capacity(num_rows);
        let mut time32_milli_values = Vec::with_capacity(num_rows);
        let mut time64_micro_values = Vec::with_capacity(num_rows);
        let mut time64_nano_values = Vec::with_capacity(num_rows);
        let mut dur_sec_values = Vec::with_capacity(num_rows);
        let mut dur_milli_values = Vec::with_capacity(num_rows);
        let mut dur_micro_values = Vec::with_capacity(num_rows);
        let mut dur_nano_values = Vec::with_capacity(num_rows);

        for i in 0..num_rows {
            // Add some null values at regular intervals
            let is_null = i % 3 == 1;

            int8_values.push(if is_null { None } else { Some((i % 100) as i8) });
            int16_values.push(if is_null {
                None
            } else {
                Some((i * 100) as i16)
            });
            int32_values.push(if is_null {
                None
            } else {
                Some((i * 1000) as i32)
            });
            int64_values.push(if is_null {
                None
            } else {
                Some((i * 10000) as i64)
            });
            uint8_values.push(if is_null { None } else { Some((i % 200) as u8) });
            uint16_values.push(if is_null {
                None
            } else {
                Some((i * 100) as u16)
            });
            uint32_values.push(if is_null {
                None
            } else {
                Some((i * 1000) as u32)
            });
            uint64_values.push(if is_null {
                None
            } else {
                Some((i * 10000) as u64)
            });
            float32_values.push(if is_null {
                None
            } else {
                Some((i as f32) * 1.5)
            });
            float64_values.push(if is_null {
                None
            } else {
                Some((i as f64) * 2.5)
            });
            string_values.push(if is_null {
                None
            } else {
                Some(format!("str_{}", i))
            });
            large_string_values.push(if is_null {
                None
            } else {
                Some(format!("large_{}", i))
            });
            string_view_values.push(if is_null {
                None
            } else {
                Some(format!("view_{}", i))
            });
            bool_values.push(if is_null { None } else { Some(i % 2 == 0) });
            binary_values.push(if is_null {
                None
            } else {
                Some(format!("bin_{}", i).into_bytes())
            });
            large_binary_values.push(if is_null {
                None
            } else {
                Some(format!("lbin_{}", i).into_bytes())
            });
            binary_view_values.push(if is_null {
                None
            } else {
                Some(format!("bv_{}", i).into_bytes())
            });
            fixed_binary_values.push(if is_null {
                None
            } else {
                Some(vec![i as u8, (i + 1) as u8, (i + 2) as u8, (i + 3) as u8])
            });
            date32_values.push(if is_null {
                None
            } else {
                Some(18000 + i as i32)
            });
            date64_values.push(if is_null {
                None
            } else {
                Some(1609459200000 + (i as i64 * 86400000))
            });
            ts_sec_values.push(if is_null {
                None
            } else {
                Some(1609459200 + i as i64)
            });
            ts_milli_values.push(if is_null {
                None
            } else {
                Some(1609459200000 + i as i64)
            });
            ts_micro_values.push(if is_null {
                None
            } else {
                Some(1609459200000000 + i as i64)
            });
            ts_nano_values.push(if is_null {
                None
            } else {
                Some(1609459200000000000 + i as i64)
            });
            time32_sec_values.push(if is_null {
                None
            } else {
                Some(3600 + (i * 10) as i32)
            });
            time32_milli_values.push(if is_null {
                None
            } else {
                Some(3600000 + (i * 1000) as i32)
            });
            time64_micro_values.push(if is_null {
                None
            } else {
                Some(3600000000 + (i * 1000000) as i64)
            });
            time64_nano_values.push(if is_null {
                None
            } else {
                Some(3600000000000 + (i * 1000000000) as i64)
            });
            dur_sec_values.push(if is_null {
                None
            } else {
                Some(86400 + i as i64)
            });
            dur_milli_values.push(if is_null {
                None
            } else {
                Some(86400000 + i as i64)
            });
            dur_micro_values.push(if is_null {
                None
            } else {
                Some(86400000000 + i as i64)
            });
            dur_nano_values.push(if is_null {
                None
            } else {
                Some(86400000000000 + i as i64)
            });
        }

        let data = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                // Integer types
                Arc::new(Int8Array::from(int8_values)),
                Arc::new(Int16Array::from(int16_values)),
                Arc::new(Int32Array::from(int32_values)),
                Arc::new(Int64Array::from(int64_values)),
                Arc::new(UInt8Array::from(uint8_values)),
                Arc::new(UInt16Array::from(uint16_values)),
                Arc::new(UInt32Array::from(uint32_values)),
                Arc::new(UInt64Array::from(uint64_values)),
                // Float types
                Arc::new(Float32Array::from(float32_values)),
                Arc::new(Float64Array::from(float64_values)),
                // String types
                Arc::new(StringArray::from(string_values)),
                Arc::new(LargeStringArray::from(large_string_values)),
                Arc::new(StringViewArray::from(string_view_values)),
                // Boolean
                Arc::new(BooleanArray::from(bool_values)),
                // Binary types
                Arc::new(BinaryArray::from(
                    binary_values
                        .iter()
                        .map(|v| v.as_ref().map(|b| b.as_slice()))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(LargeBinaryArray::from(
                    large_binary_values
                        .iter()
                        .map(|v| v.as_ref().map(|b| b.as_slice()))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(BinaryViewArray::from_iter(
                    binary_view_values
                        .iter()
                        .map(|v| v.as_ref().map(|b| b.as_slice())),
                )),
                Arc::new(
                    FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                        fixed_binary_values
                            .iter()
                            .map(|v| v.as_ref().map(|b| b.as_slice())),
                        4,
                    )
                    .unwrap(),
                ),
                // Date types
                Arc::new(Date32Array::from(date32_values)),
                Arc::new(Date64Array::from(date64_values)),
                // Timestamp types
                Arc::new(TimestampSecondArray::from(ts_sec_values)),
                Arc::new(TimestampMillisecondArray::from(ts_milli_values)),
                Arc::new(TimestampMicrosecondArray::from(ts_micro_values)),
                Arc::new(TimestampNanosecondArray::from(ts_nano_values)),
                // Time types
                Arc::new(Time32SecondArray::from(time32_sec_values)),
                Arc::new(Time32MillisecondArray::from(time32_milli_values)),
                Arc::new(Time64MicrosecondArray::from(time64_micro_values)),
                Arc::new(Time64NanosecondArray::from(time64_nano_values)),
                // Duration types
                Arc::new(DurationSecondArray::from(dur_sec_values)),
                Arc::new(DurationMillisecondArray::from(dur_milli_values)),
                Arc::new(DurationMicrosecondArray::from(dur_micro_values)),
                Arc::new(DurationNanosecondArray::from(dur_nano_values)),
            ],
        )
        .expect("data should be created");

        let exec = MockExec::new(vec![Ok(data)], Arc::clone(&schema));

        let insertion = table
            .insert_into(&ctx.state(), Arc::new(exec), InsertOp::Append)
            .await
            .expect("insertion should be successful");

        collect(insertion, ctx.task_ctx())
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "insert successful for {} rows - all Arrow types should be converted to SQLite",
                    num_rows
                )
            });
    }

    #[tokio::test]
    async fn test_filter_pushdown_support() {
        use datafusion::logical_expr::{col, lit, TableProviderFilterPushDown};

        let schema = Arc::new(Schema::new(vec![
            datafusion::arrow::datatypes::Field::new("id", DataType::Int64, false),
            datafusion::arrow::datatypes::Field::new("name", DataType::Utf8, false),
        ]));
        let df_schema = ToDFSchema::to_dfschema_ref(Arc::clone(&schema)).expect("df schema");
        let external_table = CreateExternalTable {
            schema: df_schema,
            name: TableReference::bare("test_filter_table"),
            locations: vec![],
            file_type: String::new(),
            table_partition_cols: vec![],
            if_not_exists: true,
            definition: None,
            order_exprs: vec![],
            unbounded: false,
            options: HashMap::new(),
            constraints: Constraints::default(),
            column_defaults: HashMap::default(),
            temporary: false,
            or_replace: false,
        };
        let ctx = SessionContext::new();
        let table = SqliteTableProviderFactory::default()
            .create(&ctx.state(), &external_table)
            .await
            .expect("table should be created");

        // Test that filter pushdown is supported
        let filter = col("id").gt(lit(10));
        let result = table
            .supports_filters_pushdown(&[&filter])
            .expect("should support filter pushdown");

        assert_eq!(
            result,
            vec![TableProviderFilterPushDown::Exact],
            "Filter pushdown should be exact for simple comparison"
        );
    }

    #[tokio::test]
    async fn test_concurrent_read_write_with_filter_pushdown() {
        use datafusion::logical_expr::{col, lit, TableProviderFilterPushDown};

        let schema = Arc::new(Schema::new(vec![
            datafusion::arrow::datatypes::Field::new("id", DataType::Int64, false),
            datafusion::arrow::datatypes::Field::new("value", DataType::Int64, false),
        ]));
        let df_schema = ToDFSchema::to_dfschema_ref(Arc::clone(&schema)).expect("df schema");

        let external_table = CreateExternalTable {
            schema: df_schema,
            name: TableReference::bare("concurrent_test"),
            locations: vec![],
            file_type: String::new(),
            table_partition_cols: vec![],
            if_not_exists: true,
            definition: None,
            order_exprs: vec![],
            unbounded: false,
            options: HashMap::new(),
            constraints: Constraints::default(),
            column_defaults: HashMap::default(),
            temporary: false,
            or_replace: false,
        };

        let ctx = SessionContext::new();
        let table = SqliteTableProviderFactory::default()
            .create(&ctx.state(), &external_table)
            .await
            .expect("table should be created");

        // Insert initial data
        let arr1 = Int64Array::from(vec![1, 2, 3]);
        let arr2 = Int64Array::from(vec![10, 20, 30]);
        let data = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(arr1), Arc::new(arr2)])
            .expect("data should be created");

        let exec = MockExec::new(vec![Ok(data)], Arc::clone(&schema));
        let insertion = table
            .insert_into(&ctx.state(), Arc::new(exec), InsertOp::Append)
            .await
            .expect("insertion should be successful");

        collect(insertion, ctx.task_ctx())
            .await
            .expect("insert successful");

        // Verify filter pushdown works after insert
        let filter = col("id").gt(lit(1));
        let result = table
            .supports_filters_pushdown(&[&filter])
            .expect("should support filter pushdown");

        assert_eq!(
            result,
            vec![TableProviderFilterPushDown::Exact],
            "Filter pushdown should be exact for simple comparison"
        );

        // Verify we can actually scan with the filter
        let scan = table
            .scan(&ctx.state(), None, &[filter], None)
            .await
            .expect("scan should succeed");

        let batches = collect(scan, ctx.task_ctx())
            .await
            .expect("collect should succeed");

        assert!(!batches.is_empty(), "Should have results");
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2, "Should have 2 rows with id > 1");
    }

    /// Helper: create an in-memory SQLite table with columns (id: Int64, name: Utf8),
    /// insert the given rows, and return `(table_provider, session_context, schema)`.
    async fn setup_test_table(
        table_name: &str,
        ids: Vec<i64>,
        names: Vec<&str>,
    ) -> (
        Arc<dyn datafusion::datasource::TableProvider>,
        SessionContext,
        Arc<Schema>,
    ) {
        let schema = Arc::new(Schema::new(vec![
            datafusion::arrow::datatypes::Field::new("id", DataType::Int64, false),
            datafusion::arrow::datatypes::Field::new("name", DataType::Utf8, false),
        ]));
        let df_schema = ToDFSchema::to_dfschema_ref(Arc::clone(&schema)).expect("df schema");
        let external_table = CreateExternalTable {
            schema: df_schema,
            name: TableReference::bare(table_name),
            locations: vec![],
            file_type: String::new(),
            table_partition_cols: vec![],
            if_not_exists: true,
            definition: None,
            order_exprs: vec![],
            unbounded: false,
            options: HashMap::new(),
            constraints: Constraints::default(),
            column_defaults: HashMap::default(),
            temporary: false,
            or_replace: false,
        };
        let ctx = SessionContext::new();
        let table = SqliteTableProviderFactory::default()
            .create(&ctx.state(), &external_table)
            .await
            .expect("table should be created");

        let id_arr = Int64Array::from(ids);
        let name_arr = StringArray::from(names);
        let data = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(id_arr), Arc::new(name_arr)],
        )
        .expect("data should be created");

        let exec = MockExec::new(vec![Ok(data)], Arc::clone(&schema));
        let insertion = table
            .insert_into(&ctx.state(), Arc::new(exec), InsertOp::Append)
            .await
            .expect("insertion should be successful");
        collect(insertion, ctx.task_ctx())
            .await
            .expect("insert successful");

        (table, ctx, schema)
    }

    /// Helper: extract the count (u64) from a DML plan result.
    async fn extract_count(
        plan: Arc<dyn datafusion::physical_plan::ExecutionPlan>,
        ctx: &SessionContext,
    ) -> u64 {
        let batches = collect(plan, ctx.task_ctx())
            .await
            .expect("collect should succeed");
        assert_eq!(batches.len(), 1, "expected exactly one batch");
        let batch = &batches[0];
        let count_arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("count column should be UInt64Array");
        assert_eq!(count_arr.len(), 1);
        count_arr.value(0)
    }

    /// Helper: scan the table and return all rows as `(Vec<i64>, Vec<String>)`.
    async fn scan_all_rows(
        table: &Arc<dyn datafusion::datasource::TableProvider>,
        ctx: &SessionContext,
    ) -> (Vec<i64>, Vec<String>) {
        let scan = table
            .scan(&ctx.state(), None, &[], None)
            .await
            .expect("scan should succeed");
        let batches = collect(scan, ctx.task_ctx())
            .await
            .expect("collect should succeed");
        let mut ids = Vec::new();
        let mut names = Vec::new();
        for batch in &batches {
            let id_arr = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id column should be Int64Array");
            let name_arr = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("name column should be StringArray");
            for i in 0..batch.num_rows() {
                ids.push(id_arr.value(i));
                names.push(name_arr.value(i).to_string());
            }
        }
        (ids, names)
    }

    #[tokio::test]
    async fn test_delete_from_with_filter() {
        let (table, ctx, _schema) =
            setup_test_table("test_delete_filter", vec![1, 2, 3], vec!["a", "b", "c"]).await;

        // DELETE WHERE id = 2
        let filters = vec![col("id").eq(lit(2i64))];
        let plan = table
            .delete_from(&ctx.state(), filters)
            .await
            .expect("delete_from should succeed");

        let count = extract_count(plan, &ctx).await;
        assert_eq!(count, 1, "should have deleted exactly 1 row");

        // Verify remaining rows
        let (ids, names) = scan_all_rows(&table, &ctx).await;
        assert_eq!(ids.len(), 2, "should have 2 rows remaining");
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        // Check names correspond to their ids
        for (id, name) in ids.iter().zip(names.iter()) {
            match *id {
                1 => assert_eq!(name, "a"),
                3 => assert_eq!(name, "c"),
                other => panic!("unexpected id {other}"),
            }
        }
    }

    #[tokio::test]
    async fn test_delete_from_empty_filters() {
        let (table, ctx, _schema) =
            setup_test_table("test_delete_empty", vec![1, 2, 3], vec!["a", "b", "c"]).await;

        // DELETE with empty filters should delete all rows
        let plan = table
            .delete_from(&ctx.state(), vec![])
            .await
            .expect("delete_from should succeed");

        let count = extract_count(plan, &ctx).await;
        assert_eq!(
            count, 3,
            "should have deleted all 3 rows with empty filters"
        );

        // Verify no rows remain
        let (ids, _names) = scan_all_rows(&table, &ctx).await;
        assert_eq!(ids.len(), 0, "no rows should remain");
    }

    #[tokio::test]
    async fn test_update_with_filter() {
        let (table, ctx, _schema) =
            setup_test_table("test_update_filter", vec![1, 2, 3], vec!["a", "b", "c"]).await;

        // UPDATE SET name = 'updated' WHERE id = 2
        let assignments = vec![("name".to_string(), lit("updated"))];
        let filters = vec![col("id").eq(lit(2i64))];
        let plan = table
            .update(&ctx.state(), assignments, filters)
            .await
            .expect("update should succeed");

        let count = extract_count(plan, &ctx).await;
        assert_eq!(count, 1, "should have updated exactly 1 row");

        // Verify data
        let (ids, names) = scan_all_rows(&table, &ctx).await;
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
        let (table, ctx, _schema) =
            setup_test_table("test_update_no_filter", vec![1, 2], vec!["a", "b"]).await;

        // UPDATE SET name = 'all' (no filter -> all rows)
        let assignments = vec![("name".to_string(), lit("all"))];
        let plan = table
            .update(&ctx.state(), assignments, vec![])
            .await
            .expect("update should succeed");

        let count = extract_count(plan, &ctx).await;
        assert_eq!(count, 2, "should have updated 2 rows");

        // Verify all rows have name = "all"
        let (ids, names) = scan_all_rows(&table, &ctx).await;
        assert_eq!(ids.len(), 2, "should still have 2 rows");
        for name in &names {
            assert_eq!(name, "all");
        }
    }

    /// Nested columns are stored as JSON text and must come back as the Arrow values
    /// written, through both insert paths. Regression tests for spiceai/spiceai#3551: a list
    /// of structs (the `comments` column of a GitHub issues dataset), a list with a null
    /// item, a list of lists, a list of dates, a fixed-size list, a large list and a struct.
    #[cfg(feature = "federation")]
    mod nested_columns {
        use std::{collections::HashMap, sync::Arc};

        use datafusion::arrow::{
            array::{
                ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Date32Builder,
                Date64Array, Decimal128Array, Decimal256Array, FixedSizeBinaryArray,
                FixedSizeListBuilder, Float32Array, Float64Array, Int16Array, Int32Array,
                Int32Builder, Int64Array, Int8Array, LargeBinaryArray, LargeListBuilder,
                LargeStringArray, ListArray, ListBuilder, RecordBatch, StringArray, StringBuilder,
                StringViewArray, StructArray, StructBuilder, Time32MillisecondArray,
                Time32SecondArray, Time64MicrosecondArray, Time64NanosecondArray,
                TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
                TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
            },
            buffer::{NullBuffer, OffsetBuffer},
            datatypes::{i256, DataType, Field, Fields, Schema, SchemaRef, TimeUnit},
        };
        use datafusion::{
            catalog::TableProviderFactory,
            common::{Constraints, TableReference, ToDFSchema},
            execution::context::SessionContext,
            logical_expr::{dml::InsertOp, CreateExternalTable},
            physical_plan::collect,
        };
        use datafusion_federation::schema_cast::record_convert::try_cast_to;

        use crate::sqlite::SqliteTableProviderFactory;
        use crate::util::test::MockExec;

        fn comment_fields() -> Fields {
            Fields::from(vec![
                Field::new(
                    "author",
                    DataType::Struct(Fields::from(vec![Field::new(
                        "login",
                        DataType::Utf8,
                        true,
                    )])),
                    true,
                ),
                Field::new("body", DataType::Utf8, true),
            ])
        }

        fn comment_builder() -> StructBuilder {
            StructBuilder::new(
                comment_fields(),
                vec![
                    Box::new(StructBuilder::new(
                        Fields::from(vec![Field::new("login", DataType::Utf8, true)]),
                        vec![Box::new(StringBuilder::new())],
                    )),
                    Box::new(StringBuilder::new()),
                ],
            )
        }

        fn push_comment(item: &mut StructBuilder, login: &str, body: Option<&str>) {
            let author = item.field_builder::<StructBuilder>(0).expect("author");
            author
                .field_builder::<StringBuilder>(0)
                .expect("login")
                .append_value(login);
            author.append(true);
            item.field_builder::<StringBuilder>(1)
                .expect("body")
                .append_option(body);
            item.append(true);
        }

        /// Four issues: two comments (one with a null body), no comments, a null comment
        /// list, and one comment whose body needs JSON escaping.
        fn comments_column() -> ArrayRef {
            let mut list = ListBuilder::new(comment_builder());
            push_comment(list.values(), "alice", Some("hello"));
            push_comment(list.values(), "bob", None);
            list.append(true);
            list.append(true);
            list.append_null();
            push_comment(
                list.values(),
                "carol",
                Some("a \"quoted\" body, with ünïcödé and a \\ backslash"),
            );
            list.append(true);
            Arc::new(list.finish())
        }

        fn large_comments_column() -> ArrayRef {
            let mut list = LargeListBuilder::new(comment_builder());
            push_comment(list.values(), "alice", Some("hello"));
            list.append(true);
            list.append_null();
            list.append(true);
            push_comment(list.values(), "dave", None);
            list.append(true);
            Arc::new(list.finish())
        }

        fn nested_issues_batch() -> (RecordBatch, SchemaRef) {
            let mut tags = ListBuilder::new(StringBuilder::new());
            tags.append_value([Some("bug"), None, Some("sqlite")]);
            tags.append_value(Vec::<Option<&str>>::new());
            tags.append_null();
            tags.append_value([Some("accelerator")]);

            let mut matrix = ListBuilder::new(ListBuilder::new(Int32Builder::new()));
            matrix.values().append_value([Some(1), Some(2)]);
            matrix.values().append(true);
            matrix.values().append_null();
            matrix.append(true);
            matrix.append(true);
            matrix.append_null();
            matrix.values().append_value([Some(3), None]);
            matrix.append(true);

            let mut dates = ListBuilder::new(Date32Builder::new());
            dates.append_value([Some(0), Some(19_723), None]);
            dates.append_value(Vec::<Option<i32>>::new());
            dates.append_null();
            dates.append_value([Some(1)]);

            let mut pairs = FixedSizeListBuilder::new(Int32Builder::new(), 2);
            pairs.values().append_value(1);
            pairs.values().append_value(2);
            pairs.append(true);
            pairs.values().append_null();
            pairs.values().append_value(4);
            pairs.append(true);
            pairs.values().append_null();
            pairs.values().append_null();
            pairs.append(false);
            pairs.values().append_value(7);
            pairs.values().append_value(8);
            pairs.append(true);

            let meta_fields = Fields::from(vec![
                Field::new("n", DataType::Int32, true),
                Field::new("label", DataType::Utf8, true),
            ]);
            let mut meta = StructBuilder::new(
                meta_fields.clone(),
                vec![
                    Box::new(Int32Builder::new()),
                    Box::new(StringBuilder::new()),
                ],
            );
            for (n, label, valid) in [
                (Some(1), Some("x"), true),
                (None, Some("y"), true),
                (None, None, false),
                (Some(4), None, true),
            ] {
                meta.field_builder::<Int32Builder>(0)
                    .expect("n")
                    .append_option(n);
                meta.field_builder::<StringBuilder>(1)
                    .expect("label")
                    .append_option(label);
                meta.append(valid);
            }

            let comments = comments_column();
            let large_comments = large_comments_column();
            let tags: ArrayRef = Arc::new(tags.finish());
            let matrix: ArrayRef = Arc::new(matrix.finish());
            let dates: ArrayRef = Arc::new(dates.finish());
            let pairs: ArrayRef = Arc::new(pairs.finish());
            let meta: ArrayRef = Arc::new(meta.finish());

            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("comments", comments.data_type().clone(), true),
                Field::new("large_comments", large_comments.data_type().clone(), true),
                Field::new("tags", tags.data_type().clone(), true),
                Field::new("matrix", matrix.data_type().clone(), true),
                Field::new("dates", dates.data_type().clone(), true),
                Field::new("pairs", pairs.data_type().clone(), true),
                Field::new("meta", DataType::Struct(meta_fields), true),
            ]));
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                    comments,
                    large_comments,
                    tags,
                    matrix,
                    dates,
                    pairs,
                    meta,
                ],
            )
            .expect("nested issues batch");
            (batch, schema)
        }

        fn external_table(table_name: &str, schema: &SchemaRef) -> CreateExternalTable {
            CreateExternalTable {
                schema: ToDFSchema::to_dfschema_ref(Arc::clone(schema)).expect("df schema"),
                name: TableReference::bare(table_name),
                locations: vec![],
                file_type: String::new(),
                table_partition_cols: vec![],
                if_not_exists: true,
                definition: None,
                order_exprs: vec![],
                unbounded: false,
                options: HashMap::new(),
                constraints: Constraints::new_unverified(vec![]),
                column_defaults: HashMap::default(),
                temporary: false,
                or_replace: false,
            }
        }

        /// Creates `table_name` with `batch`'s schema, inserts `batch` through the given insert
        /// path, and returns `SELECT * ... ORDER BY id` as SQLite stores it, before any cast
        /// back to the nested types.
        async fn insert_and_read_back(
            table_name: &str,
            use_prepared_statements: bool,
            batch: &RecordBatch,
        ) -> Vec<RecordBatch> {
            let ctx = SessionContext::new();
            let schema = batch.schema();

            let table = SqliteTableProviderFactory::default()
                .with_batch_insert_use_prepared_statements(use_prepared_statements)
                .create(&ctx.state(), &external_table(table_name, &schema))
                .await
                .expect("a table with nested columns should be created");

            let exec = MockExec::new(vec![Ok(batch.clone())], Arc::clone(&schema));
            let insertion = table
                .insert_into(&ctx.state(), Arc::new(exec), InsertOp::Append)
                .await
                .expect("insertion should be planned");
            collect(insertion, ctx.task_ctx())
                .await
                .expect("insert should complete");

            ctx.register_table(table_name, Arc::clone(&table))
                .expect("table should be registered");
            ctx.sql(&format!("SELECT * FROM {table_name} ORDER BY id"))
                .await
                .expect("query should plan")
                .collect()
                .await
                .expect("should collect results")
        }

        /// The accelerator accepts the schema at create time (a list item type it does not
        /// support fails create with `The field 'comments' has an unsupported data type`), and
        /// every row, item and null survives the round trip through the given insert path.
        async fn round_trip(use_prepared_statements: bool, table_name: &str) {
            let (record_batch, schema) = nested_issues_batch();
            let result_batches =
                insert_and_read_back(table_name, use_prepared_statements, &record_batch).await;
            assert_eq!(result_batches.len(), 1, "one result batch");
            let result = &result_batches[0];

            // The nested columns reach the reader as the JSON text SQLite stored, so a cast
            // from a string to the nested type is exercised rather than a pass-through.
            assert_eq!(
                result
                    .schema()
                    .field_with_name("comments")
                    .expect("comments")
                    .data_type(),
                &DataType::Utf8,
                "the raw SQLite read returns the JSON text"
            );
            let casted = try_cast_to(result.clone(), Arc::clone(&schema))
                .expect("the JSON text should decode back into every nested column");
            assert_eq!(
                casted, record_batch,
                "round-tripped data should match the original"
            );
        }

        /// A null item inside a list of strings is stored as the JSON `null`, not as its
        /// type's zero value (an empty string).
        #[tokio::test]
        async fn a_null_item_in_a_list_of_strings_reads_back_null() {
            let mut tags = ListBuilder::new(StringBuilder::new());
            tags.append_value([Some("bug"), None, Some("sqlite")]);
            tags.append_value([None::<&str>]);
            let tags: ArrayRef = Arc::new(tags.finish());
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("tags", tags.data_type().clone(), true),
            ]));
            let record_batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(vec![1, 2])), tags],
            )
            .expect("batch");

            let result = insert_and_read_back("null_tags", true, &record_batch).await;
            let stored = result[0]
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("the raw SQLite read returns the JSON text");
            assert_eq!(stored.value(0), r#"["bug",null,"sqlite"]"#);
            assert_eq!(stored.value(1), "[null]");
            let casted = try_cast_to(result[0].clone(), Arc::clone(&schema)).expect("cast");
            assert_eq!(casted, record_batch, "a null item reads back as null");
        }

        /// One struct column holding every scalar the SQLite type check admits inside a nested
        /// column, and a list of that struct: row 0 carries a value in every field, row 1 a
        /// null in every field, row 2 is a null struct; the list holds rows 0-1, a null list,
        /// and row 2.
        fn leaf_types_batch() -> (RecordBatch, SchemaRef) {
            let children: Vec<(&str, ArrayRef)> = vec![
                ("i8", Arc::new(Int8Array::from(vec![Some(-1), None, None]))),
                (
                    "i16",
                    Arc::new(Int16Array::from(vec![Some(-300), None, None])),
                ),
                (
                    "i32",
                    Arc::new(Int32Array::from(vec![Some(i32::MIN), None, None])),
                ),
                (
                    "i64",
                    Arc::new(Int64Array::from(vec![Some(i64::MAX), None, None])),
                ),
                (
                    "u8",
                    Arc::new(UInt8Array::from(vec![Some(255), None, None])),
                ),
                (
                    "u16",
                    Arc::new(UInt16Array::from(vec![Some(65_535), None, None])),
                ),
                (
                    "u32",
                    Arc::new(UInt32Array::from(vec![Some(u32::MAX), None, None])),
                ),
                (
                    "u64",
                    Arc::new(UInt64Array::from(vec![Some(u64::MAX), None, None])),
                ),
                (
                    "f32",
                    Arc::new(Float32Array::from(vec![Some(-0.25), None, None])),
                ),
                (
                    "f64",
                    Arc::new(Float64Array::from(vec![Some(1.0e300), None, None])),
                ),
                (
                    "utf8",
                    Arc::new(StringArray::from(vec![
                        Some("a \"q\" \\ \u{1F600}"),
                        None,
                        None,
                    ])),
                ),
                (
                    "large_utf8",
                    Arc::new(LargeStringArray::from(vec![Some(""), None, None])),
                ),
                (
                    "utf8_view",
                    Arc::new(StringViewArray::from(vec![
                        Some("a view longer than twelve bytes"),
                        None,
                        None,
                    ])),
                ),
                (
                    "bool",
                    Arc::new(BooleanArray::from(vec![Some(false), None, None])),
                ),
                (
                    "binary",
                    Arc::new(BinaryArray::from(vec![
                        Some(b"\x00\xff".as_slice()),
                        None,
                        None,
                    ])),
                ),
                (
                    "large_binary",
                    Arc::new(LargeBinaryArray::from(vec![
                        Some(b"".as_slice()),
                        None,
                        None,
                    ])),
                ),
                (
                    "binary_view",
                    Arc::new(BinaryViewArray::from(vec![
                        Some(b"a binary view longer than twelve".as_slice()),
                        None,
                        None,
                    ])),
                ),
                (
                    "fixed_binary",
                    Arc::new(
                        FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                            vec![Some([1u8, 2]), None, None].into_iter(),
                            2,
                        )
                        .expect("fixed-size binary"),
                    ),
                ),
                (
                    "date32",
                    Arc::new(Date32Array::from(vec![Some(19_723), None, None])),
                ),
                (
                    "date64",
                    Arc::new(Date64Array::from(vec![Some(1_700_000_000_123), None, None])),
                ),
                (
                    "time32_s",
                    Arc::new(Time32SecondArray::from(vec![Some(86_399), None, None])),
                ),
                (
                    "time32_ms",
                    Arc::new(Time32MillisecondArray::from(vec![Some(1), None, None])),
                ),
                (
                    "time64_us",
                    Arc::new(Time64MicrosecondArray::from(vec![
                        Some(86_399_999_999),
                        None,
                        None,
                    ])),
                ),
                (
                    "time64_ns",
                    Arc::new(Time64NanosecondArray::from(vec![
                        Some(1_000_000_001),
                        None,
                        None,
                    ])),
                ),
                (
                    "ts_s",
                    Arc::new(TimestampSecondArray::from(vec![Some(-1), None, None])),
                ),
                (
                    "ts_ms",
                    Arc::new(TimestampMillisecondArray::from(vec![
                        Some(1_700_000_000_123),
                        None,
                        None,
                    ])),
                ),
                (
                    "ts_us",
                    Arc::new(TimestampMicrosecondArray::from(vec![
                        Some(1_700_000_000_123_456),
                        None,
                        None,
                    ])),
                ),
                (
                    "ts_ns",
                    Arc::new(TimestampNanosecondArray::from(vec![
                        Some(1_700_000_000_123_456_789),
                        None,
                        None,
                    ])),
                ),
                (
                    "ts_us_tz",
                    Arc::new(
                        TimestampMicrosecondArray::from(vec![
                            Some(1_700_000_000_123_456),
                            None,
                            None,
                        ])
                        .with_timezone("+02:00"),
                    ),
                ),
                (
                    "decimal128",
                    Arc::new(
                        Decimal128Array::from(vec![Some(-123_456), None, None])
                            .with_precision_and_scale(10, 3)
                            .expect("decimal128"),
                    ),
                ),
                (
                    "decimal256",
                    Arc::new(
                        Decimal256Array::from(vec![Some(i256::from(123_456_789_i64)), None, None])
                            .with_precision_and_scale(50, 5)
                            .expect("decimal256"),
                    ),
                ),
            ];
            let fields = Fields::from(
                children
                    .iter()
                    .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
                    .collect::<Vec<_>>(),
            );
            let arrays: Vec<ArrayRef> = children.into_iter().map(|(_, a)| a).collect();
            let leaves = StructArray::try_new(
                fields.clone(),
                arrays,
                Some(NullBuffer::from(vec![true, true, false])),
            )
            .expect("struct of every leaf type");
            let leaves: ArrayRef = Arc::new(leaves);
            let list_of_leaves: ArrayRef = Arc::new(ListArray::new(
                Arc::new(Field::new("item", DataType::Struct(fields.clone()), true)),
                OffsetBuffer::from_lengths([2, 0, 1]),
                Arc::clone(&leaves),
                Some(NullBuffer::from(vec![true, false, true])),
            ));
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("leaves", DataType::Struct(fields), true),
                Field::new("list_of_leaves", list_of_leaves.data_type().clone(), true),
            ]));
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(vec![1, 2, 3])),
                    leaves,
                    list_of_leaves,
                ],
            )
            .expect("leaf types batch");
            (batch, schema)
        }

        async fn leaf_types_round_trip(use_prepared_statements: bool, table_name: &str) {
            let (record_batch, schema) = leaf_types_batch();
            let result =
                insert_and_read_back(table_name, use_prepared_statements, &record_batch).await;
            assert_eq!(result.len(), 1, "one result batch");
            let casted = try_cast_to(result[0].clone(), Arc::clone(&schema))
                .expect("every leaf type should decode back from the stored JSON");
            for (i, field) in schema.fields().iter().enumerate() {
                assert_eq!(
                    casted.column(i),
                    record_batch.column(i),
                    "column '{}' reads back as written",
                    field.name()
                );
            }
        }

        #[tokio::test]
        async fn every_leaf_type_round_trips_inside_a_struct_and_a_list_through_prepared_statements(
        ) {
            leaf_types_round_trip(true, "leaf_types_prepared").await;
        }

        #[tokio::test]
        async fn every_leaf_type_round_trips_inside_a_struct_and_a_list_through_inline_sql() {
            leaf_types_round_trip(false, "leaf_types_inline").await;
        }

        /// Arrow's JSON encoder writes a `Duration` as an ISO 8601 period (`PT1S`) that its
        /// reader does not parse, so a nested column holding one is refused at create rather
        /// than written and then unreadable.
        #[tokio::test]
        async fn a_duration_inside_a_list_or_a_struct_is_refused_at_create() {
            let cases = [
                (
                    "durations",
                    DataType::new_list(DataType::Duration(TimeUnit::Second), true),
                ),
                (
                    "timing",
                    DataType::Struct(Fields::from(vec![Field::new(
                        "elapsed",
                        DataType::Duration(TimeUnit::Millisecond),
                        true,
                    )])),
                ),
            ];
            for (name, data_type) in cases {
                let schema = Arc::new(Schema::new(vec![
                    Field::new("id", DataType::Int64, false),
                    Field::new(name, data_type.clone(), true),
                ]));
                let ctx = SessionContext::new();
                let err = SqliteTableProviderFactory::default()
                    .create(&ctx.state(), &external_table(name, &schema))
                    .await
                    .expect_err("a nested Duration should be refused at create");
                let expected =
                    format!("The field '{name}' has an unsupported data type: {data_type}");
                assert!(
                    err.to_string().contains(&expected),
                    "expected the error to name the field and its type ({expected}), got: {err}"
                );
            }
        }

        /// Both insert paths store a struct as the same JSON text, with every field present
        /// (`null` for a null field) and nothing after the closing brace.
        #[tokio::test]
        async fn both_insert_paths_store_the_same_json_text_for_a_struct() {
            let meta_fields = Fields::from(vec![
                Field::new("n", DataType::Int32, true),
                Field::new("label", DataType::Utf8, true),
            ]);
            let meta: ArrayRef = Arc::new(
                StructArray::try_new(
                    meta_fields.clone(),
                    vec![
                        Arc::new(Int32Array::from(vec![Some(1), None, None])),
                        Arc::new(StringArray::from(vec![Some("x"), Some("y"), None])),
                    ],
                    Some(NullBuffer::from(vec![true, true, false])),
                )
                .expect("meta struct"),
            );
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("meta", DataType::Struct(meta_fields), true),
            ]));
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(vec![1, 2, 3])), meta],
            )
            .expect("batch");

            let stored_text = |batches: &[RecordBatch]| -> Vec<Option<String>> {
                batches[0]
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("the raw SQLite read returns the JSON text")
                    .iter()
                    .map(|v| v.map(str::to_owned))
                    .collect()
            };
            let prepared = stored_text(&insert_and_read_back("meta_prepared", true, &batch).await);
            let inline = stored_text(&insert_and_read_back("meta_inline", false, &batch).await);
            assert_eq!(
                prepared,
                vec![
                    Some(r#"{"n":1,"label":"x"}"#.to_owned()),
                    Some(r#"{"n":null,"label":"y"}"#.to_owned()),
                    None,
                ]
            );
            assert_eq!(inline, prepared, "the inline path stores the same text");
        }

        #[tokio::test]
        async fn nested_columns_round_trip_through_prepared_statements() {
            round_trip(true, "nested_issues_prepared").await;
        }

        #[tokio::test]
        async fn nested_columns_round_trip_through_inline_sql() {
            round_trip(false, "nested_issues_inline").await;
        }
    }
}
