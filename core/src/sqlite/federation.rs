use crate::sql::db_connection_pool::dbconnection::{get_schema, Error as DbError};
use crate::sql::sql_provider_datafusion::{get_stream, to_execution_error};
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::sql::sqlparser::ast::{self, VisitMut};
use datafusion::sql::unparser::dialect::Dialect;
use datafusion_federation::sql::{
    ast_analyzer::AstAnalyzer, RemoteTableRef, SQLExecutor, SQLFederationProvider, SQLTableSource,
};
use datafusion_federation::{FederatedTableProviderAdaptor, FederatedTableSource};
use futures::TryStreamExt;
use snafu::ResultExt;
use std::sync::Arc;

use super::between::SQLiteBetweenVisitor;
use super::sql_table::SQLiteTable;
use super::sqlite_interval::SQLiteIntervalVisitor;
use datafusion::{
    common::TableReference,
    datasource::TableProvider,
    error::{DataFusionError, Result as DataFusionResult},
    execution::SendableRecordBatchStream,
    physical_plan::stream::RecordBatchStreamAdapter,
};

impl<T, P> SQLiteTable<T, P> {
    fn create_federated_table_source(
        self: Arc<Self>,
    ) -> DataFusionResult<Arc<dyn FederatedTableSource>> {
        let table_reference = self.base_table.table_reference.clone();
        let schema = Arc::clone(&Arc::clone(&self).base_table.schema());
        let fed_provider = Arc::new(SQLFederationProvider::new(self));
        Ok(Arc::new(SQLTableSource::new_with_schema(
            fed_provider,
            RemoteTableRef::from(table_reference),
            schema,
        )))
    }

    pub fn create_federated_table_provider(
        self: Arc<Self>,
    ) -> DataFusionResult<FederatedTableProviderAdaptor> {
        let table_source = Self::create_federated_table_source(Arc::clone(&self))?;
        Ok(FederatedTableProviderAdaptor::new_with_provider(
            table_source,
            self,
        ))
    }
}

fn sqlite_ast_analyzer(
    decimal_between: bool,
) -> impl Fn(ast::Statement) -> Result<ast::Statement, DataFusionError> {
    move |ast: ast::Statement| -> Result<ast::Statement, DataFusionError> {
        match ast {
            ast::Statement::Query(query) => {
                let mut new_query = query.clone();

                // normalize INTERVAL usage into SQLite-compatible datetime expressions
                let mut interval_visitor = SQLiteIntervalVisitor::default();
                let _ = new_query.visit(&mut interval_visitor);

                // rewrite BETWEEN clauses with numeric operands to decimal comparisons
                if decimal_between {
                    let mut between_visitor = SQLiteBetweenVisitor::default();
                    let _ = new_query.visit(&mut between_visitor);
                }

                Ok(ast::Statement::Query(new_query))
            }
            _ => Ok(ast),
        }
    }
}

#[async_trait]
impl<T, P> SQLExecutor for SQLiteTable<T, P> {
    fn name(&self) -> &str {
        self.base_table.name()
    }

    fn compute_context(&self) -> Option<String> {
        self.base_table.compute_context()
    }

    fn dialect(&self) -> Arc<dyn Dialect> {
        self.base_table.dialect()
    }

    fn ast_analyzer(&self) -> Option<AstAnalyzer> {
        let rule = Box::new(sqlite_ast_analyzer(self.decimal_between));
        Some(AstAnalyzer::new(vec![rule]))
    }

    fn can_execute_plan(&self, plan: &LogicalPlan) -> bool {
        self.base_table.can_execute_plan(plan)
    }

    fn execute(
        &self,
        query: &str,
        schema: SchemaRef,
        _filters: &[Arc<dyn PhysicalExpr>],
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let fut = get_stream(
            self.base_table.clone_pool(),
            query.to_string(),
            Arc::clone(&schema),
        );

        let stream = futures::stream::once(fut).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    async fn table_names(&self) -> DataFusionResult<Vec<String>> {
        Err(DataFusionError::NotImplemented(
            "table inference not implemented".to_string(),
        ))
    }

    async fn get_table_schema(&self, table_name: &str) -> DataFusionResult<SchemaRef> {
        let conn = self
            .base_table
            .clone_pool()
            .connect()
            .await
            .map_err(to_execution_error)?;
        get_schema(conn, &TableReference::from(table_name))
            .await
            .boxed()
            .map_err(|e| DbError::UnableToGetSchema { source: e })
            .map_err(to_execution_error)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::TableReference;
    use datafusion::execution::context::SessionContext;

    use crate::sql::arrow_sql_gen::statement::{CreateTableBuilder, InsertBuilder};
    use crate::sql::db_connection_pool::sqlitepool::SqliteConnectionPoolFactory;
    use crate::sql::db_connection_pool::{DbConnectionPool, Mode};
    use crate::sqlite::sql_table::SQLiteTable;
    use crate::sqlite::DynSqliteConnectionPool;

    /// A federated session over an in-memory SQLite table holding `batch`.
    async fn federated_session(table: &str, batch: RecordBatch) -> SessionContext {
        let schema = batch.schema();
        let pool = SqliteConnectionPoolFactory::new(
            ":memory:",
            Mode::Memory,
            std::time::Duration::from_millis(5000),
        )
        .build()
        .await
        .expect("pool");
        let conn = pool.connect().await.expect("connection");
        let conn = conn.as_async().expect("async connection");
        conn.execute(
            &CreateTableBuilder::new(Arc::clone(&schema), table).build_sqlite(),
            &[],
        )
        .await
        .expect("table created");
        conn.execute(
            &InsertBuilder::new(&TableReference::from(table), &vec![batch])
                .build_sqlite(None)
                .expect("insert statement"),
            &[],
        )
        .await
        .expect("rows inserted");
        let pool: Arc<DynSqliteConnectionPool> = Arc::new(pool);
        let provider = Arc::new(SQLiteTable::new_with_schema(&pool, schema, table, None))
            .create_federated_table_provider()
            .expect("federated provider");
        let ctx = SessionContext::new_with_state(datafusion_federation::default_session_state());
        ctx.register_table(table, Arc::new(provider))
            .expect("table registered");
        ctx
    }

    /// An error SQLite raises while running a federated statement fails the query
    /// instead of answering with the rows produced before it: an ungrouped `SUM`
    /// that overflows `i64` must not come back as an empty result.
    #[tokio::test]
    async fn an_error_sqlite_raises_mid_query_fails_the_federated_query() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(Int64Array::from(vec![9_000_000_000_000_000_000_i64; 3])),
            ],
        )
        .expect("batch");
        let ctx = federated_session("ovf", batch).await;
        let df = ctx
            .sql("SELECT sum(v) AS s FROM ovf")
            .await
            .expect("the query plans");
        let plan = df
            .clone()
            .create_physical_plan()
            .await
            .expect("physical plan");
        let display = datafusion::physical_plan::displayable(plan.as_ref())
            .indent(true)
            .to_string();
        assert!(
            display.contains("VirtualExecutionPlan") && display.contains("sum("),
            "the SUM must be pushed down to SQLite for the overflow to be its error:
{display}"
        );
        let err = df
            .collect()
            .await
            .expect_err("SQLite's integer overflow must fail the query");
        assert!(
            err.to_string()
                .contains("Failed to read the next row: integer overflow"),
            "unexpected error: {err}"
        );
    }
}
