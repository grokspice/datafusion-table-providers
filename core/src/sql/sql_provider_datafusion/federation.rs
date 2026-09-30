use crate::sql::db_connection_pool::{dbconnection::get_schema, JoinPushDown};
use async_trait::async_trait;
use datafusion::physical_expr::PhysicalExpr;
use datafusion_federation::sql::{
    RemoteTableRef, SQLExecutor, SQLFederationProvider, SQLTableSource,
};
use datafusion_federation::{FederatedTableProviderAdaptor, FederatedTableSource};
use futures::TryStreamExt;
use snafu::prelude::*;
use std::sync::Arc;

use crate::sql::sql_provider_datafusion::{
    get_stream, to_execution_error, SqlTable, UnableToGetSchemaSnafu,
};
use crate::util::supported_functions::contains_unsupported_functions;
use datafusion::{
    arrow::datatypes::SchemaRef,
    common::TableReference,
    error::{DataFusionError, Result as DataFusionResult},
    logical_expr::LogicalPlan,
    physical_plan::{stream::RecordBatchStreamAdapter, SendableRecordBatchStream},
    sql::unparser::dialect::{DefaultDialect, Dialect},
};

impl<T, P> SqlTable<T, P> {
    #[allow(dead_code)]
    fn arc_dialect(&self) -> Arc<dyn Dialect + Send + Sync> {
        match &self.dialect {
            Some(dialect) => Arc::clone(dialect),
            None => Arc::new(DefaultDialect {}),
        }
    }

    fn create_federated_table_source(self: Arc<Self>) -> Arc<dyn FederatedTableSource> {
        let table_reference = self.table_reference.clone();
        let schema = Arc::clone(&self.schema);
        let fed_provider = Arc::new(SQLFederationProvider::new(self));
        Arc::new(SQLTableSource::new_with_schema(
            fed_provider,
            RemoteTableRef::from(table_reference),
            schema,
        ))
    }

    /// Creates a federated table provider.
    ///
    /// # Errors
    ///
    /// This function currently never returns an error, but the Result type is kept for API compatibility.
    pub fn create_federated_table_provider(
        self: Arc<Self>,
    ) -> DataFusionResult<FederatedTableProviderAdaptor> {
        let table_source = Self::create_federated_table_source(Arc::clone(&self));
        Ok(FederatedTableProviderAdaptor::new_with_provider(
            table_source,
            self,
        ))
    }
}

#[async_trait]
impl<T, P> SQLExecutor for SqlTable<T, P> {
    fn name(&self) -> &str {
        self.name
    }

    fn compute_context(&self) -> Option<String> {
        match self.pool.join_push_down() {
            JoinPushDown::AllowedFor(context) => Some(context),
            // Don't return None here - it will cause incorrect federation with other providers of the same name that also have a compute_context of None.
            // Instead return a random string that will never match any other provider's context.
            JoinPushDown::Disallow => Some(format!("{}", std::ptr::from_ref(self) as usize)),
        }
    }

    /// Return the provided [`Dialect`], defaulting to [`DefaultDialect`].
    fn dialect(&self) -> Arc<dyn Dialect> {
        let Some(ref dialect) = self.dialect else {
            // TODO: Derive default from [`SQLExecutor::engine`].
            return Arc::new(DefaultDialect {});
        };
        Arc::clone(dialect) as Arc<_>
    }

    fn can_execute_plan(&self, plan: &LogicalPlan) -> bool {
        // Default to not federating the plan if a [`FunctionSupport`] policy is
        // configured and the plan contains an unsupported function; otherwise
        // allow federation. Functions that can't be federated are left for
        // DataFusion to execute locally instead of being unparsed into remote SQL.
        //
        // Fail safe: if the support check itself errors, treat the plan as
        // containing unsupported functions (i.e. do not federate) so we never
        // unparse a plan the remote can't handle.
        self.function_support.as_ref().is_none_or(|func_supp| {
            !contains_unsupported_functions(plan, func_supp).unwrap_or(true)
        })
    }

    fn execute(
        &self,
        query: &str,
        schema: SchemaRef,
        _filters: &[Arc<dyn PhysicalExpr>],
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let fut = get_stream(
            Arc::clone(&self.pool),
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
        let conn = self.pool.connect().await.map_err(to_execution_error)?;
        get_schema(conn, &TableReference::from(table_name))
            .await
            .context(UnableToGetSchemaSnafu)
            .map_err(to_execution_error)
    }
}

#[cfg(test)]
mod tests {
    use std::{any::Any, error::Error};

    use datafusion::{
        arrow::datatypes::{DataType, Field, Schema},
        common::DFSchema,
        logical_expr::{builder::LogicalTableSource, col, lit, Expr, LogicalPlanBuilder},
    };

    use crate::{
        sql::db_connection_pool::{dbconnection::DbConnection, DbConnectionPool, JoinPushDown},
        util::supported_functions::FunctionSupport,
    };

    use super::*;

    struct MockConn;

    impl DbConnection<(), &'static dyn ToString> for MockConn {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    struct MockDBPool;

    #[async_trait]
    impl DbConnectionPool<(), &'static dyn ToString> for MockDBPool {
        async fn connect(
            &self,
        ) -> Result<Box<dyn DbConnection<(), &'static dyn ToString>>, Box<dyn Error + Send + Sync>>
        {
            Ok(Box::new(MockConn))
        }

        fn join_push_down(&self) -> JoinPushDown {
            JoinPushDown::Disallow
        }
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("val", DataType::Utf8, true),
        ]))
    }

    fn test_table() -> SqlTable<(), &'static dyn ToString> {
        let pool = Arc::new(MockDBPool)
            as Arc<dyn DbConnectionPool<(), &'static dyn ToString> + Send + Sync>;
        let support = FunctionSupport::new(None, None, None).with_expression_support(Arc::new(
            |expr: &Expr, _: Option<&DFSchema>| {
                !matches!(expr, Expr::Like(like) if like.case_insensitive)
            },
        ));

        SqlTable::new_with_schema("test", &pool, schema(), "test", None)
            .with_function_support(Some(support))
    }

    fn scan_plan() -> LogicalPlan {
        let source = Arc::new(LogicalTableSource::new(schema()))
            as Arc<dyn datafusion::logical_expr::TableSource>;
        LogicalPlanBuilder::scan("test", source, None)
            .expect("scan")
            .build()
            .expect("build")
    }

    fn filter_plan(predicate: Expr) -> LogicalPlan {
        LogicalPlanBuilder::from(scan_plan())
            .filter(predicate)
            .expect("filter")
            .build()
            .expect("build")
    }

    #[test]
    fn case_insensitive_like_filter_is_not_federated() {
        let plan = filter_plan(col("val").ilike(lit("u%")));

        assert!(!SQLExecutor::can_execute_plan(&test_table(), &plan));
    }

    #[test]
    fn negated_case_insensitive_like_filter_is_not_federated() {
        let plan = filter_plan(col("val").not_ilike(lit("u%")));

        assert!(!SQLExecutor::can_execute_plan(&test_table(), &plan));
    }

    #[test]
    fn case_insensitive_like_projection_is_not_federated() {
        let plan = LogicalPlanBuilder::from(scan_plan())
            .project(vec![col("val").ilike(lit("u%")).alias("matched")])
            .expect("project")
            .build()
            .expect("build");

        assert!(!SQLExecutor::can_execute_plan(&test_table(), &plan));
    }

    #[test]
    fn ordinary_like_is_still_federated() {
        let plan = filter_plan(col("val").like(lit("u%")));

        assert!(SQLExecutor::can_execute_plan(&test_table(), &plan));
    }
}
