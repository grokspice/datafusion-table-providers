use crate::mongodb::connection::StringComparison;
use crate::mongodb::connection_pool::MongoDBConnectionPool;
use crate::mongodb::utils::expression::{translate_filter, FilterScope};
use crate::mongodb::Error;
use crate::schema_projection::SchemaProjection;
use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::project_schema;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion::sql::TableReference;
use futures::TryStreamExt;
use mongodb::bson::{doc, Document};
use mongodb::options::Collation;
use serde_json;
use std::{collections::HashSet, fmt, sync::Arc};

#[derive(Debug)]
pub struct MongoDBTable {
    pool: Arc<MongoDBConnectionPool>,
    schema: SchemaRef,
    table_reference: Arc<TableReference>,
    projection: Option<SchemaProjection>,
    filter_scope: FilterScope,
    strings: StringComparison,
}

impl MongoDBTable {
    pub async fn new(
        pool: &Arc<MongoDBConnectionPool>,
        table_reference: impl Into<TableReference>,
        declared_schema: Option<SchemaRef>,
        projection: Option<SchemaProjection>,
    ) -> Result<Self, Error> {
        let table_reference = table_reference.into();
        let connection = pool.connect().await?;
        let schema = connection
            .get_schema(&table_reference, declared_schema)
            .await?;
        let strings = connection.string_comparison(table_reference.table()).await;

        // When a JSON-nesting / declared-schema projection is configured, the
        // exposed schema is the projected one (declared columns + catch-all).
        let schema = match &projection {
            Some(p) => p.project_schema(schema),
            None => schema,
        };

        // A JSON nesting catch-all is assembled from every undeclared field, so
        // it names no field a filter could be evaluated against.
        let unpushable_columns: HashSet<String> = projection
            .as_ref()
            .and_then(SchemaProjection::catch_all_name)
            .map(str::to_string)
            .into_iter()
            .collect();
        let filter_scope = FilterScope {
            unpushable_columns,
            unnest_depth: pool.unnest_depth(),
            // A catch-all folds in every undeclared field, so each scan reads
            // whole documents.
            whole_documents: projection
                .as_ref()
                .is_some_and(SchemaProjection::has_catch_all),
            code_point_strings: strings != StringComparison::Fixed,
        };

        Ok(Self {
            pool: Arc::clone(pool),
            schema,
            table_reference: Arc::new(table_reference),
            projection,
            filter_scope,
            strings,
        })
    }
}

#[async_trait]
impl TableProvider for MongoDBTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(MongoDBExec::new(
            Arc::clone(&self.table_reference),
            Arc::clone(&self.pool),
            Arc::clone(&self.schema),
            projection,
            filters,
            &self.filter_scope,
            self.strings,
            limit,
            self.projection.clone(),
        )?))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        Ok(supports_filters_pushdown(
            filters,
            &self.schema,
            &self.filter_scope,
        ))
    }
}

fn supports_filters_pushdown(
    filters: &[&Expr],
    schema: &SchemaRef,
    scope: &FilterScope,
) -> Vec<TableProviderFilterPushDown> {
    filters
        .iter()
        .map(|f| match translate_filter(f, schema, scope) {
            Some(filter) if filter.exact => TableProviderFilterPushDown::Exact,
            Some(_) => TableProviderFilterPushDown::Inexact,
            None => TableProviderFilterPushDown::Unsupported,
        })
        .collect()
}

#[derive(Debug)]
struct MongoDBExec {
    table_reference: Arc<TableReference>,
    pool: Arc<MongoDBConnectionPool>,
    projected_schema: SchemaRef,
    filters_doc: Document,
    collation: Option<Collation>,
    limit: Option<i32>,
    properties: Arc<PlanProperties>,
    schema_projection: Option<SchemaProjection>,
}

impl MongoDBExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        table_reference: Arc<TableReference>,
        pool: Arc<MongoDBConnectionPool>,
        schema: SchemaRef,
        projections: Option<&Vec<usize>>,
        filters: &[Expr],
        filter_scope: &FilterScope,
        strings: StringComparison,
        limit: Option<usize>,
        schema_projection: Option<SchemaProjection>,
    ) -> DataFusionResult<Self> {
        let mut projected_schema = project_schema(&schema, projections)?;

        // If no columns are specified, use _id - otherwise mongo returns an error
        if projected_schema.fields.is_empty() {
            let idx = schema.index_of("_id")?;
            projected_schema = SchemaRef::from(schema.project(&[idx])?);
        }

        let limit = limit
            .map(|u| {
                let Ok(u) = u32::try_from(u) else {
                    return Err(DataFusionError::Execution(
                        "Value is too large to fit in a u32".to_string(),
                    ));
                };
                if let Ok(u) = i32::try_from(u) {
                    Ok(u)
                } else {
                    Err(DataFusionError::Execution(
                        "Value is too large to fit in an i32".to_string(),
                    ))
                }
            })
            .transpose()?;

        // `DataFusion` hands the scan only the filters `supports_filters_pushdown`
        // accepted, and translating one again yields the same document, so a
        // failure here is a bug rather than a filter to skip: an exact filter is
        // no longer applied anywhere else.
        let mut documents = filters
            .iter()
            .map(|filter| {
                translate_filter(filter, &schema, filter_scope)
                    .map(|translated| translated.document)
                    .ok_or_else(|| {
                        DataFusionError::Internal(format!(
                            "MongoDB filter {filter} was accepted for pushdown but could not be translated"
                        ))
                    })
            })
            .collect::<DataFusionResult<Vec<_>>>()?;
        let mongo_filters_doc = match documents.len() {
            0 => Document::new(),
            1 => documents.pop().unwrap_or_default(),
            _ => doc! { "$and": documents },
        };
        let collation = strings.collation_for(&mongo_filters_doc);

        Ok(Self {
            table_reference: Arc::clone(&table_reference),
            pool,
            projected_schema: Arc::clone(&projected_schema),
            filters_doc: mongo_filters_doc,
            collation,
            limit,
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(projected_schema),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
            schema_projection,
        })
    }
}

impl DisplayAs for MongoDBExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> std::fmt::Result {
        let columns = self
            .projected_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>();

        let filters = serde_json::to_string(&self.filters_doc).map_err(|_| fmt::Error)?;

        write!(
            f,
            "MongoDBExec projection=[{}] filters=[{}]",
            columns.join(", "),
            filters,
        )?;

        if let Some(collation) = &self.collation {
            write!(f, " collation=[{}]", collation.locale)?;
        }

        if let Some(limit) = self.limit {
            write!(f, " limit=[{limit}]")?;
        }

        Ok(())
    }
}

impl ExecutionPlan for MongoDBExec {
    fn name(&self) -> &'static str {
        "MongoDBExec"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.projected_schema)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let schema = self.schema();

        let table_reference = Arc::clone(&self.table_reference);
        let pool = Arc::clone(&self.pool);
        let projected_schema = Arc::clone(&self.projected_schema);
        let filters_doc = self.filters_doc.clone();
        let collation = self.collation.clone();
        let limit = self.limit;
        let schema_projection = self.schema_projection.clone();

        let stream = futures::stream::once(async move {
            let conn = pool.connect().await.map_err(to_execution_error)?;

            conn.query_arrow(
                &table_reference,
                &projected_schema,
                &filters_doc,
                collation.as_ref(),
                limit,
                schema_projection.as_ref(),
            )
            .await
            .map_err(to_execution_error)
        })
        .try_flatten();

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

#[allow(clippy::needless_pass_by_value)]
pub fn to_execution_error(
    e: impl Into<Box<dyn std::error::Error + Send + Sync>>,
) -> DataFusionError {
    DataFusionError::Execution(format!("{}", e.into()).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::logical_expr::{col, lit, BinaryExpr, Expr, Operator};
    use datafusion::physical_plan::sort_pushdown::SortOrderPushdownResult;

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("_id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int32, true),
            Field::new("active", DataType::Boolean, true),
        ]))
    }

    fn scope() -> FilterScope {
        FilterScope {
            unpushable_columns: HashSet::new(),
            unnest_depth: Some(0),
            whole_documents: false,
            code_point_strings: true,
        }
    }

    fn format_exec(exec: &MongoDBExec) -> String {
        struct Wrapper<'a>(&'a MongoDBExec);
        impl fmt::Display for Wrapper<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt_as(DisplayFormatType::Default, f)
            }
        }
        format!("{}", Wrapper(exec))
    }

    fn exec(filters: &[Expr], limit: Option<usize>) -> DataFusionResult<MongoDBExec> {
        exec_comparing(filters, StringComparison::CodePoint, limit)
    }

    fn exec_comparing(
        filters: &[Expr],
        strings: StringComparison,
        limit: Option<usize>,
    ) -> DataFusionResult<MongoDBExec> {
        MongoDBExec::new(
            Arc::new(TableReference::bare("users")),
            Arc::new(MongoDBConnectionPool::new_stub()),
            test_schema(),
            None,
            filters,
            &scope(),
            strings,
            limit,
            None,
        )
    }

    #[test]
    fn pushdown_is_exact_only_where_the_translation_is() {
        let exprs = [
            // Exact: the integer column's accepted types are guarded.
            col("age").not_eq(lit(30)),
            col("active").is_true(),
            // Inexact: a string column renders values MongoDB cannot compare as strings.
            col("name").eq(lit("alice")),
            // Unsupported: arithmetic, and a column against a column.
            Expr::BinaryExpr(BinaryExpr::new(
                Box::new(col("age")),
                Operator::Modulo,
                Box::new(lit(2)),
            ))
            .eq(lit(0)),
            col("name").eq(col("_id")),
        ];
        let refs: Vec<&Expr> = exprs.iter().collect();
        assert_eq!(
            supports_filters_pushdown(&refs, &test_schema(), &scope()),
            vec![
                TableProviderFilterPushDown::Exact,
                TableProviderFilterPushDown::Exact,
                TableProviderFilterPushDown::Inexact,
                TableProviderFilterPushDown::Unsupported,
                TableProviderFilterPushDown::Unsupported,
            ]
        );
    }

    #[test]
    fn a_catch_all_column_is_not_pushed_down() {
        let scope = FilterScope {
            unpushable_columns: HashSet::from(["name".to_string()]),
            ..scope()
        };
        assert_eq!(
            supports_filters_pushdown(&[&col("name").is_null()], &test_schema(), &scope),
            vec![TableProviderFilterPushDown::Unsupported]
        );
    }

    #[tokio::test]
    async fn filters_are_combined_with_and() {
        let exec =
            exec(&[col("age").gt(lit(18)), col("active").eq(lit(true))], None).expect("exec");
        let conditions = exec.filters_doc.get_array("$and").expect("$and");
        assert_eq!(conditions.len(), 2);
        let display = format_exec(&exec);
        // Over integers, `age > 18` is `age >= 19`.
        assert!(display.contains(r#""$gte":19"#), "{display}");
        assert!(display.contains(r#""$eq":true"#), "{display}");
    }

    #[tokio::test]
    async fn a_filter_that_cannot_be_translated_is_an_error_not_skipped() {
        let modulo = Expr::BinaryExpr(BinaryExpr::new(
            Box::new(col("age")),
            Operator::Modulo,
            Box::new(lit(2)),
        ))
        .eq(lit(0));
        assert!(exec(&[modulo], None).is_err());
    }

    #[tokio::test]
    async fn no_filters_is_an_empty_document() {
        let exec = exec(&[], Some(100)).expect("exec");
        assert!(exec.filters_doc.is_empty());
        let display = format_exec(&exec);
        assert!(display.contains("filters=[{}]"), "{display}");
        assert!(display.contains("limit=[100]"), "{display}");
    }

    #[tokio::test]
    async fn an_empty_projection_reads_the_id() {
        let exec = MongoDBExec::new(
            Arc::new(TableReference::bare("users")),
            Arc::new(MongoDBConnectionPool::new_stub()),
            test_schema(),
            Some(&vec![]),
            &[],
            &scope(),
            StringComparison::CodePoint,
            None,
            None,
        )
        .expect("exec");
        assert!(format_exec(&exec).contains("projection=[_id]"));
    }

    #[tokio::test]
    async fn a_collated_collection_compares_strings_by_code_point_only_when_a_filter_orders_them() {
        let ordered = [col("name").gt(lit("b"))];
        let excluded = [col("name").not_eq(lit("bob"))];
        let equal = [col("name").eq(lit("bob")), col("age").gt(lit(18))];

        for filters in [&ordered[..], &excluded[..]] {
            let exec = exec_comparing(filters, StringComparison::Collated, None).expect("exec");
            assert_eq!(
                exec.collation.as_ref().map(|c| c.locale.as_str()),
                Some("simple")
            );
            assert!(format_exec(&exec).contains("collation=[simple]"));
        }
        // A collation only widens `$in`, and it is the collection's indexes
        // that are built with it.
        let exec = exec_comparing(&equal, StringComparison::Collated, None).expect("exec");
        assert!(exec.collation.is_none());
        // Without a collation there is nothing to replace.
        let exec = exec_comparing(&ordered, StringComparison::CodePoint, None).expect("exec");
        assert!(exec.collation.is_none());
    }

    #[tokio::test]
    async fn a_limit_beyond_i32_is_an_error() {
        assert!(exec(&[], Some(usize::MAX)).is_err());
    }

    /// MongoDB sorts a null or missing field first, orders values by BSON type
    /// before value, puts NaN below every number and an array at its smallest
    /// element; none of that is how `DataFusion` orders the converted rows, so
    /// the scan must not claim an ordering.
    #[tokio::test]
    async fn a_sort_is_not_pushed_down() {
        use datafusion::arrow::compute::SortOptions;
        use datafusion::physical_expr::expressions::Column as PhysicalColumn;
        use datafusion::physical_expr::PhysicalSortExpr;

        let exec = exec(&[], None).expect("exec");
        let order = [PhysicalSortExpr::new(
            Arc::new(PhysicalColumn::new("age", 2)),
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        )];
        assert!(matches!(
            exec.try_pushdown_sort(&order).expect("pushdown"),
            SortOrderPushdownResult::Unsupported
        ));
        assert!(!format_exec(&exec).contains("sort="));
    }
}
