//! # SQL DataFusion TableProvider
//!
//! This module implements a SQL TableProvider for DataFusion.
//!
//! This is used as a fallback if the `datafusion-federation` optimizer is not enabled.

use crate::{
    sql::db_connection_pool::{
        self,
        dbconnection::{get_schema, query_arrow},
        DbConnectionPool,
    },
    util::supported_functions::FunctionSupport,
};
use async_trait::async_trait;
use datafusion::{
    catalog::Session,
    common::Constraints,
    sql::unparser::{
        dialect::{DefaultDialect, Dialect},
        Unparser,
    },
};
use expr::Engine;
use futures::TryStreamExt;
use snafu::prelude::*;
use std::fmt::Display;
use std::{fmt, sync::Arc};

use datafusion::{
    arrow::datatypes::SchemaRef,
    common::TableReference,
    config::ConfigOptions,
    datasource::TableProvider,
    error::{DataFusionError, Result as DataFusionResult},
    execution::TaskContext,
    logical_expr::{Expr, TableProviderFilterPushDown, TableType},
    physical_expr::{EquivalenceProperties, PhysicalSortExpr},
    physical_plan::{
        execution_plan::{Boundedness, EmissionType},
        filter_pushdown::{
            ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation, PushedDown,
        },
        sort_pushdown::SortOrderPushdownResult,
        stream::RecordBatchStreamAdapter,
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream,
    },
};

pub mod expr;
#[cfg(feature = "federation")]
pub mod federation;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Unable to get a DB connection from the pool: {source}"))]
    UnableToGetConnectionFromPool { source: db_connection_pool::Error },

    #[snafu(display("Unable to get schema: {source}"))]
    UnableToGetSchema {
        source: db_connection_pool::dbconnection::Error,
    },

    #[snafu(display("Unable to generate SQL: {source}"))]
    UnableToGenerateSQL { source: expr::Error },

    #[snafu(display("Unable to generate SQL: {source}"))]
    UnableToGenerateSQLDataFusion { source: DataFusionError },
}

impl From<Error> for DataFusionError {
    fn from(e: Error) -> Self {
        DataFusionError::External(Box::new(e))
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub struct SqlTable<T: 'static, P: 'static> {
    name: &'static str,
    pool: Arc<dyn DbConnectionPool<T, P> + Send + Sync>,
    schema: SchemaRef,
    pub table_reference: TableReference,
    engine: Option<Engine>,
    pub(crate) dialect: Option<Arc<dyn Dialect + Send + Sync>>,
    constraints: Option<Constraints>,
    pub(crate) function_support: Option<FunctionSupport>,
    allow_physical_filter_pushdown: bool,
    allow_physical_sort_pushdown: bool,
}

impl<T, P> std::fmt::Debug for SqlTable<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqlTable")
            .field("name", &self.name)
            .field("table_reference", &self.table_reference)
            .field("schema", &self.schema)
            .field("engine", &self.engine)
            .field("dialect", &self.dialect.is_some())
            .field("constraints", &self.constraints)
            .field("function_support", &self.function_support)
            .finish_non_exhaustive()
    }
}

impl<T, P> SqlTable<T, P> {
    /// Creates a new SQL table.
    ///
    /// # Errors
    ///
    /// Returns an error if the schema cannot be retrieved from the database.
    pub async fn new(
        name: &'static str,
        pool: &Arc<dyn DbConnectionPool<T, P> + Send + Sync>,
        table_reference: impl Into<TableReference>,
        engine: Option<expr::Engine>,
    ) -> Result<Self> {
        let table_reference = table_reference.into();
        let conn = pool
            .connect()
            .await
            .context(UnableToGetConnectionFromPoolSnafu)?;

        let schema = get_schema(conn, &table_reference)
            .await
            .context(UnableToGetSchemaSnafu)?;

        Ok(Self {
            name,
            pool: Arc::clone(pool),
            schema,
            table_reference,
            engine,
            dialect: None,
            constraints: None,
            function_support: None,
            allow_physical_filter_pushdown: true,
            allow_physical_sort_pushdown: true,
        })
    }

    pub fn new_with_schema(
        name: &'static str,
        pool: &Arc<dyn DbConnectionPool<T, P> + Send + Sync>,
        schema: impl Into<SchemaRef>,
        table_reference: impl Into<TableReference>,
        engine: Option<expr::Engine>,
    ) -> Self {
        Self {
            name,
            pool: Arc::clone(pool),
            schema: schema.into(),
            table_reference: table_reference.into(),
            engine,
            dialect: None,
            constraints: None,
            function_support: None,
            allow_physical_filter_pushdown: true,
            allow_physical_sort_pushdown: true,
        }
    }

    #[must_use]
    pub fn with_function_support(mut self, function_support: Option<FunctionSupport>) -> Self {
        self.function_support = function_support;
        self
    }

    #[must_use]
    pub fn with_constraints_opt(mut self, constraints: Option<Constraints>) -> Self {
        self.constraints = constraints;
        self
    }

    #[must_use]
    pub fn with_constraints(mut self, constraints: Constraints) -> Self {
        self.constraints = Some(constraints);
        self
    }

    /// Disables physical-level filter pushdown on the underlying `SqlExec`.
    ///
    /// When set to `false`, physical filters (`FilterExec` predicates) will NOT
    /// be absorbed into the generated SQL. This is required for databases with
    /// limited query-language support (e.g. ScyllaDB/CQL) where the logical-level
    /// `TableProvider::supports_filters_pushdown` intentionally rejects filters.
    #[must_use]
    pub fn with_allow_physical_filter_pushdown(mut self, allow: bool) -> Self {
        self.allow_physical_filter_pushdown = allow;
        self
    }

    /// Disables physical-level sort pushdown on the underlying `SqlExec`.
    ///
    /// When set to `false`, a `SortExec` above the scan will NOT be absorbed into
    /// the generated SQL as an `ORDER BY`, and the sort is performed by DataFusion
    /// instead. This is required for query languages that cannot order an arbitrary
    /// scan: CQL (ScyllaDB/Cassandra) accepts `ORDER BY` only when the partition key
    /// is restricted by `=` or `IN`, and rejects the statement otherwise.
    #[must_use]
    pub fn with_allow_physical_sort_pushdown(mut self, allow: bool) -> Self {
        self.allow_physical_sort_pushdown = allow;
        self
    }

    pub fn scan_to_sql(
        &self,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<String> {
        let projected_schema = project_schema_safe(&self.schema, projection)
            .context(UnableToGenerateSQLDataFusionSnafu)?;

        let dialect = self.dialect_arc();

        // Only quote when dialect has identifier. Note: `char::default()` is '\0' (which we don't want).
        let quote_identifier = |identifier: &str| match dialect.identifier_quote_style(identifier) {
            Some(quote) => format!("{quote}{identifier}{quote}"),
            None => identifier.to_string(),
        };

        let columns = projected_schema
            .fields()
            .iter()
            .map(|f| quote_identifier(f.name()))
            .collect::<Vec<_>>()
            .join(", ");

        let limit_expr = match limit {
            Some(limit) => format!("LIMIT {limit}"),
            None => String::new(),
        };

        let where_expr = if filters.is_empty() {
            String::new()
        } else {
            let unparser = Unparser::new(dialect.as_ref());
            let filter_expr = filters
                .iter()
                .map(|f| unparser.expr_to_sql(f).map(|s| s.to_string()))
                .collect::<std::result::Result<Vec<String>, DataFusionError>>()
                .context(UnableToGenerateSQLDataFusionSnafu)?;
            format!("WHERE {}", filter_expr.join(" AND "))
        };

        let table_expr = match &self.table_reference {
            TableReference::Full {
                catalog,
                schema,
                table,
            } => format!(
                "{}.{}.{}",
                quote_identifier(catalog),
                quote_identifier(schema),
                quote_identifier(table)
            ),
            TableReference::Partial { schema, table } => {
                format!("{}.{}", quote_identifier(schema), quote_identifier(table))
            }
            TableReference::Bare { table } => quote_identifier(table),
        };

        let mut sql = format!("SELECT {columns} FROM {table_expr}");
        if !where_expr.is_empty() {
            sql.push(' ');
            sql.push_str(&where_expr);
        }
        if !limit_expr.is_empty() {
            sql.push(' ');
            sql.push_str(&limit_expr);
        }

        Ok(sql)
    }

    fn create_physical_plan(
        &self,
        projection: Option<&Vec<usize>>,
        sql: String,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let exec = SqlExec::new(
            projection,
            &self.schema(),
            Arc::clone(&self.pool),
            sql,
            self.dialect_arc(),
        )?
        .with_function_support(self.function_support.clone())
        .with_allow_physical_filter_pushdown(self.allow_physical_filter_pushdown)
        .with_allow_physical_sort_pushdown(self.allow_physical_sort_pushdown);
        Ok(Arc::new(exec))
    }

    #[must_use]
    pub fn with_dialect(self, dialect: Arc<dyn Dialect + Send + Sync>) -> Self {
        Self {
            dialect: Some(dialect),
            ..self
        }
    }

    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    #[must_use]
    pub fn clone_pool(&self) -> Arc<dyn DbConnectionPool<T, P> + Send + Sync> {
        Arc::clone(&self.pool)
    }

    pub(crate) fn function_support(&self) -> Option<FunctionSupport> {
        self.function_support.clone()
    }

    /// Returns a cloneable Arc of the dialect for passing to SqlExec.
    pub fn dialect_arc(&self) -> Arc<dyn Dialect + Send + Sync> {
        match &self.dialect {
            Some(dialect) => Arc::clone(dialect),
            None => Arc::new(DefaultDialect {}),
        }
    }
}

#[async_trait]
impl<T, P> TableProvider for SqlTable<T, P> {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn constraints(&self) -> Option<&Constraints> {
        self.constraints.as_ref()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        let dialect = self.dialect_arc();
        let unparser = Unparser::new(dialect.as_ref());

        // A filter here resolves against the table's own schema, which is also
        // what carries the source-declared column metadata a per-call check may
        // need. `None` would be safe but pessimistic: a check that refuses on an
        // unknown type would drop every such filter from the pushdown.
        let scope = datafusion::common::DFSchema::try_from(self.schema()).ok();

        let filter_push_down: Vec<TableProviderFilterPushDown> = filters
            .iter()
            .map(|f| {
                // Expressions containing subqueries or outer references must not be
                // pushed down. Subqueries are handled at the plan level and may
                // reference tables in other databases not accessible from this
                // connection. Outer references refer to columns from an enclosing
                // query that the table provider cannot resolve.
                if expr::expr_contains_subquery_or_outer_ref(f).unwrap_or(true) {
                    return TableProviderFilterPushDown::Unsupported;
                }
                // Respect the configured function-support policy. A filter that
                // references a scalar/aggregate/window function the target engine
                // cannot execute must not be pushed down.
                if self
                    .function_support
                    .as_ref()
                    .is_some_and(|func_support| !func_support.supports(f, scope.as_ref()))
                {
                    return TableProviderFilterPushDown::Unsupported;
                }
                // Use the same Unparser that scan_to_sql() uses for actual SQL
                // generation, so the capability check matches what can really
                // be converted to SQL for the target dialect.
                match unparser.expr_to_sql(f) {
                    Ok(_) => TableProviderFilterPushDown::Exact,
                    Err(_) => TableProviderFilterPushDown::Unsupported,
                }
            })
            .collect();

        Ok(filter_push_down)
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let sql = self.scan_to_sql(projection, filters, limit)?;
        return self.create_physical_plan(projection, sql);
    }
}

impl<T, P> Display for SqlTable<T, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SqlTable {}", self.name)
    }
}

pub fn project_schema_safe(
    schema: &SchemaRef,
    projection: Option<&Vec<usize>>,
) -> DataFusionResult<SchemaRef> {
    let schema = match projection {
        Some(columns) => {
            if columns.is_empty() {
                Arc::clone(schema)
            } else {
                Arc::new(schema.project(columns)?)
            }
        }
        None => Arc::clone(schema),
    };
    Ok(schema)
}

pub struct SqlExec<T, P> {
    projected_schema: SchemaRef,
    pool: Arc<dyn DbConnectionPool<T, P> + Send + Sync>,
    sql: String,
    properties: Arc<PlanProperties>,
    dialect: Arc<dyn Dialect + Send + Sync>,
    function_support: Option<FunctionSupport>,
    allow_physical_filter_pushdown: bool,
    allow_physical_sort_pushdown: bool,
}

impl<T, P> Clone for SqlExec<T, P> {
    fn clone(&self) -> Self {
        Self {
            projected_schema: Arc::clone(&self.projected_schema),
            pool: Arc::clone(&self.pool),
            sql: self.sql.clone(),
            properties: self.properties.clone(),
            dialect: Arc::clone(&self.dialect),
            function_support: self.function_support.clone(),
            allow_physical_filter_pushdown: self.allow_physical_filter_pushdown,
            allow_physical_sort_pushdown: self.allow_physical_sort_pushdown,
        }
    }
}

impl<T, P> SqlExec<T, P> {
    pub fn new(
        projection: Option<&Vec<usize>>,
        schema: &SchemaRef,
        pool: Arc<dyn DbConnectionPool<T, P> + Send + Sync>,
        sql: String,
        dialect: Arc<dyn Dialect + Send + Sync>,
    ) -> DataFusionResult<Self> {
        let projected_schema = project_schema_safe(schema, projection)?;

        Ok(Self {
            projected_schema: Arc::clone(&projected_schema),
            pool,
            sql,
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(projected_schema),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Incremental,
                Boundedness::Bounded,
            )),
            dialect,
            function_support: None,
            allow_physical_filter_pushdown: true,
            allow_physical_sort_pushdown: true,
        })
    }

    /// Applies the same expression and function policy used for logical filter
    /// admission to physical filters before they are absorbed into remote SQL.
    #[must_use]
    pub fn with_function_support(mut self, function_support: Option<FunctionSupport>) -> Self {
        self.function_support = function_support;
        self
    }

    /// Disables physical-level filter pushdown (`handle_child_pushdown_result`).
    ///
    /// When set to `false`, this `SqlExec` will reject all physical filter
    /// pushdown requests, keeping filters as `FilterExec` nodes above the scan.
    /// This is required for databases with limited SQL/query-language support
    /// (e.g. ScyllaDB/CQL) where the logical-level `TableProvider::supports_filters_pushdown`
    /// intentionally rejects certain filters.
    #[must_use]
    pub fn with_allow_physical_filter_pushdown(mut self, allow: bool) -> Self {
        self.allow_physical_filter_pushdown = allow;
        self
    }

    /// Returns whether physical-level filter pushdown is enabled.
    #[must_use]
    pub fn allow_physical_filter_pushdown(&self) -> bool {
        self.allow_physical_filter_pushdown
    }

    /// Disables physical-level sort pushdown (`try_pushdown_sort`).
    ///
    /// When set to `false`, this `SqlExec` will reject sort pushdown requests, so
    /// the `SortExec` above it is kept and DataFusion performs the sort. This is
    /// required for query languages that cannot order an arbitrary scan: CQL
    /// (ScyllaDB/Cassandra) accepts `ORDER BY` only when the partition key is
    /// restricted by `=` or `IN`, and rejects the statement otherwise.
    #[must_use]
    pub fn with_allow_physical_sort_pushdown(mut self, allow: bool) -> Self {
        self.allow_physical_sort_pushdown = allow;
        self
    }

    /// Returns whether physical-level sort pushdown is enabled.
    #[must_use]
    pub fn allow_physical_sort_pushdown(&self) -> bool {
        self.allow_physical_sort_pushdown
    }

    #[must_use]
    pub fn clone_pool(&self) -> Arc<dyn DbConnectionPool<T, P> + Send + Sync> {
        Arc::clone(&self.pool)
    }

    pub fn sql(&self) -> Result<String> {
        Ok(self.sql.clone())
    }

    fn dialect(&self) -> &(dyn Dialect + Send + Sync) {
        self.dialect.as_ref()
    }
}

impl<T, P> std::fmt::Debug for SqlExec<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let sql = self.sql().unwrap_or_default();
        write!(f, "SqlExec sql={sql}")
    }
}

impl<T, P> DisplayAs for SqlExec<T, P> {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> std::fmt::Result {
        let sql = self.sql().unwrap_or_default();
        write!(f, "SqlExec sql={sql}")
    }
}

/// Convert a `PhysicalExpr` to a logical `Expr` for use with the Unparser.
/// Returns `None` if the expression cannot be safely converted.
fn physical_expr_to_logical_expr(
    expr: &Arc<dyn datafusion::physical_plan::PhysicalExpr>,
) -> Option<Expr> {
    use datafusion::physical_expr::expressions::{
        BinaryExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, Literal, NegativeExpr,
        NotExpr,
    };

    if let Some(col) = expr.downcast_ref::<Column>() {
        return Some(Expr::Column(datafusion::common::Column::new_unqualified(
            col.name(),
        )));
    }

    if let Some(lit) = expr.downcast_ref::<Literal>() {
        return Some(Expr::Literal(lit.value().clone(), None));
    }

    if let Some(bin) = expr.downcast_ref::<BinaryExpr>() {
        let left = physical_expr_to_logical_expr(bin.left())?;
        let right = physical_expr_to_logical_expr(bin.right())?;
        return Some(Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr::new(
            Box::new(left),
            *bin.op(),
            Box::new(right),
        )));
    }

    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        let inner = physical_expr_to_logical_expr(not.arg())?;
        return Some(Expr::Not(Box::new(inner)));
    }

    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        let inner = physical_expr_to_logical_expr(is_null.arg())?;
        return Some(Expr::IsNull(Box::new(inner)));
    }

    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        let inner = physical_expr_to_logical_expr(is_not_null.arg())?;
        return Some(Expr::IsNotNull(Box::new(inner)));
    }

    if let Some(neg) = expr.downcast_ref::<NegativeExpr>() {
        let inner = physical_expr_to_logical_expr(neg.arg())?;
        return Some(Expr::Negative(Box::new(inner)));
    }

    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        let inner = physical_expr_to_logical_expr(cast.expr())?;
        return Some(Expr::Cast(datafusion::logical_expr::Cast::new(
            Box::new(inner),
            cast.cast_type().clone(),
        )));
    }

    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        let value = physical_expr_to_logical_expr(in_list.expr())?;
        let list: Option<Vec<Expr>> = in_list
            .list()
            .iter()
            .map(physical_expr_to_logical_expr)
            .collect();
        let list = list?;
        return Some(Expr::InList(datafusion::logical_expr::expr::InList::new(
            Box::new(value),
            list,
            in_list.negated(),
        )));
    }

    None
}

/// Convert a `PhysicalExpr` to a SQL string fragment for filter pushdown,
/// using the Unparser with the given dialect for correct identifier quoting
/// and operator support.
/// Returns `None` if the expression cannot be safely converted.
fn physical_expr_to_sql(
    expr: &Arc<dyn datafusion::physical_plan::PhysicalExpr>,
    dialect: &dyn Dialect,
    function_support: Option<&FunctionSupport>,
) -> Option<String> {
    let logical_expr = physical_expr_to_logical_expr(expr)?;
    if function_support.is_some_and(|support| !support.supports(&logical_expr, None)) {
        return None;
    }
    let unparser = Unparser::new(dialect);
    unparser
        .expr_to_sql(&logical_expr)
        .ok()
        .map(|e| e.to_string())
}

/// Insert additional WHERE conditions into a SQL string.
/// Handles the case where a WHERE clause already exists (appends with AND)
/// and where it doesn't (inserts WHERE before ORDER BY / LIMIT / end).
fn insert_where_clause(sql: &str, conditions: &str) -> String {
    let upper = sql.to_uppercase();

    // Find existing WHERE clause
    if let Some(where_pos) = upper.find(" WHERE ") {
        // Find the end of the WHERE clause (before ORDER BY, LIMIT, GROUP BY, HAVING)
        let after_where = where_pos + 7; // skip " WHERE "

        // Find the position of the next major clause
        let next_clause_pos = [" ORDER BY ", " LIMIT ", " GROUP BY ", " HAVING "]
            .iter()
            .filter_map(|kw| upper[after_where..].find(kw).map(|p| after_where + p))
            .min()
            .unwrap_or(sql.len());

        format!(
            "{} AND ({conditions}){}",
            &sql[..next_clause_pos],
            &sql[next_clause_pos..]
        )
    } else {
        // No WHERE clause — insert before ORDER BY / LIMIT or at end
        let insert_pos = [" ORDER BY ", " LIMIT ", " GROUP BY ", " HAVING "]
            .iter()
            .filter_map(|kw| upper.find(kw))
            .min()
            .unwrap_or(sql.len());

        format!(
            "{} WHERE {conditions}{}",
            &sql[..insert_pos],
            &sql[insert_pos..]
        )
    }
}

impl<T: 'static, P: 'static> ExecutionPlan for SqlExec<T, P> {
    fn name(&self) -> &'static str {
        "SqlExec"
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

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(
            &Arc<dyn datafusion::physical_plan::PhysicalExpr>,
        ) -> datafusion::common::Result<
            datafusion::common::tree_node::TreeNodeRecursion,
        >,
    ) -> datafusion::common::Result<datafusion::common::tree_node::TreeNodeRecursion> {
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }

    fn fetch(&self) -> Option<usize> {
        let upper = self.sql.to_uppercase();
        let limit_pos = upper.rfind(" LIMIT ")?;
        let after_limit = &self.sql[limit_pos + 7..]; // skip " LIMIT "
        after_limit.trim().parse::<usize>().ok()
    }

    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
    ) -> DataFusionResult<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
        use datafusion::logical_expr::expr::Sort;
        use datafusion::physical_expr::expressions::Column;

        if !self.allow_physical_sort_pushdown || order.is_empty() {
            return Ok(SortOrderPushdownResult::Unsupported);
        }

        let unparser = Unparser::new(self.dialect());

        // Build ORDER BY clause from physical sort expressions using the Unparser
        let mut order_by_parts = Vec::with_capacity(order.len());
        for sort_expr in order {
            let Some(col) = sort_expr.expr.downcast_ref::<Column>() else {
                return Ok(SortOrderPushdownResult::Unsupported);
            };
            let logical_sort = Sort::new(
                Expr::Column(datafusion::common::Column::new_unqualified(col.name())),
                !sort_expr.options.descending,
                sort_expr.options.nulls_first,
            );
            let order_by_expr = unparser.sort_to_sql(&logical_sort).map_err(|e| {
                DataFusionError::Internal(format!("Failed to unparse sort expression: {e}"))
            })?;
            order_by_parts.push(order_by_expr.to_string());
        }

        let order_by_clause = format!("ORDER BY {}", order_by_parts.join(", "));

        // Insert ORDER BY before LIMIT (if present) or at the end
        let sql = &self.sql;
        let new_sql = if let Some(limit_pos) = sql.to_uppercase().rfind(" LIMIT ") {
            format!(
                "{} {order_by_clause}{}",
                &sql[..limit_pos],
                &sql[limit_pos..]
            )
        } else {
            format!("{sql} {order_by_clause}")
        };

        let mut new_exec = Self {
            projected_schema: Arc::clone(&self.projected_schema),
            pool: Arc::clone(&self.pool),
            sql: new_sql,
            properties: self.properties.clone(),
            dialect: Arc::clone(&self.dialect),
            function_support: self.function_support.clone(),
            allow_physical_filter_pushdown: self.allow_physical_filter_pushdown,
            allow_physical_sort_pushdown: self.allow_physical_sort_pushdown,
        };

        // Update equivalence properties to reflect the output ordering
        let eq_properties = EquivalenceProperties::new_with_orderings(
            Arc::clone(&self.projected_schema),
            vec![order.to_vec()],
        );
        new_exec.properties =
            Arc::new(PlanProperties::clone(&new_exec.properties).with_eq_properties(eq_properties));

        Ok(SortOrderPushdownResult::Exact {
            inner: Arc::new(new_exec),
        })
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        let limit = limit?;

        let sql = &self.sql;
        let upper = sql.to_uppercase();

        // Replace existing LIMIT or append one
        let new_sql = if let Some(limit_pos) = upper.rfind(" LIMIT ") {
            format!("{} LIMIT {limit}", &sql[..limit_pos])
        } else {
            format!("{sql} LIMIT {limit}")
        };

        Some(Arc::new(Self {
            projected_schema: Arc::clone(&self.projected_schema),
            pool: Arc::clone(&self.pool),
            sql: new_sql,
            properties: self.properties.clone(),
            dialect: Arc::clone(&self.dialect),
            function_support: self.function_support.clone(),
            allow_physical_filter_pushdown: self.allow_physical_filter_pushdown,
            allow_physical_sort_pushdown: self.allow_physical_sort_pushdown,
        }))
    }

    fn handle_child_pushdown_result(
        &self,
        _phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> DataFusionResult<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        if !self.allow_physical_filter_pushdown {
            let filter_results = child_pushdown_result
                .parent_filters
                .iter()
                .map(|_| PushedDown::No)
                .collect::<Vec<_>>();
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                filter_results,
            ));
        }

        let mut accepted_filters = Vec::new();
        let mut filter_results = Vec::with_capacity(child_pushdown_result.parent_filters.len());
        let mut any_accepted = false;

        for parent_filter in &child_pushdown_result.parent_filters {
            match physical_expr_to_sql(
                &parent_filter.filter,
                self.dialect(),
                self.function_support.as_ref(),
            ) {
                Some(sql_fragment) => {
                    accepted_filters.push(sql_fragment);
                    filter_results.push(PushedDown::Yes);
                    any_accepted = true;
                }
                None => {
                    filter_results.push(PushedDown::No);
                }
            }
        }

        if !any_accepted {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                filter_results,
            ));
        }

        let where_clause = accepted_filters.join(" AND ");

        // Insert WHERE conditions into the SQL
        let new_sql = insert_where_clause(&self.sql, &where_clause);

        let new_exec = Self {
            projected_schema: Arc::clone(&self.projected_schema),
            pool: Arc::clone(&self.pool),
            sql: new_sql,
            properties: self.properties.clone(),
            dialect: Arc::clone(&self.dialect),
            function_support: self.function_support.clone(),
            allow_physical_filter_pushdown: self.allow_physical_filter_pushdown,
            allow_physical_sort_pushdown: self.allow_physical_sort_pushdown,
        };

        Ok(
            FilterPushdownPropagation::with_parent_pushdown_result(filter_results)
                .with_updated_node(Arc::new(new_exec) as Arc<dyn ExecutionPlan>),
        )
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let sql = self.sql().map_err(to_execution_error)?;
        tracing::debug!("SqlExec sql: {sql}");

        let schema = self.schema();

        let fut = get_stream(Arc::clone(&self.pool), sql, Arc::clone(&schema));

        let stream = futures::stream::once(fut).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

pub async fn get_stream<T: 'static, P: 'static>(
    pool: Arc<dyn DbConnectionPool<T, P> + Send + Sync>,
    sql: String,
    projected_schema: SchemaRef,
) -> DataFusionResult<SendableRecordBatchStream> {
    let conn = pool.connect().await.map_err(to_execution_error)?;

    query_arrow(conn, sql, Some(projected_schema))
        .await
        .map_err(to_execution_error)
}

#[allow(clippy::needless_pass_by_value)]
pub fn to_execution_error(
    e: impl Into<Box<dyn std::error::Error + Send + Sync>>,
) -> DataFusionError {
    DataFusionError::Execution(format!("{}", e.into()).to_string())
}

#[cfg(test)]
mod tests {
    use std::{error::Error, sync::Arc};

    use datafusion::common::TableReference;
    use datafusion::execution::context::SessionContext;
    use tracing::{level_filters::LevelFilter, subscriber::DefaultGuard, Dispatch};

    use crate::sql::sql_provider_datafusion::SqlTable;

    fn setup_tracing() -> DefaultGuard {
        let subscriber: tracing_subscriber::FmtSubscriber = tracing_subscriber::fmt()
            .with_max_level(LevelFilter::DEBUG)
            .finish();

        let dispatch = Dispatch::new(subscriber);
        tracing::dispatcher::set_default(&dispatch)
    }

    mod sql_table_plan_to_sql_tests {
        use std::any::Any;

        use async_trait::async_trait;
        use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        use datafusion::sql::unparser::dialect::{Dialect, SqliteDialect};
        use datafusion::{
            common::TableReference,
            logical_expr::{col, lit},
        };

        use crate::sql::db_connection_pool::{
            dbconnection::DbConnection, DbConnectionPool, JoinPushDown,
        };

        use super::*;

        struct MockConn {}

        impl DbConnection<(), &'static dyn ToString> for MockConn {
            fn as_any(&self) -> &dyn Any {
                self
            }

            fn as_any_mut(&mut self) -> &mut dyn Any {
                self
            }
        }

        struct MockDBPool {}

        #[async_trait]
        impl DbConnectionPool<(), &'static dyn ToString> for MockDBPool {
            async fn connect(
                &self,
            ) -> Result<
                Box<dyn DbConnection<(), &'static dyn ToString>>,
                Box<dyn Error + Send + Sync>,
            > {
                Ok(Box::new(MockConn {}))
            }

            fn join_push_down(&self) -> JoinPushDown {
                JoinPushDown::Disallow
            }
        }

        fn new_sql_table(
            table_reference: &'static str,
            dialect: Option<Arc<dyn Dialect + Send + Sync>>,
        ) -> Result<SqlTable<(), &'static dyn ToString>, Box<dyn Error + Send + Sync>> {
            let fields = vec![
                Field::new("name", DataType::Utf8, false),
                Field::new("age", DataType::Int16, false),
                Field::new(
                    "createdDate",
                    DataType::Timestamp(TimeUnit::Millisecond, None),
                    false,
                ),
                Field::new("userId", DataType::LargeUtf8, false),
                Field::new("active", DataType::Boolean, false),
                Field::new("5e48", DataType::LargeUtf8, false),
            ];
            let schema = Arc::new(Schema::new(fields));
            let pool = Arc::new(MockDBPool {})
                as Arc<dyn DbConnectionPool<(), &'static dyn ToString> + Send + Sync>;
            let table_ref = TableReference::parse_str(table_reference);

            let sql_table =
                SqlTable::new_with_schema(table_reference, &pool, schema, table_ref, None);
            if let Some(dialect) = dialect {
                Ok(sql_table.with_dialect(dialect))
            } else {
                Ok(sql_table)
            }
        }

        #[tokio::test]
        async fn test_sql_to_string() -> Result<(), Box<dyn Error + Send + Sync>> {
            let sql_table = new_sql_table("users", Some(Arc::new(SqliteDialect {})))?;
            let result = sql_table.scan_to_sql(Some(&vec![0]), &[], None).unwrap();
            assert_eq!(result, "SELECT `name` FROM `users`");
            Ok(())
        }

        #[tokio::test]
        async fn test_sql_to_string_default_dialect_has_no_nul(
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            // Pick identifiers that should never be quoted by `DefaultDialect`.
            let schema = Arc::new(Schema::new(vec![Field::new(
                "zz_safe_identifier",
                DataType::Utf8,
                false,
            )]));
            let pool = Arc::new(MockDBPool {})
                as Arc<dyn DbConnectionPool<(), &'static dyn ToString> + Send + Sync>;
            let table_ref = TableReference::parse_str("zz_users");
            let sql_table = SqlTable::new_with_schema("users", &pool, schema, table_ref, None);

            let result = sql_table.scan_to_sql(Some(&vec![0]), &[], None).unwrap();
            assert_eq!(result, "SELECT zz_safe_identifier FROM zz_users");
            assert!(!result.contains('\0'));
            Ok(())
        }

        #[tokio::test]
        async fn test_sql_to_string_with_filters_and_limit(
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            let filters = vec![col("age").gt_eq(lit(30)).and(col("name").eq(lit("x")))];
            let sql_table = new_sql_table("users", Some(Arc::new(SqliteDialect {})))?;
            let result = sql_table
                .scan_to_sql(Some(&vec![0, 1]), &filters, Some(3))
                .unwrap();
            assert_eq!(
                result,
                "SELECT `name`, `age` FROM `users` WHERE ((`age` >= 30) AND (`name` = 'x')) LIMIT 3"
            );
            Ok(())
        }

        #[tokio::test]
        async fn test_scan_to_sql_full_table_reference() -> Result<(), Box<dyn Error + Send + Sync>>
        {
            let sql_table = new_sql_table(
                "my_catalog.my_schema.users",
                Some(Arc::new(SqliteDialect {})),
            )?;
            let result = sql_table.scan_to_sql(Some(&vec![0]), &[], None).unwrap();
            assert_eq!(
                result,
                "SELECT `name` FROM `my_catalog`.`my_schema`.`users`"
            );
            Ok(())
        }

        #[tokio::test]
        async fn test_scan_to_sql_partial_table_reference(
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            let sql_table = new_sql_table("my_schema.users", Some(Arc::new(SqliteDialect {})))?;
            let result = sql_table.scan_to_sql(Some(&vec![0]), &[], None).unwrap();
            assert_eq!(result, "SELECT `name` FROM `my_schema`.`users`");
            Ok(())
        }

        #[tokio::test]
        async fn test_scan_to_sql_bare_table_reference() -> Result<(), Box<dyn Error + Send + Sync>>
        {
            let sql_table = new_sql_table("users", Some(Arc::new(SqliteDialect {})))?;
            let result = sql_table.scan_to_sql(Some(&vec![0]), &[], None).unwrap();
            assert_eq!(result, "SELECT `name` FROM `users`");
            Ok(())
        }
    }

    mod supports_filters_pushdown_tests {
        use std::any::Any;
        use std::error::Error;
        use std::sync::Arc;

        use async_trait::async_trait;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use datafusion::common::DFSchema;
        use datafusion::common::TableReference;
        use datafusion::datasource::TableProvider;
        use datafusion::logical_expr::expr::ScalarFunction;
        use datafusion::logical_expr::{
            col, create_udf, lit, ColumnarValue, Expr, ScalarUDF, TableProviderFilterPushDown,
            Volatility,
        };
        use datafusion::sql::unparser::dialect::SqliteDialect;

        use crate::sql::db_connection_pool::{
            dbconnection::DbConnection, DbConnectionPool, JoinPushDown,
        };
        use crate::sql::sql_provider_datafusion::SqlTable;
        use crate::util::supported_functions::{FunctionRestriction, FunctionSupport};

        struct MockConn {}

        impl DbConnection<(), &'static dyn ToString> for MockConn {
            fn as_any(&self) -> &dyn Any {
                self
            }

            fn as_any_mut(&mut self) -> &mut dyn Any {
                self
            }
        }

        struct MockDBPool {}

        #[async_trait]
        impl DbConnectionPool<(), &'static dyn ToString> for MockDBPool {
            async fn connect(
                &self,
            ) -> Result<
                Box<dyn DbConnection<(), &'static dyn ToString>>,
                Box<dyn Error + Send + Sync>,
            > {
                Ok(Box::new(MockConn {}))
            }

            fn join_push_down(&self) -> JoinPushDown {
                JoinPushDown::Disallow
            }
        }

        /// Builds a table whose support policy refuses `denied_fn` and
        /// case-insensitive `LIKE` expressions.
        fn test_table() -> SqlTable<(), &'static dyn ToString> {
            let schema = Arc::new(Schema::new(vec![
                Field::new("name", DataType::Utf8, false),
                Field::new("age", DataType::Int16, false),
            ]));
            let pool = Arc::new(MockDBPool {})
                as Arc<dyn DbConnectionPool<(), &'static dyn ToString> + Send + Sync>;
            let function_support = FunctionSupport::new(
                Some(FunctionRestriction::Deny(vec!["denied_fn".to_string()])),
                None,
                None,
            )
            .with_expression_support(Arc::new(|expr: &Expr, _: Option<&DFSchema>| {
                !matches!(expr, Expr::Like(like) if like.case_insensitive)
            }));
            SqlTable::new_with_schema("users", &pool, schema, TableReference::bare("users"), None)
                .with_dialect(Arc::new(SqliteDialect {}))
                .with_function_support(Some(function_support))
        }

        /// A call to the denied scalar function: `denied_fn(name)`.
        fn denied_call() -> Expr {
            let udf: Arc<ScalarUDF> = Arc::new(create_udf(
                "denied_fn",
                vec![DataType::Utf8],
                DataType::Utf8,
                Volatility::Immutable,
                Arc::new(|args: &[ColumnarValue]| Ok(args[0].clone())),
            ));
            Expr::ScalarFunction(ScalarFunction::new_udf(udf, vec![col("name")]))
        }

        fn assert_pushdown(filter: Expr, expected: TableProviderFilterPushDown) {
            let table = test_table();
            let result = table
                .supports_filters_pushdown(&[&filter])
                .expect("supports_filters_pushdown should succeed");
            assert_eq!(result, vec![expected]);
        }

        #[test]
        fn allowed_filter_is_pushed_down() {
            assert_pushdown(col("age").gt(lit(30)), TableProviderFilterPushDown::Exact);
        }

        #[test]
        fn denied_function_alone_is_not_pushed_down() {
            assert_pushdown(
                denied_call().eq(lit("x")),
                TableProviderFilterPushDown::Unsupported,
            );
        }

        #[test]
        fn case_insensitive_like_is_not_pushed_down() {
            assert_pushdown(
                col("name").ilike(lit("u%")),
                TableProviderFilterPushDown::Unsupported,
            );
        }

        #[test]
        fn negated_case_insensitive_like_is_not_pushed_down() {
            assert_pushdown(
                col("name").not_ilike(lit("u%")),
                TableProviderFilterPushDown::Unsupported,
            );
        }

        #[test]
        fn case_insensitive_like_nested_in_a_filter_is_not_pushed_down() {
            assert_pushdown(
                col("age").gt(lit(30)).or(col("name").ilike(lit("u%"))),
                TableProviderFilterPushDown::Unsupported,
            );
        }

        #[test]
        fn ordinary_like_is_still_pushed_down() {
            assert_pushdown(
                col("name").like(lit("u%")),
                TableProviderFilterPushDown::Exact,
            );
        }

        #[test]
        fn denied_function_in_and_is_not_pushed_down() {
            let filter = col("age").gt(lit(30)).and(denied_call().eq(lit("x")));
            assert_pushdown(filter, TableProviderFilterPushDown::Unsupported);
        }

        #[test]
        fn denied_function_in_or_is_not_pushed_down() {
            let filter = col("age").gt(lit(30)).or(denied_call().eq(lit("x")));
            assert_pushdown(filter, TableProviderFilterPushDown::Unsupported);
        }

        #[test]
        fn denied_function_nested_deeply_is_not_pushed_down() {
            // (age > 30 AND name = 'a') OR (age < 5 AND denied_fn(name) = 'x')
            let filter = col("age")
                .gt(lit(30))
                .and(col("name").eq(lit("a")))
                .or(col("age").lt(lit(5)).and(denied_call().eq(lit("x"))));
            assert_pushdown(filter, TableProviderFilterPushDown::Unsupported);
        }

        #[test]
        fn denied_function_negated_is_not_pushed_down() {
            let filter = !denied_call().eq(lit("x"));
            assert_pushdown(filter, TableProviderFilterPushDown::Unsupported);
        }

        #[test]
        fn complex_allowed_expression_is_pushed_down() {
            // A nested AND/OR of only allowed expressions stays Exact.
            let filter = col("age")
                .gt(lit(30))
                .and(col("name").eq(lit("a")).or(col("age").lt(lit(5))));
            assert_pushdown(filter, TableProviderFilterPushDown::Exact);
        }

        #[test]
        fn mixed_filters_are_evaluated_independently() {
            // Each filter in the slice is judged on its own merits.
            let table = test_table();
            let allowed = col("age").gt(lit(30));
            let denied = denied_call().eq(lit("x"));
            let result = table
                .supports_filters_pushdown(&[&allowed, &denied])
                .expect("supports_filters_pushdown should succeed");
            assert_eq!(
                result,
                vec![
                    TableProviderFilterPushDown::Exact,
                    TableProviderFilterPushDown::Unsupported,
                ]
            );
        }
    }

    mod sort_pushdown_tests {
        use crate::sql::sql_provider_datafusion::{SqlExec, SqlTable};
        use crate::util::supported_functions::FunctionSupport;
        use arrow::compute::SortOptions;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use datafusion::common::config::ConfigOptions;
        use datafusion::common::TableReference;
        use datafusion::logical_expr::Operator;
        use datafusion::physical_expr::expressions::{binary, col, lit, Column};
        use datafusion::physical_expr::PhysicalSortExpr;
        use datafusion::physical_plan::filter_pushdown::{
            ChildFilterPushdownResult, ChildPushdownResult, FilterPushdownPhase, PushedDown,
        };
        use datafusion::physical_plan::sort_pushdown::SortOrderPushdownResult;
        use datafusion::physical_plan::ExecutionPlan;
        use datafusion::sql::unparser::dialect::DefaultDialect;
        use std::sync::Arc;

        use crate::sql::db_connection_pool::{
            dbconnection::DbConnection, DbConnectionPool, JoinPushDown,
        };

        struct MockConn {}

        impl DbConnection<(), &'static dyn ToString> for MockConn {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                self
            }
        }

        struct MockDBPool {}

        #[async_trait::async_trait]
        impl DbConnectionPool<(), &'static dyn ToString> for MockDBPool {
            async fn connect(
                &self,
            ) -> Result<
                Box<dyn DbConnection<(), &'static dyn ToString>>,
                Box<dyn std::error::Error + Send + Sync>,
            > {
                Ok(Box::new(MockConn {}))
            }

            fn join_push_down(&self) -> JoinPushDown {
                JoinPushDown::Disallow
            }
        }

        fn make_exec(sql: &str) -> SqlExec<(), &'static dyn ToString> {
            let schema = Arc::new(Schema::new(vec![
                Field::new("name", DataType::Utf8, false),
                Field::new("age", DataType::Int16, false),
            ]));
            let pool = Arc::new(MockDBPool {})
                as Arc<dyn DbConnectionPool<(), &'static dyn ToString> + Send + Sync>;
            SqlExec::new(
                None,
                &schema,
                pool,
                sql.to_string(),
                Arc::new(DefaultDialect {}),
            )
            .unwrap()
        }

        #[test]
        fn test_sort_pushdown_single_column_asc() {
            let exec = make_exec("SELECT \"name\", \"age\" FROM \"users\"");
            let order = vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("name", 0)),
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }];

            match exec.try_pushdown_sort(&order).unwrap() {
                SortOrderPushdownResult::Exact { inner } => {
                    let sql_exec = inner
                        .downcast_ref::<SqlExec<(), &'static dyn ToString>>()
                        .unwrap();
                    assert_eq!(
                        sql_exec.sql().unwrap(),
                        "SELECT \"name\", \"age\" FROM \"users\" ORDER BY \"name\" ASC NULLS FIRST"
                    );
                }
                other => panic!("Expected Exact, got {:?}", sort_result_name(&other)),
            }
        }

        #[test]
        fn test_sort_pushdown_desc_nulls_last() {
            let exec = make_exec("SELECT \"name\", \"age\" FROM \"users\"");
            let order = vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("age", 1)),
                options: SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            }];

            match exec.try_pushdown_sort(&order).unwrap() {
                SortOrderPushdownResult::Exact { inner } => {
                    let sql_exec = inner
                        .downcast_ref::<SqlExec<(), &'static dyn ToString>>()
                        .unwrap();
                    assert_eq!(
                        sql_exec.sql().unwrap(),
                        "SELECT \"name\", \"age\" FROM \"users\" ORDER BY age DESC NULLS LAST"
                    );
                }
                other => panic!("Expected Exact, got {:?}", sort_result_name(&other)),
            }
        }

        #[test]
        fn test_sort_pushdown_multiple_columns() {
            let exec = make_exec("SELECT \"name\", \"age\" FROM \"users\"");
            let order = vec![
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("age", 1)),
                    options: SortOptions {
                        descending: true,
                        nulls_first: true,
                    },
                },
                PhysicalSortExpr {
                    expr: Arc::new(Column::new("name", 0)),
                    options: SortOptions {
                        descending: false,
                        nulls_first: false,
                    },
                },
            ];

            match exec.try_pushdown_sort(&order).unwrap() {
                SortOrderPushdownResult::Exact { inner } => {
                    let sql_exec = inner
                        .downcast_ref::<SqlExec<(), &'static dyn ToString>>()
                        .unwrap();
                    assert_eq!(
                        sql_exec.sql().unwrap(),
                        "SELECT \"name\", \"age\" FROM \"users\" ORDER BY age DESC NULLS FIRST, \"name\" ASC NULLS LAST"
                    );
                }
                other => panic!("Expected Exact, got {:?}", sort_result_name(&other)),
            }
        }

        #[test]
        fn test_sort_pushdown_with_limit() {
            let exec =
                make_exec("SELECT \"name\", \"age\" FROM \"users\" WHERE \"age\" > 30 LIMIT 10");
            let order = vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("name", 0)),
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }];

            match exec.try_pushdown_sort(&order).unwrap() {
                SortOrderPushdownResult::Exact { inner } => {
                    let sql_exec = inner
                        .downcast_ref::<SqlExec<(), &'static dyn ToString>>()
                        .unwrap();
                    assert_eq!(
                        sql_exec.sql().unwrap(),
                        "SELECT \"name\", \"age\" FROM \"users\" WHERE \"age\" > 30 ORDER BY \"name\" ASC NULLS FIRST LIMIT 10"
                    );
                }
                other => panic!("Expected Exact, got {:?}", sort_result_name(&other)),
            }
        }

        #[test]
        fn test_sort_pushdown_empty_order_returns_unsupported() {
            let exec = make_exec("SELECT \"name\" FROM \"users\"");
            match exec.try_pushdown_sort(&[]).unwrap() {
                SortOrderPushdownResult::Unsupported => {}
                other => panic!("Expected Unsupported, got {:?}", sort_result_name(&other)),
            }
        }

        #[test]
        fn test_sort_pushdown_updates_output_ordering() {
            let exec = make_exec("SELECT \"name\", \"age\" FROM \"users\"");
            let order = vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("name", 0)),
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }];

            match exec.try_pushdown_sort(&order).unwrap() {
                SortOrderPushdownResult::Exact { inner } => {
                    let orderings = inner.properties().output_ordering();
                    assert!(orderings.is_some(), "Output ordering should be set");
                    let output_order = orderings.unwrap();
                    assert_eq!(output_order.len(), 1);
                }
                other => panic!("Expected Exact, got {:?}", sort_result_name(&other)),
            }
        }

        #[test]
        fn test_sort_pushdown_is_allowed_by_default() {
            assert!(
                make_exec("SELECT \"name\" FROM \"users\"").allow_physical_sort_pushdown(),
                "sort pushdown must stay on unless a provider opts out"
            );
        }

        #[test]
        fn test_sort_pushdown_disabled_returns_unsupported() {
            // ScyllaDB/CQL: `ORDER BY` is rejected unless the partition key is
            // restricted by `=`/`IN`, so the sort must stay above the scan.
            // https://github.com/spiceai/spiceai/issues/10775
            let exec = make_exec("SELECT \"name\", \"age\" FROM \"users\"")
                .with_allow_physical_sort_pushdown(false);

            match exec.try_pushdown_sort(&order_by_name()).unwrap() {
                SortOrderPushdownResult::Unsupported => {}
                other => panic!("Expected Unsupported, got {:?}", sort_result_name(&other)),
            }
        }

        #[test]
        fn test_disabled_sort_pushdown_survives_limit_pushdown() {
            // LimitPushdown runs before PushdownSort, so the node the sort rule sees
            // is the one `with_fetch` produced, not the one the table built.
            let exec = make_exec("SELECT \"name\", \"age\" FROM \"users\"")
                .with_allow_physical_sort_pushdown(false);
            let with_limit = exec.with_fetch(Some(10)).expect("with_fetch should apply");

            match with_limit.try_pushdown_sort(&order_by_name()).unwrap() {
                SortOrderPushdownResult::Unsupported => {}
                other => panic!("Expected Unsupported, got {:?}", sort_result_name(&other)),
            }
        }

        #[test]
        fn test_disabled_sort_pushdown_survives_filter_pushdown() {
            // FilterPushdown runs before PushdownSort, so the node the sort rule sees
            // can be the one a filter rewrite produced.
            let exec = make_exec("SELECT \"name\", \"age\" FROM \"users\"")
                .with_allow_physical_sort_pushdown(false);
            let schema = exec.schema();
            let predicate = binary(
                col("age", &schema).expect("column should resolve"),
                Operator::Gt,
                lit(30i16),
                &schema,
            )
            .expect("predicate should build");

            let propagation = exec
                .handle_child_pushdown_result(
                    FilterPushdownPhase::Pre,
                    ChildPushdownResult {
                        parent_filters: vec![ChildFilterPushdownResult {
                            filter: predicate,
                            child_results: vec![PushedDown::Yes],
                        }],
                        self_filters: vec![],
                    },
                    &ConfigOptions::default(),
                )
                .expect("filter pushdown should succeed");
            let rewritten = propagation
                .updated_node
                .expect("the filter should be absorbed into the SQL");

            match rewritten.try_pushdown_sort(&order_by_name()).unwrap() {
                SortOrderPushdownResult::Unsupported => {}
                other => panic!("Expected Unsupported, got {:?}", sort_result_name(&other)),
            }
        }

        #[test]
        fn test_disabled_sort_pushdown_survives_clone() {
            let exec = make_exec("SELECT \"name\", \"age\" FROM \"users\"")
                .with_allow_physical_sort_pushdown(false);

            match exec.clone().try_pushdown_sort(&order_by_name()).unwrap() {
                SortOrderPushdownResult::Unsupported => {}
                other => panic!("Expected Unsupported, got {:?}", sort_result_name(&other)),
            }
        }

        #[test]
        fn test_table_threads_sort_pushdown_setting_into_its_plan() {
            let schema = Arc::new(Schema::new(vec![
                Field::new("name", DataType::Utf8, false),
                Field::new("age", DataType::Int16, false),
            ]));
            let pool = Arc::new(MockDBPool {})
                as Arc<dyn DbConnectionPool<(), &'static dyn ToString> + Send + Sync>;
            let table = SqlTable::new_with_schema(
                "mock",
                &pool,
                Arc::clone(&schema),
                TableReference::bare("users"),
                None,
            )
            .with_allow_physical_sort_pushdown(false);

            let plan = table
                .create_physical_plan(None, "SELECT \"name\", \"age\" FROM \"users\"".to_string())
                .expect("physical plan should be created");
            let exec = plan
                .downcast_ref::<SqlExec<(), &'static dyn ToString>>()
                .expect("plan should be a SqlExec");

            assert!(!exec.allow_physical_sort_pushdown());
            match exec.try_pushdown_sort(&order_by_name()).unwrap() {
                SortOrderPushdownResult::Unsupported => {}
                other => panic!("Expected Unsupported, got {:?}", sort_result_name(&other)),
            }
        }

        #[test]
        fn test_table_threads_expression_support_into_physical_filter_pushdown() {
            let schema = Arc::new(Schema::new(vec![
                Field::new("name", DataType::Utf8, false),
                Field::new("age", DataType::Int16, false),
            ]));
            let pool = Arc::new(MockDBPool {})
                as Arc<dyn DbConnectionPool<(), &'static dyn ToString> + Send + Sync>;
            let support = FunctionSupport::new(None, None, None).with_expression_support(Arc::new(
                |expr, _| {
                    !matches!(
                        expr,
                        datafusion::logical_expr::Expr::BinaryExpr(binary)
                            if binary.op == Operator::Gt
                    )
                },
            ));
            let table = SqlTable::new_with_schema(
                "mock",
                &pool,
                Arc::clone(&schema),
                TableReference::bare("users"),
                None,
            )
            .with_function_support(Some(support));
            let plan = table
                .create_physical_plan(None, "SELECT \"name\", \"age\" FROM \"users\"".to_string())
                .expect("physical plan should be created");
            let exec = plan
                .downcast_ref::<SqlExec<(), &'static dyn ToString>>()
                .expect("plan should be a SqlExec");

            let denied = binary(
                col("age", &schema).expect("column should resolve"),
                Operator::Gt,
                lit(30i16),
                &schema,
            )
            .expect("predicate should build");
            let denied_result = exec
                .handle_child_pushdown_result(
                    FilterPushdownPhase::Pre,
                    ChildPushdownResult {
                        parent_filters: vec![ChildFilterPushdownResult {
                            filter: denied,
                            child_results: vec![PushedDown::Yes],
                        }],
                        self_filters: vec![],
                    },
                    &ConfigOptions::default(),
                )
                .expect("filter pushdown should succeed");
            assert!(matches!(denied_result.filters.as_slice(), [PushedDown::No]));
            assert!(denied_result.updated_node.is_none());

            let allowed = binary(
                col("age", &schema).expect("column should resolve"),
                Operator::Eq,
                lit(30i16),
                &schema,
            )
            .expect("predicate should build");
            let allowed_result = exec
                .handle_child_pushdown_result(
                    FilterPushdownPhase::Pre,
                    ChildPushdownResult {
                        parent_filters: vec![ChildFilterPushdownResult {
                            filter: allowed,
                            child_results: vec![PushedDown::Yes],
                        }],
                        self_filters: vec![],
                    },
                    &ConfigOptions::default(),
                )
                .expect("filter pushdown should succeed");
            assert!(matches!(
                allowed_result.filters.as_slice(),
                [PushedDown::Yes]
            ));
            let rewritten = allowed_result
                .updated_node
                .expect("the allowed filter should be absorbed into SQL");
            let rewritten = rewritten
                .downcast_ref::<SqlExec<(), &'static dyn ToString>>()
                .expect("updated plan should be a SqlExec");
            assert_eq!(
                rewritten.sql().expect("SQL should be available"),
                "SELECT \"name\", \"age\" FROM \"users\" WHERE (age = 30)"
            );
        }

        fn order_by_name() -> Vec<PhysicalSortExpr> {
            vec![PhysicalSortExpr {
                expr: Arc::new(Column::new("name", 0)),
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            }]
        }

        fn sort_result_name(
            result: &SortOrderPushdownResult<Arc<dyn ExecutionPlan>>,
        ) -> &'static str {
            match result {
                SortOrderPushdownResult::Exact { .. } => "Exact",
                SortOrderPushdownResult::Inexact { .. } => "Inexact",
                SortOrderPushdownResult::Unsupported => "Unsupported",
            }
        }
    }

    mod with_fetch_tests {
        use crate::sql::sql_provider_datafusion::SqlExec;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use datafusion::physical_plan::ExecutionPlan;
        use datafusion::sql::unparser::dialect::DefaultDialect;
        use std::sync::Arc;

        use crate::sql::db_connection_pool::{
            dbconnection::DbConnection, DbConnectionPool, JoinPushDown,
        };

        struct MockConn {}

        impl DbConnection<(), &'static dyn ToString> for MockConn {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                self
            }
        }

        struct MockDBPool {}

        #[async_trait::async_trait]
        impl DbConnectionPool<(), &'static dyn ToString> for MockDBPool {
            async fn connect(
                &self,
            ) -> Result<
                Box<dyn DbConnection<(), &'static dyn ToString>>,
                Box<dyn std::error::Error + Send + Sync>,
            > {
                Ok(Box::new(MockConn {}))
            }

            fn join_push_down(&self) -> JoinPushDown {
                JoinPushDown::Disallow
            }
        }

        fn make_exec(sql: &str) -> SqlExec<(), &'static dyn ToString> {
            let schema = Arc::new(Schema::new(vec![
                Field::new("name", DataType::Utf8, false),
                Field::new("age", DataType::Int16, false),
            ]));
            let pool = Arc::new(MockDBPool {})
                as Arc<dyn DbConnectionPool<(), &'static dyn ToString> + Send + Sync>;
            SqlExec::new(
                None,
                &schema,
                pool,
                sql.to_string(),
                Arc::new(DefaultDialect {}),
            )
            .unwrap()
        }

        #[test]
        fn test_with_fetch_appends_limit() {
            let exec = make_exec("SELECT \"name\", \"age\" FROM \"users\"");
            let result = exec.with_fetch(Some(10)).unwrap();
            let sql_exec = result
                .downcast_ref::<SqlExec<(), &'static dyn ToString>>()
                .unwrap();
            assert_eq!(
                sql_exec.sql().unwrap(),
                "SELECT \"name\", \"age\" FROM \"users\" LIMIT 10"
            );
        }

        #[test]
        fn test_with_fetch_replaces_existing_limit() {
            let exec = make_exec("SELECT \"name\", \"age\" FROM \"users\" LIMIT 100");
            let result = exec.with_fetch(Some(5)).unwrap();
            let sql_exec = result
                .downcast_ref::<SqlExec<(), &'static dyn ToString>>()
                .unwrap();
            assert_eq!(
                sql_exec.sql().unwrap(),
                "SELECT \"name\", \"age\" FROM \"users\" LIMIT 5"
            );
        }

        #[test]
        fn test_with_fetch_none_returns_none() {
            let exec = make_exec("SELECT \"name\" FROM \"users\"");
            assert!(exec.with_fetch(None).is_none());
        }

        #[test]
        fn test_with_fetch_preserves_order_by() {
            let exec = make_exec(
                "SELECT \"name\", \"age\" FROM \"users\" ORDER BY \"name\" ASC NULLS FIRST LIMIT 50",
            );
            let result = exec.with_fetch(Some(10)).unwrap();
            let sql_exec = result
                .downcast_ref::<SqlExec<(), &'static dyn ToString>>()
                .unwrap();
            assert_eq!(
                sql_exec.sql().unwrap(),
                "SELECT \"name\", \"age\" FROM \"users\" ORDER BY \"name\" ASC NULLS FIRST LIMIT 10"
            );
        }
    }

    #[test]
    fn test_references() {
        let table_ref = TableReference::bare("test");
        assert_eq!(format!("{table_ref}"), "test");
    }

    // XXX move this to duckdb mod??
    #[cfg(feature = "duckdb")]
    mod duckdb_tests {
        use super::*;
        use crate::sql::db_connection_pool::dbconnection::duckdbconn::{
            DuckDBSyncParameter, DuckDbConnection,
        };
        use crate::sql::db_connection_pool::{duckdbpool::DuckDbConnectionPool, DbConnectionPool};
        use duckdb::DuckdbConnectionManager;

        #[tokio::test]
        async fn test_duckdb_table() -> Result<(), Box<dyn Error + Send + Sync>> {
            let t = setup_tracing();
            let ctx = SessionContext::new();
            let pool: Arc<
                dyn DbConnectionPool<
                        r2d2::PooledConnection<DuckdbConnectionManager>,
                        Box<dyn DuckDBSyncParameter>,
                    > + Send
                    + Sync,
            > = Arc::new(DuckDbConnectionPool::new_memory()?)
                as Arc<
                    dyn DbConnectionPool<
                            r2d2::PooledConnection<DuckdbConnectionManager>,
                            Box<dyn DuckDBSyncParameter>,
                        > + Send
                        + Sync,
                >;
            let conn = pool.connect().await?;
            let db_conn = conn
                .as_any()
                .downcast_ref::<DuckDbConnection>()
                .expect("Unable to downcast to DuckDbConnection");
            db_conn.conn.execute_batch(
                "CREATE TABLE test (a INTEGER, b VARCHAR); INSERT INTO test VALUES (3, 'bar');",
            )?;
            let duckdb_table = SqlTable::new("duckdb", &pool, "test", None).await?;
            ctx.register_table("test_datafusion", Arc::new(duckdb_table))?;
            let sql = "SELECT * FROM test_datafusion limit 1";
            let df = ctx.sql(sql).await?;
            df.show().await?;
            drop(t);
            Ok(())
        }

        #[tokio::test]
        async fn test_duckdb_table_filter() -> Result<(), Box<dyn Error + Send + Sync>> {
            let t = setup_tracing();
            let ctx = SessionContext::new();
            let pool: Arc<
                dyn DbConnectionPool<
                        r2d2::PooledConnection<DuckdbConnectionManager>,
                        Box<dyn DuckDBSyncParameter>,
                    > + Send
                    + Sync,
            > = Arc::new(DuckDbConnectionPool::new_memory()?)
                as Arc<
                    dyn DbConnectionPool<
                            r2d2::PooledConnection<DuckdbConnectionManager>,
                            Box<dyn DuckDBSyncParameter>,
                        > + Send
                        + Sync,
                >;
            let conn = pool.connect().await?;
            let db_conn = conn
                .as_any()
                .downcast_ref::<DuckDbConnection>()
                .expect("Unable to downcast to DuckDbConnection");
            db_conn.conn.execute_batch(
                "CREATE TABLE test (a INTEGER, b VARCHAR); INSERT INTO test VALUES (3, 'bar');",
            )?;
            let duckdb_table = SqlTable::new("duckdb", &pool, "test", None).await?;
            ctx.register_table("test_datafusion", Arc::new(duckdb_table))?;
            let sql = "SELECT * FROM test_datafusion where a > 1 and b = 'bar' limit 1";
            let df = ctx.sql(sql).await?;
            df.show().await?;
            drop(t);
            Ok(())
        }
    }

    mod fetch_tests {
        use crate::sql::sql_provider_datafusion::SqlExec;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use datafusion::physical_plan::ExecutionPlan;
        use datafusion::sql::unparser::dialect::DefaultDialect;
        use std::sync::Arc;

        use crate::sql::db_connection_pool::{
            dbconnection::DbConnection, DbConnectionPool, JoinPushDown,
        };

        struct MockConn {}
        impl DbConnection<(), &'static dyn ToString> for MockConn {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                self
            }
        }
        struct MockDBPool {}
        #[async_trait::async_trait]
        impl DbConnectionPool<(), &'static dyn ToString> for MockDBPool {
            async fn connect(
                &self,
            ) -> Result<
                Box<dyn DbConnection<(), &'static dyn ToString>>,
                Box<dyn std::error::Error + Send + Sync>,
            > {
                Ok(Box::new(MockConn {}))
            }
            fn join_push_down(&self) -> JoinPushDown {
                JoinPushDown::Disallow
            }
        }

        fn make_exec(sql: &str) -> SqlExec<(), &'static dyn ToString> {
            let schema = Arc::new(Schema::new(vec![
                Field::new("a", DataType::Int32, false),
                Field::new("b", DataType::Utf8, false),
            ]));
            let pool = Arc::new(MockDBPool {})
                as Arc<dyn DbConnectionPool<(), &'static dyn ToString> + Send + Sync>;
            SqlExec::new(
                None,
                &schema,
                pool,
                sql.to_string(),
                Arc::new(DefaultDialect {}),
            )
            .unwrap()
        }

        #[test]
        fn test_fetch_returns_none_when_no_limit() {
            let exec = make_exec("SELECT * FROM t");
            assert_eq!(exec.fetch(), None);
        }

        #[test]
        fn test_fetch_returns_limit_value() {
            let exec = make_exec("SELECT * FROM t LIMIT 42");
            assert_eq!(exec.fetch(), Some(42));
        }

        #[test]
        fn test_fetch_with_order_by_and_limit() {
            let exec = make_exec("SELECT * FROM t ORDER BY a LIMIT 10");
            assert_eq!(exec.fetch(), Some(10));
        }

        #[test]
        fn test_supports_limit_pushdown() {
            let exec = make_exec("SELECT * FROM t");
            assert!(exec.supports_limit_pushdown());
        }
    }

    mod physical_expr_to_sql_tests {
        use super::super::*;
        use datafusion::logical_expr::Operator;
        use datafusion::physical_expr::expressions::{
            BinaryExpr, Column, IsNotNullExpr, IsNullExpr, Literal, NotExpr,
        };
        use datafusion::scalar::ScalarValue;
        use datafusion::sql::unparser::dialect::DefaultDialect;

        fn col(name: &str) -> Arc<dyn datafusion::physical_plan::PhysicalExpr> {
            Arc::new(Column::new(name, 0))
        }

        fn lit_i32(v: i32) -> Arc<dyn datafusion::physical_plan::PhysicalExpr> {
            Arc::new(Literal::new(ScalarValue::Int32(Some(v))))
        }

        fn lit_str(s: &str) -> Arc<dyn datafusion::physical_plan::PhysicalExpr> {
            Arc::new(Literal::new(ScalarValue::Utf8(Some(s.to_string()))))
        }

        fn default_dialect() -> &'static dyn Dialect {
            &DefaultDialect {}
        }

        #[test]
        fn test_column() {
            let result = physical_expr_to_sql(&col("name"), default_dialect(), None);
            assert_eq!(result, Some("\"name\"".to_string()));
        }

        #[test]
        fn test_literal_int() {
            let result = physical_expr_to_sql(&lit_i32(42), default_dialect(), None);
            assert_eq!(result, Some("42".to_string()));
        }

        #[test]
        fn test_literal_string() {
            let result = physical_expr_to_sql(&lit_str("hello"), default_dialect(), None);
            assert_eq!(result, Some("'hello'".to_string()));
        }

        #[test]
        fn test_literal_string_with_quote() {
            let result = physical_expr_to_sql(&lit_str("it's"), default_dialect(), None);
            assert_eq!(result, Some("'it''s'".to_string()));
        }

        #[test]
        fn test_literal_null() {
            let expr: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
                Arc::new(Literal::new(ScalarValue::Null));
            let result = physical_expr_to_sql(&expr, default_dialect(), None);
            assert_eq!(result, Some("NULL".to_string()));
        }

        #[test]
        fn test_binary_eq() {
            let expr: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
                Arc::new(BinaryExpr::new(col("age"), Operator::Gt, lit_i32(30)));
            let result = physical_expr_to_sql(&expr, default_dialect(), None);
            assert_eq!(result, Some("(age > 30)".to_string()));
        }

        #[test]
        fn test_and_expression() {
            let left: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
                Arc::new(BinaryExpr::new(col("a"), Operator::Gt, lit_i32(1)));
            let right: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
                Arc::new(BinaryExpr::new(col("b"), Operator::Eq, lit_str("x")));
            let expr: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
                Arc::new(BinaryExpr::new(left, Operator::And, right));
            let result = physical_expr_to_sql(&expr, default_dialect(), None);
            assert_eq!(result, Some("((a > 1) AND (b = 'x'))".to_string()));
        }

        #[test]
        fn test_is_null() {
            let expr: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
                Arc::new(IsNullExpr::new(col("x")));
            let result = physical_expr_to_sql(&expr, default_dialect(), None);
            assert_eq!(result, Some("x IS NULL".to_string()));
        }

        #[test]
        fn test_is_not_null() {
            let expr: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
                Arc::new(IsNotNullExpr::new(col("x")));
            let result = physical_expr_to_sql(&expr, default_dialect(), None);
            assert_eq!(result, Some("x IS NOT NULL".to_string()));
        }

        #[test]
        fn test_not() {
            let inner: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
                Arc::new(BinaryExpr::new(col("active"), Operator::Eq, lit_i32(1)));
            let expr: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
                Arc::new(NotExpr::new(inner));
            let result = physical_expr_to_sql(&expr, default_dialect(), None);
            assert_eq!(result, Some("NOT (active = 1)".to_string()));
        }

        #[test]
        fn test_literal_bool() {
            let expr: Arc<dyn datafusion::physical_plan::PhysicalExpr> =
                Arc::new(Literal::new(ScalarValue::Boolean(Some(true))));
            let result = physical_expr_to_sql(&expr, default_dialect(), None);
            assert_eq!(result, Some("true".to_string()));
        }
    }

    mod insert_where_clause_tests {
        use super::super::insert_where_clause;

        #[test]
        fn test_no_existing_where() {
            let sql = "SELECT * FROM t";
            let result = insert_where_clause(sql, "\"a\" > 1");
            assert_eq!(result, "SELECT * FROM t WHERE \"a\" > 1");
        }

        #[test]
        fn test_existing_where() {
            let sql = "SELECT * FROM t WHERE \"b\" = 2";
            let result = insert_where_clause(sql, "\"a\" > 1");
            assert_eq!(result, "SELECT * FROM t WHERE \"b\" = 2 AND (\"a\" > 1)");
        }

        #[test]
        fn test_no_where_with_order_by() {
            let sql = "SELECT * FROM t ORDER BY a";
            let result = insert_where_clause(sql, "\"x\" = 1");
            assert_eq!(result, "SELECT * FROM t WHERE \"x\" = 1 ORDER BY a");
        }

        #[test]
        fn test_no_where_with_limit() {
            let sql = "SELECT * FROM t LIMIT 10";
            let result = insert_where_clause(sql, "\"x\" = 1");
            assert_eq!(result, "SELECT * FROM t WHERE \"x\" = 1 LIMIT 10");
        }

        #[test]
        fn test_existing_where_with_order_by_and_limit() {
            let sql = "SELECT * FROM t WHERE \"a\" = 1 ORDER BY b LIMIT 5";
            let result = insert_where_clause(sql, "\"c\" > 3");
            assert_eq!(
                result,
                "SELECT * FROM t WHERE \"a\" = 1 AND (\"c\" > 3) ORDER BY b LIMIT 5"
            );
        }

        #[test]
        fn test_no_where_with_order_by_and_limit() {
            let sql = "SELECT * FROM t ORDER BY b LIMIT 5";
            let result = insert_where_clause(sql, "\"c\" > 3");
            assert_eq!(result, "SELECT * FROM t WHERE \"c\" > 3 ORDER BY b LIMIT 5");
        }
    }
}
