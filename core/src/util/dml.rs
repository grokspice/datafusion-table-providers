use std::{fmt, sync::Arc};

use arrow::datatypes::Schema;

use async_trait::async_trait;
use datafusion::{
    common::TableReference,
    error::DataFusionError,
    execution::{SendableRecordBatchStream, TaskContext},
    logical_expr::Expr,
    physical_plan::{
        stream::RecordBatchStreamAdapter, DisplayAs, DisplayFormatType, ExecutionPlan,
        PlanProperties,
    },
};

use super::count_exec::{count_schema, count_to_record_batch};
use crate::sql::sql_provider_datafusion::expr;

/// Converts filter expressions to a SQL WHERE clause string.
pub fn filters_to_sql(
    filters: &[Expr],
    engine: Option<expr::Engine>,
) -> datafusion::error::Result<String> {
    filters_to_sql_with_schema(filters, engine, None)
}

/// [`filters_to_sql`], given the schema the filters' columns belong to.
///
/// The schema is what lets the DuckDB rendering tell a column that needs its timestamp
/// normalization from one the normalization would only truncate — see
/// [`expr::to_sql_with_engine_and_schema`].
pub fn filters_to_sql_with_schema(
    filters: &[Expr],
    engine: Option<expr::Engine>,
    schema: Option<&Schema>,
) -> datafusion::error::Result<String> {
    let sql_parts: Result<Vec<String>, _> = filters
        .iter()
        .map(|f| expr::to_sql_with_engine_and_schema(f, engine, schema))
        .collect();
    sql_parts
        .map(|parts| match parts.as_slice() {
            [] => String::new(),
            // Nothing is composed with a lone filter, so it needs no parentheses.
            [only] => only.clone(),
            // Every filter is parenthesized before being joined, because `AND` binds tighter
            // than `OR`: a filter that is itself a disjunction would otherwise be re-grouped by
            // the join. `[a OR b, c]` has to mean `(a OR b) AND c`, not `a OR (b AND c)`, which
            // matches different rows.
            _ => parts
                .iter()
                .map(|part| format!("({part})"))
                .collect::<Vec<String>>()
                .join(" AND "),
        })
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))
}

/// Converts assignment expressions to a SQL SET clause string.
pub fn assignments_to_sql(
    assignments: &[(String, Expr)],
    engine: Option<expr::Engine>,
) -> datafusion::error::Result<String> {
    assignments_to_sql_with_schema(assignments, engine, None)
}

/// [`assignments_to_sql`], given the schema the assignments' columns belong to.
pub fn assignments_to_sql_with_schema(
    assignments: &[(String, Expr)],
    engine: Option<expr::Engine>,
    schema: Option<&Schema>,
) -> datafusion::error::Result<String> {
    let parts: Result<Vec<String>, _> = assignments
        .iter()
        .map(|(col, val)| {
            expr::to_sql_with_engine_and_schema(val, engine, schema)
                .map(|sql_val| format!("{col} = {sql_val}", col = expr::quoted_identifier(col)))
        })
        .collect();
    parts
        .map(|p| p.join(", "))
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))
}

/// Renders `DELETE FROM <table> [WHERE <sql_where>]`, quoting each part of `table` so that a quote
/// in it cannot close the identifier and leave the rest of the name to be read as statement text.
///
/// Doubling is the whole escape here, unlike in a string literal: none of `DuckDB`, `SQLite` and
/// `PostgreSQL` give `\` any meaning inside a delimited identifier.
///
/// The statement addresses the reference in full rather than its bare table part, so that a
/// qualified dataset's `DELETE` reaches the table its truncate path already reaches instead of
/// whatever the session's `search_path` resolves the bare name to.
pub fn delete_statement(table: &TableReference, sql_where: Option<&str>) -> String {
    let table = expr::quoted_table_reference(table);
    match sql_where {
        Some(sql_where) => format!("DELETE FROM {table} WHERE {sql_where}"),
        None => format!("DELETE FROM {table}"),
    }
}

/// Renders [`delete_statement`] wrapped in a CTE that returns the removed rows, so the count comes
/// back as a result row: `WITH deleted AS (DELETE ... RETURNING *) SELECT COUNT(*) FROM deleted`.
///
/// This is for a caller that reads the count from a query rather than from the driver's
/// affected-row tag — the `PostgreSQL` deletion sink runs this with `query_one` inside its
/// transaction, so the count and the delete stay one statement.
pub fn delete_statement_returning_count(table: &TableReference, sql_where: Option<&str>) -> String {
    let delete = delete_statement(table, sql_where);
    format!("WITH deleted AS ({delete} RETURNING *) SELECT COUNT(*) FROM deleted")
}

/// Renders `UPDATE <table> SET <set_clause> [WHERE <sql_where>]`, naming the table as
/// [`delete_statement`] describes.
pub fn update_statement(
    table: &TableReference,
    set_clause: &str,
    sql_where: Option<&str>,
) -> String {
    let table = expr::quoted_table_reference(table);
    match sql_where {
        Some(sql_where) => format!("UPDATE {table} SET {set_clause} WHERE {sql_where}"),
        None => format!("UPDATE {table} SET {set_clause}"),
    }
}

#[async_trait]
pub trait DeletionSink: Send + Sync {
    async fn delete_from(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>>;
}

pub struct DeletionExec {
    deletion_sink: Arc<dyn DeletionSink + 'static>,
    properties: Arc<PlanProperties>,
}

impl DeletionExec {
    pub fn new(deletion_sink: Arc<dyn DeletionSink>) -> Self {
        let properties = PlanProperties::new(
            datafusion::physical_expr::EquivalenceProperties::new(count_schema()),
            datafusion::physical_plan::Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );
        Self {
            deletion_sink,
            properties: Arc::new(properties),
        }
    }
}

impl fmt::Debug for DeletionExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeletionExec").finish_non_exhaustive()
    }
}

impl DisplayAs for DeletionExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DeletionExec")
    }
}

impl ExecutionPlan for DeletionExec {
    fn name(&self) -> &'static str {
        "DeletionExec"
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
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let schema = count_schema();
        let deletion_sink = Arc::clone(&self.deletion_sink);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&schema),
            futures::stream::once(async move {
                let count = deletion_sink
                    .delete_from()
                    .await
                    .map_err(DataFusionError::External)?;
                count_to_record_batch(schema, count)
            }),
        )))
    }
}

#[async_trait]
pub trait UpdateSink: Send + Sync {
    async fn execute_update(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>>;
}

pub struct UpdateExec {
    update_sink: Arc<dyn UpdateSink + 'static>,
    properties: Arc<PlanProperties>,
}

impl UpdateExec {
    pub fn new(update_sink: Arc<dyn UpdateSink>) -> Self {
        let properties = PlanProperties::new(
            datafusion::physical_expr::EquivalenceProperties::new(count_schema()),
            datafusion::physical_plan::Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );
        Self {
            update_sink,
            properties: Arc::new(properties),
        }
    }
}

impl fmt::Debug for UpdateExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UpdateExec").finish_non_exhaustive()
    }
}

impl DisplayAs for UpdateExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UpdateExec")
    }
}

impl ExecutionPlan for UpdateExec {
    fn name(&self) -> &'static str {
        "UpdateExec"
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
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let schema = count_schema();
        let update_sink = Arc::clone(&self.update_sink);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&schema),
            futures::stream::once(async move {
                let count = update_sink
                    .execute_update()
                    .await
                    .map_err(DataFusionError::External)?;
                count_to_record_batch(schema, count)
            }),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::*;

    #[test]
    fn test_filters_to_sql_single() {
        let filter = col("id").eq(lit(1i32));
        let sql = filters_to_sql(&[filter], None).expect("filters_to_sql should succeed");
        assert_eq!(sql, r#""id" = 1"#);
    }

    #[test]
    fn test_filters_to_sql_multiple() {
        let f1 = col("id").eq(lit(1i32));
        let f2 = col("name").eq(lit("foo"));
        let sql = filters_to_sql(&[f1, f2], None).expect("filters_to_sql should succeed");
        assert_eq!(sql, r#"("id" = 1) AND ("name" = 'foo')"#);
    }

    /// A filter that is itself a disjunction must keep its grouping when joined with the others:
    /// `AND` binds tighter than `OR`, so joining the fragments bare turns `(a OR b) AND c` into
    /// `a OR (b AND c)`, which matches a different set of rows.
    #[test]
    fn test_filters_to_sql_parenthesizes_a_disjunction() {
        let disjunction = col("a").eq(lit(1i32)).or(col("b").eq(lit(1i32)));
        let conjunct = col("c").eq(lit(1i32));

        let sql =
            filters_to_sql(&[disjunction, conjunct], None).expect("filters_to_sql should succeed");

        assert_eq!(sql, r#"(("a" = 1) OR ("b" = 1)) AND ("c" = 1)"#);
    }

    #[test]
    fn test_filters_to_sql_empty() {
        let sql = filters_to_sql(&[], None).expect("filters_to_sql should succeed");
        assert_eq!(sql, "");
    }

    #[test]
    fn test_assignments_to_sql_single() {
        let assignments = vec![("name".to_string(), lit("foo"))];
        let sql =
            assignments_to_sql(&assignments, None).expect("assignments_to_sql should succeed");
        assert_eq!(sql, r#""name" = 'foo'"#);
    }

    #[test]
    fn test_assignments_to_sql_multiple() {
        let assignments = vec![
            ("name".to_string(), lit("foo")),
            ("age".to_string(), lit(30i32)),
        ];
        let sql =
            assignments_to_sql(&assignments, None).expect("assignments_to_sql should succeed");
        assert_eq!(sql, r#""name" = 'foo', "age" = 30"#);
    }

    /// The WHERE clause a `DELETE`/`UPDATE` is built from must keep a quote-bearing value
    /// inside its literal: rendered bare, `= 'x' OR 1=1 --'` is a tautology and the statement
    /// matches every row.
    #[test]
    fn test_filters_to_sql_escapes_a_quote_bearing_value() {
        let filter = col("name").eq(lit("x' OR 1=1 --"));
        let sql = filters_to_sql(&[filter], None).expect("filters_to_sql should succeed");
        assert_eq!(sql, r#""name" = 'x'' OR 1=1 --'"#);

        let apostrophe = col("name").eq(lit("O'Brien"));
        let sql = filters_to_sql(&[apostrophe], None).expect("filters_to_sql should succeed");
        assert_eq!(sql, r#""name" = 'O''Brien'"#);
    }

    /// A predicate reaching a DML sink can carry a qualifier, and it must render as the column it
    /// names rather than as one flat identifier.
    #[test]
    fn test_filters_to_sql_renders_a_qualified_column_as_its_own_column() {
        let filter = Expr::Column(datafusion::common::Column::new(Some("t"), "id")).eq(lit(1i32));

        let sql = filters_to_sql(&[filter], None).expect("filters_to_sql should succeed");
        assert_eq!(sql, r#""id" = 1"#);
    }

    /// The same applies to a SET clause, whose value expression can carry a column reference.
    #[test]
    fn test_assignments_to_sql_renders_a_qualified_column_value() {
        let assignments = vec![(
            "name".to_string(),
            Expr::Column(datafusion::common::Column::new(Some("t"), "other")),
        )];

        let sql =
            assignments_to_sql(&assignments, None).expect("assignments_to_sql should succeed");
        assert_eq!(sql, r#""name" = "other""#);
    }

    /// A name needing no escape must render exactly as the interpolated form did, so that quoting
    /// the table name cannot change the statement built for any ordinary dataset.
    #[test]
    fn test_statements_render_an_ordinary_table_name_unchanged() {
        assert_eq!(
            delete_statement(&TableReference::bare("orders"), Some(r#""id" = 1"#)),
            r#"DELETE FROM "orders" WHERE "id" = 1"#
        );
        assert_eq!(
            delete_statement(&TableReference::bare("orders"), None),
            r#"DELETE FROM "orders""#
        );
        assert_eq!(
            update_statement(
                &TableReference::bare("orders"),
                r#""qty" = 2"#,
                Some(r#""id" = 1"#)
            ),
            r#"UPDATE "orders" SET "qty" = 2 WHERE "id" = 1"#
        );
        assert_eq!(
            update_statement(&TableReference::bare("orders"), r#""qty" = 2"#, None),
            r#"UPDATE "orders" SET "qty" = 2"#
        );
        assert_eq!(
            delete_statement_returning_count(&TableReference::bare("orders"), Some(r#""id" = 1"#)),
            r#"WITH deleted AS (DELETE FROM "orders" WHERE "id" = 1 RETURNING *) SELECT COUNT(*) FROM deleted"#
        );
        assert_eq!(
            delete_statement_returning_count(&TableReference::bare("orders"), None),
            r#"WITH deleted AS (DELETE FROM "orders" RETURNING *) SELECT COUNT(*) FROM deleted"#
        );
    }

    /// A quote in the table name closes its identifier early when interpolated, so the rest of the
    /// name becomes statement text — the identifier counterpart of an unescaped string literal.
    #[test]
    fn test_statements_escape_a_quote_bearing_table_name() {
        assert_eq!(
            delete_statement(&TableReference::bare(r#"we"ird"#), Some(r#""id" = 1"#)),
            r#"DELETE FROM "we""ird" WHERE "id" = 1"#
        );
        assert_eq!(
            update_statement(&TableReference::bare(r#"we"ird"#), r#""qty" = 2"#, None),
            r#"UPDATE "we""ird" SET "qty" = 2"#
        );
        assert_eq!(
            delete_statement_returning_count(&TableReference::bare(r#"we"ird"#), None),
            r#"WITH deleted AS (DELETE FROM "we""ird" RETURNING *) SELECT COUNT(*) FROM deleted"#
        );

        // A quote at either end has no ordinary character beside it to make the doubling obvious.
        assert_eq!(
            delete_statement(&TableReference::bare(r#"trailing""#), None),
            r#"DELETE FROM "trailing""""#
        );
        assert_eq!(
            delete_statement(&TableReference::bare(r#""leading"#), None),
            r#"DELETE FROM """leading""#
        );
    }

    /// A name carrying SQL of its own is what the missing escape actually costs: interpolated, the
    /// `WHERE` below escapes the identifier and widens the statement to the whole table. Quoted, the
    /// name stays one identifier and the statement can only ever name that table.
    #[test]
    fn test_a_table_name_carrying_sql_stays_one_identifier() {
        assert_eq!(
            delete_statement(
                &TableReference::bare(r#"t" WHERE 1=1 --"#),
                Some(r#""id" = 1"#)
            ),
            r#"DELETE FROM "t"" WHERE 1=1 --" WHERE "id" = 1"#
        );
    }

    /// Doubling a delimiter is only a complete escape where nothing else escapes: a backslash next
    /// to the quote is the composition that defeats it for a *string literal* on a backslash-reading
    /// engine. Inside a delimited identifier none of the engines here give `\` any meaning, so it
    /// must pass through untouched — asserted so that adding an engine which does has to face it.
    #[test]
    fn test_a_backslash_in_a_table_name_is_not_escaped() {
        assert_eq!(
            delete_statement(&TableReference::bare(r#"back\slash"quote"#), None),
            r#"DELETE FROM "back\slash""quote""#
        );
    }

    /// A qualified dataset must be addressed by the name it was given. Naming only the bare table
    /// part leaves the schema to the session's `search_path`, which is how a `DELETE`/`UPDATE`
    /// came to address a different schema's table than the truncate path on the same provider.
    #[test]
    fn test_statements_address_a_qualified_table_in_full() {
        let qualified = TableReference::partial("myschema", "orders");

        assert_eq!(
            delete_statement(&qualified, Some(r#""id" = 1"#)),
            r#"DELETE FROM "myschema"."orders" WHERE "id" = 1"#
        );
        assert_eq!(
            update_statement(&qualified, r#""qty" = 2"#, None),
            r#"UPDATE "myschema"."orders" SET "qty" = 2"#
        );
        assert_eq!(
            delete_statement_returning_count(&qualified, None),
            r#"WITH deleted AS (DELETE FROM "myschema"."orders" RETURNING *) SELECT COUNT(*) FROM deleted"#
        );

        let full = TableReference::full("mycatalog", "myschema", "orders");
        assert_eq!(
            delete_statement(&full, None),
            r#"DELETE FROM "mycatalog"."myschema"."orders""#
        );
    }

    /// Each part is a separate identifier, so a quote in one may not leak into another and a dot
    /// inside a part may not split it.
    #[test]
    fn test_a_qualified_name_escapes_each_part_separately() {
        assert_eq!(
            delete_statement(&TableReference::partial(r#"we"ird"#, r#"ta"ble"#), None),
            r#"DELETE FROM "we""ird"."ta""ble""#
        );

        // A dot inside a part names a table whose name contains a dot, not two identifiers.
        assert_eq!(
            delete_statement(&TableReference::bare("has.dot"), None),
            r#"DELETE FROM "has.dot""#
        );
    }

    /// Every part is quoted unconditionally, so a table named for a reserved word stays a name.
    /// `TableReference::to_quoted_string` leaves an all-lowercase part bare, which would render
    /// this as syntax.
    #[test]
    fn test_a_reserved_word_table_name_is_still_quoted() {
        assert_eq!(
            delete_statement(&TableReference::partial("public", "order"), None),
            r#"DELETE FROM "public"."order""#
        );
        assert_eq!(
            TableReference::partial("public", "order").to_quoted_string(),
            "public.order",
            "the assertion above is only load-bearing while to_quoted_string leaves these bare"
        );
    }

    /// A SET clause carries both a value and a column name, and interpolates the name itself.
    #[test]
    fn test_assignments_to_sql_escapes_value_and_column_name() {
        let assignments = vec![("name".to_string(), lit("O'Brien"))];
        let sql =
            assignments_to_sql(&assignments, None).expect("assignments_to_sql should succeed");
        assert_eq!(sql, r#""name" = 'O''Brien'"#);

        let quoted_column = vec![("we\"ird".to_string(), lit(1i32))];
        let sql =
            assignments_to_sql(&quoted_column, None).expect("assignments_to_sql should succeed");
        assert_eq!(sql, r#""we""ird" = 1"#);
    }
}

/// The rendered SQL is only correct if the target engine agrees, so these run the statement the
/// DML paths build and assert on the rows that actually survive it. A rendering assertion alone
/// cannot distinguish "escaped" from "escaped in a form this engine accepts".
#[cfg(all(test, feature = "duckdb"))]
mod duckdb_execution_tests {
    use super::*;
    use crate::sql::sql_provider_datafusion::expr::Engine;
    use datafusion::prelude::*;
    use duckdb::Connection;

    const ROWS: [&str; 4] = ["O'Brien", "plain", "x' OR 1=1 --", r"\' OR 1=1 --"];

    /// Deletes with the WHERE clause `filters_to_sql` renders, and returns the surviving names.
    fn surviving_after_delete(filters: &[Expr]) -> Vec<String> {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch("CREATE TABLE t (name VARCHAR)")
            .expect("create");
        for row in ROWS {
            conn.execute("INSERT INTO t VALUES (?)", [row])
                .expect("insert");
        }

        let where_clause =
            filters_to_sql(filters, Some(Engine::DuckDB)).expect("filters_to_sql should succeed");
        conn.execute_batch(&format!("DELETE FROM t WHERE {where_clause}"))
            .expect("the rendered DELETE must be valid SQL for DuckDB");

        let mut stmt = conn
            .prepare("SELECT name FROM t ORDER BY name")
            .expect("prepare");
        let mut names: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<duckdb::Result<Vec<String>>>()
            .expect("rows");
        names.sort();
        names
    }

    fn all_but(excluded: &str) -> Vec<String> {
        let mut rest: Vec<String> = ROWS
            .iter()
            .filter(|row| **row != excluded)
            .map(|row| (*row).to_string())
            .collect();
        rest.sort();
        rest
    }

    /// An apostrophe rendered bare makes the statement fail to parse, so the delete cannot run.
    #[test]
    fn an_apostrophe_bearing_value_deletes_exactly_its_row() {
        let filters = vec![col("name").eq(lit("O'Brien"))];

        assert_eq!(surviving_after_delete(&filters), all_but("O'Brien"));
    }

    /// The unbounded-deletion shape: rendered bare this binds as a tautology and removes every
    /// row. It must remove exactly the one row whose value it is.
    #[test]
    fn a_tautology_shaped_value_deletes_exactly_its_row() {
        let filters = vec![col("name").eq(lit("x' OR 1=1 --"))];

        assert_eq!(surviving_after_delete(&filters), all_but("x' OR 1=1 --"));
    }

    /// A backslash immediately before the quote is the composition quote doubling cannot handle
    /// on a backslash-escaping engine. DuckDB treats `\` as ordinary, so it must stay contained.
    #[test]
    fn a_backslash_before_a_quote_deletes_exactly_its_row() {
        let filters = vec![col("name").eq(lit(r"\' OR 1=1 --"))];

        assert_eq!(surviving_after_delete(&filters), all_but(r"\' OR 1=1 --"));
    }

    /// Two filters are joined with `AND`, so a quote in either must stay inside its own literal.
    #[test]
    fn a_conjunction_of_quote_bearing_values_matches_nothing() {
        let filters = vec![
            col("name").eq(lit("O'Brien")),
            col("name").eq(lit("x' OR 1=1 --")),
        ];

        let mut expected: Vec<String> = ROWS.iter().map(|row| (*row).to_string()).collect();
        expected.sort();
        assert_eq!(surviving_after_delete(&filters), expected);
    }

    /// The flat rendering of a qualified column is not merely invalid — against a table that has a
    /// column of that literal name it is *bindable*, so the statement succeeds against the wrong
    /// column. This runs the `DELETE` the DML path builds for `t.a = 1` over a table holding both
    /// `a` and `t.a`, and asserts the row the caller selected is the one that goes.
    #[test]
    fn a_qualified_column_deletes_by_the_column_it_names() {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch(r#"CREATE TABLE t ("a" INTEGER, "t.a" INTEGER)"#)
            .expect("create");
        conn.execute_batch("INSERT INTO t VALUES (1, 99), (99, 1)")
            .expect("insert");

        let filters =
            vec![Expr::Column(datafusion::common::Column::new(Some("t"), "a")).eq(lit(1i32))];
        let where_clause =
            filters_to_sql(&filters, Some(Engine::DuckDB)).expect("a qualified column must render");
        conn.execute_batch(&format!("DELETE FROM t WHERE {where_clause}"))
            .expect("the rendered DELETE must be valid SQL for DuckDB");

        let mut stmt = conn.prepare(r#"SELECT "a" FROM t"#).expect("prepare");
        let surviving: Vec<i32> = stmt
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<duckdb::Result<Vec<i32>>>()
            .expect("rows");

        // Rendered flat, `"t.a" = 1` binds the decoy column and leaves `a = 1` behind instead.
        assert_eq!(surviving, vec![99], "the wrong row was deleted");
    }

    /// The table name is a delimited identifier, and doubling is only the right escape if the engine
    /// reads it that way. This creates a table whose name holds a quote, runs the `DELETE` the DML
    /// path builds for it, and asserts the row selected is the row that goes.
    #[test]
    fn a_quote_bearing_table_name_deletes_by_the_table_it_names() {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch(r#"CREATE TABLE "we""ird" (id INTEGER)"#)
            .expect("create");
        conn.execute_batch(r#"INSERT INTO "we""ird" VALUES (1), (2), (3)"#)
            .expect("insert");

        conn.execute_batch(&delete_statement(
            &TableReference::bare(r#"we"ird"#),
            Some(r#""id" = 2"#),
        ))
        .expect("the rendered DELETE must be valid SQL for DuckDB");

        let mut stmt = conn
            .prepare(r#"SELECT id FROM "we""ird" ORDER BY id"#)
            .expect("prepare");
        let surviving: Vec<i32> = stmt
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<duckdb::Result<Vec<i32>>>()
            .expect("rows");
        assert_eq!(surviving, vec![1, 3]);

        // Interpolated instead of quoted, the name closes its identifier early and DuckDB cannot
        // parse what follows — which is why the delete could not run at all before.
        assert!(
            conn.execute_batch(r#"DELETE FROM "we"ird" WHERE "id" = 2"#)
                .is_err(),
            "the interpolated form must not be valid SQL"
        );
    }

    /// The same for an `UPDATE`, which names the table and carries a SET clause.
    /// The wrong-table case the qualifier exists to prevent: two tables share a name in different
    /// schemas, so a statement naming only the bare part is still valid SQL and still reports a
    /// count — it just empties the wrong one. Asserting on the surviving rows of *both* tables is
    /// what distinguishes addressing the right table from merely running.
    #[test]
    fn a_qualified_delete_empties_its_own_schemas_table_and_leaves_the_other() {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch(
            "CREATE SCHEMA myschema;
             CREATE TABLE myschema.orders (id INTEGER);
             INSERT INTO myschema.orders VALUES (1), (2);
             CREATE TABLE orders (id INTEGER);
             INSERT INTO orders VALUES (3), (4);",
        )
        .expect("create both tables");

        conn.execute_batch(&delete_statement(
            &TableReference::partial("myschema", "orders"),
            None,
        ))
        .expect("the rendered DELETE must be valid SQL for DuckDB");

        let count = |sql: &str| -> i64 {
            conn.query_row(sql, [], |row| row.get(0))
                .expect("count query")
        };
        assert_eq!(
            count("SELECT COUNT(*) FROM myschema.orders"),
            0,
            "the qualified table is the one the statement names"
        );
        assert_eq!(
            count("SELECT COUNT(*) FROM main.orders"),
            2,
            "the bare name resolves to this table, which the statement must not touch"
        );
    }

    /// The same for `UPDATE`, whose wrong-table form also succeeds and reports a count.
    #[test]
    fn a_qualified_update_writes_to_its_own_schemas_table() {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch(
            "CREATE SCHEMA myschema;
             CREATE TABLE myschema.orders (id INTEGER, qty INTEGER);
             INSERT INTO myschema.orders VALUES (1, 10);
             CREATE TABLE orders (id INTEGER, qty INTEGER);
             INSERT INTO orders VALUES (1, 10);",
        )
        .expect("create both tables");

        conn.execute_batch(&update_statement(
            &TableReference::partial("myschema", "orders"),
            r#""qty" = 99"#,
            None,
        ))
        .expect("the rendered UPDATE must be valid SQL for DuckDB");

        let qty = |sql: &str| -> i32 {
            conn.query_row(sql, [], |row| row.get(0))
                .expect("qty query")
        };
        assert_eq!(qty("SELECT qty FROM myschema.orders"), 99);
        assert_eq!(
            qty("SELECT qty FROM main.orders"),
            10,
            "the bare name resolves to this table, which the statement must not touch"
        );
    }

    #[test]
    fn a_quote_bearing_table_name_updates_by_the_table_it_names() {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch(r#"CREATE TABLE "we""ird" (id INTEGER, qty INTEGER)"#)
            .expect("create");
        conn.execute_batch(r#"INSERT INTO "we""ird" VALUES (1, 10), (2, 20)"#)
            .expect("insert");

        conn.execute_batch(&update_statement(
            &TableReference::bare(r#"we"ird"#),
            r#""qty" = 99"#,
            Some(r#""id" = 2"#),
        ))
        .expect("the rendered UPDATE must be valid SQL for DuckDB");

        let mut stmt = conn
            .prepare(r#"SELECT qty FROM "we""ird" ORDER BY id"#)
            .expect("prepare");
        let quantities: Vec<i32> = stmt
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<duckdb::Result<Vec<i32>>>()
            .expect("rows");
        assert_eq!(quantities, vec![10, 99]);
    }

    /// What the escape is worth: a name carrying a `WHERE` of its own widens an interpolated
    /// `DELETE` to the whole table, because the name's quote ends the identifier and the rest is
    /// read as statement text. Quoted, the statement can only name a table — and no table by that
    /// name exists, so it is refused rather than silently widened.
    #[test]
    fn a_table_name_carrying_sql_cannot_widen_a_delete() {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch("CREATE TABLE t (id INTEGER)")
            .expect("create");
        conn.execute_batch("INSERT INTO t VALUES (1), (2), (3)")
            .expect("insert");

        let name = r#"t" WHERE 1=1 --"#;
        assert!(
            conn.execute_batch(&delete_statement(
                &TableReference::bare(name),
                Some(r#""id" = 1"#)
            ))
            .is_err(),
            "the quoted name must not resolve to table t"
        );
        let remaining: i64 = conn
            .query_row("SELECT count(*) FROM t", [], |row| row.get(0))
            .expect("count");
        assert_eq!(remaining, 3, "no row may be removed from t");

        // What the interpolated rendering meant instead.
        conn.execute_batch(&format!(r#"DELETE FROM "{name}" WHERE "id" = 1"#))
            .expect("the interpolated rendering is valid SQL");
        let remaining: i64 = conn
            .query_row("SELECT count(*) FROM t", [], |row| row.get(0))
            .expect("count");
        assert_eq!(remaining, 0, "the interpolated name emptied the table");
    }

    /// A name that needs no escape has to reach the engine as the same table it always did.
    #[test]
    fn an_ordinary_table_name_still_names_its_table() {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch("CREATE TABLE orders (id INTEGER)")
            .expect("create");
        conn.execute_batch("INSERT INTO orders VALUES (1), (2)")
            .expect("insert");

        conn.execute_batch(&delete_statement(
            &TableReference::bare("orders"),
            Some(r#""id" = 1"#),
        ))
        .expect("the rendered DELETE must be valid SQL for DuckDB");

        let remaining: i64 = conn
            .query_row("SELECT count(*) FROM orders", [], |row| row.get(0))
            .expect("count");
        assert_eq!(remaining, 1);
    }

    /// The reason a cross-relation predicate must be refused rather than rendered by column name:
    /// erased to `"id" = "id"` it is a tautology, and the `DELETE` built from it empties the table.
    #[test]
    fn a_cross_relation_predicate_cannot_become_a_tautology() {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch("CREATE TABLE t (id INTEGER)")
            .expect("create");
        conn.execute_batch("INSERT INTO t VALUES (1), (2), (3)")
            .expect("insert");

        let filters = vec![
            Expr::Column(datafusion::common::Column::new(Some("t1"), "id")).eq(Expr::Column(
                datafusion::common::Column::new(Some("t2"), "id"),
            )),
        ];
        assert!(
            filters_to_sql(&filters, Some(Engine::DuckDB)).is_err(),
            "a cross-relation predicate must not render into a WHERE clause"
        );

        // What the erased rendering would have meant.
        conn.execute_batch(r#"DELETE FROM t WHERE "id" = "id""#)
            .expect("the erased rendering is valid SQL");
        let remaining: i64 = conn
            .query_row("SELECT count(*) FROM t", [], |row| row.get(0))
            .expect("count");
        assert_eq!(remaining, 0, "the erased predicate is a tautology");
    }
}

/// `SQLite` is the second engine these statements are built for, and it reads a delimited
/// identifier by its own rules. Postgres, the third, needs a running server and so has no
/// in-process equivalent to assert against.
#[cfg(all(test, feature = "sqlite"))]
mod sqlite_execution_tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn a_quote_bearing_table_name_deletes_by_the_table_it_names() {
        let conn = Connection::open_in_memory().expect("in-memory SQLite");
        conn.execute_batch(r#"CREATE TABLE "we""ird" (id INTEGER)"#)
            .expect("create");
        conn.execute_batch(r#"INSERT INTO "we""ird" VALUES (1), (2), (3)"#)
            .expect("insert");

        conn.execute_batch(&delete_statement(
            &TableReference::bare(r#"we"ird"#),
            Some(r#""id" = 2"#),
        ))
        .expect("the rendered DELETE must be valid SQL for SQLite");

        let mut stmt = conn
            .prepare(r#"SELECT id FROM "we""ird" ORDER BY id"#)
            .expect("prepare");
        let surviving: Vec<i32> = stmt
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<rusqlite::Result<Vec<i32>>>()
            .expect("rows");
        assert_eq!(surviving, vec![1, 3]);
    }

    #[test]
    fn a_quote_bearing_table_name_updates_by_the_table_it_names() {
        let conn = Connection::open_in_memory().expect("in-memory SQLite");
        conn.execute_batch(r#"CREATE TABLE "we""ird" (id INTEGER, qty INTEGER)"#)
            .expect("create");
        conn.execute_batch(r#"INSERT INTO "we""ird" VALUES (1, 10), (2, 20)"#)
            .expect("insert");

        conn.execute_batch(&update_statement(
            &TableReference::bare(r#"we"ird"#),
            r#""qty" = 99"#,
            Some(r#""id" = 2"#),
        ))
        .expect("the rendered UPDATE must be valid SQL for SQLite");

        let mut stmt = conn
            .prepare(r#"SELECT qty FROM "we""ird" ORDER BY id"#)
            .expect("prepare");
        let quantities: Vec<i32> = stmt
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<rusqlite::Result<Vec<i32>>>()
            .expect("rows");
        assert_eq!(quantities, vec![10, 99]);
    }
}

/// The DuckDB timestamp normalization is only correct if the rows DuckDB then removes are the
/// rows the filter named, so these run the `DELETE` the DML path builds and assert on what
/// survives it. A rendering assertion cannot tell "normalized" from "normalized into a different
/// instant".
#[cfg(all(test, feature = "duckdb"))]
mod duckdb_timestamp_precision_tests {
    use super::*;
    use crate::sql::sql_provider_datafusion::expr::Engine;
    use arrow::datatypes::{DataType, Field, TimeUnit};
    use datafusion::prelude::*;
    use datafusion::scalar::ScalarValue;
    use duckdb::Connection;

    /// `2026-01-01 00:00:00` UTC, in microseconds.
    const EPOCH_US: i64 = 1_767_225_600_000_000;

    /// Row 1 sits 999µs past the second, row 2 sits on it. A filter naming any instant between
    /// the two must remove exactly row 1 — and cannot, once both operands collapse onto the
    /// second.
    const MICROSECOND_APART: [&str; 2] = [
        "2026-01-01 00:00:00.000999+00",
        "2026-01-01 00:00:00.000000+00",
    ];

    fn schema(ts: DataType) -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("ts", ts, true),
        ])
    }

    fn ts_literal(value: ScalarValue) -> Expr {
        Expr::Literal(value, None)
    }

    /// Runs `setup` against a fresh in-memory DuckDB, then returns the single column `query`
    /// selects. Every test here asserts on rows that survived a statement, so the statement and
    /// the read have to share one connection.
    fn one_column<T: duckdb::types::FromSql>(setup: &[String], query: &str) -> Vec<T> {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        for statement in setup {
            conn.execute_batch(statement).unwrap_or_else(|e| {
                panic!("the rendered statement must be valid DuckDB SQL: {statement}: {e}")
            });
        }

        let mut stmt = conn.prepare(query).expect("prepare");
        stmt.query_map([], |row| row.get(0))
            .expect("query")
            .collect::<duckdb::Result<Vec<T>>>()
            .expect("rows")
    }

    /// Runs the `DELETE` the DML path builds for `where_clause` over a two-row table of
    /// `column_type`, and returns the ids that survive it.
    fn surviving_ids(column_type: &str, values: [&str; 2], where_clause: &str) -> Vec<i32> {
        one_column(
            &[
                format!("CREATE TABLE t (id INTEGER, ts {column_type})"),
                format!(
                    "INSERT INTO t VALUES (1, '{}'), (2, '{}')",
                    values[0], values[1]
                ),
                format!("DELETE FROM t WHERE {where_clause}"),
            ],
            "SELECT id FROM t ORDER BY id",
        )
    }

    /// The reported defect, end to end. Row 1 sits 999µs past the second and row 2 sits on it;
    /// the filter names the 500µs mark, so exactly row 1 goes.
    ///
    /// Normalized, both operands collapse onto the second, the predicate is false for every row,
    /// and the `DELETE` silently removes nothing.
    #[test]
    fn a_microsecond_timestamptz_deletes_the_row_inside_the_millisecond() {
        let schema = schema(DataType::Timestamp(
            TimeUnit::Microsecond,
            Some("UTC".into()),
        ));
        let filters = vec![col("ts").gt(ts_literal(ScalarValue::TimestampMicrosecond(
            Some(EPOCH_US + 500),
            Some("UTC".into()),
        )))];

        let with_schema = filters_to_sql_with_schema(&filters, Some(Engine::DuckDB), Some(&schema))
            .expect("filters_to_sql should succeed");
        assert_eq!(with_schema, "\"ts\" > make_timestamptz(1767225600000500)");
        assert_eq!(
            surviving_ids("TIMESTAMPTZ", MICROSECOND_APART, &with_schema),
            vec![2],
            "the row past the filter's instant must be the one removed"
        );

        // The same filter with the column-side truncation left in place, which isolates it from
        // the literal rendering: the column collapses onto the millisecond it started in, and the
        // row the caller selected survives even though the literal is exact.
        let without_schema =
            filters_to_sql(&filters, Some(Engine::DuckDB)).expect("filters_to_sql should succeed");
        assert_eq!(
            without_schema,
            "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) > make_timestamptz(1767225600000500)"
        );
        assert_eq!(
            surviving_ids("TIMESTAMPTZ", MICROSECOND_APART, &without_schema),
            vec![1, 2],
            "this is the defect being fixed: the truncated comparison removes nothing"
        );
    }

    /// The types the normalization exists for keep it, and keep working. Without it DuckDB v1.5.5
    /// refuses the statement outright — *"Cannot compare values of type TIMESTAMP_MS and type
    /// TIMESTAMP WITH TIME ZONE"* — so this is the check that the type test did not narrow the
    /// rewrite past the case it was written for.
    #[test]
    fn a_naive_millisecond_column_is_still_normalized_and_still_binds() {
        let schema = schema(DataType::Timestamp(TimeUnit::Millisecond, None));
        let filters = vec![col("ts").gt(ts_literal(ScalarValue::TimestampMillisecond(
            Some(1_767_225_600_001),
            None,
        )))];

        let where_clause =
            filters_to_sql_with_schema(&filters, Some(Engine::DuckDB), Some(&schema))
                .expect("filters_to_sql should succeed");
        assert_eq!(
            where_clause,
            "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) > make_timestamptz(1767225600001000)"
        );
        assert_eq!(
            surviving_ids(
                "TIMESTAMP_MS",
                ["2026-01-01 00:00:00.002", "2026-01-01 00:00:00.001"],
                &where_clause,
            ),
            vec![2],
        );
    }

    /// Subtraction is the one non-comparison operator the rewrite covers, and the type test now
    /// takes it off a timezone-aware column too. DuckDB v1.5.5 refuses `TIMESTAMP_MS - TIMESTAMPTZ`
    /// — which is why the rewrite reaches subtraction at all — but subtracts two `TIMESTAMPTZ`
    /// values happily, returning the same interval the normalized form does.
    #[test]
    fn subtracting_a_timestamp_from_a_timezone_aware_column_still_binds() {
        let schema = schema(DataType::Timestamp(
            TimeUnit::Microsecond,
            Some("UTC".into()),
        ));
        let difference = col("ts")
            - ts_literal(ScalarValue::TimestampMicrosecond(
                Some(EPOCH_US),
                Some("UTC".into()),
            ));

        let rendered = assignments_to_sql_with_schema(
            &[("d".to_string(), difference)],
            Some(Engine::DuckDB),
            Some(&schema),
        )
        .expect("assignments_to_sql should succeed");
        assert_eq!(
            rendered,
            "\"d\" = \"ts\" - make_timestamptz(1767225600000000)"
        );

        let difference: Vec<String> = one_column(
            &[
                "CREATE TABLE t (ts TIMESTAMPTZ)".to_string(),
                "INSERT INTO t VALUES ('2026-01-01 00:00:01.5+00')".to_string(),
            ],
            "SELECT (\"ts\" - make_timestamptz(1767225600000000))::VARCHAR FROM t",
        );

        assert_eq!(difference, vec!["00:00:01.5".to_string()]);
    }

    /// A `DATE` column is the case where the normalization is load-bearing for the *reference
    /// frame* rather than for binding. DuckDB promotes a bare `DATE` to a `TIMESTAMPTZ` at midnight
    /// in the session's `TimeZone`; the rendered literal is midnight UTC. Under a session west of
    /// UTC the two are different instants, so declining the rewrite for a resolved date type would
    /// make a `DELETE` remove different rows depending on the host.
    #[test]
    fn a_date_column_is_compared_in_utc_whatever_the_session_timezone_is() {
        let schema = schema(DataType::Date32);
        let filters = vec![col("ts").eq(ts_literal(ScalarValue::TimestampMicrosecond(
            Some(EPOCH_US),
            Some("UTC".into()),
        )))];

        let where_clause =
            filters_to_sql_with_schema(&filters, Some(Engine::DuckDB), Some(&schema))
                .expect("filters_to_sql should succeed");
        assert_eq!(
            where_clause,
            "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) = make_timestamptz(1767225600000000)"
        );

        // Row 1 is the date the filter names. It must go under either session timezone.
        for timezone in ["UTC", "America/Los_Angeles"] {
            let surviving: Vec<i32> = one_column(
                &[
                    format!("SET TimeZone='{timezone}'"),
                    "CREATE TABLE t (id INTEGER, ts DATE)".to_string(),
                    "INSERT INTO t VALUES (1, '2026-01-01'), (2, '2026-01-02')".to_string(),
                    format!("DELETE FROM t WHERE {where_clause}"),
                ],
                "SELECT id FROM t ORDER BY id",
            );

            assert_eq!(surviving, vec![2], "session TimeZone {timezone}");
        }
    }

    /// A `SET` value carries the same literal rendering as a filter, so an `UPDATE` writing a
    /// sub-second instant has to store the instant it was given.
    #[test]
    fn an_update_stores_the_sub_second_instant_it_was_assigned() {
        let schema = schema(DataType::Timestamp(
            TimeUnit::Microsecond,
            Some("UTC".into()),
        ));
        let assignments = vec![(
            "ts".to_string(),
            ts_literal(ScalarValue::TimestampMicrosecond(
                Some(EPOCH_US + 999),
                Some("UTC".into()),
            )),
        )];

        let set_clause =
            assignments_to_sql_with_schema(&assignments, Some(Engine::DuckDB), Some(&schema))
                .expect("assignments_to_sql should succeed");
        assert_eq!(set_clause, "\"ts\" = make_timestamptz(1767225600000999)");

        // Pinned so the rendered instant is read back in the frame it was written in, rather
        // than in whatever zone the host happens to be set to.
        let stored: Vec<String> = one_column(
            &[
                "SET TimeZone='UTC'".to_string(),
                "CREATE TABLE t (id INTEGER, ts TIMESTAMPTZ)".to_string(),
                "INSERT INTO t VALUES (1, '2020-01-01 00:00:00+00')".to_string(),
                update_statement(&TableReference::bare("t"), &set_clause, None),
            ],
            "SELECT ts::VARCHAR FROM t",
        );

        assert_eq!(stored, vec!["2026-01-01 00:00:00.000999+00".to_string()]);
    }
}
