//! Runs the SQL that the shared filter and DML rendering builds against an in-process DuckDB and
//! asserts on the rows that survive it.

use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use datafusion::common::TableReference;
use datafusion::scalar::ScalarValue;

/// A schema whose `ts` column has `data_type`, for the type-aware normalization tests.
fn schema_with_ts(data_type: DataType) -> Schema {
    Schema::new(vec![
        Field::new("ts", data_type, true),
        Field::new("other", DataType::Int64, true),
    ])
}

/// A rendering assertion cannot tell a literal that names the right instant from one that names a
/// neighbouring microsecond — both are well-formed SQL. These run the comparison DuckDB actually
/// evaluates against rows one microsecond apart and assert on which row survives it.
mod duckdb_timestamp_execution_tests {
    use super::*;
    use datafusion::prelude::*;
    use datafusion_table_providers_common::sql::sql_provider_datafusion::expr::*;
    use duckdb::Connection;

    /// `2^53` microseconds since the epoch — the last microsecond an `f64` holds alongside both
    /// its neighbours.
    const DOUBLE_BOUND_MICROS: i64 = 9_007_199_254_740_992;

    /// The two rows are built by a route the rendering under test does not use, so a fixture and
    /// the literal that has to find it cannot be wrong together.
    const ROWS: [(&str, &str); 2] = [
        ("at_bound", "2255-06-05 23:47:34.740992+00"),
        ("one_past_bound", "2255-06-05 23:47:34.740993+00"),
    ];

    /// Runs `filter` as DuckDB would receive it and returns the labels of the rows it keeps.
    fn rows_selected_by(filter: &Expr, session_timezone: &str) -> Vec<String> {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch(&format!(
            "SET TimeZone='{session_timezone}'; CREATE TABLE t (label VARCHAR, ts TIMESTAMPTZ)"
        ))
        .expect("create");
        for (label, instant) in ROWS {
            conn.execute_batch(&format!(
                "INSERT INTO t VALUES ('{label}', CAST('{instant}' AS TIMESTAMPTZ))"
            ))
            .expect("insert");
        }

        let schema = schema_with_ts(DataType::Timestamp(
            TimeUnit::Microsecond,
            Some("UTC".into()),
        ));
        let where_clause =
            to_sql_with_engine_and_schema(filter, Some(Engine::DuckDB), Some(&schema))
                .expect("to unparse");

        let mut statement = conn
            .prepare(&format!(
                "SELECT label FROM t WHERE {where_clause} ORDER BY label"
            ))
            .expect("prepare");
        statement
            .query_map([], |row| row.get::<_, String>(0))
            .expect("query")
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("rows")
    }

    fn micros_literal(micros: i64) -> Expr {
        Expr::Literal(ScalarValue::TimestampMicrosecond(Some(micros), None), None)
    }

    /// The instant one microsecond past the `f64` bound is the one the caller named; the row at
    /// the bound is the one the old rendering selected in its place.
    #[test]
    fn a_microsecond_past_the_double_bound_selects_the_row_it_names() {
        let filter = col("ts").eq(micros_literal(DOUBLE_BOUND_MICROS + 1));

        assert_eq!(
            rows_selected_by(&filter, "UTC"),
            vec!["one_past_bound".to_string()],
            "the literal must select the microsecond it names, not its neighbour"
        );
    }

    /// The row at the bound is still reachable — the repair moves which instant each literal
    /// names, and both must land on their own.
    #[test]
    fn the_microsecond_at_the_double_bound_still_selects_its_own_row() {
        let filter = col("ts").eq(micros_literal(DOUBLE_BOUND_MICROS));

        assert_eq!(
            rows_selected_by(&filter, "UTC"),
            vec!["at_bound".to_string()]
        );
    }

    /// A literal is an instant, not a wall clock. The session's `TimeZone` decides how DuckDB
    /// *prints* a `TIMESTAMPTZ` and must not decide which rows a comparison against one keeps —
    /// which is what rules out reaching the same exactness through a naive `make_timestamp`.
    #[test]
    fn the_selected_row_does_not_depend_on_the_session_timezone() {
        let filter = col("ts").eq(micros_literal(DOUBLE_BOUND_MICROS + 1));

        for timezone in ["UTC", "America/Los_Angeles", "Australia/Sydney"] {
            assert_eq!(
                rows_selected_by(&filter, timezone),
                vec!["one_past_bound".to_string()],
                "under TimeZone={timezone}"
            );
        }
    }

    /// A range predicate reads the same boundary the equality does, and is the shape a filter
    /// pushdown usually takes.
    #[test]
    fn a_range_boundary_past_the_double_bound_excludes_the_row_below_it() {
        let filter = col("ts").gt_eq(micros_literal(DOUBLE_BOUND_MICROS + 1));

        assert_eq!(
            rows_selected_by(&filter, "UTC"),
            vec!["one_past_bound".to_string()],
            "a bound one microsecond above the row at the f64 limit must exclude it"
        );
    }

    /// The counts DuckDB reserves are refused before any SQL is built, and the ones beside them
    /// are not — this is the half of that guard a rendering assertion cannot check, because both
    /// spellings are well-formed SQL and only one of them survives execution.
    #[test]
    fn the_counts_beside_the_reserved_sentinels_still_execute() {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");

        for micros in [i64::MAX - 1, i64::MIN, i64::MIN + 2] {
            let sql = to_sql_with_engine(&micros_literal(micros), Some(Engine::DuckDB))
                .expect("to unparse");
            let round_tripped: i64 = conn
                .query_row(&format!("SELECT epoch_us({sql})"), [], |row| row.get(0))
                .unwrap_or_else(|e| panic!("DuckDB refused {sql}: {e}"));

            assert_eq!(round_tripped, micros, "{sql}");
        }
    }
}

/// A rendering assertion pins the constant this file computes; it cannot tell a literal that names
/// the right day from one that names a different day, or none at all, because every one of those
/// is well-formed SQL. These run the comparison DuckDB actually evaluates against rows one day
/// apart and assert on which row survives it.
mod duckdb_date_execution_tests {
    use super::*;
    use datafusion::prelude::*;
    use datafusion_table_providers_common::sql::sql_provider_datafusion::expr::*;
    use duckdb::Connection;

    /// `2038-01-19` is the last day whose epoch second an `i32` holds (`i32::MAX / 86_400` =
    /// 24 855). The row beside it is the first day a product computed in `i32` could not name.
    const ROWS: [(&str, &str, i32); 2] = [
        ("last_i32_day", "2038-01-19", 24_855),
        ("first_day_past_it", "2038-01-20", 24_856),
    ];

    /// Runs `filter` against a real `DATE` column as DuckDB would receive it, with the column's
    /// Arrow type declared as `column_type`, and returns the labels of the rows it keeps.
    fn rows_selected_by(
        filter: &Expr,
        column_type: DataType,
        session_timezone: &str,
    ) -> Vec<String> {
        let conn = Connection::open_in_memory().expect("in-memory DuckDB");
        conn.execute_batch(&format!(
            "SET TimeZone='{session_timezone}'; CREATE TABLE t (label VARCHAR, ts DATE)"
        ))
        .expect("create");
        // The rows are built from the calendar day itself, a route the rendering under test does
        // not use, so a fixture and the literal that has to find it cannot be wrong together.
        for (label, day, _) in ROWS {
            conn.execute_batch(&format!(
                "INSERT INTO t VALUES ('{label}', CAST('{day}' AS DATE))"
            ))
            .expect("insert");
        }

        let schema = schema_with_ts(column_type);
        let where_clause =
            to_sql_with_engine_and_schema(filter, Some(Engine::DuckDB), Some(&schema))
                .expect("to unparse");

        let mut statement = conn
            .prepare(&format!(
                "SELECT label FROM t WHERE {where_clause} ORDER BY label"
            ))
            .expect("prepare");
        statement
            .query_map([], |row| row.get::<_, String>(0))
            .expect("query")
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("rows")
    }

    /// Every day names its own row and only its own row, on both sides of the bound the `i32`
    /// product stopped at. Asserting the row *beside* it too is what separates "the literal names
    /// this day" from "the literal matches everything".
    #[test]
    fn a_date32_literal_selects_the_day_it_names() {
        for session_timezone in ["UTC", "America/Los_Angeles"] {
            for (label, _, days) in ROWS {
                let filter = col("ts").eq(Expr::Literal(ScalarValue::Date32(Some(days)), None));

                assert_eq!(
                    rows_selected_by(&filter, DataType::Date32, session_timezone),
                    vec![label.to_string()],
                    "Date32({days}) under TimeZone {session_timezone}"
                );
            }
        }
    }

    /// The same days spelled as `Date64` millisecond counts select the same rows. Scaling such a
    /// count by the length of a day named an instant past the year 238 000, so the comparison it
    /// built selected nothing at all.
    #[test]
    fn a_date64_literal_selects_the_same_day_as_the_date32_spelling_of_it() {
        for session_timezone in ["UTC", "America/Los_Angeles"] {
            for (label, _, days) in ROWS {
                let millis = i64::from(days) * 86_400_000;
                let filter = col("ts").eq(Expr::Literal(ScalarValue::Date64(Some(millis)), None));

                assert_eq!(
                    rows_selected_by(&filter, DataType::Date64, session_timezone),
                    vec![label.to_string()],
                    "Date64({millis}) under TimeZone {session_timezone}"
                );
            }
        }
    }
}

/// The rendered SQL is only correct if the target engine agrees, so these run the statement the
/// DML paths build and assert on the rows that actually survive it. A rendering assertion alone
/// cannot distinguish "escaped" from "escaped in a form this engine accepts".
mod duckdb_execution_tests {
    use super::*;
    use datafusion::prelude::*;
    use datafusion_table_providers_common::sql::sql_provider_datafusion::expr::Engine;
    use datafusion_table_providers_common::util::dml::*;
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

/// The DuckDB timestamp normalization is only correct if the rows DuckDB then removes are the
/// rows the filter named, so these run the `DELETE` the DML path builds and assert on what
/// survives it. A rendering assertion cannot tell "normalized" from "normalized into a different
/// instant".
mod duckdb_timestamp_precision_tests {
    use super::*;
    use datafusion::prelude::*;
    use datafusion::scalar::ScalarValue;
    use datafusion_table_providers_common::sql::sql_provider_datafusion::expr::Engine;
    use datafusion_table_providers_common::util::dml::*;
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
