use std::{borrow::Cow, sync::Arc};

use arrow::datatypes::{DataType, Schema, TimeUnit};
use bigdecimal::{num_bigint::BigInt, BigDecimal};
use datafusion::{
    common::TableReference,
    logical_expr::{Cast, Expr, Operator},
    scalar::ScalarValue,
    sql::unparser::dialect::{
        DefaultDialect, Dialect, DuckDBDialect, MySqlDialect, PostgreSqlDialect, SqliteDialect,
    },
};

pub const SECONDS_IN_DAY: i32 = 86_400;

#[derive(Debug, snafu::Snafu)]
pub enum Error {
    #[snafu(display("Expression not supported {expr}"))]
    UnsupportedFilterExpr { expr: String },

    #[snafu(display("Engine {engine} not supported for expression {expr}"))]
    EngineNotSupportedForExpression { engine: String, expr: String },
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug)]
pub enum Engine {
    Spark,
    SQLite,
    DuckDB,
    ODBC,
    Postgres,
    MySQL,
}

impl Engine {
    /// Get the corresponding `Dialect` to use for unparsing
    pub fn dialect(&self) -> Arc<dyn Dialect + Send + Sync> {
        match self {
            Engine::SQLite => Arc::new(SqliteDialect {}),
            Engine::Postgres => Arc::new(PostgreSqlDialect {}),
            Engine::MySQL => Arc::new(MySqlDialect {}),
            Engine::DuckDB => Arc::new(DuckDBDialect::new()),
            Engine::Spark | Engine::ODBC => Arc::new(DefaultDialect {}),
        }
    }
}

/// Renders `value` as a SQL string literal for `engine`, escaping the characters that engine
/// reads as significant inside one.
///
/// Interpolating the value bare lets a single quote close the literal early: an ordinary
/// apostrophe then makes the statement fail to parse, and a value shaped like `x' OR 1=1 --`
/// binds as a wider predicate than the caller wrote, so a `DELETE` can match rows the filter
/// excluded. Every engine here ends a literal on `'` and accepts a doubled `''` for an embedded
/// one, which is also what `DataFusion`'s own unparser emits.
///
/// `Engine::MySQL` and `Engine::Spark` additionally read `\` as an escape character inside a
/// literal, so for them a backslash must be doubled as well — otherwise `\'` consumes the first
/// quote of a doubled pair and the second one closes the literal, which is worse than not
/// escaping at all. `Engine::DuckDB`, `Engine::SQLite`, `Engine::ODBC` and Postgres (under
/// `standard_conforming_strings`, on by default since 9.1) take `\` literally and must not gain
/// an extra one.
///
/// Two escape-mode residues remain, both of them server settings this function cannot see:
/// MySQL's `NO_BACKSLASH_ESCAPES` and Postgres's `standard_conforming_strings = off`. Binding the
/// value instead of rendering it is the only fix that removes them.
///
/// The match is exhaustive over [`Engine`] on purpose: a new engine must state which escaping it
/// needs rather than inherit whichever arm happens to be the catch-all.
pub(crate) fn string_literal(value: &str, engine: Option<Engine>) -> String {
    let escaped = match engine {
        Some(Engine::MySQL | Engine::Spark) => {
            if value.contains(['\\', '\'']) {
                Cow::Owned(value.replace('\\', r"\\").replace('\'', "''"))
            } else {
                Cow::Borrowed(value)
            }
        }
        Some(Engine::DuckDB | Engine::SQLite | Engine::Postgres | Engine::ODBC) | None => {
            if value.contains('\'') {
                Cow::Owned(value.replace('\'', "''"))
            } else {
                Cow::Borrowed(value)
            }
        }
    };
    format!("'{escaped}'")
}

/// Renders `name` as a double-quoted SQL identifier, doubling any embedded quote so that a
/// quote in a column name cannot close the identifier early.
pub(crate) fn quoted_identifier(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Renders `table` as a qualified SQL identifier chain, quoting each part with
/// [`quoted_identifier`] so that a statement addresses the same table the reference names.
///
/// Every part is quoted unconditionally, unlike [`TableReference::to_quoted_string`], which leaves
/// an all-lowercase part bare and so renders a reserved word (`order`, `user`) as syntax rather
/// than as a name.
pub(crate) fn quoted_table_reference(table: &TableReference) -> String {
    match table {
        TableReference::Bare { table } => quoted_identifier(table),
        TableReference::Partial { schema, table } => {
            format!("{}.{}", quoted_identifier(schema), quoted_identifier(table))
        }
        TableReference::Full {
            catalog,
            schema,
            table,
        } => format!(
            "{}.{}.{}",
            quoted_identifier(catalog),
            quoted_identifier(schema),
            quoted_identifier(table)
        ),
    }
}

/// Renders `expr` as a SQL fragment for `engine`.
///
/// The fragment goes into a statement that names exactly one table, so every qualified column in
/// `expr` must carry the same relation. A predicate spanning two relations has no meaning in such a
/// statement, and rendering it by column name alone would silently *widen* it: `t1.id = t2.id` would
/// become `"id" = "id"`, a tautology that a `DELETE` applies to every row. `DataFusion` never offers
/// a cross-relation predicate to a single-table provider — it pushes a filter into a scan only when
/// that scan covers it, and it validates ownership before unqualifying a DML filter — but this
/// function is public and its callers have no such guarantee, so the check belongs here.
///
/// References are compared for exact equality rather than with `TableReference::resolved_eq`, which
/// treats a component the other side omits as matching: a bare `t` is `resolved_eq` to both
/// `a.schema.t` and `b.schema.t`, so it is not transitive, and comparing every reference against one
/// arbitrarily chosen pivot could then admit two genuinely different tables. Without the session's
/// default catalog and schema this function cannot canonicalise a partial reference, so mixed
/// qualification of one table (`t` alongside `public.t`) is refused too — no caller produces it, and
/// refusing is the direction that cannot widen a statement.
pub fn to_sql_with_engine(expr: &Expr, engine: Option<Engine>) -> Result<String> {
    to_sql_with_engine_and_schema(expr, engine, None)
}

/// [`to_sql_with_engine`], given the schema the expression's columns belong to.
///
/// Only [`Engine::DuckDB`] consults the schema, and only to decide whether a comparison's operand
/// needs the timestamp normalization described on [`duckdb_normalizes_timestamp_operand`]. Passing
/// `None` — or a schema that does not carry the column — renders exactly what
/// [`to_sql_with_engine`] renders, so a caller with no schema to offer loses nothing.
pub fn to_sql_with_engine_and_schema(
    expr: &Expr,
    engine: Option<Engine>,
    schema: Option<&Schema>,
) -> Result<String> {
    let mut relations = expr
        .column_refs()
        .into_iter()
        .filter_map(|column| column.relation.as_ref());
    if let Some(first) = relations.next() {
        if relations.any(|other| other != first) {
            return Err(Error::UnsupportedFilterExpr {
                expr: format!("{expr}"),
            });
        }
    }

    render_expr(expr, engine, schema)
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_precision_loss)]
fn render_expr(expr: &Expr, engine: Option<Engine>, schema: Option<&Schema>) -> Result<String> {
    match expr {
        Expr::BinaryExpr(binary_expr) => {
            let left = render_expr(&binary_expr.left, engine, schema)?;
            let right = render_expr(&binary_expr.right, engine, schema)?;

            if let Some(Engine::DuckDB) = engine {
                // TODO: DuckDB doesn't support comparison between timestamp_s /timestamp_ms with timestampz as of v1
                // Revisit in future DuckDB versions
                //
                // Which operand is a timestamp is decided by its `Expr` shape, never by whether
                // its rendered SQL starts with `TO_TIMESTAMP`. A nested comparison is itself
                // normalized, so it renders as `TO_TIMESTAMP(..)` while being a boolean, and
                // normalizing it emits `EPOCH_MS(<predicate>)`, for which DuckDB has no overload.
                //
                // `Minus` is included because subtraction is the only arithmetic DuckDB defines
                // between two timestamps. `+`, `*` and `/` have no timestamp/timestamp overload
                // at all, so normalizing their operands could not make them bind.
                let normalizes_timestamp_operands = matches!(
                    binary_expr.op,
                    Operator::Eq
                        | Operator::NotEq
                        | Operator::Lt
                        | Operator::LtEq
                        | Operator::Gt
                        | Operator::GtEq
                        | Operator::IsDistinctFrom
                        | Operator::IsNotDistinctFrom
                        | Operator::Minus
                );
                // Only the left operand is normalized, and only when the right is a timestamp
                // and the left is not. Normalizing a bare *right* operand would extend the
                // rewrite's cost to comparisons that bind correctly today, trading a loud binder
                // error for a silently shifted result, so the mismatch is left to fail loudly in
                // that direction.
                if normalizes_timestamp_operands
                    && is_duckdb_timestamp_operand(&binary_expr.right)
                    && !is_duckdb_timestamp_operand(&binary_expr.left)
                    && duckdb_normalizes_timestamp_operand(&binary_expr.left, schema)
                {
                    // `EPOCH_MS(..)` parenthesizes the left operand on its own.
                    return Ok(format!(
                        "TO_TIMESTAMP(EPOCH_MS({}) / 1000) {} {}",
                        left,
                        binary_expr.op,
                        group_if_binary(&binary_expr.right, &right)
                    ));
                }
            }

            match binary_expr.op {
                Operator::And | Operator::Or => {
                    Ok(format!("({}) {} ({})", left, binary_expr.op, right))
                }
                // The `AND`/`OR` arm above parenthesizes its own operands; every other
                // operator needs a nested operand grouped so precedence cannot regroup it.
                _ => Ok(format!(
                    "{} {} {}",
                    group_if_binary(&binary_expr.left, &left),
                    binary_expr.op,
                    group_if_binary(&binary_expr.right, &right)
                )),
            }
        }
        Expr::Column(column) => {
            // Render the column's own name, dropping any qualifier. Rendering `Column`'s
            // `Display` would emit its *flat* name, making a qualified `t.a` the single identifier
            // `"t.a"` — a column that usually does not exist, and where it does (dotted names are
            // legal in Arrow and in every engine here) is not the one the caller named.
            //
            // Dropping it is safe because `to_sql_with_engine` has already established that the
            // whole fragment names at most one relation, and the statement it goes into names
            // exactly one table. `DataFusion` unqualifies at the same boundary for the same reason
            // (`strip_column_qualifiers` for DML filters, `unnormalize_cols` for pushed-down ones).
            //
            // Rejecting a lone qualifier instead would break the callers that use this function as
            // a *capability probe*: `supports_filters_pushdown` is asked about the qualified
            // predicate while `scan` receives the unqualified one, and `ClickHouseTable` (plus the
            // runtime's DuckDB vector index) answers it by rendering here and mapping `Err(_)` to
            // `Unsupported`. An error would silently keep those predicates — `LIKE` among them —
            // out of the remote statement.
            match engine {
                Some(Engine::Spark | Engine::ODBC) => Ok(column.name.clone()),
                _ => Ok(quoted_identifier(&column.name)),
            }
        }
        Expr::Cast(cast) => handle_cast(cast, engine, schema, expr),
        Expr::Literal(value, _) => match value {
            // `Date32` and `Date64` both name a whole day since the UNIX epoch, but they
            // count in different units — days and milliseconds — so the epoch second each
            // one names is a different computation. Only one of them scales by the length
            // of a day, and it has to be computed in a width that holds the result.
            ScalarValue::Date32(Some(value)) => {
                // Widen before multiplying. `value` and `SECONDS_IN_DAY` are both `i32`, so
                // the product would be computed in `i32` and overflow for every date past
                // `i32::MAX / 86_400` = 24 855 days (2038-01-19), and every date below
                // -24 855 (1901-12-13). A debug build panics there; a release build wraps to
                // a literal naming an unrelated instant, so the comparison built from it
                // selects a row set the query never asked for. `i32::MIN` days is the widest
                // count the variant holds and still leaves the product four orders of
                // magnitude inside `i64`.
                let seconds = i64::from(*value) * i64::from(SECONDS_IN_DAY);
                match engine {
                    Some(Engine::SQLite) => Ok(format!("date({seconds}, 'unixepoch')")),
                    _ => Ok(format!("TO_TIMESTAMP({seconds})")),
                }
            }
            ScalarValue::Date64(Some(value)) => {
                // Arrow defines `Date64` as "the elapsed time since UNIX epoch in
                // milliseconds", so the count is already an instant and scales *down* to the
                // second. Scaling it up by the length of a day instead reads each
                // millisecond as a day, which puts 1970-01-02 past the year 238 000 — an
                // instant no row can hold, so the comparison matches nothing.
                //
                // `div_euclid` floors, so a count that is not a whole number of days — which
                // Arrow's definition excludes but its type does not — lands on the same side
                // of the instant it names above the epoch as below. Every whole-day count,
                // which is the entire well-formed domain, divides exactly and is unaffected.
                let seconds = value.div_euclid(1_000);
                match engine {
                    Some(Engine::SQLite) => Ok(format!("date({seconds}, 'unixepoch')")),
                    _ => Ok(format!("TO_TIMESTAMP({seconds})")),
                }
            }
            ScalarValue::Null => Ok(value.to_string()),
            ScalarValue::Int16(Some(value)) => Ok(value.to_string()),
            ScalarValue::Int32(Some(value)) => Ok(value.to_string()),
            ScalarValue::Int64(Some(value)) => Ok(value.to_string()),
            ScalarValue::Boolean(Some(value)) => Ok(value.to_string()),
            ScalarValue::Utf8(Some(value))
            | ScalarValue::LargeUtf8(Some(value))
            | ScalarValue::Utf8View(Some(value)) => Ok(string_literal(value, engine)),
            ScalarValue::Float32(Some(value)) => Ok(value.to_string()),
            ScalarValue::Float64(Some(value)) => Ok(value.to_string()),
            ScalarValue::Int8(Some(value)) => Ok(value.to_string()),
            ScalarValue::UInt8(Some(value)) => Ok(value.to_string()),
            ScalarValue::UInt16(Some(value)) => Ok(value.to_string()),
            ScalarValue::UInt32(Some(value)) => Ok(value.to_string()),
            ScalarValue::UInt64(Some(value)) => Ok(value.to_string()),
            ScalarValue::TimestampNanosecond(Some(value), timezone) => match engine {
                Some(Engine::SQLite) => {
                    Ok(format!("datetime({}, 'unixepoch')", value / 1_000_000_000))
                }
                Some(Engine::Postgres) => {
                    if timezone.is_none() {
                        Ok(format!(
                            "TO_TIMESTAMP({}) AT TIME ZONE 'UTC'",
                            *value as f64 / 1_000_000_000.0
                        ))
                    } else {
                        Ok(format!("TO_TIMESTAMP({})", *value as f64 / 1_000_000_000.0))
                    }
                }
                Some(Engine::DuckDB) => {
                    duckdb_timestamp_literal(*value, TimeUnit::Nanosecond, expr)
                }
                _ => Ok(format!("TO_TIMESTAMP({})", value / 1_000_000_000)),
            },
            ScalarValue::TimestampMicrosecond(Some(value), timezone) => {
                let seconds = *value as f64 / 1_000_000.0;
                match engine {
                    Some(Engine::SQLite) => {
                        Ok(format!("datetime({}, 'unixepoch')", value / 1_000_000))
                    }
                    Some(Engine::Postgres) => {
                        if timezone.is_none() {
                            Ok(format!("TO_TIMESTAMP({seconds}) AT TIME ZONE 'UTC'"))
                        } else {
                            Ok(format!("TO_TIMESTAMP({seconds})"))
                        }
                    }
                    Some(Engine::MySQL) => Ok(format!("FROM_UNIXTIME({seconds})")),
                    Some(Engine::DuckDB) => {
                        duckdb_timestamp_literal(*value, TimeUnit::Microsecond, expr)
                    }
                    _ => Ok(format!("TO_TIMESTAMP({})", value / 1_000_000)),
                }
            }
            ScalarValue::TimestampMillisecond(Some(value), timezone) => {
                let seconds = *value as f64 / 1000.0;
                match engine {
                    Some(Engine::SQLite) => Ok(format!("datetime({}, 'unixepoch')", value / 1000)),
                    Some(Engine::Postgres) => {
                        if timezone.is_none() {
                            Ok(format!("TO_TIMESTAMP({seconds}) AT TIME ZONE 'UTC'"))
                        } else {
                            Ok(format!("TO_TIMESTAMP({seconds})"))
                        }
                    }
                    Some(Engine::DuckDB) => {
                        duckdb_timestamp_literal(*value, TimeUnit::Millisecond, expr)
                    }
                    _ => Ok(format!("TO_TIMESTAMP({})", value / 1000)),
                }
            }
            ScalarValue::TimestampSecond(Some(value), timezone) => match engine {
                Some(Engine::SQLite) => Ok(format!("datetime({value}, 'unixepoch')")),
                Some(Engine::Postgres) => {
                    if timezone.is_none() {
                        Ok(format!("TO_TIMESTAMP({value}) AT TIME ZONE 'UTC'"))
                    } else {
                        Ok(format!("TO_TIMESTAMP({value})"))
                    }
                }
                Some(Engine::DuckDB) => duckdb_timestamp_literal(*value, TimeUnit::Second, expr),
                _ => Ok(format!("TO_TIMESTAMP({value})")),
            },
            ScalarValue::Decimal128(Some(v), _, s) => {
                let decimal = BigDecimal::new(BigInt::from(*v), *s as i64);
                Ok(decimal.to_string())
            }
            _ => Err(Error::UnsupportedFilterExpr {
                expr: format!("{expr}"),
            }),
        },
        Expr::Like(like_expr) => {
            if like_expr.escape_char.is_some() {
                // Escape char is not supported
                return Err(Error::UnsupportedFilterExpr {
                    expr: format!("{expr}"),
                });
            }
            let expr = render_expr(&like_expr.expr, engine, schema)?;
            let pattern = render_expr(&like_expr.pattern, engine, schema)?;

            let mut op_and_pattern = match (engine, like_expr.case_insensitive) {
                (Some(Engine::Postgres | Engine::DuckDB), true) => format!("ILIKE {pattern}"),
                (Some(Engine::Postgres | Engine::DuckDB), false) => format!("LIKE {pattern}"),
                (Some(Engine::SQLite), true) => format!("LIKE {pattern}"),
                (Some(Engine::SQLite), false) => format!("LIKE {pattern} COLLATE BINARY"),
                // Standard SQL LIKE for unknown/unspecified engines (e.g. FlightSQL federation)
                (None | Some(Engine::MySQL | Engine::Spark | Engine::ODBC), false) => {
                    format!("LIKE {pattern}")
                }
                // Case-insensitive LIKE via UPPER() for engines without ILIKE
                (None | Some(Engine::MySQL | Engine::Spark | Engine::ODBC), true) => {
                    return Ok(format!("UPPER({expr}) LIKE UPPER({pattern})"));
                }
            };

            if like_expr.negated {
                op_and_pattern = format!("NOT {}", op_and_pattern)
            };

            Ok(format!("{expr} {op_and_pattern}"))
        }
        Expr::InList(in_list) => {
            let expr = render_expr(&in_list.expr, engine, schema)?;
            let list = in_list
                .list
                .iter()
                .map(|expr| render_expr(expr, engine, schema))
                .collect::<Result<Vec<String>>>()?;

            let op = if in_list.negated { "NOT IN" } else { "IN" };

            Ok(format!("{expr} {op} ({list})", list = list.join(", ")))
        }
        Expr::ScalarFunction(scalar_function) => {
            let args = scalar_function
                .args
                .iter()
                .map(|expr| render_expr(expr, engine, schema))
                .collect::<Result<Vec<String>>>()?;

            Ok(format!("{}({})", scalar_function.name(), args.join(", ")))
        }
        _ => Err(Error::UnsupportedFilterExpr {
            expr: format!("{expr}"),
        }),
    }
}

pub fn to_sql(expr: &Expr) -> Result<String> {
    to_sql_with_engine(expr, None)
}

/// Parenthesize `rendered` when `expr` is itself a binary expression, so that composing it into a
/// larger expression cannot lose the grouping the `Expr` tree carries.
///
/// Left to SQL operator precedence, the grouping is re-derived and can differ: `a * (b + c)`
/// rendered as `a * b + c` is a different value that binds and returns silently; `x - (t + d)`
/// rendered as `x - t + d` means `(x - t) + d`, since `-` and `+` share a precedence level and
/// associate left; and `flag = (ts < t)` rendered as `flag = ts < t` is a chained comparison,
/// which DuckDB rejects outright.
fn group_if_binary(expr: &Expr, rendered: &str) -> String {
    if matches!(expr, Expr::BinaryExpr(_)) {
        format!("({rendered})")
    } else {
        rendered.to_string()
    }
}

/// Whether `expr` is a timestamp-valued operand under [`Engine::DuckDB`] — that is, whether it
/// renders as a term DuckDB types as `TIMESTAMPTZ`.
///
/// This is deliberately a test of what the expression *is*, not of what its SQL looks like. A
/// nested comparison the normalization has already rewritten also renders with a `TO_TIMESTAMP`
/// prefix, yet it is a boolean; conflating the two is what made the normalization emit
/// `EPOCH_MS(<predicate>)`.
///
/// The leaf shapes mirror the arms that produce a `TIMESTAMPTZ` rendering — the date and
/// timestamp [`ScalarValue`] literals, and [`handle_cast`]'s DuckDB arm for a cast to a
/// timestamp. Keep them in step with those arms.
///
/// Timestamp arithmetic is timestamp-valued in turn, following DuckDB's own typing: adding an
/// interval to a timestamp yields a timestamp either way round, and subtracting one does too,
/// while subtracting a timestamp *from* a timestamp yields an interval instead. That is decided
/// from the operator and the operands' own shapes.
///
/// A bare [`Expr::Column`] is *not* treated as a timestamp here, so a comparison between two
/// timestamp columns is left un-normalized and DuckDB refuses it. Resolving the operand's type
/// against a schema — which [`duckdb_normalizes_timestamp_operand`] now does for the other side of
/// the comparison — would settle that case too, but it would extend the rewrite to expressions it
/// has never covered rather than narrow it: <https://github.com/spiceai/spiceai/issues/13145>.
fn is_duckdb_timestamp_operand(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(value, _) => matches!(
            value,
            ScalarValue::Date32(Some(_))
                | ScalarValue::Date64(Some(_))
                | ScalarValue::TimestampSecond(Some(_), _)
                | ScalarValue::TimestampMillisecond(Some(_), _)
                | ScalarValue::TimestampMicrosecond(Some(_), _)
                | ScalarValue::TimestampNanosecond(Some(_), _)
        ),
        Expr::Cast(cast) => matches!(
            cast.field.data_type(),
            arrow::datatypes::DataType::Timestamp(_, _)
        ),
        Expr::BinaryExpr(binary_expr) => match binary_expr.op {
            Operator::Plus => {
                is_duckdb_timestamp_operand(&binary_expr.left)
                    || is_duckdb_timestamp_operand(&binary_expr.right)
            }
            Operator::Minus => {
                is_duckdb_timestamp_operand(&binary_expr.left)
                    && !is_duckdb_timestamp_operand(&binary_expr.right)
            }
            _ => false,
        },
        _ => false,
    }
}

/// Renders a `ScalarValue::Timestamp*` count as the DuckDB literal all four of those arms of
/// [`render_expr`] emit for that engine. (The `Date32` and `Date64` arms have no DuckDB arm and
/// reach the shared catch-all instead. They render a whole-second count, and `TO_TIMESTAMP`'s
/// `DOUBLE` represents consecutive integers exactly up to `2^53` seconds — past the end of the
/// timestamp range DuckDB itself holds — so nothing they can name loses a digit to it.)
///
/// The engines sharing that catch-all render a timestamp by integer division down to whole
/// seconds, which drops everything below the second and moves which rows a comparison against the
/// literal selects. DuckDB carries its own arm to keep those digits, and this is the whole of it —
/// `unit` is what each arm contributes, so a call site cannot name a scale its `ScalarValue`
/// variant disagrees with.
///
/// `TO_TIMESTAMP` takes a `DOUBLE`, so reaching it costs the count a widening to `f64`, which
/// represents consecutive integers exactly only up to `2^53` — about **2255-06-05** as a count of
/// microseconds. `make_timestamptz` takes the count as a `BIGINT`, so nothing is widened, and it
/// yields the same `TIMESTAMP WITH TIME ZONE` fixed to UTC, which is the type
/// [`is_duckdb_timestamp_operand`] and the normalization both rely on the literal having.
///
/// That UTC frame is what rules out the alternative: `make_timestamp` is exact too, but it yields
/// a naive `TIMESTAMP`, and casting one to `TIMESTAMPTZ` reads the count in the session's
/// `TimeZone` — measured under `America/Los_Angeles` it lands seven hours from the instant the
/// value names, while this rendering measures identical there and under `UTC`.
///
/// Two kinds of count have no literal to render. A second or millisecond count can name an instant
/// no microsecond count holds, and DuckDB holds nothing else; and DuckDB reserves `i64::MAX` and
/// `-i64::MAX` as its infinity sentinels, refusing both with *"Timestamp microseconds out of
/// range"*, so rendering one would report the filter pushdown-capable and then fail the statement
/// built from it — which a `DELETE` or `UPDATE` reaches only after its SQL is generated. Declining
/// leaves `supports_filters_pushdown` reporting the filter unsupported and DataFusion applying it
/// itself. `i64::MIN` is not a sentinel: v1.5.5 renders it, round-trips it exactly, and compares it
/// as the finite instant it is.
fn duckdb_timestamp_literal(value: i64, unit: TimeUnit, expr: &Expr) -> Result<String> {
    let micros = match unit {
        // A `TIMESTAMPTZ` holds microseconds, so the nanosecond digits cannot survive whatever
        // this renders. `div_euclid` floors, so they are dropped towards the same side of the
        // epoch above it as below.
        TimeUnit::Nanosecond => Some(value.div_euclid(1_000)),
        TimeUnit::Microsecond => Some(value),
        TimeUnit::Millisecond => value.checked_mul(1_000),
        TimeUnit::Second => value.checked_mul(1_000_000),
    }
    .filter(|micros| *micros != i64::MAX && *micros != -i64::MAX)
    .ok_or_else(|| Error::UnsupportedFilterExpr {
        expr: format!("{expr}"),
    })?;

    Ok(format!("make_timestamptz({micros})"))
}

/// Whether the DuckDB timestamp normalization has to be applied to `expr`, given the schema its
/// columns come from.
///
/// The normalization renders its operand as `TO_TIMESTAMP(EPOCH_MS(<operand>) / 1000)`, which makes
/// DuckDB read it as a `TIMESTAMPTZ` fixed to UTC. Whether the operand carries a timezone is what
/// decides it; the unit does not.
///
/// A **naive** timestamp needs it, at every unit. DuckDB v1.5.5 refuses to compare `TIMESTAMP_S`,
/// `TIMESTAMP_MS` or `TIMESTAMP_NS` with a `TIMESTAMPTZ` at all — *"Cannot compare values of type
/// `TIMESTAMP_S` and type `TIMESTAMP WITH TIME ZONE`"* — and a microsecond `TIMESTAMP`, which does
/// compare, would be read in the session's `TimeZone` where the rendered literal is a UTC instant.
///
/// It is not free for all of them: `EPOCH_MS` yields whole milliseconds, so a naive `TIMESTAMP` or
/// `TIMESTAMP_NS` is still truncated to the millisecond by the very rewrite that makes it compare
/// in the right frame. `TIMESTAMP_S` and `TIMESTAMP_MS` carry nothing finer and lose nothing. The
/// remaining truncation is <https://github.com/spiceai/spiceai/issues/13146>; pinning UTC losslessly
/// needs `AT TIME ZONE`, which is an ICU-extension dependency this rendering does not take.
///
/// A **date** needs it for the reference frame alone. DuckDB promotes a bare `DATE` to a
/// `TIMESTAMPTZ` at midnight in the *session's* `TimeZone`, while `EPOCH_MS` pins the same midnight
/// UTC the rendered literal is in — measured against v1.5.5, `"dt" = TO_TIMESTAMP(..)` and its
/// normalized form disagree under `America/Los_Angeles` and agree under `UTC`.
///
/// A **timezone-aware** timestamp needs neither: it is already the type and the reference frame the
/// literal renders as, so normalizing only truncates it to whole milliseconds. That is the case this
/// predicate exists for — without a schema the rewrite could not tell it from the types above, so it
/// truncated a column that needed nothing done to it, silently moving which rows a comparison inside
/// that millisecond selects.
///
/// A **non-temporal** operand has no `EPOCH_MS` overload that would help it: DuckDB reads
/// `EPOCH_MS(<integer>)` as the *constructor* and then fails on dividing a timestamp, naming neither
/// the column nor the comparison it came from. Rendered bare it still does not bind, but the error
/// names the types that could not be compared.
///
/// A column the schema does not carry, a non-`Column` operand, or no schema at all leaves the
/// decision where it was — normalize — because that is what makes the types the rewrite exists for
/// bind. Only a *resolved* type declines it.
fn duckdb_normalizes_timestamp_operand(expr: &Expr, schema: Option<&Schema>) -> bool {
    let (Expr::Column(column), Some(schema)) = (expr, schema) else {
        return true;
    };
    // `column_with_name` rather than `field_with_name`: the latter builds a `Vec` of every field
    // name and formats it into an error string that this discards, and a caller passing a schema
    // without the column is a supported path rather than an accident.
    let Some((_, field)) = schema.column_with_name(&column.name) else {
        return true;
    };

    matches!(
        field.data_type(),
        DataType::Timestamp(_, None) | DataType::Date32 | DataType::Date64
    )
}

fn handle_cast(
    cast: &Cast,
    engine: Option<Engine>,
    schema: Option<&Schema>,
    expr: &Expr,
) -> Result<String> {
    match cast.field.data_type() {
        arrow::datatypes::DataType::Timestamp(_, Some(_) | None) => match engine {
            Some(Engine::ODBC) => Ok(format!(
                "CAST({} AS TIMESTAMP)",
                render_expr(&cast.expr, engine, schema)?,
            )),
            // `EPOCH` yields whole seconds, so a cast renders a coarser term than the timestamp
            // literals below, which keep their microseconds. Both are `TIMESTAMPTZ`, which is
            // what `is_duckdb_timestamp_operand` relies on; the precisions differ.
            Some(Engine::DuckDB) => Ok(format!(
                "TO_TIMESTAMP(EPOCH(CAST({} AS TIMESTAMP)))",
                render_expr(&cast.expr, engine, schema)?,
            )),
            Some(Engine::SQLite) => Ok(format!(
                "datetime({}, 'subsec', 'utc')",
                render_expr(&cast.expr, engine, schema)?,
            )),
            Some(Engine::Spark) => EngineNotSupportedForExpressionSnafu {
                engine: "Spark".to_string(),
                expr: format!("{expr}"),
            }
            .fail()?,
            _ => Ok(format!(
                "CAST({} AS TIMESTAMPTZ)",
                render_expr(&cast.expr, engine, schema)?,
            )),
        },
        arrow::datatypes::DataType::Int64 => Ok(format!(
            "CAST({} AS BIGINT)",
            render_expr(&cast.expr, engine, schema)?,
        )),
        _ => Err(Error::UnsupportedFilterExpr {
            expr: format!("{expr}"),
        }),
    }
}

// Helper function to check if expression contains subquery or outer reference
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};

pub(super) fn expr_contains_subquery_or_outer_ref(expr: &Expr) -> datafusion::error::Result<bool> {
    let mut found = false;
    expr.apply(|expr| match expr {
        Expr::ScalarSubquery(_)
        | Expr::InSubquery(_)
        | Expr::Exists(_)
        | Expr::OuterReferenceColumn(_, _) => {
            found = true;
            Ok(TreeNodeRecursion::Stop)
        }
        _ => Ok(TreeNodeRecursion::Continue),
    })?;
    Ok(found)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use arrow::datatypes::{DataType, TimeUnit};
    use datafusion::{
        logical_expr::{
            expr::{InList, ScalarFunction},
            ColumnarValue, Expr, Like, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
            Volatility,
        },
        prelude::col,
        scalar::ScalarValue,
    };

    #[test]
    fn test_like_expr_to_sql() -> Result<()> {
        for (engine, case_insensitive, negated, expected) in [
            (Engine::Postgres, false, false, "\"name\" LIKE '%John%'"),
            (Engine::Postgres, true, true, "\"name\" NOT ILIKE '%John%'"),
            (Engine::DuckDB, false, false, "\"name\" LIKE '%John%'"),
            (Engine::DuckDB, true, true, "\"name\" NOT ILIKE '%John%'"),
            (
                Engine::SQLite,
                false,
                false,
                "\"name\" LIKE '%John%' COLLATE BINARY",
            ),
            (Engine::SQLite, true, true, "\"name\" NOT LIKE '%John%'"),
        ] {
            let expr = Expr::Like(Like {
                expr: Box::new(col("name")),
                pattern: Box::new(Expr::Literal(
                    ScalarValue::Utf8(Some("%John%".to_string())),
                    None,
                )),
                case_insensitive,
                negated,
                escape_char: None,
            });

            let sql = to_sql_with_engine(&expr, Some(engine))?;
            assert_eq!(sql, expected);
        }
        Ok(())
    }

    #[test]
    fn test_utf8view_literal_to_sql() -> Result<()> {
        let expr = Expr::Literal(
            ScalarValue::Utf8View(Some("ECONOMY ANODIZED STEEL".to_string())),
            None,
        );
        assert_eq!(to_sql_with_engine(&expr, None)?, "'ECONOMY ANODIZED STEEL'");

        // Utf8View equality filter (common after DataFusion type coercion)
        let eq_expr = Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
            left: Box::new(col("p_type")),
            op: datafusion::logical_expr::Operator::Eq,
            right: Box::new(Expr::Literal(
                ScalarValue::Utf8View(Some("ECONOMY ANODIZED STEEL".to_string())),
                None,
            )),
        });
        assert_eq!(to_sql(&eq_expr)?, "\"p_type\" = 'ECONOMY ANODIZED STEEL'");

        Ok(())
    }

    #[test]
    fn test_utf8view_inlist_to_sql() -> Result<()> {
        let expr = Expr::InList(InList {
            expr: Box::new(col("l_shipmode")),
            list: vec![
                Expr::Literal(ScalarValue::Utf8View(Some("MAIL".to_string())), None),
                Expr::Literal(ScalarValue::Utf8View(Some("SHIP".to_string())), None),
            ],
            negated: false,
        });
        assert_eq!(
            to_sql_with_engine(&expr, None)?,
            "\"l_shipmode\" IN ('MAIL', 'SHIP')"
        );

        Ok(())
    }

    #[test]
    fn test_like_expr_no_engine_to_sql() -> Result<()> {
        // LIKE with no engine should produce standard SQL LIKE
        let expr = Expr::Like(Like {
            expr: Box::new(col("name")),
            pattern: Box::new(Expr::Literal(
                ScalarValue::Utf8(Some("%John%".to_string())),
                None,
            )),
            case_insensitive: false,
            negated: false,
            escape_char: None,
        });
        assert_eq!(to_sql(&expr)?, "\"name\" LIKE '%John%'");

        // NOT LIKE with no engine
        let expr_negated = Expr::Like(Like {
            expr: Box::new(col("name")),
            pattern: Box::new(Expr::Literal(
                ScalarValue::Utf8(Some("MEDIUM POLISHED%".to_string())),
                None,
            )),
            case_insensitive: false,
            negated: true,
            escape_char: None,
        });
        assert_eq!(
            to_sql(&expr_negated)?,
            "\"name\" NOT LIKE 'MEDIUM POLISHED%'"
        );

        // Case-insensitive LIKE with no engine uses UPPER()
        let expr_ci = Expr::Like(Like {
            expr: Box::new(col("name")),
            pattern: Box::new(Expr::Literal(
                ScalarValue::Utf8(Some("%john%".to_string())),
                None,
            )),
            case_insensitive: true,
            negated: false,
            escape_char: None,
        });
        assert_eq!(to_sql(&expr_ci)?, "UPPER(\"name\") LIKE UPPER('%john%')");

        Ok(())
    }

    #[test]
    fn test_like_expr_utf8view_no_engine_to_sql() -> Result<()> {
        // LIKE with Utf8View pattern and no engine
        let expr = Expr::Like(Like {
            expr: Box::new(col("p_name")),
            pattern: Box::new(Expr::Literal(
                ScalarValue::Utf8View(Some("forest%".to_string())),
                None,
            )),
            case_insensitive: false,
            negated: false,
            escape_char: None,
        });
        assert_eq!(to_sql(&expr)?, "\"p_name\" LIKE 'forest%'");

        Ok(())
    }

    #[test]
    fn test_decimal128_literal_to_sql() -> Result<()> {
        let expr = Expr::Literal(ScalarValue::Decimal128(Some(1234567890), 38, 2), None);
        assert_eq!(to_sql_with_engine(&expr, None)?, "12345678.90");

        let expr_negative = Expr::Literal(ScalarValue::Decimal128(Some(-1234567890), 38, 4), None);
        assert_eq!(to_sql_with_engine(&expr_negative, None)?, "-123456.7890");

        let expr_int = Expr::Literal(ScalarValue::Decimal128(Some(1234567890), 38, 0), None);
        assert_eq!(to_sql_with_engine(&expr_int, None)?, "1234567890");

        Ok(())
    }

    #[test]
    fn test_expr_inlist_to_sql() -> Result<()> {
        let expr = Expr::InList(InList {
            expr: Box::new(col("a")),
            list: vec![
                Expr::Literal(ScalarValue::Int32(Some(1)), None),
                Expr::Literal(ScalarValue::Int32(Some(2)), None),
                Expr::Literal(ScalarValue::Int32(Some(3)), None),
            ],
            negated: false,
        });
        assert_eq!(to_sql_with_engine(&expr, None)?, "\"a\" IN (1, 2, 3)");

        let expr_negated = Expr::InList(InList {
            expr: Box::new(col("a")),
            list: vec![
                Expr::Literal(ScalarValue::Int32(Some(4)), None),
                Expr::Literal(ScalarValue::Int32(Some(5)), None),
                Expr::Literal(ScalarValue::Int32(Some(6)), None),
            ],
            negated: true,
        });

        assert_eq!(
            to_sql_with_engine(&expr_negated, None)?,
            "\"a\" NOT IN (4, 5, 6)"
        );

        Ok(())
    }

    #[test]
    fn test_expr_scalar_function_to_sql() -> Result<()> {
        #[derive(Debug, Hash, Eq, PartialEq)]
        struct TestScalarUDF {
            signature: Signature,
        }
        impl ScalarUDFImpl for TestScalarUDF {
            fn name(&self) -> &str {
                "substring"
            }

            fn signature(&self) -> &Signature {
                &self.signature
            }

            fn return_type(&self, _arg_types: &[DataType]) -> datafusion::error::Result<DataType> {
                Ok(DataType::Utf8)
            }

            fn invoke_with_args(
                &self,
                _args: ScalarFunctionArgs,
            ) -> datafusion::error::Result<ColumnarValue> {
                Ok(ColumnarValue::Scalar(ScalarValue::from("a")))
            }
        }
        let substring_udf = Arc::new(ScalarUDF::from(TestScalarUDF {
            signature: Signature::uniform(
                3,
                vec![DataType::Utf8, DataType::Int32, DataType::Int32],
                Volatility::Stable,
            ),
        }));

        let expr = Expr::ScalarFunction(ScalarFunction {
            func: substring_udf,
            args: vec![
                Expr::Literal(ScalarValue::Utf8(Some("hello world".to_string())), None),
                Expr::Literal(ScalarValue::Int32(Some(1)), None),
                Expr::Literal(ScalarValue::Int32(Some(5)), None),
            ],
        });
        assert_eq!(
            to_sql_with_engine(&expr, None)?,
            "substring('hello world', 1, 5)"
        );

        Ok(())
    }

    fn binary(left: Expr, op: Operator, right: Expr) -> Expr {
        Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
            left: Box::new(left),
            op,
            right: Box::new(right),
        })
    }

    fn int(v: i32) -> Expr {
        Expr::Literal(ScalarValue::Int32(Some(v)), None)
    }

    #[test]
    fn test_and_or_binary_exprs_are_parenthesized() -> Result<()> {
        // AND operands are wrapped so the tree shape is preserved in SQL
        let and_expr = binary(col("a"), Operator::And, col("b"));
        assert_eq!(to_sql(&and_expr)?, "(\"a\") AND (\"b\")");

        // OR operands are wrapped for the same reason
        let or_expr = binary(col("a"), Operator::Or, col("b"));
        assert_eq!(to_sql(&or_expr)?, "(\"a\") OR (\"b\")");

        Ok(())
    }

    #[test]
    fn test_non_boolean_binary_exprs_not_parenthesized() -> Result<()> {
        // Comparison and arithmetic operators must NOT add extra parens
        let eq_expr = binary(col("k1"), Operator::Eq, int(1));
        assert_eq!(to_sql(&eq_expr)?, "\"k1\" = 1");

        let lt_expr = binary(col("v"), Operator::Lt, int(42));
        assert_eq!(to_sql(&lt_expr)?, "\"v\" < 42");

        let plus_expr = binary(col("x"), Operator::Plus, int(5));
        assert_eq!(to_sql(&plus_expr)?, "\"x\" + 5");

        Ok(())
    }

    #[test]
    fn test_composite_key_or_chain_parenthesized() -> Result<()> {
        // Simulates a two-row composite-key DELETE:
        //   (k1 = 1 AND k2 = 2) OR (k1 = 2 AND k2 = 4)
        //
        // Without parens this serialises as a flat chain that DuckDB's optimizer
        // reconstructs as a left-recursive tree of depth N, causing a stack overflow
        // for large N. With parens the structure is explicit.
        let row1 = binary(
            binary(col("k1"), Operator::Eq, int(1)),
            Operator::And,
            binary(col("k2"), Operator::Eq, int(2)),
        );
        let row2 = binary(
            binary(col("k1"), Operator::Eq, int(2)),
            Operator::And,
            binary(col("k2"), Operator::Eq, int(4)),
        );
        let or_chain = binary(row1, Operator::Or, row2);

        assert_eq!(
            to_sql(&or_chain)?,
            "((\"k1\" = 1) AND (\"k2\" = 2)) OR ((\"k1\" = 2) AND (\"k2\" = 4))"
        );

        Ok(())
    }

    #[test]
    fn test_expr_timestamp_scalar_value_to_sql() -> Result<()> {
        let expr = Expr::Literal(
            ScalarValue::TimestampNanosecond(Some(1_693_219_803_001_000_000), None),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803.001) AT TIME ZONE 'UTC'"
        );
        let expr = Expr::Literal(
            ScalarValue::TimestampNanosecond(
                Some(1_693_219_803_001_000_000),
                Some(Arc::from("+10:00")),
            ),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803.001)"
        );

        let expr = Expr::Literal(
            ScalarValue::TimestampMicrosecond(Some(1_693_219_803_001_000), None),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803.001) AT TIME ZONE 'UTC'"
        );
        let expr = Expr::Literal(
            ScalarValue::TimestampMicrosecond(
                Some(1_693_219_803_001_000),
                Some(Arc::from("+10:00")),
            ),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803.001)"
        );

        let expr = Expr::Literal(
            ScalarValue::TimestampMillisecond(Some(1_693_219_803_001), None),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803.001) AT TIME ZONE 'UTC'"
        );
        let expr = Expr::Literal(
            ScalarValue::TimestampMillisecond(Some(1_693_219_803_001), Some(Arc::from("+10:00"))),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803.001)"
        );

        let expr = Expr::Literal(
            ScalarValue::TimestampSecond(Some(1_693_219_803), None),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803) AT TIME ZONE 'UTC'"
        );
        let expr = Expr::Literal(
            ScalarValue::TimestampSecond(Some(1_693_219_803), Some(Arc::from("+10:00"))),
            None,
        );
        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::Postgres))?,
            "TO_TIMESTAMP(1693219803)"
        );

        Ok(())
    }

    fn bool_lit(v: bool) -> Expr {
        Expr::Literal(ScalarValue::Boolean(Some(v)), None)
    }

    fn int_lit(v: i64) -> Expr {
        Expr::Literal(ScalarValue::Int64(Some(v)), None)
    }

    fn ts_lit() -> Expr {
        Expr::Literal(
            ScalarValue::TimestampNanosecond(Some(1_767_225_600_000_000_000), None),
            None,
        )
    }

    /// A logical connective is not a timestamp comparison. The DuckDB timestamp
    /// normalization must not fire on `AND`/`OR`: the left operand of a connective is a
    /// predicate, and wrapping a predicate in `EPOCH_MS` makes DuckDB's binder reject the
    /// whole statement with `Binder Error: epoch_ms(BOOLEAN)`.
    #[test]
    fn test_duckdb_or_with_timestamp_right_operand_is_not_rewritten() {
        let expr = col("flag").eq(bool_lit(true)).or(col("ts").lt(ts_lit()));

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert!(
            !sql.contains("EPOCH_MS(\"flag\""),
            "the boolean operand of OR must not be wrapped in EPOCH_MS: {sql}"
        );
        assert_eq!(
            sql,
            "(\"flag\" = true) OR (TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < make_timestamptz(1767225600000000))"
        );
    }

    #[test]
    fn test_duckdb_and_with_timestamp_right_operand_is_not_rewritten() {
        let expr = col("flag").eq(bool_lit(true)).and(col("ts").lt(ts_lit()));

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert!(
            !sql.contains("EPOCH_MS(\"flag\""),
            "the boolean operand of AND must not be wrapped in EPOCH_MS: {sql}"
        );
        assert_eq!(
            sql,
            "(\"flag\" = true) AND (TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < make_timestamptz(1767225600000000))"
        );
    }

    /// The left operand of the outer connective is itself a connective, so it renders
    /// parenthesised rather than as a bare predicate — it must still not be wrapped.
    #[test]
    fn test_duckdb_nested_connective_operand_is_not_rewritten() {
        let inner = col("x").eq(int_lit(1)).and(col("y").eq(int_lit(2)));
        let expr = inner.or(col("ts").lt(ts_lit()));

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert!(
            !sql.contains("EPOCH_MS((") && !sql.contains("EPOCH_MS(\"x\""),
            "a predicate operand must not be wrapped in EPOCH_MS: {sql}"
        );
    }

    /// The normalization must be preserved for a genuine timestamp comparison — this is
    /// the case the rewrite exists to serve.
    #[test]
    fn test_duckdb_timestamp_comparison_is_still_rewritten() {
        let expr = col("ts").lt(ts_lit());

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(
            sql,
            "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < make_timestamptz(1767225600000000)"
        );
    }

    fn utf8(value: &str) -> Expr {
        Expr::Literal(ScalarValue::Utf8(Some(value.to_string())), None)
    }

    /// An apostrophe is ordinary in real data (names, addresses, free text). Rendered bare it
    /// closes the literal early and the statement fails to parse, so a `DELETE`/`UPDATE`
    /// filtering on such a value cannot run at all.
    #[test]
    fn test_apostrophe_in_string_literal_is_doubled() -> Result<()> {
        assert_eq!(to_sql(&utf8("O'Brien"))?, "'O''Brien'");

        // Every engine ends a literal on `'` and accepts the doubled form for an embedded one.
        for engine in [
            Engine::DuckDB,
            Engine::SQLite,
            Engine::Postgres,
            Engine::ODBC,
            Engine::MySQL,
            Engine::Spark,
        ] {
            assert_eq!(
                to_sql_with_engine(&utf8("O'Brien"), Some(engine))?,
                "'O''Brien'",
                "engine {engine:?} must escape the quote"
            );
        }

        Ok(())
    }

    /// A backslash is only significant inside a literal for the engines that read it as an escape
    /// character; the others must not gain an extra one, since doubling a literal backslash
    /// silently changes the value.
    #[test]
    fn test_backslash_is_escaped_only_where_it_is_significant() -> Result<()> {
        for engine in [Engine::MySQL, Engine::Spark] {
            assert_eq!(
                to_sql_with_engine(&utf8(r"C:\temp\"), Some(engine))?,
                r"'C:\\temp\\'",
                "engine {engine:?} must double the backslash"
            );
        }

        for engine in [
            None,
            Some(Engine::DuckDB),
            Some(Engine::SQLite),
            Some(Engine::Postgres),
            Some(Engine::ODBC),
        ] {
            assert_eq!(
                to_sql_with_engine(&utf8(r"C:\temp\"), engine)?,
                r"'C:\temp\'",
                "engine {engine:?} must leave the backslash alone"
            );
        }

        Ok(())
    }

    /// A backslash immediately before a quote is the composition quote doubling alone cannot
    /// handle: on a backslash-escaping engine `'\''` would be `\'` followed by a closing quote,
    /// which is worse than not escaping. Both directions are pinned here.
    #[test]
    fn test_backslash_before_a_quote_is_escaped_per_engine() -> Result<()> {
        // `\` ordinary: `''` is the escaped quote and the whole value stays inside the literal.
        assert_eq!(to_sql(&utf8(r"\' OR 1=1 --"))?, r"'\'' OR 1=1 --'");

        // `\` significant: it is doubled first, so it can no longer consume the escaped quote.
        assert_eq!(
            to_sql_with_engine(&utf8(r"\' OR 1=1 --"), Some(Engine::MySQL))?,
            r"'\\'' OR 1=1 --'"
        );

        Ok(())
    }

    /// `LargeUtf8`/`Utf8View` share the arm — DataFusion's type coercion picks between them, so
    /// escaping only one variant would leave the defect reachable.
    #[test]
    fn test_every_string_scalar_variant_is_escaped() -> Result<()> {
        for value in [
            ScalarValue::Utf8(Some("it's".to_string())),
            ScalarValue::LargeUtf8(Some("it's".to_string())),
            ScalarValue::Utf8View(Some("it's".to_string())),
        ] {
            assert_eq!(to_sql(&Expr::Literal(value, None))?, "'it''s'");
        }

        Ok(())
    }

    /// The unbounded-deletion shape: rendered bare, `WHERE "name" = 'x' OR 1=1 --'` binds as a
    /// tautology and a `DELETE` matches every row instead of none.
    #[test]
    fn test_quote_bearing_value_stays_a_single_literal() -> Result<()> {
        let expr = col("name").eq(utf8("x' OR 1=1 --"));

        assert_eq!(to_sql(&expr)?, "\"name\" = 'x'' OR 1=1 --'");

        Ok(())
    }

    /// Every construct that carries a string value routes through the same literal arm.
    #[test]
    fn test_quotes_are_escaped_in_like_and_in_list() -> Result<()> {
        let like = Expr::Like(Like {
            expr: Box::new(col("name")),
            pattern: Box::new(utf8("%O'B%")),
            case_insensitive: false,
            negated: false,
            escape_char: None,
        });
        assert_eq!(to_sql(&like)?, "\"name\" LIKE '%O''B%'");

        let in_list = Expr::InList(InList {
            expr: Box::new(col("name")),
            list: vec![utf8("O'Brien"), utf8("plain")],
            negated: false,
        });
        assert_eq!(to_sql(&in_list)?, "\"name\" IN ('O''Brien', 'plain')");

        Ok(())
    }

    #[test]
    fn test_string_literal_edge_cases() -> Result<()> {
        assert_eq!(to_sql(&utf8(""))?, "''");
        assert_eq!(to_sql(&utf8("'"))?, "''''");
        assert_eq!(to_sql(&utf8("''"))?, "''''''");
        assert_eq!(to_sql(&utf8("a'b'c"))?, "'a''b''c'");

        // A backslash is an ordinary character for the engines reached through this function,
        // so it must pass through unchanged rather than be doubled.
        assert_eq!(to_sql(&utf8(r"a\b"))?, r"'a\b'");

        Ok(())
    }

    /// An Arrow field name may contain a double quote, which would otherwise close the
    /// identifier early and leave the rest of the name as SQL text.
    #[test]
    fn test_quote_in_column_name_is_doubled() -> Result<()> {
        let column = Expr::Column(datafusion::common::Column::new_unqualified("we\"ird"));
        let expr = column.eq(utf8("v"));

        assert_eq!(to_sql(&expr)?, "\"we\"\"ird\" = 'v'");

        Ok(())
    }

    /// A value with no quote in it must render exactly as before — the escape is a no-op for
    /// the overwhelmingly common case.
    #[test]
    fn test_ordinary_values_are_unchanged() -> Result<()> {
        assert_eq!(to_sql(&col("a").eq(utf8("plain")))?, "\"a\" = 'plain'");
        assert_eq!(to_sql(&col("a").eq(int(1)))?, "\"a\" = 1");

        Ok(())
    }

    /// A qualified column must render as the column it names. Rendering `Column`'s flat name made
    /// `t.a` one identifier, which either fails to bind or — where a column of that literal name
    /// exists — binds the wrong one.
    #[test]
    fn a_qualified_column_renders_as_its_own_column() -> Result<()> {
        let qualified = Expr::Column(datafusion::common::Column::new(Some("t"), "a"));

        for (engine, expected) in [
            (None, "\"a\""),
            (Some(Engine::DuckDB), "\"a\""),
            (Some(Engine::SQLite), "\"a\""),
            (Some(Engine::Postgres), "\"a\""),
            (Some(Engine::MySQL), "\"a\""),
            (Some(Engine::ODBC), "a"),
            (Some(Engine::Spark), "a"),
        ] {
            assert_eq!(
                to_sql_with_engine(&qualified, engine)?,
                expected,
                "engine {engine:?}"
            );
        }

        Ok(())
    }

    /// A predicate spanning two relations cannot be rendered for a single-table statement, and
    /// rendering it by column name alone would *widen* it into `"id" = "id"` — a tautology that a
    /// `DELETE` applies to every row. It has to be refused, not erased.
    #[test]
    fn a_cross_relation_predicate_is_refused() {
        let left = Expr::Column(datafusion::common::Column::new(Some("t1"), "id"));
        let right = Expr::Column(datafusion::common::Column::new(Some("t2"), "id"));

        let err = to_sql(&left.eq(right)).expect_err("a cross-relation predicate must be refused");
        assert!(
            matches!(err, Error::UnsupportedFilterExpr { .. }),
            "unexpected error: {err}"
        );
    }

    /// One relation used by two columns is not a cross-relation predicate — this is the shape a
    /// planner-produced filter over two columns of the same table has.
    #[test]
    fn one_relation_used_twice_renders() -> Result<()> {
        let left = Expr::Column(datafusion::common::Column::new(Some("t"), "a"));
        let right = Expr::Column(datafusion::common::Column::new(Some("t"), "b"));

        assert_eq!(to_sql(&left.eq(right))?, "\"a\" = \"b\"");

        Ok(())
    }

    /// Mixed qualification of one table cannot be told apart from two tables without the session's
    /// default catalog and schema, so it is refused rather than resolved.
    #[test]
    fn mixed_qualification_of_one_relation_is_refused() {
        let bare = Expr::Column(datafusion::common::Column::new(Some("t"), "a"));
        let partial = Expr::Column(datafusion::common::Column::new(
            Some(datafusion::common::TableReference::partial("public", "t")),
            "b",
        ));

        assert!(matches!(
            to_sql(&bare.eq(partial)),
            Err(Error::UnsupportedFilterExpr { .. })
        ));
    }

    /// `TableReference::resolved_eq` treats an omitted component as matching, so a bare reference is
    /// `resolved_eq` to two different fully qualified tables. Comparing against one arbitrarily
    /// chosen pivot under those semantics would admit this predicate — over columns of two distinct
    /// catalogs — and then erase both qualifiers into `"id" = "id"`.
    #[test]
    fn two_catalogs_bridged_by_a_bare_reference_are_refused() {
        let left = Expr::Column(datafusion::common::Column::new(
            Some(datafusion::common::TableReference::full("a", "public", "t")),
            "id",
        ));
        let bridge = Expr::Column(datafusion::common::Column::new(Some("t"), "id"));
        let right = Expr::Column(datafusion::common::Column::new(
            Some(datafusion::common::TableReference::full("b", "public", "t")),
            "id",
        ));

        // Every ordering of the three references must refuse, whatever order `column_refs` yields.
        for predicate in [
            left.clone()
                .eq(bridge.clone())
                .and(right.clone().eq(bridge.clone())),
            bridge
                .clone()
                .eq(left.clone())
                .and(bridge.clone().eq(right.clone())),
            right.eq(bridge.clone()).and(bridge.eq(left)),
        ] {
            assert!(
                matches!(to_sql(&predicate), Err(Error::UnsupportedFilterExpr { .. })),
                "accepted a predicate spanning two catalogs: {predicate}"
            );
        }
    }

    /// The qualifier is what gets dropped, not every dot: a column whose own name contains a dot is
    /// a legal Arrow field name and must still render as itself. This is also the column a
    /// qualified `t.a` would have bound to had it kept rendering flat.
    #[test]
    fn an_unqualified_dotted_column_name_is_not_a_qualifier() -> Result<()> {
        let dotted = Expr::Column(datafusion::common::Column::new_unqualified("t.a"));

        assert_eq!(to_sql(&dotted.clone().eq(int(1)))?, "\"t.a\" = 1");
        assert_eq!(
            to_sql_with_engine(&dotted, Some(Engine::DuckDB))?,
            "\"t.a\""
        );

        Ok(())
    }

    /// The qualifier has to be dropped wherever the column sits, not only at the top of the tree.
    #[test]
    fn a_qualified_column_inside_a_predicate_renders_as_its_own_column() -> Result<()> {
        let predicate = Expr::Column(datafusion::common::Column::new(Some("t"), "a"))
            .eq(int(1))
            .and(col("b").eq(utf8("x")));

        assert_eq!(to_sql(&predicate)?, "(\"a\" = 1) AND (\"b\" = 'x')");

        Ok(())
    }

    /// An unqualified column keeps rendering exactly as before, for both the quoted and the bare
    /// engine arms.
    #[test]
    fn an_unqualified_column_renders_unchanged() -> Result<()> {
        assert_eq!(to_sql(&col("a"))?, "\"a\"");
        assert_eq!(
            to_sql_with_engine(&col("a"), Some(Engine::DuckDB))?,
            "\"a\""
        );
        assert_eq!(to_sql_with_engine(&col("a"), Some(Engine::Spark))?, "a");
        assert_eq!(to_sql_with_engine(&col("a"), Some(Engine::ODBC))?, "a");

        Ok(())
    }

    /// `supports_filters_pushdown` is asked about the **qualified** predicate while `scan` receives
    /// the unqualified one, and `ClickHouseTable` answers that question by rendering here and
    /// mapping `Err(_)` to `Unsupported` (as does the runtime's DuckDB vector index). So this
    /// function must render the shape the probe passes it: refusing a qualifier would silently stop
    /// those predicates from reaching the remote statement, and the physical-pushdown fallback does
    /// not cover `LIKE`.
    ///
    /// The predicate is captured from a real `SessionContext` rather than hand-built, so the test
    /// keeps holding if `DataFusion` changes when it normalizes.
    #[tokio::test]
    async fn a_predicate_as_the_pushdown_probe_receives_it_still_renders() {
        use datafusion::catalog::{Session, TableProvider};
        use datafusion::logical_expr::{TableProviderFilterPushDown, TableType};
        use datafusion::physical_plan::{empty::EmptyExec, ExecutionPlan};
        use datafusion::prelude::SessionContext;
        use std::sync::Mutex;

        #[derive(Debug)]
        struct ProbeRecorder {
            schema: arrow::datatypes::SchemaRef,
            probed: Arc<Mutex<Vec<Expr>>>,
        }

        #[async_trait::async_trait]
        impl TableProvider for ProbeRecorder {
            fn schema(&self) -> arrow::datatypes::SchemaRef {
                Arc::clone(&self.schema)
            }
            fn table_type(&self) -> TableType {
                TableType::Base
            }
            fn supports_filters_pushdown(
                &self,
                filters: &[&Expr],
            ) -> datafusion::error::Result<Vec<TableProviderFilterPushDown>> {
                self.probed
                    .lock()
                    .expect("lock")
                    .extend(filters.iter().map(|f| (*f).clone()));
                Ok(vec![TableProviderFilterPushDown::Exact; filters.len()])
            }
            async fn scan(
                &self,
                _state: &dyn Session,
                _projection: Option<&Vec<usize>>,
                _filters: &[Expr],
                _limit: Option<usize>,
            ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
                Ok(Arc::new(EmptyExec::new(Arc::clone(&self.schema))))
            }
        }

        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("name", DataType::Utf8, true),
        ]));
        let probed = Arc::new(Mutex::new(Vec::new()));
        let ctx = SessionContext::new();
        ctx.register_table(
            "docs",
            Arc::new(ProbeRecorder {
                schema,
                probed: Arc::clone(&probed),
            }),
        )
        .expect("register");

        ctx.sql("SELECT * FROM docs WHERE name LIKE '%x%'")
            .await
            .expect("plan")
            .collect()
            .await
            .expect("collect");

        let filters = probed.lock().expect("lock").clone();
        assert!(!filters.is_empty(), "the pushdown probe saw no filter");
        for filter in &filters {
            assert!(
                filter
                    .column_refs()
                    .iter()
                    .any(|column| column.relation.is_some()),
                "the probe is expected to receive a qualified column: {filter:?}"
            );
            assert_eq!(
                to_sql_with_engine(filter, None).expect("the probed predicate must render"),
                "\"name\" LIKE '%x%'"
            );
        }
    }

    fn cast_to_timestamp(expr: Expr) -> Expr {
        Expr::Cast(Cast::new(
            Box::new(expr),
            DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
        ))
    }

    /// A comparison whose right operand is itself a normalized comparison renders as
    /// `TO_TIMESTAMP(..)` without being a timestamp. Deciding by rendered text wrapped that
    /// boolean in `EPOCH_MS`, which has no `BOOLEAN` overload; deciding by `Expr` shape does
    /// not. Verified against DuckDB v1.5.5: the asserted SQL evaluates, and the text-sniffed
    /// form is rejected at parse time as a chained comparison, before the binder is reached.
    #[test]
    fn test_duckdb_predicate_compared_to_a_timestamp_comparison_is_not_normalized() {
        let expr = col("flag").eq(col("ts").lt(ts_lit()));

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert!(
            !sql.contains("EPOCH_MS(\"flag\""),
            "a boolean operand must not be wrapped in EPOCH_MS: {sql}"
        );
        assert_eq!(
            sql,
            "\"flag\" = (TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < make_timestamptz(1767225600000000))"
        );
    }

    /// Subtraction between a timestamp column and a timestamp literal is normalized: DuckDB
    /// v1.5.5 rejects `"ts" - TO_TIMESTAMP(..)` with
    /// `No function matches '-(TIMESTAMP_MS, TIMESTAMP WITH TIME ZONE)'`, and accepts the
    /// normalized form, returning the interval between the two instants.
    #[test]
    fn test_duckdb_timestamp_subtraction_is_normalized() {
        let expr = col("ts") - ts_lit();

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(
            sql,
            "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) - make_timestamptz(1767225600000000)"
        );
    }

    /// A bare *right* operand is deliberately not normalized. Doing so would apply `EPOCH_MS`'s
    /// whole-millisecond truncation to a microsecond `TIMESTAMPTZ` column that compares exactly
    /// without it, turning a loud binder error into a silently shifted comparison. DuckDB
    /// rejects this shape, which is the safe outcome.
    #[test]
    fn test_duckdb_timestamp_literal_on_the_left_is_not_normalized() {
        let expr = ts_lit().gt(col("ts"));

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(sql, "make_timestamptz(1767225600000000) > \"ts\"");
    }

    /// Two timestamp operands are already comparable and must be left alone. This is the shape
    /// retention emits — a cast to a timestamp against a timestamp literal.
    #[test]
    fn test_duckdb_two_timestamp_operands_are_not_normalized() {
        let expr = cast_to_timestamp(col("ts")).lt(ts_lit());

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(
            sql,
            "TO_TIMESTAMP(EPOCH(CAST(\"ts\" AS TIMESTAMP))) < make_timestamptz(1767225600000000)"
        );
    }

    /// Adding an interval to a timestamp is itself timestamp-valued, so a comparison against it
    /// still normalizes the bare side. `"delta"` is an interval column, whose type is invisible
    /// here — the operand is recognised from the operator and the literal's shape. Verified
    /// against DuckDB v1.5.5: the asserted SQL evaluates, while leaving `"ts"` bare fails with
    /// `Cannot compare values of type TIMESTAMP_MS and type TIMESTAMP WITH TIME ZONE`.
    #[test]
    fn test_duckdb_comparison_against_timestamp_plus_interval_is_normalized() {
        let expr = col("ts").lt(ts_lit() + col("delta"));

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(
            sql,
            "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < (make_timestamptz(1767225600000000) + \"delta\")"
        );
    }

    /// Subtracting an interval from a timestamp is timestamp-valued too, and the operands may be
    /// the other way round for `+`, since DuckDB accepts `INTERVAL + TIMESTAMPTZ` as well.
    #[test]
    fn test_duckdb_timestamp_interval_arithmetic_is_timestamp_valued() {
        for (expr, expected) in [
            (
                col("ts").lt(ts_lit() - col("delta")),
                "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < (make_timestamptz(1767225600000000) - \"delta\")",
            ),
            (
                col("ts").lt(col("delta") + ts_lit()),
                "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < (\"delta\" + make_timestamptz(1767225600000000))",
            ),
        ] {
            let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

            assert_eq!(sql, expected);
        }
    }

    /// The normalized rendering must group its right operand too. `-` and `+` share a precedence
    /// level and associate left, so `x - (t + d)` rendered flat as `x - t + d` means
    /// `(x - t) + d` — a different value, silently.
    #[test]
    fn test_duckdb_normalized_subtraction_groups_its_right_operand() {
        let expr = col("x") - (ts_lit() + col("delta"));

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(
            sql,
            "TO_TIMESTAMP(EPOCH_MS(\"x\") / 1000) - (make_timestamptz(1767225600000000) + \"delta\")"
        );
    }

    /// Subtracting a timestamp *from* a timestamp yields an interval, not a timestamp, so the
    /// bare operand opposite it must not be normalized — DuckDB would then be asked to compare a
    /// timestamp against an interval.
    #[test]
    fn test_duckdb_difference_of_two_timestamps_is_not_timestamp_valued() {
        let expr = col("ts").lt(ts_lit() - ts_lit());

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(
            sql,
            "\"ts\" < (make_timestamptz(1767225600000000) - make_timestamptz(1767225600000000))"
        );
    }

    /// `Minus` is in the normalization's operator set, but neither operand here is a timestamp,
    /// so ordinary arithmetic must not be pulled into the rewrite.
    #[test]
    fn test_duckdb_subtraction_without_a_timestamp_operand_is_not_normalized() {
        let expr = col("a") - col("b");

        let sql = to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse");

        assert_eq!(sql, "\"a\" - \"b\"");
    }

    /// A nested binary operand keeps its parentheses so the SQL means what the `Expr` meant.
    /// Rendered flat, `a * (b + c)` becomes `a * b + c` — a different value that binds and
    /// returns silently (10 rather than 14 for a=2, b=3, c=4).
    #[test]
    fn test_nested_arithmetic_keeps_its_grouping() {
        let expr = col("a") * (col("b") + col("c"));

        for engine in [
            None,
            Some(Engine::DuckDB),
            Some(Engine::Postgres),
            Some(Engine::SQLite),
            Some(Engine::MySQL),
        ] {
            let sql = to_sql_with_engine(&expr, engine).expect("to unparse");

            assert_eq!(sql, "\"a\" * (\"b\" + \"c\")", "engine {engine:?}");
        }
    }

    /// Subtraction is not associative, so grouping is load-bearing for `-` as well:
    /// `a - (b - c)` rendered flat is `(a - b) - c`.
    #[test]
    fn test_nested_subtraction_keeps_its_grouping() {
        let expr = col("a") - (col("b") - col("c"));

        let sql = to_sql(&expr).expect("to unparse");

        assert_eq!(sql, "\"a\" - (\"b\" - \"c\")");
    }

    /// The normalization is DuckDB-specific and must not leak into other engines.
    #[test]
    fn test_non_duckdb_engines_do_not_normalize_timestamps() {
        let expr = col("ts").lt(ts_lit());

        for engine in [None, Some(Engine::Postgres), Some(Engine::MySQL)] {
            let sql = to_sql_with_engine(&expr, engine).expect("to unparse");

            assert!(
                !sql.contains("EPOCH_MS"),
                "engine {engine:?} must not normalize: {sql}"
            );
        }
    }

    /// A schema whose `ts` column has `data_type`, for the type-aware normalization tests.
    pub(super) fn schema_with_ts(data_type: DataType) -> Schema {
        Schema::new(vec![
            arrow::datatypes::Field::new("ts", data_type, true),
            arrow::datatypes::Field::new("other", DataType::Int64, true),
        ])
    }

    /// The defect this predicate exists for: a `TIMESTAMPTZ` column compares against the rendered
    /// literal exactly, and the normalization only truncates it to whole milliseconds — silently
    /// moving which rows a comparison inside that millisecond selects.
    #[test]
    fn test_duckdb_does_not_normalize_a_timezone_aware_column() {
        let expr = col("ts").lt(ts_lit());

        for unit in [
            TimeUnit::Second,
            TimeUnit::Millisecond,
            TimeUnit::Microsecond,
            TimeUnit::Nanosecond,
        ] {
            let schema = schema_with_ts(DataType::Timestamp(unit, Some("UTC".into())));

            let sql = to_sql_with_engine_and_schema(&expr, Some(Engine::DuckDB), Some(&schema))
                .expect("to unparse");

            assert_eq!(
                sql, "\"ts\" < make_timestamptz(1767225600000000)",
                "a TIMESTAMPTZ column ({unit:?}) must be compared as it stands"
            );
        }
    }

    /// The types the normalization exists for keep it. `TIMESTAMP_S`, `TIMESTAMP_MS` and
    /// `TIMESTAMP_NS` cannot be compared with a `TIMESTAMPTZ` at all, and a naive `TIMESTAMP`
    /// would be compared in the session's `TimeZone` instead of in UTC.
    #[test]
    fn test_duckdb_still_normalizes_a_naive_timestamp_column() {
        let expr = col("ts").lt(ts_lit());

        for unit in [
            TimeUnit::Second,
            TimeUnit::Millisecond,
            TimeUnit::Microsecond,
            TimeUnit::Nanosecond,
        ] {
            let schema = schema_with_ts(DataType::Timestamp(unit, None));

            let sql = to_sql_with_engine_and_schema(&expr, Some(Engine::DuckDB), Some(&schema))
                .expect("to unparse");

            assert_eq!(
                sql, "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < make_timestamptz(1767225600000000)",
                "a naive timestamp column ({unit:?}) must still be normalized"
            );
        }
    }

    /// A `Date32`/`Date64` column keeps the normalization too, and for the reference frame rather
    /// than to bind: DuckDB promotes a bare `DATE` to a `TIMESTAMPTZ` at midnight in the *session's*
    /// `TimeZone`, while the rendered literal is a UTC instant. Declining it here would make a date
    /// comparison depend on the host's timezone.
    #[test]
    fn test_duckdb_still_normalizes_a_date_column() {
        let expr = col("ts").lt(ts_lit());

        for date in [DataType::Date32, DataType::Date64] {
            let schema = schema_with_ts(date.clone());

            let sql = to_sql_with_engine_and_schema(&expr, Some(Engine::DuckDB), Some(&schema))
                .expect("to unparse");

            assert_eq!(
                sql, "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < make_timestamptz(1767225600000000)",
                "a {date:?} column must still be normalized"
            );
        }
    }

    /// `EPOCH_MS(<integer>)` is DuckDB's *constructor*, so normalizing a non-temporal operand
    /// produced `No function matches '/(TIMESTAMP, INTEGER_LITERAL)'` — an error naming neither
    /// the column nor the comparison. Rendered bare it still does not bind, which is the right
    /// outcome for a comparison that has no meaning, but DuckDB v1.5.5 now names the mismatch that
    /// is actually there: *"Cannot compare values of type BIGINT and type TIMESTAMP WITH TIME
    /// ZONE"*.
    #[test]
    fn test_duckdb_does_not_normalize_a_non_temporal_column() {
        let expr = col("other").lt(ts_lit());
        let schema = schema_with_ts(DataType::Timestamp(
            TimeUnit::Microsecond,
            Some("UTC".into()),
        ));

        let sql = to_sql_with_engine_and_schema(&expr, Some(Engine::DuckDB), Some(&schema))
            .expect("to unparse");

        assert_eq!(sql, "\"other\" < make_timestamptz(1767225600000000)");
    }

    /// Only a resolved type declines the normalization. With nothing to resolve against, the
    /// rendering is what it was before the schema could be passed at all, so the types the
    /// rewrite exists for keep binding.
    #[test]
    fn test_duckdb_normalizes_when_the_column_type_is_not_resolvable() {
        let expr = col("ts").lt(ts_lit());
        let unrelated = Schema::new(vec![arrow::datatypes::Field::new(
            "elsewhere",
            DataType::Int64,
            true,
        )]);
        let normalized =
            "TO_TIMESTAMP(EPOCH_MS(\"ts\") / 1000) < make_timestamptz(1767225600000000)";

        assert_eq!(
            to_sql_with_engine(&expr, Some(Engine::DuckDB)).expect("to unparse"),
            normalized,
            "no schema"
        );
        assert_eq!(
            to_sql_with_engine_and_schema(&expr, Some(Engine::DuckDB), None).expect("to unparse"),
            normalized,
            "an explicit absence of schema"
        );
        assert_eq!(
            to_sql_with_engine_and_schema(&expr, Some(Engine::DuckDB), Some(&unrelated))
                .expect("to unparse"),
            normalized,
            "a schema that does not carry the column"
        );
    }

    /// The literal is the other half of the comparison the test above leaves exact. Rendered by
    /// integer division it lost every digit below the second, so a comparison against a
    /// sub-second instant selected by a different instant than the caller named.
    #[test]
    fn test_duckdb_renders_a_sub_second_timestamp_literal_at_full_precision() {
        let micros = 1_767_225_600_000_999i64;

        for (value, expected) in [
            (
                ScalarValue::TimestampMicrosecond(Some(micros), None),
                "make_timestamptz(1767225600000999)",
            ),
            (
                ScalarValue::TimestampMillisecond(Some(1_767_225_600_001), None),
                "make_timestamptz(1767225600001000)",
            ),
            // A `TIMESTAMPTZ` holds microseconds, so the trailing nanoseconds go — but only
            // those, rather than the whole fractional second.
            (
                ScalarValue::TimestampNanosecond(Some(micros * 1_000 + 999), None),
                "make_timestamptz(1767225600000999)",
            ),
        ] {
            let sql = render_duckdb(&value).expect("to unparse");

            assert_eq!(sql, expected, "{value:?}");
        }
    }

    /// One instant has one rendering however the caller spelled it.
    #[test]
    fn test_duckdb_renders_every_unit_of_one_instant_identically() {
        for value in [
            ScalarValue::TimestampSecond(Some(1_767_225_600), None),
            ScalarValue::TimestampMillisecond(Some(1_767_225_600_000), None),
            ScalarValue::TimestampMicrosecond(Some(1_767_225_600_000_000), None),
            ScalarValue::TimestampNanosecond(Some(1_767_225_600_000_000_000), None),
        ] {
            let sql = render_duckdb(&value).expect("to unparse");

            assert_eq!(sql, "make_timestamptz(1767225600000000)", "{value:?}");
        }
    }

    fn render_duckdb(value: &ScalarValue) -> Result<String> {
        to_sql_with_engine(&Expr::Literal(value.clone(), None), Some(Engine::DuckDB))
    }

    /// `2^53` microseconds is about 2255-06-05, and it is the last microsecond an `f64` can hold
    /// alongside its neighbours. One past it is the first value the old rendering could not name:
    /// `9007199254740993` widened to `9007199254.740992`, a literal that is a whole microsecond
    /// short of the instant the caller asked for.
    #[test]
    fn test_duckdb_renders_a_microsecond_past_the_double_bound_exactly() {
        // 2^53 + 1 microseconds.
        let micros = 9_007_199_254_740_993i64;

        for (value, expected) in [
            (
                ScalarValue::TimestampMicrosecond(Some(micros), None),
                "make_timestamptz(9007199254740993)",
            ),
            // The nanosecond arm reduces to microseconds before rendering, so it inherits the
            // same bound and the same repair.
            (
                ScalarValue::TimestampNanosecond(Some(micros * 1_000 + 999), None),
                "make_timestamptz(9007199254740993)",
            ),
        ] {
            let sql = render_duckdb(&value).expect("to unparse");

            assert_eq!(sql, expected, "{value:?}");
        }
    }

    /// The bound is a magnitude, not a date, so a pre-epoch instant the same distance out is
    /// exposed to it too — and `div_euclid` floors, so the nanosecond arm truncates towards the
    /// same side of the epoch there as it does above it.
    #[test]
    fn test_duckdb_renders_a_pre_epoch_microsecond_past_the_double_bound_exactly() {
        let micros = -9_007_199_254_740_993i64;

        for (value, expected) in [
            (
                ScalarValue::TimestampMicrosecond(Some(micros), None),
                "make_timestamptz(-9007199254740993)",
            ),
            (
                ScalarValue::TimestampNanosecond(Some(micros * 1_000 - 999), None),
                "make_timestamptz(-9007199254740994)",
            ),
        ] {
            let sql = render_duckdb(&value).expect("to unparse");

            assert_eq!(sql, expected, "{value:?}");
        }
    }

    /// A second or millisecond count can name an instant no microsecond count can hold. Rendering
    /// is declined rather than wrapped.
    #[test]
    fn test_duckdb_declines_a_timestamp_literal_microseconds_cannot_hold() {
        for value in [
            ScalarValue::TimestampSecond(Some(i64::MAX), None),
            ScalarValue::TimestampSecond(Some(i64::MIN), None),
            ScalarValue::TimestampMillisecond(Some(i64::MAX), None),
            ScalarValue::TimestampMillisecond(Some(i64::MIN), None),
        ] {
            let rendered = render_duckdb(&value);

            assert!(
                matches!(rendered, Err(Error::UnsupportedFilterExpr { .. })),
                "{value:?} rendered {rendered:?} instead of declining"
            );
        }
    }

    /// Only the microsecond arm can reach DuckDB's two reserved counts: no product of a coarser
    /// count and its tick size is either value, and no nanosecond count divides down to one.
    #[test]
    fn test_duckdb_declines_the_microsecond_counts_reserved_as_infinity() {
        for micros in [i64::MAX, -i64::MAX] {
            let rendered = render_duckdb(&ScalarValue::TimestampMicrosecond(Some(micros), None));

            assert!(
                matches!(rendered, Err(Error::UnsupportedFilterExpr { .. })),
                "{micros} rendered {rendered:?} instead of declining"
            );
        }
    }

    /// The counts either side of the sentinels are ordinary instants DuckDB holds, and `i64::MIN`
    /// is one of them — v1.5.5 reserves only the two above, and `i64::MIN` is one below
    /// `-i64::MAX`, not equal to it. Declining more than those two would refuse a filter the
    /// engine can serve.
    #[test]
    fn test_duckdb_still_renders_the_counts_the_sentinels_sit_beside() {
        for micros in [i64::MAX - 1, i64::MIN, i64::MIN + 2] {
            let sql = render_duckdb(&ScalarValue::TimestampMicrosecond(Some(micros), None))
                .expect("to unparse");

            assert_eq!(sql, format!("make_timestamptz({micros})"));
        }
    }

    /// `2038-01-19` is the last day whose epoch second an `i32` holds (`i32::MAX / 86_400` =
    /// 24 855). One day further is where computing the product in `i32` stops naming a date at
    /// all, so it is the day every `Date32` assertion below is anchored on.
    const FIRST_DAY_PAST_THE_I32_SECOND_BOUND: i32 = 24_856;

    fn date32_lit(days: i32) -> Expr {
        Expr::Literal(ScalarValue::Date32(Some(days)), None)
    }

    fn date64_lit(millis: i64) -> Expr {
        Expr::Literal(ScalarValue::Date64(Some(millis)), None)
    }

    /// Asserts a date literal names `seconds` through BOTH arms the date `match` has — SQLite's
    /// own `date(..)` and the catch-all every other engine shares. Checking only one would pass a
    /// fix that reached that arm and left the other naming a different day, which is the shape
    /// both defects had.
    fn assert_date_renders(expr: &Expr, seconds: i64, what: &str) {
        assert_eq!(
            to_sql_with_engine(expr, None).expect("to unparse"),
            format!("TO_TIMESTAMP({seconds})"),
            "{what}"
        );
        assert_eq!(
            to_sql_with_engine(expr, Some(Engine::SQLite)).expect("to unparse"),
            format!("date({seconds}, 'unixepoch')"),
            "{what} on SQLite"
        );
    }

    /// A `Date32` past the `i32` second bound has no rendering at all on a product computed in
    /// `i32`: a debug build panics on the multiply and a release build wraps to a literal naming
    /// an unrelated instant, so a comparison built from it selects a row set nobody asked for.
    /// Every date here is an ordinary one a user can write.
    #[test]
    fn test_date32_past_the_i32_second_bound_names_its_own_instant() {
        for (days, seconds) in [
            (FIRST_DAY_PAST_THE_I32_SECOND_BOUND, 2_147_558_400_i64), // 2038-01-20
            (-FIRST_DAY_PAST_THE_I32_SECOND_BOUND, -2_147_558_400),   // 1901-12-13
            (30_000, 2_592_000_000),                                  // 2052-02-20
        ] {
            assert_date_renders(&date32_lit(days), seconds, &format!("Date32({days})"));
        }
    }

    /// The widest counts the variant holds, so the widening cannot be sized to the dates that
    /// happen to be in a test above it.
    #[test]
    fn test_date32_renders_the_widest_counts_it_can_hold() {
        for (days, seconds) in [
            (i32::MAX, 185_542_587_100_800_i64),
            (i32::MIN, -185_542_587_187_200),
        ] {
            assert_date_renders(&date32_lit(days), seconds, &format!("Date32({days})"));
        }
    }

    /// `Date64` counts milliseconds since the epoch, so its epoch second is a division. Scaling
    /// it by the length of a day instead put 1970-01-02 past the year 238 000.
    #[test]
    fn test_date64_renders_its_millisecond_count_as_the_second_it_names() {
        for (millis, seconds) in [
            (86_400_000_i64, 86_400_i64),       // 1970-01-02
            (-86_400_000, -86_400),             // 1969-12-31
            (0, 0),                             // 1970-01-01
            (1_767_225_600_000, 1_767_225_600), // 2026-01-01
            (2_147_558_400_000, 2_147_558_400), // 2038-01-20
        ] {
            assert_date_renders(&date64_lit(millis), seconds, &format!("Date64({millis})"));
        }
    }

    /// The `div_euclid` half of the `Date64` arm, which every whole-day count above leaves
    /// untested: each of those divides exactly, so all of them pass just as well under a
    /// truncating `/`. These do not.
    ///
    /// A count that is not a whole number of days is outside what Arrow's definition of `Date64`
    /// describes but inside what its type holds, so the arm has to render *something* for it, and
    /// below the epoch the two spellings disagree — `div_euclid` floors to the second at or before
    /// the instant named, `/` truncates towards zero and lands after it. Flooring is what keeps the
    /// arm consistent on both sides of the epoch, and what `duckdb_timestamp_literal` chose one arm
    /// over for the same reason.
    #[test]
    fn test_date64_floors_a_sub_second_count_below_the_epoch() {
        for (millis, seconds) in [
            (-1_i64, -1_i64),       // truncating `/` would render 0
            (-999, -1),             // truncating `/` would render 0
            (-86_400_500, -86_401), // truncating `/` would render -86_400
        ] {
            assert_date_renders(&date64_lit(millis), seconds, &format!("Date64({millis})"));
        }

        // Above the epoch the two agree, so this half pins that flooring costs nothing there.
        for (millis, seconds) in [(1_i64, 0_i64), (999, 0), (86_400_500, 86_400)] {
            assert_date_renders(&date64_lit(millis), seconds, &format!("Date64({millis})"));
        }
    }

    /// The two variants spell the same day in different units, so a filter on one has to render
    /// the same instant as a filter on the other. This is what the units error broke: the same
    /// day rendered two instants that were not even the same order of magnitude apart.
    #[test]
    fn test_date32_and_date64_render_one_day_identically() {
        for days in [
            0,
            1,
            -1,
            19_000,
            FIRST_DAY_PAST_THE_I32_SECOND_BOUND,
            -FIRST_DAY_PAST_THE_I32_SECOND_BOUND,
        ] {
            let millis = i64::from(days) * 86_400_000;

            for engine in [None, Some(Engine::SQLite), Some(Engine::Postgres)] {
                assert_eq!(
                    to_sql_with_engine(&date32_lit(days), engine).expect("to unparse"),
                    to_sql_with_engine(&date64_lit(millis), engine).expect("to unparse"),
                    "day {days} on {engine:?}"
                );
            }
        }
    }

    /// Both arms are shared by every engine, so the fix has to reach the catch-all and the
    /// SQLite arm alike — patching one would leave the other naming a different day.
    #[test]
    fn test_every_engine_renders_a_date_literal_past_the_i32_bound() {
        for (engine, expected) in [
            (None, "TO_TIMESTAMP(2147558400)"),
            (Some(Engine::DuckDB), "TO_TIMESTAMP(2147558400)"),
            (Some(Engine::Postgres), "TO_TIMESTAMP(2147558400)"),
            (Some(Engine::MySQL), "TO_TIMESTAMP(2147558400)"),
            (Some(Engine::ODBC), "TO_TIMESTAMP(2147558400)"),
            (Some(Engine::Spark), "TO_TIMESTAMP(2147558400)"),
            (Some(Engine::SQLite), "date(2147558400, 'unixepoch')"),
        ] {
            assert_eq!(
                to_sql_with_engine(&date32_lit(FIRST_DAY_PAST_THE_I32_SECOND_BOUND), engine)
                    .expect("to unparse"),
                expected,
                "Date32 on {engine:?}"
            );
            assert_eq!(
                to_sql_with_engine(&date64_lit(2_147_558_400_000), engine).expect("to unparse"),
                expected,
                "Date64 on {engine:?}"
            );
        }
    }

    /// The half that must NOT move: a `Date32` inside the bound renders byte-identically, because
    /// widening changes the width the product is computed in and not the instant any date names.
    ///
    /// This is the one assertion here that holds on both sides of the fix, and deliberately so —
    /// it pins the property the fix has to preserve, and it is what would refuse a "repair" that
    /// shifted every date to buy the ones past the bound. Read it as a control, not as a
    /// regression pin; the pins are the tests above it.
    #[test]
    fn test_a_date_inside_the_bound_renders_unchanged() {
        for (days, expected) in [
            (0, "TO_TIMESTAMP(0)"),
            (1, "TO_TIMESTAMP(86400)"),
            (-1, "TO_TIMESTAMP(-86400)"),
            (19_000, "TO_TIMESTAMP(1641600000)"),
            (24_855, "TO_TIMESTAMP(2147472000)"),
        ] {
            assert_eq!(
                to_sql_with_engine(&date32_lit(days), None).expect("to unparse"),
                expected,
                "Date32({days})"
            );
        }

        assert_eq!(
            to_sql_with_engine(&date32_lit(19_000), Some(Engine::SQLite)).expect("to unparse"),
            "date(1641600000, 'unixepoch')"
        );
        // The one `Date64` count the units error left alone: zero scales either way to zero.
        assert_eq!(
            to_sql_with_engine(&date64_lit(0), None).expect("to unparse"),
            "TO_TIMESTAMP(0)"
        );
    }

    /// The sub-second rendering is DuckDB-specific: the engines sharing the catch-all arm are
    /// untouched, and the ones with an arm of their own keep it.
    #[test]
    fn test_sub_second_literal_rendering_is_duckdb_specific() {
        let expr = Expr::Literal(
            ScalarValue::TimestampMicrosecond(Some(1_767_225_600_000_999), None),
            None,
        );

        for (engine, expected) in [
            (None, "TO_TIMESTAMP(1767225600)"),
            (Some(Engine::ODBC), "TO_TIMESTAMP(1767225600)"),
            (Some(Engine::Spark), "TO_TIMESTAMP(1767225600)"),
            (
                Some(Engine::Postgres),
                "TO_TIMESTAMP(1767225600.000999) AT TIME ZONE 'UTC'",
            ),
            (Some(Engine::MySQL), "FROM_UNIXTIME(1767225600.000999)"),
            (Some(Engine::SQLite), "datetime(1767225600, 'unixepoch')"),
        ] {
            let sql = to_sql_with_engine(&expr, engine).expect("to unparse");

            assert_eq!(sql, expected, "engine {engine:?}");
        }
    }
}

/// A rendering assertion cannot tell a literal that names the right instant from one that names a
/// neighbouring microsecond — both are well-formed SQL. These run the comparison DuckDB actually
/// evaluates against rows one microsecond apart and assert on which row survives it.
#[cfg(all(test, feature = "duckdb"))]
mod duckdb_timestamp_execution_tests {
    use super::tests::schema_with_ts;
    use super::*;
    use datafusion::prelude::*;
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
#[cfg(all(test, feature = "duckdb"))]
mod duckdb_date_execution_tests {
    use super::tests::schema_with_ts;
    use super::*;
    use datafusion::prelude::*;
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
