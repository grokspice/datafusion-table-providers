use std::ops::ControlFlow;
use std::str::FromStr;
use std::sync::Arc;

use arrow::datatypes::{DataType, FieldRef};
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::common::{DFSchema, ScalarValue};
use datafusion::error::DataFusionError;
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::expr_rewriter::NamePreserver;
use datafusion::logical_expr::{
    BinaryExpr, ColumnarValue, Expr as LogicalExpr, ExprSchemable, LogicalPlan, Operator,
    ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion::sql::sqlparser::ast::{
    self, BinaryOperator, Expr, FunctionArg, FunctionArgExpr, FunctionArgumentList, Ident,
    ObjectNamePart, VisitorMut,
};
use datafusion_federation::FederatedPlanNode;

/// The name of the identity function [`mark_date_operands`] wraps a `DATE` operand of
/// interval arithmetic in, so that [`SQLiteIntervalVisitor`] can tell it from a
/// timestamp once the plan is unparsed to SQL, where both are bare `TEXT`.
const DATE_OPERAND_MARKER: &str = "sqlite_date_operand";

/// The identity function over a `DATE` that [`mark_date_operands`] wraps a `DATE`
/// operand in. It only ever appears in a plan that is unparsed for SQLite, where the
/// visitor consumes it; it is still a faithful identity should it be evaluated.
#[derive(Debug, PartialEq, Eq, Hash)]
struct DateOperand {
    signature: Signature,
}

impl DateOperand {
    fn new() -> Self {
        Self {
            signature: Signature::uniform(
                1,
                vec![DataType::Date32, DataType::Date64],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for DateOperand {
    fn name(&self) -> &str {
        DATE_OPERAND_MARKER
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType, DataFusionError> {
        arg_types.first().cloned().ok_or_else(|| {
            DataFusionError::Plan(format!("{DATE_OPERAND_MARKER} takes exactly one argument"))
        })
    }

    /// The field of the operand this returns, rather than the default
    /// `Field::new(self.name(), self.return_type(..)?, true)`.
    ///
    /// An identity reports the field it was handed. The default reports every call
    /// nullable, which would widen a `NOT NULL` operand and, with it, every
    /// expression whose field is computed from it.
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef, DataFusionError> {
        args.arg_fields.first().map(Arc::clone).ok_or_else(|| {
            DataFusionError::Plan(format!("{DATE_OPERAND_MARKER} takes exactly one argument"))
        })
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue, DataFusionError> {
        args.args.into_iter().next().ok_or_else(|| {
            DataFusionError::Execution(format!("{DATE_OPERAND_MARKER} takes exactly one argument"))
        })
    }
}

/// Wraps the `DATE` operand of every `expr +/- INTERVAL` in `plan` in
/// [`DATE_OPERAND_MARKER`], so the SQL rendered for it shifts through SQLite's
/// `date()` rather than `datetime()`.
///
/// Whether an operand is a date or a timestamp is only known here, on the typed
/// logical plan: unparsed, both are `TEXT`, and SQLite's `date()` and `datetime()`
/// return a day and a day with a time of day respectively whatever they are given.
/// Run it as the scan's logical optimizer, before the plan is unparsed; the
/// [`SQLiteIntervalVisitor`] the executor's AST analyzer runs then consumes the
/// marker.
///
/// Every expression keeps the name it had: the plan's schema must not change under
/// a logical optimizer, and the unparser finds a window expression by its name. An
/// expression the marker renames is aliased back to its original name.
///
/// # Errors
///
/// Returns an error if the plan cannot be rebuilt around a rewritten expression.
pub fn mark_date_operands(plan: LogicalPlan) -> Result<LogicalPlan, DataFusionError> {
    // `with_subqueries`: an `EXISTS`, `IN` or scalar subquery is a plan of its own
    // inside an expression, which a plain `transform_up` never reaches, and a
    // federated statement keeps its same-source subqueries.
    plan.transform_up_with_subqueries(|node| {
        // The expressions of a node are evaluated against its inputs, or against its
        // own schema when it has none (a scan's pushed-down filters).
        let mut schema = DFSchema::empty();
        for input in node.inputs() {
            schema.merge(input.schema());
        }
        if node.inputs().is_empty() {
            schema.merge(node.schema());
        }
        let name_preserver = NamePreserver::new(&node);
        node.map_expressions(|expr| {
            let name = name_preserver.save(&expr);
            expr.transform_up(|expr| mark_date_operand(expr, &schema))
                .map(|marked| marked.update_data(|expr| name.restore(expr)))
        })
    })
    .data()
}

/// [`mark_date_operands`] for the plan the federation analyzer hands the executor's
/// logical optimizer: the federated statement wrapped in its [`FederatedPlanNode`],
/// whose own plan is what gets marked.
///
/// The scan's logical optimizer ([`mark_date_operands`] on the scan's `SQLTable`)
/// is gathered by walking the statement's plan nodes, which does not reach a table
/// that only appears inside an `EXISTS` or scalar subquery; this hook runs for every
/// federated statement regardless. Marking is idempotent, so a statement reached by
/// both is marked once.
///
/// # Errors
///
/// Returns an error if the plan cannot be rebuilt around a rewritten expression.
pub fn mark_federated_date_operands(plan: LogicalPlan) -> Result<LogicalPlan, DataFusionError> {
    let LogicalPlan::Extension(extension) = &plan else {
        return mark_date_operands(plan);
    };
    let Some(node) = extension.node.as_any().downcast_ref::<FederatedPlanNode>() else {
        return Ok(plan);
    };
    let marked = mark_date_operands(node.plan().clone())?;
    Ok(LogicalPlan::Extension(
        datafusion::logical_expr::Extension {
            node: Arc::new(FederatedPlanNode::new_with_query_type(
                marked,
                Arc::clone(node.planner()),
                node.query_type(),
            )),
        },
    ))
}

fn mark_date_operand(
    expr: LogicalExpr,
    schema: &DFSchema,
) -> Result<Transformed<LogicalExpr>, DataFusionError> {
    let LogicalExpr::BinaryExpr(BinaryExpr {
        left,
        op: op @ (Operator::Plus | Operator::Minus),
        right,
    }) = &expr
    else {
        return Ok(Transformed::no(expr));
    };
    let interval_on_the_right = is_interval_literal(right);
    let operand = if interval_on_the_right {
        left
    } else if is_interval_literal(left) {
        right
    } else {
        return Ok(Transformed::no(expr));
    };
    // An operand whose type cannot be resolved is left alone: it renders as it did.
    let Ok(DataType::Date32 | DataType::Date64) = operand.get_type(schema) else {
        return Ok(Transformed::no(expr));
    };
    if is_date_operand_marker(operand) {
        return Ok(Transformed::no(expr));
    }
    let marked = LogicalExpr::ScalarFunction(ScalarFunction::new_udf(
        Arc::new(ScalarUDF::new_from_impl(DateOperand::new())),
        vec![operand.as_ref().clone()],
    ));
    let (left, right) = if interval_on_the_right {
        (marked, right.as_ref().clone())
    } else {
        (left.as_ref().clone(), marked)
    };
    Ok(Transformed::yes(LogicalExpr::BinaryExpr(BinaryExpr::new(
        Box::new(left),
        *op,
        Box::new(right),
    ))))
}

fn is_interval_literal(expr: &LogicalExpr) -> bool {
    matches!(
        expr,
        LogicalExpr::Literal(
            ScalarValue::IntervalYearMonth(_)
                | ScalarValue::IntervalDayTime(_)
                | ScalarValue::IntervalMonthDayNano(_),
            _
        )
    )
}

fn is_date_operand_marker(expr: &LogicalExpr) -> bool {
    matches!(expr, LogicalExpr::ScalarFunction(function) if function.name() == DATE_OPERAND_MARKER)
}

/// The operand a [`DATE_OPERAND_MARKER`] call in the unparsed SQL wraps, if `expr` is one.
fn date_operand(expr: &Expr) -> Option<Expr> {
    let Expr::Function(function) = expr else {
        return None;
    };
    if function.name.to_string() != DATE_OPERAND_MARKER {
        return None;
    }
    let ast::FunctionArguments::List(arguments) = &function.args else {
        return None;
    };
    match arguments.args.as_slice() {
        [FunctionArg::Unnamed(FunctionArgExpr::Expr(operand))] => Some(operand.clone()),
        _ => None,
    }
}

#[derive(Default)]
pub struct SQLiteIntervalVisitor {}

#[derive(Default, Debug)]
struct IntervalParts {
    years: i64,
    months: i64,
    days: i64,
    hours: i64,
    minutes: i64,
    seconds: i64,
    nanos: u32,
}

type IntervalSetter = fn(IntervalParts, i64) -> IntervalParts;

impl IntervalParts {
    fn new() -> Self {
        Self::default()
    }

    fn negate(mut self) -> Self {
        self.years = -self.years;
        self.months = -self.months;
        self.days = -self.days;
        self.hours = -self.hours;
        self.minutes = -self.minutes;
        self.seconds = -self.seconds;
        self
    }

    fn with_years(mut self, years: i64) -> Self {
        self.years = years;
        self
    }

    fn with_months(mut self, months: i64) -> Self {
        self.months = months;
        self
    }

    fn with_days(mut self, days: i64) -> Self {
        self.days = days;
        self
    }

    fn with_hours(mut self, hours: i64) -> Self {
        self.hours = hours;
        self
    }

    fn with_minutes(mut self, minutes: i64) -> Self {
        self.minutes = minutes;
        self
    }

    fn with_seconds(mut self, seconds: i64) -> Self {
        self.seconds = seconds;
        self
    }

    fn with_nanos(mut self, nanos: u32) -> Self {
        self.nanos = nanos;
        self
    }
}

impl VisitorMut for SQLiteIntervalVisitor {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &mut Expr) -> ControlFlow<Self::Break> {
        // for each INTERVAL, find the previous (or next, if the INTERVAL is first) expression or column name that is associated with it
        // e.g. `column_name + INTERVAL '1' DAY``, we should find the `column_name`
        // then replace the `INTERVAL` with a shift of it: `datetime(column_name, '+1 days')`, or
        // `date(column_name, '+1 days')` when `mark_date_operands` marked it as a date
        // this should also apply to expressions though, like `CAST(column_name AS TEXT) + INTERVAL '1' DAY`
        // in this example, the shifted operand would be `CAST(column_name AS TEXT)`
        // a chain, `column_name + INTERVAL '1' DAY - INTERVAL '1' HOUR`, is rewritten from the
        // outside in: the outer shift's operand is the inner `column_name + INTERVAL '1' DAY`,
        // which this visitor then reaches and rewrites in turn

        if let Some(operand) = date_operand(expr) {
            // A marker left over from an interval the rewrite below could not parse
            // (the marked operand then renders as it did, without its marker).
            *expr = operand;
        } else if let Expr::BinaryOp { op, left, right } = expr {
            if *op != BinaryOperator::Plus && *op != BinaryOperator::Minus {
                return ControlFlow::Continue(());
            }

            let (target, interval) = SQLiteIntervalVisitor::normalize_interval_expr(left, right);

            if let Expr::Interval(_) = interval.as_ref() {
                // parse the INTERVAL and get the bits out of it
                // e.g. INTERVAL 0 YEARS 0 MONS 1 DAYS 0 HOURS 0 MINUTES 0.000000000 SECS -> IntervalParts { days: 1 }
                if let Ok(interval_parts) = SQLiteIntervalVisitor::parse_interval(interval) {
                    // negate the interval parts if the operator is minus
                    let interval_parts = if *op == BinaryOperator::Minus {
                        interval_parts.negate()
                    } else {
                        interval_parts
                    };

                    *expr = SQLiteIntervalVisitor::create_shifted_expr(target, &interval_parts);
                }
            }
        }
        ControlFlow::Continue(())
    }
}

impl SQLiteIntervalVisitor {
    // normalize the sides of the operation to make sure the INTERVAL is always on the right
    fn normalize_interval_expr<'a>(
        left: &'a mut Box<Expr>,
        right: &'a mut Box<Expr>,
    ) -> (&'a mut Box<Expr>, &'a mut Box<Expr>) {
        if let Expr::Interval { .. } = left.as_ref() {
            (right, left)
        } else {
            (left, right)
        }
    }

    fn parse_interval(interval: &Expr) -> Result<IntervalParts, DataFusionError> {
        if let Expr::Interval(interval_expr) = interval {
            if let Expr::Value(ast::ValueWithSpan {
                value: ast::Value::SingleQuotedString(value),
                span: _,
            }) = interval_expr.value.as_ref()
            {
                return SQLiteIntervalVisitor::parse_interval_string(value);
            }
        }
        Err(DataFusionError::Plan(
            "Invalid interval expression".to_string(),
        ))
    }

    fn parse_interval_string(value: &str) -> Result<IntervalParts, DataFusionError> {
        let mut parts = IntervalParts::new();
        let mut remaining = value;

        let components: [(_, IntervalSetter); 5] = [
            ("YEARS", IntervalParts::with_years),
            ("MONS", IntervalParts::with_months),
            ("DAYS", IntervalParts::with_days),
            ("HOURS", IntervalParts::with_hours),
            ("MINS", IntervalParts::with_minutes),
        ];

        for (unit, setter) in &components {
            if let Some((value, rest)) = remaining.split_once(unit) {
                let parsed_value: i64 = SQLiteIntervalVisitor::parse_value(value.trim())?;
                parts = setter(parts, parsed_value);
                remaining = rest;
            }
        }

        // Parse seconds and nanoseconds separately
        if let Some((secs, _)) = remaining.split_once("SECS") {
            let (seconds, nanos) = SQLiteIntervalVisitor::parse_seconds_and_nanos(secs.trim())?;
            parts = parts.with_seconds(seconds).with_nanos(nanos);
        }

        Ok(parts)
    }

    fn parse_seconds_and_nanos(value: &str) -> Result<(i64, u32), DataFusionError> {
        let parts: Vec<&str> = value.split('.').collect();
        let seconds = SQLiteIntervalVisitor::parse_value(parts[0])?;
        let nanos = if parts.len() > 1 {
            let nanos_str = format!("{:0<9}", parts[1]);
            nanos_str[..9].parse().map_err(|_| {
                DataFusionError::Plan(format!("Failed to parse nanoseconds: {}", parts[1]))
            })?
        } else {
            0
        };
        Ok((seconds, nanos))
    }

    fn parse_value<T: FromStr>(value: &str) -> Result<T, DataFusionError> {
        value
            .parse()
            .map_err(|_| DataFusionError::Plan(format!("Failed to parse interval value: {value}")))
    }

    /// Renders `target` shifted by `interval`, cast to `TEXT`: `datetime(target,
    /// <modifiers>)`, or `date(target, <modifiers>)` when [`mark_date_operands`] marked
    /// `target` as a `DATE`.
    ///
    /// The choice is by the operand, never by the interval. SQLite's `date()` returns
    /// the calendar day alone, so choosing it for a whole-day or a subtracted intraday
    /// interval discarded the time of day of a timestamp operand; `datetime()` keeps
    /// it. A `DATE` still shifts through `date()`, so it stays date-only text and
    /// compares equal to a date literal, which is rendered date-only too.
    fn create_shifted_expr(target: &Expr, interval: &IntervalParts) -> Expr {
        let (function, operand) = match date_operand(target) {
            Some(operand) => ("date", operand),
            None => ("datetime", target.clone()),
        };
        let args = std::iter::once(Some(FunctionArg::Unnamed(FunctionArgExpr::Expr(operand))))
            .chain([
                SQLiteIntervalVisitor::create_interval_arg("years", interval.years),
                SQLiteIntervalVisitor::create_interval_arg("months", interval.months),
                SQLiteIntervalVisitor::create_interval_arg("days", interval.days),
                SQLiteIntervalVisitor::create_interval_arg("hours", interval.hours),
                SQLiteIntervalVisitor::create_interval_arg("minutes", interval.minutes),
                SQLiteIntervalVisitor::create_interval_arg_with_fraction(
                    "seconds",
                    interval.seconds,
                    interval.nanos,
                ),
            ])
            .flatten() // flatten the list of arguments to exclude 0 values
            .collect();

        Expr::Cast {
            expr: Box::new(SQLiteIntervalVisitor::call(function, args)),
            data_type: ast::DataType::Text,
            format: None,
            kind: ast::CastKind::Cast,
            array: false,
        }
    }

    fn call(function: &str, args: Vec<FunctionArg>) -> Expr {
        Expr::Function(ast::Function {
            name: ast::ObjectName(vec![ObjectNamePart::Identifier(Ident::new(function))]),
            args: ast::FunctionArguments::List(FunctionArgumentList {
                duplicate_treatment: None,
                args,
                clauses: Vec::new(),
            }),
            filter: None,
            null_treatment: None,
            over: None,
            within_group: Vec::new(),
            parameters: ast::FunctionArguments::None,
            uses_odbc_syntax: false,
        })
    }

    fn create_interval_arg(unit: &str, value: i64) -> Option<FunctionArg> {
        if value == 0 {
            None
        } else {
            Some(FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::value(
                ast::Value::SingleQuotedString(format!("{value:+} {unit}")),
            ))))
        }
    }

    fn create_interval_arg_with_fraction(
        unit: &str,
        value: i64,
        fraction: u32,
    ) -> Option<FunctionArg> {
        if value == 0 && fraction == 0 {
            None
        } else {
            let fraction_str = if fraction > 0 {
                format!(".{fraction:09}")
            } else {
                String::new()
            };

            Some(FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::value(
                ast::Value::SingleQuotedString(format!("{value:+}{fraction_str} {unit}")),
            ))))
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::datatypes::{Field, Schema, TimeUnit};
    use datafusion::logical_expr::{cast, col, lit, LogicalPlanBuilder, LogicalTableSource};
    use datafusion::sql::sqlparser::ast::VisitMut;
    use datafusion::sql::sqlparser::{dialect::GenericDialect, parser::Parser};

    #[test]
    fn test_interval_parts_parse() {
        let parts = SQLiteIntervalVisitor::parse_interval_string(
            "0 YEARS 0 MONS 1 DAYS 0 HOURS 0 MINS 0.000000000 SECS",
        )
        .expect("interval parts should be parsed");

        assert_eq!(parts.years, 0);
        assert_eq!(parts.months, 0);
        assert_eq!(parts.days, 1);
        assert_eq!(parts.hours, 0);
        assert_eq!(parts.minutes, 0);
        assert_eq!(parts.seconds, 0);
        assert_eq!(parts.nanos, 0);
    }

    #[test]
    fn test_interval_parts_parse_with_nanos() {
        let parts = SQLiteIntervalVisitor::parse_interval_string(
            "0 YEARS 0 MONS 0 DAYS 0 HOURS 0 MINS 0.000000001 SECS",
        )
        .expect("interval parts should be parsed");

        assert_eq!(parts.years, 0);
        assert_eq!(parts.months, 0);
        assert_eq!(parts.days, 0);
        assert_eq!(parts.hours, 0);
        assert_eq!(parts.minutes, 0);
        assert_eq!(parts.seconds, 0);
        assert_eq!(parts.nanos, 1);
    }

    #[test]
    fn test_interval_parts_parse_negative() {
        let parts = SQLiteIntervalVisitor::parse_interval_string(
            "0 YEARS 0 MONS -1 DAYS 0 HOURS 0 MINS 0.000000000 SECS",
        )
        .expect("interval parts should be parsed");

        assert_eq!(parts.years, 0);
        assert_eq!(parts.months, 0);
        assert_eq!(parts.days, -1);
        assert_eq!(parts.hours, 0);
        assert_eq!(parts.minutes, 0);
        assert_eq!(parts.seconds, 0);
        assert_eq!(parts.nanos, 0);
    }

    #[test]
    fn test_interval_parts_parse_intraday() {
        let parts = SQLiteIntervalVisitor::parse_interval_string(
            "0 YEARS 0 MONS 0 DAYS 1 HOURS 1 MINS 1.000000001 SECS",
        )
        .expect("interval parts should be parsed");

        assert_eq!(parts.years, 0);
        assert_eq!(parts.months, 0);
        assert_eq!(parts.days, 0);
        assert_eq!(parts.hours, 1);
        assert_eq!(parts.minutes, 1);
        assert_eq!(parts.seconds, 1);
        assert_eq!(parts.nanos, 1);
    }

    fn target() -> Expr {
        Expr::value(ast::Value::SingleQuotedString("1995-01-01".to_string()))
    }

    /// `target()` wrapped in the marker `mark_date_operands` puts on a `DATE`.
    fn marked_target() -> Expr {
        SQLiteIntervalVisitor::call(
            DATE_OPERAND_MARKER,
            vec![FunctionArg::Unnamed(FunctionArgExpr::Expr(target()))],
        )
    }

    /// A whole-day interval shifts an unmarked operand through `datetime`, not
    /// `date`: `date` returns the calendar day alone and would discard the time.
    #[test]
    fn a_whole_day_interval_keeps_a_datetime_operands_time() {
        let interval = IntervalParts::new()
            .with_years(1)
            .with_months(2)
            .with_days(3);

        assert_eq!(
            SQLiteIntervalVisitor::create_shifted_expr(&target(), &interval).to_string(),
            "CAST(datetime('1995-01-01', '+1 years', '+2 months', '+3 days') AS TEXT)"
        );
    }

    #[test]
    fn an_intraday_interval_renders_its_modifiers() {
        let interval = IntervalParts::new()
            .with_hours(1)
            .with_minutes(2)
            .with_seconds(3);

        assert_eq!(
            SQLiteIntervalVisitor::create_shifted_expr(&target(), &interval).to_string(),
            "CAST(datetime('1995-01-01', '+1 hours', '+2 minutes', '+3 seconds') AS TEXT)"
        );
    }

    /// A subtracted interval negates every part; the modifiers carry the sign, and a
    /// negated intraday part still shifts through `datetime`.
    #[test]
    fn a_negated_interval_renders_signed_modifiers() {
        let interval = IntervalParts::new().with_hours(1).with_minutes(30).negate();

        assert_eq!(
            SQLiteIntervalVisitor::create_shifted_expr(&target(), &interval).to_string(),
            "CAST(datetime('1995-01-01', '-1 hours', '-30 minutes') AS TEXT)"
        );
    }

    /// A marked `DATE` operand shifts through `date`, whatever the interval holds,
    /// and the marker itself is consumed.
    #[test]
    fn a_marked_date_operand_shifts_through_date() {
        let whole_day = IntervalParts::new().with_days(1);
        let intraday = IntervalParts::new().with_hours(1);

        assert_eq!(
            SQLiteIntervalVisitor::create_shifted_expr(&marked_target(), &whole_day).to_string(),
            "CAST(date('1995-01-01', '+1 days') AS TEXT)"
        );
        assert_eq!(
            SQLiteIntervalVisitor::create_shifted_expr(&marked_target(), &intraday).to_string(),
            "CAST(date('1995-01-01', '+1 hours') AS TEXT)"
        );
    }

    /// The rewrite a federated statement reaches SQLite with, for the shapes the
    /// report in spiceai/spiceai#14714 ran, a marked date, a chain of intervals, and a
    /// marker the rewrite did not consume.
    #[test]
    fn interval_arithmetic_is_rewritten_to_a_shift_of_the_operands_own_kind() {
        for (sql, expected) in [
            (
                "SELECT ts - INTERVAL '0 YEARS 0 MONS 0 DAYS 1 HOURS 0 MINS 0.000000000 SECS' FROM t",
                "SELECT CAST(datetime(ts, '-1 hours') AS TEXT) FROM t",
            ),
            (
                "SELECT ts + INTERVAL '0 YEARS 0 MONS 1 DAYS 0 HOURS 0 MINS 0.000000000 SECS' FROM t",
                "SELECT CAST(datetime(ts, '+1 days') AS TEXT) FROM t",
            ),
            (
                "SELECT ts - INTERVAL '0 YEARS 0 MONS 1 DAYS 0 HOURS 0 MINS 0.000000000 SECS' FROM t",
                "SELECT CAST(datetime(ts, '-1 days') AS TEXT) FROM t",
            ),
            (
                "SELECT ts + INTERVAL '0 YEARS 0 MONS 0 DAYS 0 HOURS 0 MINS 1.500000000 SECS' FROM t",
                "SELECT CAST(datetime(ts, '+1.500000000 seconds') AS TEXT) FROM t",
            ),
            (
                "SELECT sqlite_date_operand(d) + INTERVAL '0 YEARS 0 MONS 1 DAYS 0 HOURS 0 MINS 0.000000000 SECS' FROM t",
                "SELECT CAST(date(d, '+1 days') AS TEXT) FROM t",
            ),
            // A chain is rewritten from the outside in, each shift's operand being the
            // shift inside it: the SQL grows by one call per interval.
            (
                "SELECT ts + INTERVAL '0 YEARS 0 MONS 1 DAYS 0 HOURS 0 MINS 0.000000000 SECS' - INTERVAL '0 YEARS 0 MONS 0 DAYS 1 HOURS 0 MINS 0.000000000 SECS' FROM t",
                "SELECT CAST(datetime(CAST(datetime(ts, '+1 days') AS TEXT), '-1 hours') AS TEXT) FROM t",
            ),
            (
                "SELECT sqlite_date_operand(sqlite_date_operand(d) + INTERVAL '0 YEARS 0 MONS 1 DAYS 0 HOURS 0 MINS 0.000000000 SECS') + INTERVAL '0 YEARS 0 MONS 1 DAYS 0 HOURS 0 MINS 0.000000000 SECS' FROM t",
                "SELECT CAST(date(CAST(date(d, '+1 days') AS TEXT), '+1 days') AS TEXT) FROM t",
            ),
            // A marker no interval consumed renders its operand alone.
            (
                "SELECT sqlite_date_operand(d) FROM t",
                "SELECT d FROM t",
            ),
        ] {
            let mut statements = Parser::parse_sql(&GenericDialect {}, sql).expect("parses");
            let mut statement = statements.pop().expect("one statement");
            let _ = statement.visit(&mut SQLiteIntervalVisitor::default());
            assert_eq!(statement.to_string(), expected, "{sql}");
        }
    }

    /// On the typed plan, only a `DATE` operand of interval arithmetic is marked: a
    /// timestamp, and a date cast to a timestamp, are left as they are.
    #[test]
    fn only_a_date_operand_of_interval_arithmetic_is_marked() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Date32, true),
            Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        ]));
        let one_day = || lit(ScalarValue::new_interval_mdn(0, 1, 0));
        let plan = LogicalPlanBuilder::scan("t", Arc::new(LogicalTableSource::new(schema)), None)
            .expect("scan")
            .project(vec![
                (col("d") + one_day()).alias("date_plus"),
                (one_day() + col("d")).alias("plus_date"),
                (col("ts") + one_day()).alias("timestamp_plus"),
                (cast(col("d"), DataType::Timestamp(TimeUnit::Microsecond, None)) + one_day())
                    .alias("cast_plus"),
                ((col("d") + one_day()) - one_day()).alias("chain"),
            ])
            .expect("projection")
            .build()
            .expect("plan");

        let marked = mark_date_operands(plan).expect("marked");
        // The interval literal's own rendering is not under test.
        let interval = one_day().to_string();
        let rendered: Vec<String> = marked
            .expressions()
            .iter()
            .map(|expr| expr.to_string().replace(&interval, "<interval>"))
            .collect();
        assert_eq!(
            rendered,
            [
                "sqlite_date_operand(t.d) + <interval> AS date_plus",
                "<interval> + sqlite_date_operand(t.d) AS plus_date",
                "t.ts + <interval> AS timestamp_plus",
                "CAST(t.d AS Timestamp(µs)) + <interval> AS cast_plus",
                "sqlite_date_operand(sqlite_date_operand(t.d) + <interval>) - <interval> AS chain",
            ]
        );
    }

    /// A marked expression keeps the name it had, so the plan's schema is unchanged:
    /// an unaliased projection is aliased back to its original name.
    #[test]
    fn a_marked_expression_keeps_its_name() {
        let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date32, true)]));
        let plan = LogicalPlanBuilder::scan("t", Arc::new(LogicalTableSource::new(schema)), None)
            .expect("scan")
            .project(vec![col("d") + lit(ScalarValue::new_interval_mdn(0, 1, 0))])
            .expect("projection")
            .build()
            .expect("plan");
        let original_schema = plan.schema().clone();
        let original_name = plan.expressions()[0].to_string();

        let marked = mark_date_operands(plan).expect("marked");

        assert_eq!(marked.schema(), &original_schema);
        assert_eq!(
            marked.expressions()[0].to_string(),
            format!(
                "sqlite_date_operand(t.d) + {} AS {original_name}",
                lit(ScalarValue::new_interval_mdn(0, 1, 0))
            )
        );
    }

    /// The marker reports the field of the operand it returns, `NOT NULL` included.
    ///
    /// It is an identity, so the field it reports has to be the field it is handed.
    /// `ScalarUDFImpl::return_field_from_args` defaults to
    /// `Field::new(self.name(), self.return_type(..)?, true)` — nullable whatever it
    /// wrapped — which widens a `NOT NULL` date operand and, with it, every
    /// expression whose field is computed from it.
    #[test]
    fn the_marker_reports_the_field_of_the_operand_it_returns() {
        use datafusion::logical_expr::ExprSchemable;

        let schema = Schema::new(vec![Field::new("d", DataType::Date32, false)]);
        let schema = DFSchema::try_from(schema).expect("schema");
        let marker = ScalarUDF::new_from_impl(DateOperand::new());

        let (_, marked) = marker
            .call(vec![col("d")])
            .to_field(&schema)
            .expect("marked field");
        assert_eq!(marked.data_type(), &DataType::Date32);
        assert!(
            !marked.is_nullable(),
            "a NOT NULL operand must stay NOT NULL through the marker, got {marked:?}"
        );

        // And so must the interval expression whose field is computed from it.
        let (_, shifted) = (marker.call(vec![col("d")])
            + lit(ScalarValue::new_interval_mdn(0, 1, 0)))
        .to_field(&schema)
        .expect("shifted field");
        assert!(
            !shifted.is_nullable(),
            "the shift of a NOT NULL operand must stay NOT NULL, got {shifted:?}"
        );
    }

    mod federated {
        use std::sync::Arc;

        use arrow::array::{Date32Array, Int64Array, RecordBatch, TimestampMicrosecondArray};
        use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        use arrow::util::pretty::pretty_format_batches;
        use datafusion::common::TableReference;
        use datafusion::execution::context::SessionContext;

        use crate::sql::arrow_sql_gen::statement::{CreateTableBuilder, InsertBuilder};
        use crate::sql::db_connection_pool::sqlitepool::SqliteConnectionPoolFactory;
        use crate::sql::db_connection_pool::{DbConnectionPool, Mode};
        use crate::sqlite::sql_table::SQLiteTable;
        use crate::sqlite::DynSqliteConnectionPool;

        /// Microseconds since the epoch of `YYYY-MM-DDTHH:MM:SS`.
        fn micros(timestamp: &str) -> i64 {
            let (date, time) = timestamp.split_once('T').expect("date and time");
            let mut date = date
                .split('-')
                .map(|part| part.parse::<i64>().expect("number"));
            let (year, month, day) = (
                date.next().expect("year"),
                date.next().expect("month"),
                date.next().expect("day"),
            );
            let mut time = time
                .split(':')
                .map(|part| part.parse::<i64>().expect("number"));
            let (hours, minutes, seconds) = (
                time.next().expect("hours"),
                time.next().expect("minutes"),
                time.next().expect("seconds"),
            );
            (days(year, month, day) * 86_400 + hours * 3_600 + minutes * 60 + seconds) * 1_000_000
        }

        /// Days since the epoch of a civil date.
        fn days(year: i64, month: i64, day: i64) -> i64 {
            let year = if month <= 2 { year - 1 } else { year };
            let era = year.div_euclid(400);
            let year_of_era = year - era * 400;
            let day_of_year =
                (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
            let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
            era * 146_097 + day_of_era - 719_468
        }

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
            let ctx =
                SessionContext::new_with_state(datafusion_federation::default_session_state());
            ctx.register_table(table, Arc::new(provider))
                .expect("table registered");
            ctx
        }

        /// Runs `sql` through the federated session and returns its rows and the
        /// physical plan's display, which carries the SQL sent to SQLite; asserts the
        /// statement was pushed down.
        async fn run(ctx: &SessionContext, sql: &str) -> (String, String) {
            let df = ctx.sql(sql).await.expect("the query plans");
            let plan = df
                .clone()
                .create_physical_plan()
                .await
                .expect("physical plan");
            let display = datafusion::physical_plan::displayable(plan.as_ref())
                .indent(true)
                .to_string();
            assert!(
                display.contains("VirtualExecutionPlan"),
                "{sql} must be pushed down to SQLite:\n{display}"
            );
            let batches = df.collect().await.expect("the query runs");
            let rows = pretty_format_batches(&batches)
                .expect("rows format")
                .to_string();
            (rows, display)
        }

        /// Asserts `sql` answers `expected` and that the SQL sent to SQLite carries
        /// `pushed`, so a case that stopped going through the rewrite cannot pass on
        /// a local evaluation.
        async fn check(ctx: &SessionContext, sql: &str, pushed: &str, expected: &str) {
            let (rows, display) = run(ctx, sql).await;
            assert!(
                display.contains(pushed),
                "{sql}: expected the SQL sent to SQLite to carry `{pushed}`:\n{display}"
            );
            assert_eq!(rows, expected, "{sql}");
        }

        /// The three-row table every case below runs over: `ts` is a timestamp, `d`
        /// the date of the same row.
        async fn session() -> SessionContext {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
                Field::new("d", DataType::Date32, true),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int64Array::from(vec![1, 2, 3])),
                    Arc::new(TimestampMicrosecondArray::from(vec![
                        micros("2026-10-02T12:34:56"),
                        micros("2026-10-02T00:30:00"),
                        micros("2026-01-31T23:59:59"),
                    ])),
                    Arc::new(Date32Array::from(vec![
                        i32::try_from(days(2026, 10, 2)).expect("fits"),
                        i32::try_from(days(2026, 10, 2)).expect("fits"),
                        i32::try_from(days(2026, 1, 31)).expect("fits"),
                    ])),
                ],
            )
            .expect("batch");
            federated_session("rows", batch).await
        }

        /// Timestamp interval arithmetic federated to SQLite keeps the time of day, for
        /// a subtracted intraday interval, a whole-day interval in either direction,
        /// and a chain of both; a filter over it keeps exactly the rows the shifted
        /// timestamp satisfies (spiceai/spiceai#14714).
        #[tokio::test]
        async fn interval_arithmetic_on_a_timestamp_keeps_the_time_of_day() {
            let ctx = session().await;
            for (sql, pushed, expected) in [
                (
                    "SELECT id, ts - INTERVAL '1 hour' AS shifted FROM rows ORDER BY id",
                    "datetime(`rows`.`ts`, '-1 hours')",
                    "+----+---------------------+\n\
                     | id | shifted             |\n\
                     +----+---------------------+\n\
                     | 1  | 2026-10-02T11:34:56 |\n\
                     | 2  | 2026-10-01T23:30:00 |\n\
                     | 3  | 2026-01-31T22:59:59 |\n\
                     +----+---------------------+",
                ),
                (
                    "SELECT id, ts + INTERVAL '1 day' AS shifted FROM rows ORDER BY id",
                    "datetime(`rows`.`ts`, '+1 days')",
                    "+----+---------------------+\n\
                     | id | shifted             |\n\
                     +----+---------------------+\n\
                     | 1  | 2026-10-03T12:34:56 |\n\
                     | 2  | 2026-10-03T00:30:00 |\n\
                     | 3  | 2026-02-01T23:59:59 |\n\
                     +----+---------------------+",
                ),
                (
                    "SELECT id, ts - INTERVAL '1 day' AS shifted FROM rows ORDER BY id",
                    "datetime(`rows`.`ts`, '-1 days')",
                    "+----+---------------------+\n\
                     | id | shifted             |\n\
                     +----+---------------------+\n\
                     | 1  | 2026-10-01T12:34:56 |\n\
                     | 2  | 2026-10-01T00:30:00 |\n\
                     | 3  | 2026-01-30T23:59:59 |\n\
                     +----+---------------------+",
                ),
                (
                    "SELECT id, ts + INTERVAL '1 day' - INTERVAL '1 hour' AS shifted FROM rows ORDER BY id",
                    "datetime(",
                    "+----+---------------------+\n\
                     | id | shifted             |\n\
                     +----+---------------------+\n\
                     | 1  | 2026-10-03T11:34:56 |\n\
                     | 2  | 2026-10-02T23:30:00 |\n\
                     | 3  | 2026-02-01T22:59:59 |\n\
                     +----+---------------------+",
                ),
                (
                    "SELECT id FROM rows WHERE ts - INTERVAL '1 hour' < TIMESTAMP '2026-10-02 10:00:00' ORDER BY id",
                    "datetime(`rows`.`ts`, '-1 hours')",
                    "+----+\n\
                     | id |\n\
                     +----+\n\
                     | 2  |\n\
                     | 3  |\n\
                     +----+",
                ),
            ] {
                check(&ctx, sql, pushed, expected).await;
            }
        }

        /// A shifted timestamp compares against a whole-second timestamp literal at
        /// the exact boundary: `=`, `<=` and `>` each keep the rows they should. Row 2
        /// shifted by an hour is exactly the literal.
        #[tokio::test]
        async fn a_shifted_timestamp_compares_exactly_at_a_whole_second_boundary() {
            let ctx = session().await;
            for (sql, expected) in [
                (
                    "SELECT id FROM rows WHERE ts + INTERVAL '1 hour' = TIMESTAMP '2026-10-02 01:30:00' ORDER BY id",
                    "+----+\n\
                     | id |\n\
                     +----+\n\
                     | 2  |\n\
                     +----+",
                ),
                (
                    "SELECT id FROM rows WHERE ts + INTERVAL '1 hour' <= TIMESTAMP '2026-10-02 01:30:00' ORDER BY id",
                    "+----+\n\
                     | id |\n\
                     +----+\n\
                     | 2  |\n\
                     | 3  |\n\
                     +----+",
                ),
                (
                    "SELECT id FROM rows WHERE ts + INTERVAL '1 hour' > TIMESTAMP '2026-10-02 01:30:00' ORDER BY id",
                    "+----+\n\
                     | id |\n\
                     +----+\n\
                     | 1  |\n\
                     +----+",
                ),
            ] {
                check(&ctx, sql, "datetime(`rows`.`ts`, '+1 hours')", expected).await;
            }
        }

        /// A `DATE` operand shifts as a day, through `date()`: it reads back as the
        /// day, an intraday remainder does not move it (as in DataFusion's own date
        /// arithmetic), a chain stays a day, and it compares equal to a date literal
        /// at the boundary, which a time-of-day rendering would not.
        #[tokio::test]
        async fn interval_arithmetic_on_a_date_stays_a_date() {
            let ctx = session().await;
            for (sql, pushed, expected) in [
                (
                    "SELECT id, d + INTERVAL '1 day' AS next_day FROM rows ORDER BY id",
                    "date(`rows`.`d`, '+1 days')",
                    "+----+------------+\n\
                     | id | next_day   |\n\
                     +----+------------+\n\
                     | 1  | 2026-10-03 |\n\
                     | 2  | 2026-10-03 |\n\
                     | 3  | 2026-02-01 |\n\
                     +----+------------+",
                ),
                (
                    "SELECT id, d - INTERVAL '1 day' AS prev_day FROM rows ORDER BY id",
                    "date(`rows`.`d`, '-1 days')",
                    "+----+------------+\n\
                     | id | prev_day   |\n\
                     +----+------------+\n\
                     | 1  | 2026-10-01 |\n\
                     | 2  | 2026-10-01 |\n\
                     | 3  | 2026-01-30 |\n\
                     +----+------------+",
                ),
                (
                    "SELECT id, d + INTERVAL '25 hours' AS next_day FROM rows ORDER BY id",
                    "date(`rows`.`d`, '+25 hours')",
                    "+----+------------+\n\
                     | id | next_day   |\n\
                     +----+------------+\n\
                     | 1  | 2026-10-03 |\n\
                     | 2  | 2026-10-03 |\n\
                     | 3  | 2026-02-01 |\n\
                     +----+------------+",
                ),
                (
                    "SELECT id, d + INTERVAL '1 day' + INTERVAL '1 day' AS later FROM rows ORDER BY id",
                    "date((CAST(date(`rows`.`d`, '+1 days') AS TEXT)), '+1 days')",
                    "+----+------------+\n\
                     | id | later      |\n\
                     +----+------------+\n\
                     | 1  | 2026-10-04 |\n\
                     | 2  | 2026-10-04 |\n\
                     | 3  | 2026-02-02 |\n\
                     +----+------------+",
                ),
                (
                    "SELECT id FROM rows WHERE d + INTERVAL '1 day' = DATE '2026-10-03' ORDER BY id",
                    "date(`rows`.`d`, '+1 days')",
                    "+----+\n\
                     | id |\n\
                     +----+\n\
                     | 1  |\n\
                     | 2  |\n\
                     +----+",
                ),
                (
                    "SELECT id FROM rows WHERE d + INTERVAL '1 day' <= DATE '2026-10-03' ORDER BY id",
                    "date(`rows`.`d`, '+1 days')",
                    "+----+\n\
                     | id |\n\
                     +----+\n\
                     | 1  |\n\
                     | 2  |\n\
                     | 3  |\n\
                     +----+",
                ),
                (
                    "SELECT id FROM rows WHERE d + INTERVAL '1 day' > DATE '2026-10-03' ORDER BY id",
                    "date(`rows`.`d`, '+1 days')",
                    "++\n\
                     ++",
                ),
            ] {
                check(&ctx, sql, pushed, expected).await;
            }
        }

        /// An unaliased shifted date keeps the plan's own name for it: the marker does
        /// not change the schema the federated statement is checked against.
        #[tokio::test]
        async fn an_unaliased_shifted_date_keeps_its_name() {
            let ctx = session().await;
            let sql = "SELECT d + INTERVAL '1 day' FROM rows ORDER BY 1";
            let (rows, display) = run(&ctx, sql).await;
            assert!(
                display.contains("date(`rows`.`d`, '+1 days')"),
                "{sql}: expected the SQL sent to SQLite to shift through `date()`:\n{display}"
            );
            let batches = ctx
                .sql(sql)
                .await
                .expect("plans")
                .collect()
                .await
                .expect("runs");
            let name = batches[0].schema().field(0).name().clone();
            assert!(
                name.starts_with("rows.d + IntervalMonthDayNano"),
                "the column keeps the plan's name for the expression, got `{name}`\n{rows}"
            );
            let shifted: Vec<i32> = batches
                .iter()
                .flat_map(|batch| {
                    batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<Date32Array>()
                        .expect("a date column")
                        .values()
                        .to_vec()
                })
                .collect();
            assert_eq!(
                shifted,
                [days(2026, 2, 1), days(2026, 10, 3), days(2026, 10, 3)]
                    .map(|day| i32::try_from(day).expect("fits"))
            );
        }

        /// A shifted date that only orders the rows, without being projected, orders
        /// them as a day.
        #[tokio::test]
        async fn a_shifted_date_orders_the_rows() {
            let ctx = session().await;
            check(
                &ctx,
                "SELECT id FROM rows ORDER BY d + INTERVAL '1 day', id",
                "date(`rows`.`d`, '+1 days')",
                "+----+\n| id |\n+----+\n| 3  |\n| 1  |\n| 2  |\n+----+",
            )
            .await;
        }

        /// A shifted date inside a window's `ORDER BY` is still found by the unparser
        /// (which looks a window expression up by its name) and orders the rows.
        #[tokio::test]
        async fn a_shifted_date_orders_a_window() {
            let ctx = session().await;
            check(
                &ctx,
                "SELECT id, row_number() OVER (ORDER BY d + INTERVAL '1 day', id) AS rn FROM rows ORDER BY id",
                "date(`rows`.`d`, '+1 days')",
                "+----+----+\n\
                 | id | rn |\n\
                 +----+----+\n\
                 | 1  | 2  |\n\
                 | 2  | 3  |\n\
                 | 3  | 1  |\n\
                 +----+----+",
            )
            .await;
        }

        /// A shifted date inside an `EXISTS` subquery, which is a plan of its own that
        /// the statement keeps, is marked too and still compares equal to a date
        /// literal there.
        #[tokio::test]
        async fn a_shifted_date_inside_a_subquery_stays_a_date() {
            let ctx = session().await;
            check(
                &ctx,
                "SELECT r.id FROM rows r WHERE EXISTS (SELECT 1 FROM rows s WHERE s.id = r.id AND s.d + INTERVAL '1 day' = DATE '2026-10-03') ORDER BY r.id",
                "date(",
                "+----+\n| id |\n+----+\n| 1  |\n| 2  |\n+----+",
            )
            .await;
        }

        /// A statement whose only table is inside a subquery is marked too: the
        /// scan's own optimizer is never gathered for it, so the executor's hook is
        /// what marks it.
        #[tokio::test]
        async fn a_shifted_date_in_a_subquery_only_statement_stays_a_date() {
            let ctx = session().await;
            for (sql, expected) in [
                (
                    "SELECT EXISTS (SELECT 1 FROM rows WHERE d + INTERVAL '1 day' = DATE '2026-10-03') AS found",
                    "+-------+\n| found |\n+-------+\n| true  |\n+-------+",
                ),
                (
                    "SELECT (SELECT count(*) FROM rows WHERE d + INTERVAL '1 day' = DATE '2026-10-03') AS n",
                    "+---+\n| n |\n+---+\n| 2 |\n+---+",
                ),
            ] {
                check(&ctx, sql, "date(", expected).await;
            }
        }

        /// A `DATE` cast to a timestamp is a timestamp operand: the hour it is shifted
        /// by is kept, as the plan's type says, even though SQLite stores the cast
        /// value as the same date-only text.
        #[tokio::test]
        async fn a_date_cast_to_a_timestamp_shifts_as_a_timestamp() {
            let ctx = session().await;
            check(
                &ctx,
                "SELECT id, CAST(d AS TIMESTAMP) + INTERVAL '1 hour' AS shifted FROM rows ORDER BY id",
                "datetime(CAST(`rows`.`d` AS TEXT), '+1 hours')",
                "+----+---------------------+\n\
                 | id | shifted             |\n\
                 +----+---------------------+\n\
                 | 1  | 2026-10-02T01:00:00 |\n\
                 | 2  | 2026-10-02T01:00:00 |\n\
                 | 3  | 2026-01-31T01:00:00 |\n\
                 +----+---------------------+",
            )
            .await;
        }
    }
}
