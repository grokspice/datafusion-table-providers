//! Translation of `DataFusion` filter expressions into MongoDB query documents.
//!
//! A pushed-down filter has to select exactly the documents whose row — as
//! [`super::arrow::mongo_docs_to_arrow`] converts it — satisfies the SQL
//! predicate, or, when it is reported inexact, a superset of them that
//! `DataFusion` then filters again. MongoDB matches documents differently from
//! how SQL evaluates the converted rows, so a literal operator-for-operator
//! translation selects the wrong documents:
//!
//! * `$ne`, `$nin`, `$not` and `$nor` match a document whose field is null or
//!   missing, where SQL's three-valued logic makes the predicate NULL.
//! * Comparisons are bracketed by BSON type (`{f: {$gt: "6"}}` never matches the
//!   number 7), while the conversion renders a non-string value into a `Utf8`
//!   column — an `ObjectId` as its hex string, a document as JSON.
//! * A comparison matches an array when any element matches, while the
//!   conversion turns an array in a scalar column into NULL.
//! * Numeric comparisons cross `int`, `long` and `double`, while the conversion
//!   keeps only the BSON types the column's Arrow type accepts.
//! * NaN compares false against every number, while Arrow orders it above them
//!   (or, with the sign bit set, below them); `-0.0` equals `0.0`, while Arrow
//!   orders it below.
//!
//! So each comparison is guarded by the BSON types its column accepts, arrays
//! are excluded wherever the conversion nulls them, and a negation is pushed
//! down to the comparisons instead of being wrapped in `$nor` or `$not`.

use std::collections::HashSet;
use std::fmt::Write as _;

use chrono::{DateTime, Utc};
use datafusion::arrow::datatypes::{DataType, Schema, TimeUnit};
use datafusion::logical_expr::expr::{Between, Cast, InList, Like, TryCast};
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use datafusion::scalar::ScalarValue;
use mongodb::bson::oid::ObjectId;
use mongodb::bson::{
    doc, Bson, DateTime as BsonDateTime, Document, Regex as BsonRegex, Timestamp as BsonTimestamp,
};

/// A MongoDB query document translated from a SQL predicate.
#[derive(Debug, Clone, PartialEq)]
pub struct MongoFilter {
    /// The query document to pass to `find`.
    pub document: Document,
    /// Whether `document` selects exactly the rows the predicate keeps. When
    /// `false` it selects a superset of them, and the predicate still has to be
    /// applied to what MongoDB returns.
    pub exact: bool,
}

impl MongoFilter {
    fn exact(document: Document) -> Self {
        Self {
            document,
            exact: true,
        }
    }

    fn inexact(document: Document) -> Self {
        Self {
            document,
            exact: false,
        }
    }

    /// Whether this selects exactly no row: a predicate that is never true.
    fn is_nothing(&self) -> bool {
        self.exact && self.document == nothing()
    }
}

/// What a collection's columns are, beyond their Arrow types.
#[derive(Debug, Clone, Default)]
pub struct FilterScope {
    /// Columns that are not a MongoDB field of the same name, such as a JSON
    /// nesting catch-all.
    pub unpushable_columns: HashSet<String>,
    /// How many levels of embedded documents are flattened into dotted
    /// columns, or `None` when the flattening is not depth based.
    pub unnest_depth: Option<usize>,
    /// Whether rows are read from whole documents rather than from a
    /// projection of the columns. Unnesting a whole document reads a field
    /// whose name contains a dot into the dotted column of that name, and no
    /// query path addresses such a field, so a dotted column is then not
    /// pushed down.
    pub whole_documents: bool,
    /// Whether the query can compare strings by code point, as SQL does: the
    /// collection has no collation, or one the query replaces with the simple
    /// collation. A collation can hold unequal strings equal and order them
    /// differently, which narrows `$nin` and the range operators, so without
    /// this only the string predicates a collation cannot narrow are pushed
    /// down.
    pub code_point_strings: bool,
}

/// Translates a filter over a collection whose rows have `schema`, or returns
/// `None` when it cannot be expressed as a MongoDB query.
#[must_use]
pub fn translate_filter(expr: &Expr, schema: &Schema, scope: &FilterScope) -> Option<MongoFilter> {
    Translator { schema, scope }.translate(expr, false)
}

/// Every BSON type alias `$type` accepts, except `string` and `null`.
const NON_STRING_TYPES: &[&str] = &[
    "double",
    "object",
    "array",
    "binData",
    "undefined",
    "objectId",
    "bool",
    "date",
    "regex",
    "dbPointer",
    "javascript",
    "symbol",
    "javascriptWithScope",
    "int",
    "timestamp",
    "long",
    "decimal",
    "minKey",
    "maxKey",
];

const NUMBER_TYPES: &[&str] = &["double", "int", "long"];
const INSTANT_TYPES: &[&str] = &["date", "timestamp"];

/// The largest magnitude below which every `int` and `long` converts to an
/// `f64` without rounding, so that comparing the converted value with a
/// literal agrees with MongoDB comparing the exact one.
const EXACT_F64_INTEGERS: f64 = 9_007_199_254_740_992.0; // 2^53

const MILLIS_PER_DAY: i128 = 86_400_000;

/// How a column's rows are produced from its field's BSON values, which
/// decides the MongoDB conditions that select a given SQL value.
#[derive(Debug, Clone, Copy)]
enum Kind {
    /// `Utf8`/`LargeUtf8`: a string as itself, an `ObjectId` as its hex string,
    /// an embedded document as JSON, and anything else as its display string.
    Utf8,
    Boolean,
    /// An integer column: the value of an accepted BSON integer within range.
    Integer {
        types: &'static [&'static str],
        min: i128,
        max: i128,
    },
    /// `double`, `int` and `long` values, as `f64`.
    Float64,
    /// A timestamp in milliseconds or finer, or `Date64`: a `date` as its
    /// milliseconds, a BSON `timestamp` as its seconds, each only within the
    /// milliseconds from `min` to `max` the unit can represent.
    Instant {
        min: i128,
        max: i128,
    },
    /// The UTC day of a `date`.
    Date32,
    Binary,
    /// `List<Utf8>`: an array, whatever its elements.
    List,
}

impl Kind {
    fn of(data_type: &DataType) -> Option<Self> {
        let integer = |types, min: i128, max: i128| Some(Self::Integer { types, min, max });
        match data_type {
            DataType::Utf8 | DataType::LargeUtf8 => Some(Self::Utf8),
            DataType::Boolean => Some(Self::Boolean),
            DataType::Int8 => integer(&["int"], i8::MIN.into(), i8::MAX.into()),
            DataType::Int16 => integer(&["int"], i16::MIN.into(), i16::MAX.into()),
            DataType::Int32 => integer(&["int", "long"], i32::MIN.into(), i32::MAX.into()),
            DataType::Int64 => integer(&["int", "long"], i64::MIN.into(), i64::MAX.into()),
            DataType::UInt8 => integer(&["int"], 0, u8::MAX.into()),
            DataType::UInt16 => integer(&["int"], 0, u16::MAX.into()),
            DataType::UInt32 => integer(&["int", "long"], 0, u32::MAX.into()),
            DataType::UInt64 => integer(&["int", "long"], 0, i64::MAX.into()),
            DataType::Float64 => Some(Self::Float64),
            DataType::Timestamp(TimeUnit::Millisecond, _) | DataType::Date64 => {
                Some(Self::Instant {
                    min: i64::MIN.into(),
                    max: i64::MAX.into(),
                })
            }
            // The conversion nulls a `date` whose milliseconds overflow the unit.
            DataType::Timestamp(TimeUnit::Microsecond, _) => Some(Self::instant_per_milli(1_000)),
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                Some(Self::instant_per_milli(1_000_000))
            }
            DataType::Date32 => Some(Self::Date32),
            DataType::Binary | DataType::LargeBinary => Some(Self::Binary),
            DataType::List(_) => Some(Self::List),
            _ => None,
        }
    }

    /// An instant in a unit `per_milli` times finer than a millisecond, whose
    /// `i64` holds only the milliseconds that do not overflow once scaled.
    fn instant_per_milli(per_milli: i64) -> Self {
        Self::Instant {
            min: (i64::MIN / per_milli).into(),
            max: (i64::MAX / per_milli).into(),
        }
    }
}

/// A column resolved to the MongoDB field it is read from.
struct Field<'a> {
    path: &'a str,
    kind: Kind,
    /// Whether an embedded document at `path` is flattened away by unnesting,
    /// which leaves the column NULL, rather than rendered into it.
    documents_flattened: bool,
}

impl Field<'_> {
    /// The proper prefixes of a dotted path. Each must hold a document rather
    /// than an array for the path to reach the value the flattened row holds:
    /// MongoDB traverses arrays along a path, unnesting does not.
    fn parents(&self) -> impl Iterator<Item = &str> {
        self.path.match_indices('.').map(|(i, _)| &self.path[..i])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cmp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

impl Cmp {
    fn from_operator(op: Operator) -> Option<Self> {
        Some(match op {
            Operator::Eq => Self::Eq,
            Operator::NotEq => Self::NotEq,
            Operator::Lt => Self::Lt,
            Operator::LtEq => Self::LtEq,
            Operator::Gt => Self::Gt,
            Operator::GtEq => Self::GtEq,
            _ => return None,
        })
    }

    /// The comparison with its operands exchanged: `a < b` is `b > a`.
    fn swapped(self) -> Self {
        match self {
            Self::Lt => Self::Gt,
            Self::LtEq => Self::GtEq,
            Self::Gt => Self::Lt,
            Self::GtEq => Self::LtEq,
            other => other,
        }
    }

    /// The comparison that is true exactly when this one is false. Both are
    /// NULL for a NULL operand, which is what lets a negation be pushed down to
    /// a comparison without changing what a NULL row evaluates to.
    fn negated(self) -> Self {
        match self {
            Self::Eq => Self::NotEq,
            Self::NotEq => Self::Eq,
            Self::Lt => Self::GtEq,
            Self::LtEq => Self::Gt,
            Self::Gt => Self::LtEq,
            Self::GtEq => Self::Lt,
        }
    }

    fn operator(self) -> &'static str {
        match self {
            Self::Eq => "$eq",
            Self::NotEq => "$ne",
            Self::Lt => "$lt",
            Self::LtEq => "$lte",
            Self::Gt => "$gt",
            Self::GtEq => "$gte",
        }
    }
}

/// A set of integers: the values a comparison over an integer domain keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntSet {
    Empty,
    /// Every integer from the first bound to the second, inclusive.
    Range(i128, i128),
    /// Every integer but one.
    AllBut(i128),
}

impl IntSet {
    const ALL: Self = Self::Range(i128::MIN, i128::MAX);

    /// The integers `v` for which `v <cmp> k` holds.
    fn compare(cmp: Cmp, k: i128) -> Self {
        match cmp {
            Cmp::Eq => Self::Range(k, k),
            Cmp::NotEq => Self::AllBut(k),
            Cmp::Lt => Self::Range(i128::MIN, k.saturating_sub(1)),
            Cmp::LtEq => Self::Range(i128::MIN, k),
            Cmp::Gt => Self::Range(k.saturating_add(1), i128::MAX),
            Cmp::GtEq => Self::Range(k, i128::MAX),
        }
    }

    /// The integers `v` for which `v <cmp> num/den` holds, `den` positive.
    fn compare_ratio(cmp: Cmp, num: i128, den: i128) -> Self {
        let floor = num.div_euclid(den);
        let whole = num.rem_euclid(den) == 0;
        let ceil = if whole { floor } else { floor + 1 };
        match cmp {
            Cmp::Eq if whole => Self::Range(floor, floor),
            Cmp::Eq => Self::Empty,
            Cmp::NotEq if whole => Self::AllBut(floor),
            Cmp::NotEq => Self::ALL,
            Cmp::Lt => Self::Range(i128::MIN, ceil.saturating_sub(1)),
            Cmp::LtEq => Self::Range(i128::MIN, floor),
            Cmp::Gt => Self::Range(floor.saturating_add(1), i128::MAX),
            Cmp::GtEq => Self::Range(ceil, i128::MAX),
        }
    }

    /// The integers `v` for which `(v as f64) <cmp> x` holds, for `x` neither
    /// NaN nor `-0.0` and, when `v` can reach beyond 2^53, below it in
    /// magnitude.
    fn compare_float(cmp: Cmp, x: f64) -> Self {
        // `as` saturates, which is what an infinite bound needs.
        let floor = x.floor() as i128;
        let ceil = x.ceil() as i128;
        let whole = x.fract() == 0.0;
        match cmp {
            Cmp::Eq if whole => Self::Range(floor, floor),
            Cmp::Eq => Self::Empty,
            Cmp::NotEq if whole => Self::AllBut(floor),
            Cmp::NotEq => Self::ALL,
            Cmp::Lt => Self::Range(i128::MIN, ceil.saturating_sub(1)),
            Cmp::LtEq => Self::Range(i128::MIN, floor),
            Cmp::Gt => Self::Range(floor.saturating_add(1), i128::MAX),
            Cmp::GtEq => Self::Range(ceil, i128::MAX),
        }
    }

    /// This set limited to `min..=max`.
    fn within(self, min: i128, max: i128) -> Self {
        match self {
            Self::Empty => Self::Empty,
            Self::Range(lo, hi) => {
                let (lo, hi) = (lo.max(min), hi.min(max));
                if lo > hi {
                    Self::Empty
                } else {
                    Self::Range(lo, hi)
                }
            }
            Self::AllBut(k) if k < min || k > max => Self::Range(min, max),
            Self::AllBut(k) => Self::AllBut(k),
        }
    }

    /// The set of `v` with `v * factor` in this set: the values of a column
    /// stored in units `factor` times as coarse as the set's.
    fn scaled_down(self, factor: i128) -> Self {
        match self {
            Self::Empty => Self::Empty,
            Self::Range(lo, hi) => {
                let lo = if lo == i128::MIN {
                    lo
                } else {
                    lo.div_euclid(factor) + i128::from(lo.rem_euclid(factor) != 0)
                };
                let hi = if hi == i128::MAX {
                    hi
                } else {
                    hi.div_euclid(factor)
                };
                if lo > hi {
                    Self::Empty
                } else {
                    Self::Range(lo, hi)
                }
            }
            Self::AllBut(k) if k.rem_euclid(factor) == 0 => Self::AllBut(k.div_euclid(factor)),
            Self::AllBut(_) => Self::ALL,
        }
    }
}

enum Numeric {
    Int(i128),
    Float(f64),
}

fn numeric(value: &ScalarValue) -> Option<Numeric> {
    Some(match value {
        ScalarValue::Int8(Some(v)) => Numeric::Int((*v).into()),
        ScalarValue::Int16(Some(v)) => Numeric::Int((*v).into()),
        ScalarValue::Int32(Some(v)) => Numeric::Int((*v).into()),
        ScalarValue::Int64(Some(v)) => Numeric::Int((*v).into()),
        ScalarValue::UInt8(Some(v)) => Numeric::Int((*v).into()),
        ScalarValue::UInt16(Some(v)) => Numeric::Int((*v).into()),
        ScalarValue::UInt32(Some(v)) => Numeric::Int((*v).into()),
        ScalarValue::UInt64(Some(v)) => Numeric::Int((*v).into()),
        ScalarValue::Float32(Some(v)) => Numeric::Float((*v).into()),
        ScalarValue::Float64(Some(v)) => Numeric::Float(*v),
        _ => return None,
    })
}

/// An instant literal as an exact fraction of milliseconds since the epoch.
fn instant_millis(value: &ScalarValue) -> Option<(i128, i128)> {
    Some(match value {
        ScalarValue::TimestampSecond(Some(v), _) => (i128::from(*v) * 1_000, 1),
        ScalarValue::TimestampMillisecond(Some(v), _) | ScalarValue::Date64(Some(v)) => {
            ((*v).into(), 1)
        }
        ScalarValue::TimestampMicrosecond(Some(v), _) => ((*v).into(), 1_000),
        ScalarValue::TimestampNanosecond(Some(v), _) => ((*v).into(), 1_000_000),
        ScalarValue::Date32(Some(days)) => (i128::from(*days) * MILLIS_PER_DAY, 1),
        _ => return None,
    })
}

fn string(value: &ScalarValue) -> Option<&str> {
    match value {
        ScalarValue::Utf8(Some(s))
        | ScalarValue::LargeUtf8(Some(s))
        | ScalarValue::Utf8View(Some(s)) => Some(s),
        _ => None,
    }
}

/// The `ObjectId` whose hex rendering is `s`. The conversion renders lowercase
/// hex, so an uppercase spelling names no row.
fn object_id(s: &str) -> Option<ObjectId> {
    let lowercase_hex = s.len() == 24 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    lowercase_hex.then(|| ObjectId::parse_str(s).ok()).flatten()
}

fn integer(v: i128) -> Bson {
    match i32::try_from(v) {
        Ok(v) => Bson::Int32(v),
        // Every bound this is given lies within the column's range, which is
        // within `i64`'s.
        Err(_) => Bson::Int64(i64::try_from(v).unwrap_or(if v < 0 { i64::MIN } else { i64::MAX })),
    }
}

fn millis(v: i128) -> Bson {
    Bson::DateTime(BsonDateTime::from_millis(
        i64::try_from(v).unwrap_or(if v < 0 { i64::MIN } else { i64::MAX }),
    ))
}

fn types(types: &[&str]) -> Bson {
    match types {
        [one] => Bson::String((*one).to_string()),
        many => Bson::Array(
            many.iter()
                .map(|t| Bson::String((*t).to_string()))
                .collect(),
        ),
    }
}

/// Matches no document. `$in` with no values matches nothing, a missing
/// field included, which matters because a view can project `_id` away.
fn nothing() -> Document {
    doc! { "_id": { "$in": [] } }
}

fn all_of(documents: Vec<Document>) -> Document {
    combine("$and", documents)
}

fn any_of(documents: Vec<Document>) -> Document {
    combine("$or", documents)
}

fn none_of(document: Document) -> Document {
    doc! { "$nor": [document] }
}

/// Joins `documents` under `operator`, splicing in the operands of any that
/// are already joined by it.
fn combine(operator: &str, documents: Vec<Document>) -> Document {
    let mut operands = Vec::with_capacity(documents.len());
    for document in documents {
        let nested = document.len() == 1 && document.contains_key(operator);
        match document.get_array(operator) {
            Ok(inner) if nested => operands.extend(inner.iter().cloned()),
            _ => operands.push(Bson::Document(document)),
        }
    }
    match operands.len() {
        1 => match operands.pop() {
            Some(Bson::Document(document)) => document,
            _ => Document::new(),
        },
        _ => doc! { operator: operands },
    }
}

fn and(left: Option<MongoFilter>, right: Option<MongoFilter>) -> Option<MongoFilter> {
    // Where either side is never true, so is the conjunction.
    if let Some(never) = [&left, &right]
        .into_iter()
        .flatten()
        .find(|filter| filter.is_nothing())
    {
        return Some(never.clone());
    }
    match (left, right) {
        (Some(l), Some(r)) => Some(MongoFilter {
            document: all_of(vec![l.document, r.document]),
            exact: l.exact && r.exact,
        }),
        // A conjunction keeps a subset of either side's rows, so one side
        // alone selects a superset of them.
        (Some(one), None) | (None, Some(one)) => Some(MongoFilter::inexact(one.document)),
        (None, None) => None,
    }
}

fn or(left: Option<MongoFilter>, right: Option<MongoFilter>) -> Option<MongoFilter> {
    let (l, r) = (left?, right?);
    // A side that is never true adds no row.
    if l.is_nothing() {
        return Some(r);
    }
    if r.is_nothing() {
        return Some(l);
    }
    Some(MongoFilter {
        document: any_of(vec![l.document, r.document]),
        exact: l.exact && r.exact,
    })
}

/// The value of a literal operand, folding a cast of one.
fn literal(expr: &Expr) -> Option<ScalarValue> {
    match expr {
        Expr::Literal(value, _) => Some(value.clone()),
        Expr::Cast(Cast { expr, field }) | Expr::TryCast(TryCast { expr, field }) => {
            match expr.as_ref() {
                Expr::Literal(value, _) => value.cast_to(field.data_type()).ok(),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Whether casting a column from `from` to `to` keeps every value, so that
/// comparing the cast value is comparing the stored one.
fn preserves_values(from: &DataType, to: &DataType) -> bool {
    use DataType::{
        Float64, Int16, Int32, Int64, Int8, LargeUtf8, Timestamp, UInt16, UInt32, UInt64, UInt8,
        Utf8, Utf8View,
    };
    match (from, to) {
        _ if from == to => true,
        (Utf8 | LargeUtf8, Utf8 | LargeUtf8 | Utf8View) => true,
        (Int8, Int16 | Int32 | Int64 | Float64)
        | (Int16, Int32 | Int64 | Float64)
        | (Int32, Int64 | Float64)
        | (UInt8, Int16 | Int32 | Int64 | UInt16 | UInt32 | UInt64 | Float64)
        | (UInt16, Int32 | Int64 | UInt32 | UInt64 | Float64)
        | (UInt32, Int64 | UInt64 | Float64)
        // Rounds beyond 2^53, which the comparison of such a cast accounts for.
        | (Int64 | UInt64, Float64) => true,
        // A finer unit overflows for a distant instant, which a cast turns into
        // an error and `TRY_CAST` into NULL, and a coarser one truncates. A zone
        // added to a timestamp without one reads it as wall-clock time there,
        // through chrono, whose range is narrower than a BSON date's, so even
        // UTC can make it NULL. Otherwise only the zone it displays in changes.
        (Timestamp(from_unit, from_tz), Timestamp(to_unit, to_tz)) => {
            from_unit == to_unit && (from_tz.is_some() || to_tz.is_none())
        }
        _ => false,
    }
}

/// Whether `name` can be addressed as a field path in a query document. A
/// leading `$` would be read as an operator, and a NUL cannot be encoded.
fn is_addressable(name: &str) -> bool {
    !name.contains('\0')
        && name
            .split('.')
            .all(|segment| !segment.is_empty() && !segment.starts_with('$'))
}

struct Translator<'a> {
    schema: &'a Schema,
    scope: &'a FilterScope,
}

impl Translator<'_> {
    /// The rows for which `expr` is true, or, when `negated`, false.
    fn translate(&self, expr: &Expr, negated: bool) -> Option<MongoFilter> {
        match expr {
            Expr::Not(inner) => self.translate(inner, !negated),
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => match op {
                Operator::And | Operator::Or => {
                    let left = self.translate(left, negated);
                    let right = self.translate(right, negated);
                    // De Morgan: a negated conjunction is a disjunction of the
                    // negations, which holds in three-valued logic too.
                    if (*op == Operator::And) != negated {
                        and(left, right)
                    } else {
                        or(left, right)
                    }
                }
                Operator::IsDistinctFrom | Operator::IsNotDistinctFrom => {
                    let distinct = (*op == Operator::IsDistinctFrom) != negated;
                    self.distinct(left, right, distinct)
                }
                op => {
                    let cmp = Cmp::from_operator(*op)?;
                    let (field, value, cmp) = match (self.field(left), self.field(right)) {
                        (Some(field), None) => (field, literal(right)?, cmp),
                        (None, Some(field)) => (field, literal(left)?, cmp.swapped()),
                        _ => return None,
                    };
                    let cmp = if negated { cmp.negated() } else { cmp };
                    self.compare(&field, cmp, &value)
                }
            },
            // A boolean column used as a predicate is true exactly when it holds `true`.
            Expr::Column(_) | Expr::Cast(_) => {
                let field = self.field(expr)?;
                matches!(field.kind, Kind::Boolean)
                    .then(|| self.compare(&field, Cmp::Eq, &ScalarValue::Boolean(Some(!negated))))
                    .flatten()
            }
            Expr::IsNull(inner) => {
                let field = self.field(inner)?;
                Some(if negated {
                    self.not_null(&field)
                } else {
                    self.is_null(&field)
                })
            }
            Expr::IsNotNull(inner) => {
                let field = self.field(inner)?;
                Some(if negated {
                    self.is_null(&field)
                } else {
                    self.not_null(&field)
                })
            }
            Expr::IsTrue(inner) => self.truth(inner, true, negated),
            Expr::IsFalse(inner) => self.truth(inner, false, negated),
            Expr::IsNotTrue(inner) => self.truth(inner, true, !negated),
            Expr::IsNotFalse(inner) => self.truth(inner, false, !negated),
            Expr::Between(Between {
                expr,
                negated: outside,
                low,
                high,
            }) => {
                let field = self.field(expr)?;
                let (low, high) = (literal(low)?, literal(high)?);
                if low.is_null() || high.is_null() {
                    return None;
                }
                if *outside == negated {
                    and(
                        self.compare(&field, Cmp::GtEq, &low),
                        self.compare(&field, Cmp::LtEq, &high),
                    )
                } else {
                    or(
                        self.compare(&field, Cmp::Lt, &low),
                        self.compare(&field, Cmp::Gt, &high),
                    )
                }
            }
            Expr::InList(InList {
                expr,
                list,
                negated: excluded,
            }) => {
                let field = self.field(expr)?;
                let values = list.iter().map(literal).collect::<Option<Vec<_>>>()?;
                if values.is_empty() {
                    return None;
                }
                if *excluded == negated {
                    // A NULL element can make the membership NULL, never true.
                    let values: Vec<_> = values.into_iter().filter(|v| !v.is_null()).collect();
                    if values.is_empty() {
                        return Some(MongoFilter::exact(nothing()));
                    }
                    self.in_list(&field, &values)
                } else {
                    // With a NULL element, NOT IN is never true.
                    if values.iter().any(ScalarValue::is_null) {
                        return None;
                    }
                    self.not_in_list(&field, &values)
                }
            }
            Expr::Like(like) => self.like(like, negated),
            // A NULL used as a predicate, which is what `DataFusion` leaves of
            // `x = NULL` in a simplified `x IN (1, NULL)`, is neither true nor
            // false, so it keeps no row either way.
            Expr::Literal(ScalarValue::Boolean(None) | ScalarValue::Null, _) => {
                Some(MongoFilter::exact(nothing()))
            }
            _ => None,
        }
    }

    /// Resolves a column, or a cast of one that keeps its values.
    fn field<'e>(&self, expr: &'e Expr) -> Option<Field<'e>> {
        let (column, stored) = match expr {
            Expr::Column(column) => (column, self.column_type(&column.name)?),
            Expr::Cast(Cast { expr, field }) | Expr::TryCast(TryCast { expr, field }) => {
                let Expr::Column(column) = expr.as_ref() else {
                    return None;
                };
                let stored = self.column_type(&column.name)?;
                if !preserves_values(stored, field.data_type()) {
                    return None;
                }
                (column, stored)
            }
            _ => return None,
        };
        let path = column.name.as_str();
        if self.scope.unpushable_columns.contains(path) || !is_addressable(path) {
            return None;
        }
        // With depth-based unnesting a dotted column is a flattened path; without
        // it, a field whose name has a dot, which a query path cannot address.
        let unnest_depth = self.scope.unnest_depth?;
        let depth = path.matches('.').count();
        if depth > unnest_depth || (depth > 0 && self.scope.whole_documents) {
            return None;
        }
        Some(Field {
            path,
            kind: Kind::of(stored)?,
            documents_flattened: depth < unnest_depth,
        })
    }

    fn column_type(&self, name: &str) -> Option<&DataType> {
        self.schema
            .field_with_name(name)
            .ok()
            .map(|f| f.data_type())
    }

    /// `spec` applied to `field`, requiring each parent along its path to be a
    /// document rather than an array.
    fn at(&self, field: &Field<'_>, spec: impl Into<Bson>) -> Document {
        let leaf = doc! { field.path: spec.into() };
        let mut guards: Vec<Document> = field
            .parents()
            .map(|parent| doc! { parent: { "$not": { "$type": "array" } } })
            .collect();
        if guards.is_empty() {
            return leaf;
        }
        guards.push(leaf);
        all_of(guards)
    }

    /// A value of one of `accepted`, not an array, meeting `conditions`.
    fn scalar(&self, field: &Field<'_>, accepted: &[&str], conditions: Document) -> Document {
        let mut spec = doc! { "$type": types(accepted) };
        spec.extend(conditions);
        spec.insert("$not", doc! { "$type": "array" });
        self.at(field, spec)
    }

    /// The rows whose column is not NULL.
    fn not_null(&self, field: &Field<'_>) -> MongoFilter {
        let document = match field.kind {
            Kind::Utf8 => {
                // Every value but null renders, arrays of nulls included; an
                // embedded document does not when unnesting flattens it away.
                let absent: &[&str] = if field.documents_flattened {
                    &["null", "object"]
                } else {
                    &["null"]
                };
                any_of(vec![
                    self.at(field, doc! { "$type": "array" }),
                    self.at(
                        field,
                        doc! { "$exists": true, "$not": { "$type": types(absent) } },
                    ),
                ])
            }
            Kind::Boolean => self.scalar(field, &["bool"], Document::new()),
            Kind::Integer { types, min, max } => {
                self.integers(field, types, min, max, IntSet::ALL.within(min, max))
            }
            Kind::Float64 => self.scalar(field, NUMBER_TYPES, Document::new()),
            Kind::Instant { min, max } if min <= i64::MIN.into() && max >= i64::MAX.into() => {
                self.scalar(field, INSTANT_TYPES, Document::new())
            }
            Kind::Instant { min, max } => self.instants(field, IntSet::ALL, min, max),
            Kind::Date32 => self.days(field, IntSet::ALL),
            Kind::Binary => self.scalar(field, &["binData"], Document::new()),
            Kind::List => self.at(field, doc! { "$type": "array" }),
        };
        MongoFilter::exact(document)
    }

    /// The rows whose column is NULL.
    fn is_null(&self, field: &Field<'_>) -> MongoFilter {
        // `not_null` is exact and never NULL itself, so its complement is too.
        MongoFilter::exact(none_of(self.not_null(field).document))
    }

    /// `inner IS value`, or when `negated`, `inner IS NOT value`.
    fn truth(&self, inner: &Expr, value: bool, negated: bool) -> Option<MongoFilter> {
        let holds = self.translate(inner, !value)?;
        if !negated {
            return Some(holds);
        }
        holds
            .exact
            .then(|| MongoFilter::exact(none_of(holds.document)))
    }

    fn distinct(&self, left: &Expr, right: &Expr, distinct: bool) -> Option<MongoFilter> {
        let (field, value) = match (self.field(left), self.field(right)) {
            (Some(field), None) => (field, literal(right)?),
            (None, Some(field)) => (field, literal(left)?),
            _ => return None,
        };
        if value.is_null() {
            return Some(if distinct {
                self.not_null(&field)
            } else {
                self.is_null(&field)
            });
        }
        // Not distinct from a value is equal to it: NULL is not.
        let equal = self.compare(&field, Cmp::Eq, &value)?;
        if !distinct {
            return Some(equal);
        }
        if equal.exact {
            return Some(MongoFilter::exact(none_of(equal.document)));
        }
        let unequal = self.compare(&field, Cmp::NotEq, &value)?;
        Some(MongoFilter::inexact(any_of(vec![
            unequal.document,
            self.is_null(&field).document,
        ])))
    }

    fn compare(&self, field: &Field<'_>, cmp: Cmp, value: &ScalarValue) -> Option<MongoFilter> {
        if value.is_null() {
            return None;
        }
        match field.kind {
            Kind::Utf8 => self.compare_utf8(field, cmp, string(value)?),
            Kind::Boolean => self.compare_boolean(field, cmp, value),
            Kind::Integer { types, min, max } => {
                let set = match numeric(value)? {
                    Numeric::Int(k) => IntSet::compare(cmp, k),
                    Numeric::Float(x) => {
                        let beyond_exact = x.is_finite() && x.abs() >= EXACT_F64_INTEGERS;
                        if beyond_exact && (min < -(1 << 53) || max > 1 << 53) {
                            return None;
                        }
                        integer_float_set(cmp, x)
                    }
                };
                Some(MongoFilter::exact(self.integers(
                    field,
                    types,
                    min,
                    max,
                    set.within(min, max),
                )))
            }
            Kind::Float64 => self.compare_float(field, cmp, value),
            Kind::Instant { min, max } => {
                let (num, den) = instant_millis(value)?;
                let set = IntSet::compare_ratio(cmp, num, den);
                Some(MongoFilter::exact(self.instants(field, set, min, max)))
            }
            Kind::Date32 => {
                let ScalarValue::Date32(Some(day)) = value else {
                    return None;
                };
                Some(MongoFilter::exact(
                    self.days(field, IntSet::compare(cmp, (*day).into())),
                ))
            }
            Kind::Binary | Kind::List => None,
        }
    }

    /// Integers of `accepted` types in `set`, which already lies within the
    /// column's range.
    fn integers(
        &self,
        field: &Field<'_>,
        accepted: &[&str],
        min: i128,
        max: i128,
        set: IntSet,
    ) -> Document {
        // The range the accepted BSON types can hold on their own.
        let (natural_min, natural_max) = if accepted.contains(&"long") {
            (i64::MIN.into(), i64::MAX.into())
        } else {
            (i32::MIN.into(), i32::MAX.into())
        };
        let mut conditions = Document::new();
        match set {
            IntSet::Empty => return nothing(),
            IntSet::Range(lo, hi) if lo == hi => {
                conditions.insert("$eq", integer(lo));
            }
            IntSet::Range(lo, hi) => {
                if lo > natural_min {
                    conditions.insert("$gte", integer(lo));
                }
                if hi < natural_max {
                    conditions.insert("$lte", integer(hi));
                }
            }
            IntSet::AllBut(k) => {
                conditions.insert("$ne", integer(k));
                if min > natural_min {
                    conditions.insert("$gte", integer(min));
                }
                if max < natural_max {
                    conditions.insert("$lte", integer(max));
                }
            }
        }
        self.scalar(field, accepted, conditions)
    }

    /// Instants whose milliseconds lie in `set` and in `min..=max`, the range
    /// the column's unit represents: a `date` by its milliseconds, a BSON
    /// `timestamp` by the milliseconds of its seconds.
    fn instants(&self, field: &Field<'_>, set: IntSet, min: i128, max: i128) -> Document {
        let set = set.within(min, max);
        let date = |set: IntSet| {
            self.bounded(field, "date", set, i64::MIN.into(), i64::MAX.into(), millis)
        };
        let mut branches = Vec::with_capacity(3);
        match set {
            // `$ne` alone would keep the dates beyond the unit's range too.
            IntSet::AllBut(k) if min > i64::MIN.into() || max < i64::MAX.into() => {
                branches.extend(date(IntSet::Range(min, k - 1).within(min, max)));
                branches.extend(date(IntSet::Range(k + 1, max).within(min, max)));
            }
            set => branches.extend(date(set)),
        }
        // Every BSON `timestamp` lies within any unit's range.
        let seconds = set.scaled_down(1_000).within(0, u32::MAX.into());
        let first = |second: i128| timestamp(second, 0);
        let last = |second: i128| timestamp(second, u32::MAX);
        let timestamps = match seconds {
            IntSet::Empty => None,
            IntSet::Range(lo, hi) => {
                let mut conditions = Document::new();
                if lo > 0 {
                    conditions.insert("$gte", first(lo));
                }
                if hi < u32::MAX.into() {
                    conditions.insert("$lte", last(hi));
                }
                Some(self.scalar(field, &["timestamp"], conditions))
            }
            IntSet::AllBut(second) => Some(any_of(vec![
                self.scalar(field, &["timestamp"], doc! { "$lt": first(second) }),
                self.scalar(field, &["timestamp"], doc! { "$gt": last(second) }),
            ])),
        };
        branches.extend(timestamps);
        if branches.is_empty() {
            return nothing();
        }
        any_of(branches)
    }

    /// `date`s whose UTC day lies in `set`. The conversion nulls a `date`
    /// chrono cannot represent, so those are excluded too.
    fn days(&self, field: &Field<'_>, set: IntSet) -> Document {
        let min: i128 = DateTime::<Utc>::MIN_UTC.timestamp_millis().into();
        let max: i128 = DateTime::<Utc>::MAX_UTC.timestamp_millis().into();
        let to_millis = |set: IntSet| match set {
            IntSet::Empty => IntSet::Empty,
            IntSet::Range(lo, hi) => IntSet::Range(
                lo.saturating_mul(MILLIS_PER_DAY),
                hi.saturating_add(1)
                    .saturating_mul(MILLIS_PER_DAY)
                    .saturating_sub(1),
            ),
            IntSet::AllBut(_) => IntSet::ALL,
        };
        match set {
            IntSet::AllBut(day) => {
                let before = to_millis(IntSet::Range(i128::MIN, day - 1)).within(min, max);
                let after = to_millis(IntSet::Range(day + 1, i128::MAX)).within(min, max);
                let branches: Vec<Document> = [before, after]
                    .into_iter()
                    .filter_map(|set| {
                        self.bounded(field, "date", set, i64::MIN.into(), i64::MAX.into(), millis)
                    })
                    .collect();
                if branches.is_empty() {
                    nothing()
                } else {
                    any_of(branches)
                }
            }
            set => self
                .bounded(
                    field,
                    "date",
                    to_millis(set).within(min, max),
                    i64::MIN.into(),
                    i64::MAX.into(),
                    millis,
                )
                .unwrap_or_else(nothing),
        }
    }

    /// Values of BSON type `accepted` in `set`, omitting a bound the type
    /// already implies. `None` when the set is empty.
    fn bounded(
        &self,
        field: &Field<'_>,
        accepted: &str,
        set: IntSet,
        natural_min: i128,
        natural_max: i128,
        encode: fn(i128) -> Bson,
    ) -> Option<Document> {
        let mut conditions = Document::new();
        match set {
            IntSet::Empty => return None,
            IntSet::Range(lo, hi) if lo == hi => {
                conditions.insert("$eq", encode(lo));
            }
            IntSet::Range(lo, hi) => {
                if lo > natural_min {
                    conditions.insert("$gte", encode(lo));
                }
                if hi < natural_max {
                    conditions.insert("$lte", encode(hi));
                }
            }
            IntSet::AllBut(k) => {
                conditions.insert("$ne", encode(k));
            }
        }
        Some(self.scalar(field, &[accepted], conditions))
    }

    fn compare_boolean(
        &self,
        field: &Field<'_>,
        cmp: Cmp,
        value: &ScalarValue,
    ) -> Option<MongoFilter> {
        let ScalarValue::Boolean(Some(b)) = value else {
            return None;
        };
        let one = |value: bool| self.scalar(field, &["bool"], doc! { "$eq": value });
        let any = || self.scalar(field, &["bool"], Document::new());
        // `false` orders before `true`.
        let document = match (cmp, *b) {
            (Cmp::Eq, v) => one(v),
            (Cmp::NotEq, v) => one(!v),
            (Cmp::Gt, false) | (Cmp::GtEq, true) => one(true),
            (Cmp::Lt, true) | (Cmp::LtEq, false) => one(false),
            (Cmp::GtEq, false) | (Cmp::LtEq, true) => any(),
            (Cmp::Gt, true) | (Cmp::Lt, false) => nothing(),
        };
        Some(MongoFilter::exact(document))
    }

    fn compare_float(
        &self,
        field: &Field<'_>,
        cmp: Cmp,
        value: &ScalarValue,
    ) -> Option<MongoFilter> {
        let x = match numeric(value)? {
            Numeric::Float(x) => x,
            Numeric::Int(k) if k.unsigned_abs() <= 1 << 53 => k as f64,
            Numeric::Int(_) => return None,
        };
        if x.is_nan() {
            // Arrow orders NaNs by sign and payload, which MongoDB cannot tell
            // apart, so only the NULL rows can be ruled out.
            return Some(MongoFilter::inexact(self.not_null(field).document));
        }
        // An `int` or `long` beyond 2^53 rounds on conversion, so comparing it
        // with a literal that large can disagree with MongoDB's exact compare.
        if x.is_finite() && x.abs() >= EXACT_F64_INTEGERS {
            return None;
        }
        let number = |cmp: Cmp| self.scalar(field, NUMBER_TYPES, doc! { cmp.operator(): x });
        // MongoDB's ordered comparisons never match NaN, which Arrow orders
        // above every number, or with its sign bit set, below them.
        let or_nan = |document: Document| {
            MongoFilter::inexact(any_of(vec![
                document,
                self.scalar(field, &["double"], doc! { "$eq": f64::NAN }),
            ]))
        };
        Some(match cmp {
            // MongoDB holds -0.0 equal to 0.0, which Arrow orders apart.
            Cmp::Eq if x == 0.0 => MongoFilter::inexact(number(Cmp::Eq)),
            Cmp::NotEq if x == 0.0 => MongoFilter::inexact(self.not_null(field).document),
            Cmp::Eq | Cmp::NotEq => MongoFilter::exact(number(cmp)),
            // A zero bound is widened to take in both zeros.
            Cmp::Lt | Cmp::LtEq if x == 0.0 => or_nan(number(Cmp::LtEq)),
            Cmp::Gt | Cmp::GtEq if x == 0.0 => or_nan(number(Cmp::GtEq)),
            ordered => or_nan(number(ordered)),
        })
    }

    fn compare_utf8(&self, field: &Field<'_>, cmp: Cmp, s: &str) -> Option<MongoFilter> {
        // A collation can only widen `$in`, which is why equality survives one.
        if cmp != Cmp::Eq && !self.scope.code_point_strings {
            return None;
        }
        let object_id = object_id(s);
        let document = match cmp {
            Cmp::Eq => {
                // An `ObjectId` renders as hex, so it equals `s` only when `s`
                // names it, and then `$in` selects it.
                let mut matches = vec![Bson::String(s.to_string())];
                matches.extend(object_id.map(Bson::ObjectId));
                any_of(vec![
                    self.at(field, doc! { "$in": matches }),
                    self.rendered(field, true),
                ])
            }
            Cmp::NotEq => self.utf8_excluding(field, &[s]),
            ordered => {
                let op = ordered.operator();
                let mut branches = vec![self.at(field, doc! { op: s })];
                branches.extend(object_id.map(|id| self.at(field, doc! { op: id })));
                branches.push(self.rendered(field, object_id.is_some()));
                any_of(branches)
            }
        };
        Some(MongoFilter::inexact(document))
    }

    /// Values a `Utf8` column renders from a type other than a string, which a
    /// string comparison cannot select, leaving out `ObjectId`s when the caller
    /// has accounted for them.
    fn rendered(&self, field: &Field<'_>, object_ids_handled: bool) -> Document {
        let rendered: Vec<&str> = NON_STRING_TYPES
            .iter()
            .copied()
            .filter(|t| !(object_ids_handled && *t == "objectId"))
            .filter(|t| !(field.documents_flattened && *t == "object"))
            .collect();
        self.at(field, doc! { "$type": types(&rendered) })
    }

    /// A superset of the rows of a `Utf8` column not equal to any of `values`.
    fn utf8_excluding(&self, field: &Field<'_>, values: &[&str]) -> Document {
        let mut excluded: Vec<Bson> = values
            .iter()
            .map(|s| Bson::String((*s).to_string()))
            .collect();
        excluded.extend(
            values
                .iter()
                .filter_map(|s| object_id(s))
                .map(Bson::ObjectId),
        );
        excluded.push(Bson::Null);
        any_of(vec![
            self.at(field, doc! { "$nin": excluded }),
            // `$nin` also rules out a symbol spelled like an excluded string,
            // and an array holding an excluded value, each of which the
            // conversion renders as a string that equals none of them.
            self.rendered(field, true),
        ])
    }

    fn in_list(&self, field: &Field<'_>, values: &[ScalarValue]) -> Option<MongoFilter> {
        match field.kind {
            Kind::Utf8 => {
                let strings = values.iter().map(string).collect::<Option<Vec<_>>>()?;
                let mut matches: Vec<Bson> = strings
                    .iter()
                    .map(|s| Bson::String((*s).to_string()))
                    .collect();
                matches.extend(
                    strings
                        .iter()
                        .filter_map(|s| object_id(s))
                        .map(Bson::ObjectId),
                );
                Some(MongoFilter::inexact(any_of(vec![
                    self.at(field, doc! { "$in": matches }),
                    self.rendered(field, true),
                ])))
            }
            Kind::Integer { types, min, max } => {
                let mut members = Vec::with_capacity(values.len());
                for value in values {
                    match numeric(value)? {
                        Numeric::Int(k) if (min..=max).contains(&k) => members.push(integer(k)),
                        Numeric::Int(_) => {}
                        Numeric::Float(_) => return self.in_list_by_comparison(field, values),
                    }
                }
                if members.is_empty() {
                    return Some(MongoFilter::exact(nothing()));
                }
                Some(MongoFilter::exact(self.scalar(
                    field,
                    types,
                    doc! { "$in": members },
                )))
            }
            _ => self.in_list_by_comparison(field, values),
        }
    }

    fn in_list_by_comparison(
        &self,
        field: &Field<'_>,
        values: &[ScalarValue],
    ) -> Option<MongoFilter> {
        let mut filters = values.iter().map(|v| self.compare(field, Cmp::Eq, v));
        let first = filters.next()?;
        filters.fold(first, or)
    }

    fn not_in_list(&self, field: &Field<'_>, values: &[ScalarValue]) -> Option<MongoFilter> {
        match field.kind {
            Kind::Utf8 if !self.scope.code_point_strings => None,
            Kind::Utf8 => {
                let strings = values.iter().map(string).collect::<Option<Vec<_>>>()?;
                Some(MongoFilter::inexact(self.utf8_excluding(field, &strings)))
            }
            Kind::Integer { types, min, max } => {
                let mut excluded = Vec::with_capacity(values.len());
                for value in values {
                    match numeric(value)? {
                        Numeric::Int(k) if (min..=max).contains(&k) => excluded.push(integer(k)),
                        Numeric::Int(_) => {}
                        Numeric::Float(_) => return self.not_in_list_by_comparison(field, values),
                    }
                }
                let mut conditions = doc! { "$nin": excluded };
                let natural_min: i128 = if types.contains(&"long") {
                    i64::MIN.into()
                } else {
                    i32::MIN.into()
                };
                let natural_max: i128 = if types.contains(&"long") {
                    i64::MAX.into()
                } else {
                    i32::MAX.into()
                };
                if min > natural_min {
                    conditions.insert("$gte", integer(min));
                }
                if max < natural_max {
                    conditions.insert("$lte", integer(max));
                }
                Some(MongoFilter::exact(self.scalar(field, types, conditions)))
            }
            _ => self.not_in_list_by_comparison(field, values),
        }
    }

    fn not_in_list_by_comparison(
        &self,
        field: &Field<'_>,
        values: &[ScalarValue],
    ) -> Option<MongoFilter> {
        let mut filters = values.iter().map(|v| self.compare(field, Cmp::NotEq, v));
        let first = filters.next()?;
        filters.fold(first, and)
    }

    fn like(&self, like: &Like, negated: bool) -> Option<MongoFilter> {
        let field = self.field(&like.expr)?;
        if !matches!(field.kind, Kind::Utf8) {
            return None;
        }
        // DataFusion evaluates LIKE with `\` as the escape and refuses any other.
        if like.escape_char.is_some_and(|c| c != '\\') {
            return None;
        }
        let pattern = literal(&like.pattern)?;
        let regex = like_regex(string(&pattern)?, like.case_insensitive)?;
        let strings = if like.negated == negated {
            self.at(&field, regex)
        } else {
            self.at(&field, doc! { "$type": "string", "$not": regex })
        };
        Some(MongoFilter::inexact(any_of(vec![
            strings,
            self.rendered(&field, false),
        ])))
    }
}

/// Whether `document` holds a string comparison that a collation other than
/// the simple one could narrow: `$nin`, `$ne` or a range operator over a
/// string, or an equality under a negation. Such a query has to run under the
/// simple collation to compare strings by code point, as SQL does.
#[must_use]
pub fn orders_strings(document: &Document) -> bool {
    document
        .iter()
        .any(|(key, value)| orders_strings_in(key, value, false))
}

fn orders_strings_in(key: &str, value: &Bson, negated: bool) -> bool {
    let holds_string = |value: &Bson| match value {
        Bson::String(_) | Bson::Symbol(_) => true,
        Bson::Array(values) => values
            .iter()
            .any(|v| matches!(v, Bson::String(_) | Bson::Symbol(_))),
        _ => false,
    };
    let negated = negated || matches!(key, "$nor" | "$not");
    match key {
        "$nin" | "$ne" | "$lt" | "$lte" | "$gt" | "$gte" if holds_string(value) => true,
        "$in" | "$eq" | "$all" if negated && holds_string(value) => true,
        _ => match value {
            Bson::Document(inner) => inner.iter().any(|(k, v)| orders_strings_in(k, v, negated)),
            Bson::Array(items) => items
                .iter()
                .any(|item| orders_strings_in(key, item, negated)),
            // A bare string under a field name is an implicit `$eq`.
            Bson::String(_) | Bson::Symbol(_) => negated && !key.starts_with('$'),
            _ => false,
        },
    }
}

/// The integers `v` for which `(v as f64) <cmp> x` holds.
fn integer_float_set(cmp: Cmp, x: f64) -> IntSet {
    if x.is_nan() {
        // Arrow orders a NaN above every number, or with its sign bit set,
        // below them, and equal to no integer.
        let above = x.is_sign_positive();
        return match cmp {
            Cmp::Eq => IntSet::Empty,
            Cmp::NotEq => IntSet::ALL,
            Cmp::Lt | Cmp::LtEq if above => IntSet::ALL,
            Cmp::Gt | Cmp::GtEq if !above => IntSet::ALL,
            _ => IntSet::Empty,
        };
    }
    if x == 0.0 && x.is_sign_negative() {
        // An integer zero converts to 0.0, which Arrow orders above -0.0.
        return match cmp {
            Cmp::Eq => IntSet::Empty,
            Cmp::NotEq => IntSet::ALL,
            Cmp::Gt | Cmp::GtEq => IntSet::Range(0, i128::MAX),
            Cmp::Lt | Cmp::LtEq => IntSet::Range(i128::MIN, -1),
        };
    }
    IntSet::compare_float(cmp, x)
}

fn timestamp(seconds: i128, increment: u32) -> Bson {
    Bson::Timestamp(BsonTimestamp {
        time: u32::try_from(seconds).unwrap_or(if seconds < 0 { 0 } else { u32::MAX }),
        increment,
    })
}

/// A MongoDB regular expression matching the strings `pattern` matches as a
/// SQL LIKE pattern, as `DataFusion` evaluates it: `\` escapes the next
/// character (a trailing one is literal), `%` matches any run of characters
/// and `_` any one, newlines included. For ILIKE, only ASCII letters are
/// folded, to the characters Unicode simple case folding relates them to;
/// `None` for a pattern with a letter beyond ASCII.
fn like_regex(pattern: &str, case_insensitive: bool) -> Option<BsonRegex> {
    let mut regex = String::with_capacity(pattern.len() + 8);
    regex.push_str("\\A");
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => push_literal(&mut regex, chars.next().unwrap_or('\\'), case_insensitive)?,
            '%' => regex.push_str(".*"),
            '_' => regex.push('.'),
            c => push_literal(&mut regex, c, case_insensitive)?,
        }
    }
    regex.push_str("\\z");
    Some(BsonRegex {
        pattern: regex,
        // `s`: `.` matches a newline, as `%` and `_` do.
        options: "s".to_string(),
    })
}

fn push_literal(regex: &mut String, c: char, case_insensitive: bool) -> Option<()> {
    if case_insensitive && c.is_alphabetic() {
        if !c.is_ascii() {
            return None;
        }
        let lower = c.to_ascii_lowercase();
        regex.push('[');
        regex.push(lower);
        regex.push(lower.to_ascii_uppercase());
        match lower {
            'k' => regex.push_str("\\x{212A}"),
            's' => regex.push_str("\\x{17F}"),
            _ => {}
        }
        regex.push(']');
    } else if c.is_ascii_alphanumeric() || (!c.is_ascii() && !c.is_control()) {
        regex.push(c);
    } else if c.is_ascii_graphic() || c == ' ' {
        regex.push('\\');
        regex.push(c);
    } else {
        // Control characters, NUL included, which a BSON regex cannot hold.
        let _ = write!(regex, "\\x{{{:X}}}", u32::from(c));
    }
    Some(())
}

#[cfg(test)]
mod tests;
