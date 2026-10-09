//! Cuts an upsert write into `INSERT` statements so that `ON CONFLICT DO UPDATE`
//! keeps the last copy of a key the write repeats.
//!
//! `DuckDB` resolves a conflict between two rows of the *same* `INSERT`
//! statement by keeping whichever row it processes first, and its parallel scan
//! of the statement's input decides that order. One statement over a whole
//! write therefore keeps a random copy of a key the write repeats. A conflict
//! between a row and a row an *earlier* statement of the same transaction
//! inserted is resolved as `DO UPDATE` says: the later row replaces the earlier
//! one.
//!
//! So the write is cut at the first row whose key an earlier row of the same
//! statement carried. No key repeats within a statement, and across statements
//! the last copy wins, in arrival order. A write that repeats no key is one
//! statement. The keys of the current statement are held as hashes; a hash
//! collision only ends a statement early, which costs one more statement and
//! changes no result.
//!
//! A NULL key is compared like any value. Within one statement `DuckDB` takes
//! two rows whose unique key is NULL as one conflict and drops the later row,
//! where across statements, as for a plain insert, it keeps both; so a repeated
//! NULL key starts a new statement too, and every row is kept.

use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex, PoisonError};

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch, RecordBatchReader};
use arrow::compute::cast;
use arrow::datatypes::{
    DataType, Date32Type, Date64Type, Float32Type, Float64Type, Schema, SchemaRef,
    Time64MicrosecondType, Time64NanosecondType, TimeUnit, TimestampMicrosecondType,
    TimestampNanosecondType,
};
use arrow::row::{RowConverter, SortField};
use arrow_schema::ArrowError;
use tokio::sync::mpsc::Receiver;

/// The most distinct keys one statement holds before the next statement starts,
/// which bounds the memory of a write that repeats no key: the set is
/// `u64` hashes, so about 16 MiB at this size.
const MAX_KEYS_PER_STATEMENT: usize = 1 << 20;

/// Writes every batch of `source` as one `insert` per statement, cutting
/// statements at the first row whose key an earlier row of the same statement
/// carried. Returns the rows `insert` reported in total.
///
/// `schema` is the written data's. The key is the `key_columns` of the
/// destination's `table_schema`, taken from the written data by position: the
/// insert is positional (`INSERT INTO table SELECT * FROM …`), so the written
/// data may name its columns differently.
pub(super) fn write_statements<'a, E>(
    source: Receiver<RecordBatch>,
    schema: &SchemaRef,
    table_schema: &Schema,
    key_columns: impl IntoIterator<Item = &'a str>,
    mut insert: impl FnMut(Box<dyn RecordBatchReader + Send>) -> Result<u64, E>,
) -> Result<u64, E>
where
    E: From<ArrowError>,
{
    let groups = Arc::new(Mutex::new(UpsertGroups::try_new(
        source,
        schema,
        table_schema,
        key_columns,
    )?));
    let mut rows = 0;
    loop {
        rows += insert(Box::new(StatementReader::start(
            Arc::clone(&groups),
            Arc::clone(schema),
        )))?;
        if groups
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .exhausted
        {
            return Ok(rows);
        }
    }
}

/// The batches of one write, handed out one statement at a time.
struct UpsertGroups {
    source: Receiver<RecordBatch>,
    /// Each key column's position in the written data, and the table's type
    /// for it, which the written column is cast to before it is compared.
    keys: Vec<(usize, DataType)>,
    /// A key column is written as a type whose conversion to the table's the
    /// insert may not perform as arrow does, so no two rows share a statement.
    isolate_rows: bool,
    converter: RowConverter,
    /// Hashes of the keys the current statement holds.
    seen: HashSet<u64>,
    max_keys: usize,
    /// The rows that start the next statement: a batch the current statement
    /// ended inside, with the hash of every row's key and the row the next
    /// statement starts at.
    held: Option<(RecordBatch, Vec<u64>, usize)>,
    /// The current statement ended at a repeated key; cleared by
    /// [`Self::start_statement`].
    ended: bool,
    /// The source gave its last batch.
    exhausted: bool,
}

impl UpsertGroups {
    /// Groups `source` by the conflict target `key_columns`, matched to the
    /// destination's `table_schema` the way `DuckDB` matches an identifier
    /// (exactly, else ignoring ASCII case) and read from the written data, of
    /// `schema`, by position.
    fn try_new<'a>(
        source: Receiver<RecordBatch>,
        schema: &Schema,
        table_schema: &Schema,
        key_columns: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self, ArrowError> {
        let keys = key_columns
            .into_iter()
            .map(|column| {
                let index = key_index(table_schema, column)?;
                if index >= schema.fields().len() {
                    return Err(ArrowError::SchemaError(format!(
                        "on_conflict column '{column}' is column {} of the table, but the \
                         written data has only {} columns",
                        index + 1,
                        schema.fields().len()
                    )));
                }
                Ok((index, table_schema.field(index).data_type().clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let isolate_rows = keys.iter().any(|(index, data_type)| {
            !converts_exactly(schema.field(*index).data_type(), data_type)
        });
        let converter = RowConverter::new(
            keys.iter()
                .map(|(_, data_type)| SortField::new(canonical_type(data_type)))
                .collect(),
        )?;
        Ok(Self {
            source,
            keys,
            isolate_rows,
            converter,
            seen: HashSet::new(),
            max_keys: MAX_KEYS_PER_STATEMENT,
            held: None,
            ended: false,
            exhausted: false,
        })
    }

    #[cfg(test)]
    fn with_max_keys(mut self, max_keys: usize) -> Self {
        self.max_keys = max_keys.max(1);
        self
    }

    /// Starts the next statement: its first rows are the ones the previous
    /// statement ended at, and it holds no keys yet.
    fn start_statement(&mut self) {
        self.seen.clear();
        self.ended = false;
    }

    /// The next rows of the current statement, or `None` when it has ended,
    /// either at a repeated key or at the end of the write.
    fn next_in_statement(&mut self) -> Result<Option<RecordBatch>, ArrowError> {
        if self.ended {
            return Ok(None);
        }
        let (batch, hashes, start) = if let Some(held) = self.held.take() {
            held
        } else {
            let Some(batch) = self.source.blocking_recv() else {
                self.exhausted = true;
                self.ended = true;
                return Ok(None);
            };
            let hashes = self.key_hashes(&batch)?;
            (batch, hashes, 0)
        };
        let Some(row) = self.first_repeat(&hashes[start..]).map(|row| start + row) else {
            return Ok(Some(batch.slice(start, batch.num_rows() - start)));
        };
        // The statement ends before `row`; the rest starts the next one. `row`
        // is `start` only when this statement already took rows from an
        // earlier batch, since it holds no key when it starts.
        self.ended = true;
        let head = (row > start).then(|| batch.slice(start, row - start));
        self.held = Some((batch, hashes, row));
        Ok(head)
    }

    /// The first row of `hashes` whose key the current statement already holds
    /// (or that would take it past `max_keys`), recording the keys of the rows
    /// before it.
    fn first_repeat(&mut self, hashes: &[u64]) -> Option<usize> {
        if self.isolate_rows {
            // One row per statement: an empty statement admits its first row, and
            // the row after it ends the statement.
            let first = hashes.first()?;
            if !self.seen.is_empty() {
                return Some(0);
            }
            self.seen.insert(*first);
            return (hashes.len() > 1).then_some(1);
        }
        hashes
            .iter()
            .position(|hash| self.seen.len() >= self.max_keys || !self.seen.insert(*hash))
    }

    /// The hash of each row's key; a NULL is encoded like any value. A key
    /// column of another type than the table's is cast to it first, as the
    /// insert casts it; see [`converts_exactly`] for the casts trusted to
    /// agree with `DuckDB`'s.
    fn key_hashes(&self, batch: &RecordBatch) -> Result<Vec<u64>, ArrowError> {
        let columns = self
            .keys
            .iter()
            .map(|(index, data_type)| {
                let column = batch.column(*index);
                let column = if column.data_type() == data_type {
                    Arc::clone(column)
                } else {
                    cast(column, data_type)?
                };
                canonical_key_column(&column)
            })
            .collect::<Result<Vec<ArrayRef>, _>>()?;
        let rows = self.converter.convert_columns(&columns)?;
        Ok((0..batch.num_rows())
            .map(|row| {
                let mut hasher = DefaultHasher::new();
                rows.row(row).as_ref().hash(&mut hasher);
                hasher.finish()
            })
            .collect())
    }
}

/// Whether arrow's cast of a key `written` as one type to the table's
/// `stored` type gives the value `DuckDB`'s insert gives it, so two written
/// values are one key exactly when the table holds them as one. Only exact
/// conversions are trusted: no change, a string or binary spelling, a dictionary
/// or run-end encoding of a trusted type, an integer or float widened, a
/// timestamp given a time zone, and a `Date32` widened. Anything else, such as
/// a float written to an integer key (arrow truncates `1.6`, `DuckDB` rounds
/// it) or text written to a date, is not: such a write puts every row in a
/// statement of its own, which is slow and right.
fn converts_exactly(written: &DataType, stored: &DataType) -> bool {
    use DataType::{
        Binary, BinaryView, Date32, Date64, Dictionary, Float32, Float64, Int16, Int32, Int64,
        Int8, LargeBinary, LargeUtf8, RunEndEncoded, Timestamp, UInt16, UInt32, UInt64, UInt8,
        Utf8, Utf8View,
    };
    match (written, stored) {
        (written, stored) if written == stored => true,
        (Dictionary(_, written), stored) => converts_exactly(written, stored),
        (RunEndEncoded(_, written), stored) => converts_exactly(written.data_type(), stored),
        (Timestamp(written_unit, None), Timestamp(stored_unit, Some(_))) => {
            written_unit == stored_unit
        }
        (Utf8 | LargeUtf8 | Utf8View, stored) => matches!(stored, Utf8 | LargeUtf8 | Utf8View),
        (Binary | LargeBinary | BinaryView, stored) => {
            matches!(stored, Binary | LargeBinary | BinaryView)
        }
        (Int8, stored) => matches!(stored, Int16 | Int32 | Int64 | Float32 | Float64),
        (Int16, stored) => matches!(stored, Int32 | Int64 | Float32 | Float64),
        (Int32, stored) => matches!(stored, Int64 | Float64),
        (UInt8, stored) => matches!(
            stored,
            UInt16 | UInt32 | UInt64 | Int16 | Int32 | Int64 | Float32 | Float64
        ),
        (UInt16, stored) => matches!(stored, UInt32 | UInt64 | Int32 | Int64 | Float32 | Float64),
        (UInt32, stored) => matches!(stored, UInt64 | Int64 | Float64),
        (Float32, Float64) | (Date32, Date64) => true,
        _ => false,
    }
}

/// The index of the field of the table's `schema` that `column` names: the
/// field with exactly that name, else the only one equal to it ignoring ASCII
/// case, which is how `DuckDB` matches the conflict target to a column.
fn key_index(schema: &Schema, column: &str) -> Result<usize, ArrowError> {
    if let Ok(index) = schema.index_of(column) {
        return Ok(index);
    }
    let mut matches = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| field.name().eq_ignore_ascii_case(column))
        .map(|(index, _)| index);
    match (matches.next(), matches.next()) {
        (Some(index), None) => Ok(index),
        (None, _) => Err(ArrowError::SchemaError(format!(
            "on_conflict column '{column}' is not a column of the table"
        ))),
        (Some(_), Some(_)) => Err(ArrowError::SchemaError(format!(
            "on_conflict column '{column}' matches more than one column of the table ignoring \
             case"
        ))),
    }
}

const MILLISECONDS_PER_DAY: i64 = 86_400_000;
const NANOSECONDS_PER_MICROSECOND: i64 = 1_000;

/// The type [`canonical_key_column`] compares a key column of `data_type` as.
fn canonical_type(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Dictionary(_, value) => canonical_type(value),
        DataType::RunEndEncoded(_, value) => canonical_type(value.data_type()),
        DataType::Date64 => DataType::Date32,
        DataType::Time64(TimeUnit::Nanosecond) => DataType::Time64(TimeUnit::Microsecond),
        DataType::Timestamp(TimeUnit::Nanosecond, Some(_)) => {
            DataType::Timestamp(TimeUnit::Microsecond, None)
        }
        other => other.clone(),
    }
}

/// `column` as the key `DuckDB` compares it as, so two values the table holds
/// as one key hash alike. The write still carries the values it was given;
/// only the keys compared change.
///
/// `DuckDB` keys `-0.0` as a repeat of `0.0` and every NaN as one key, while
/// their bits, and arrow's row format, tell them apart. It stores a dictionary
/// or run-end-encoded column as its values; a `Date64` as a `DATE`, a nanosecond `Time64` as a
/// microsecond `TIME`, and a nanosecond timestamp with a time zone as a
/// microsecond `TIMESTAMP WITH TIME ZONE`, each truncated toward zero, as the
/// `/` here does (a nanosecond timestamp without a time zone is a
/// `TIMESTAMP_NS`, kept exactly). A duration or interval is stored as an
/// `INTERVAL`, which `DuckDB` refuses as an index key, so it is never a
/// conflict target.
fn canonical_key_column(column: &ArrayRef) -> Result<ArrayRef, ArrowError> {
    Ok(match column.data_type() {
        DataType::Dictionary(_, value) => canonical_key_column(&cast(column, value)?)?,
        DataType::RunEndEncoded(_, value) => {
            canonical_key_column(&cast(column, value.data_type())?)?
        }
        DataType::Float32 => Arc::new(
            column
                .as_primitive::<Float32Type>()
                .unary::<_, Float32Type>(canonical_f32),
        ),
        DataType::Float64 => Arc::new(
            column
                .as_primitive::<Float64Type>()
                .unary::<_, Float64Type>(canonical_f64),
        ),
        DataType::Date64 => Arc::new(column.as_primitive::<Date64Type>().unary::<_, Date32Type>(
            |milliseconds| i32::try_from(milliseconds / MILLISECONDS_PER_DAY).unwrap_or(i32::MAX),
        )),
        DataType::Time64(TimeUnit::Nanosecond) => Arc::new(
            column
                .as_primitive::<Time64NanosecondType>()
                .unary::<_, Time64MicrosecondType>(|nanoseconds| {
                    nanoseconds / NANOSECONDS_PER_MICROSECOND
                }),
        ),
        DataType::Timestamp(TimeUnit::Nanosecond, Some(_)) => Arc::new(
            column
                .as_primitive::<TimestampNanosecondType>()
                .unary::<_, TimestampMicrosecondType>(|nanoseconds| {
                    nanoseconds / NANOSECONDS_PER_MICROSECOND
                }),
        ),
        _ => Arc::clone(column),
    })
}

fn canonical_f32(value: f32) -> f32 {
    if value.is_nan() {
        f32::NAN
    } else if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn canonical_f64(value: f64) -> f64 {
    if value.is_nan() {
        f64::NAN
    } else if value == 0.0 {
        0.0
    } else {
        value
    }
}

/// Reads one statement's rows out of shared [`UpsertGroups`], as the arrow
/// stream `DuckDB` scans for that statement.
struct StatementReader {
    groups: Arc<Mutex<UpsertGroups>>,
    schema: SchemaRef,
}

impl StatementReader {
    /// Starts the next statement and returns the reader of its rows.
    fn start(groups: Arc<Mutex<UpsertGroups>>, schema: SchemaRef) -> Self {
        groups
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .start_statement();
        Self { groups, schema }
    }
}

impl Iterator for StatementReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.groups
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .next_in_statement()
            .transpose()
    }
}

impl RecordBatchReader for StatementReader {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Date64Array, DictionaryArray, Float32Array, Float64Array, Int32Array, Int64Array, RunArray,
        StringArray, Time64NanosecondArray, TimestampNanosecondArray,
    };
    use arrow::datatypes::{Field, Int32Type, Int64Type};
    use tokio::sync::mpsc;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("v", DataType::Utf8, true),
        ]))
    }

    fn batch(rows: &[(Option<i64>, &str)]) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(
                    rows.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
                )),
            ],
        )
        .expect("batch")
    }

    fn float_schema(data_type: DataType) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", data_type, true),
            Field::new("v", DataType::Utf8, true),
        ]))
    }

    fn f64_batch(ids: &[Option<f64>]) -> RecordBatch {
        RecordBatch::try_new(
            float_schema(DataType::Float64),
            vec![
                Arc::new(Float64Array::from(ids.to_vec())),
                Arc::new(StringArray::from(vec!["v"; ids.len()])),
            ],
        )
        .expect("batch")
    }

    fn f32_batch(ids: &[Option<f32>]) -> RecordBatch {
        RecordBatch::try_new(
            float_schema(DataType::Float32),
            vec![
                Arc::new(Float32Array::from(ids.to_vec())),
                Arc::new(StringArray::from(vec!["v"; ids.len()])),
            ],
        )
        .expect("batch")
    }

    /// A source already holding every batch of the write.
    fn source(batches: Vec<RecordBatch>) -> Receiver<RecordBatch> {
        let (sender, receiver) = mpsc::channel(batches.len().max(1));
        for batch in batches {
            sender.try_send(batch).expect("queue a batch");
        }
        receiver
    }

    /// Drives the production loop over `batches` with an `insert` that drains
    /// each statement's reader, and returns `f` of every batch read, per
    /// statement.
    fn drive<T>(
        batches: Vec<RecordBatch>,
        keys: &[&str],
        mut f: impl FnMut(&RecordBatch) -> T,
    ) -> Vec<Vec<T>> {
        let schema = batches.first().map_or_else(schema, RecordBatch::schema);
        let expected_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        let mut statements = Vec::new();
        let rows = write_statements(
            source(batches),
            &schema,
            &schema,
            keys.iter().copied(),
            |reader| {
                let mut statement = Vec::new();
                let mut rows = 0;
                for batch in reader {
                    let batch = batch?;
                    rows += batch.num_rows() as u64;
                    statement.push(f(&batch));
                }
                statements.push(statement);
                Ok::<u64, ArrowError>(rows)
            },
        )
        .expect("write");
        assert_eq!(
            rows, expected_rows as u64,
            "every row of the write reaches exactly one statement"
        );
        statements
    }

    /// The `(id, v)` rows of each statement.
    fn statements(batches: Vec<RecordBatch>, keys: &[&str]) -> Vec<Vec<(Option<i64>, String)>> {
        drive(batches, keys, |batch| {
            let ids = batch.column(0).as_primitive::<Int64Type>();
            let vs = batch.column(1).as_string::<i32>();
            (0..batch.num_rows())
                .map(|row| {
                    let id = (!ids.is_null(row)).then(|| ids.value(row));
                    (id, vs.value(row).to_string())
                })
                .collect::<Vec<_>>()
        })
        .into_iter()
        .map(|statement| statement.into_iter().flatten().collect())
        .collect()
    }

    /// The number of rows in each statement.
    fn statement_sizes(batches: Vec<RecordBatch>, keys: &[&str]) -> Vec<usize> {
        drive(batches, keys, RecordBatch::num_rows)
            .into_iter()
            .map(|statement| statement.into_iter().sum())
            .collect()
    }

    fn rows(pairs: &[(Option<i64>, &str)]) -> Vec<(Option<i64>, String)> {
        pairs
            .iter()
            .map(|(id, v)| (*id, (*v).to_string()))
            .collect()
    }

    #[test]
    fn a_write_that_repeats_no_key_is_one_statement() {
        let got = statements(
            vec![
                batch(&[(Some(1), "a"), (Some(2), "b")]),
                batch(&[(Some(3), "c")]),
            ],
            &["id"],
        );
        assert_eq!(
            got,
            vec![rows(&[(Some(1), "a"), (Some(2), "b"), (Some(3), "c")])]
        );
    }

    #[test]
    fn a_key_repeated_in_a_later_batch_starts_a_new_statement_at_that_batch() {
        let got = statements(
            vec![
                batch(&[(Some(1), "a"), (Some(2), "b")]),
                batch(&[(Some(1), "c"), (Some(3), "d")]),
            ],
            &["id"],
        );
        assert_eq!(
            got,
            vec![
                rows(&[(Some(1), "a"), (Some(2), "b")]),
                rows(&[(Some(1), "c"), (Some(3), "d")]),
            ]
        );
    }

    #[test]
    fn a_key_repeated_inside_a_batch_splits_that_batch_at_the_repeat() {
        let got = statements(
            vec![batch(&[
                (Some(1), "a"),
                (Some(2), "b"),
                (Some(1), "c"),
                (Some(2), "d"),
            ])],
            &["id"],
        );
        assert_eq!(
            got,
            vec![
                rows(&[(Some(1), "a"), (Some(2), "b")]),
                rows(&[(Some(1), "c"), (Some(2), "d")]),
            ]
        );
    }

    #[test]
    fn every_copy_of_a_key_lands_in_its_own_statement_in_arrival_order() {
        let got = statements(
            vec![
                batch(&[(Some(1), "a")]),
                batch(&[(Some(1), "b")]),
                batch(&[(Some(1), "c")]),
            ],
            &["id"],
        );
        assert_eq!(
            got,
            vec![
                rows(&[(Some(1), "a")]),
                rows(&[(Some(1), "b")]),
                rows(&[(Some(1), "c")]),
            ]
        );
    }

    #[test]
    fn a_repeated_null_key_starts_a_new_statement_so_duckdb_keeps_every_row() {
        let got = statements(
            vec![
                batch(&[(None, "a"), (Some(1), "b")]),
                batch(&[(None, "c"), (None, "d")]),
            ],
            &["id"],
        );
        assert_eq!(
            got,
            vec![
                rows(&[(None, "a"), (Some(1), "b")]),
                rows(&[(None, "c")]),
                rows(&[(None, "d")]),
            ]
        );
    }

    #[test]
    fn an_empty_write_is_one_empty_statement() {
        assert_eq!(statements(vec![], &["id"]), vec![Vec::new()]);
    }

    #[test]
    fn an_empty_batch_is_passed_through() {
        let got = statements(
            vec![
                batch(&[]),
                batch(&[(Some(1), "a")]),
                batch(&[]),
                batch(&[(Some(1), "b")]),
            ],
            &["id"],
        );
        assert_eq!(got, vec![rows(&[(Some(1), "a")]), rows(&[(Some(1), "b")])]);
    }

    #[test]
    fn the_key_cap_ends_a_statement_before_a_new_key_would_pass_it() {
        let batches = vec![batch(&[
            (Some(1), "a"),
            (Some(2), "b"),
            (Some(3), "c"),
            (Some(4), "d"),
            (Some(1), "e"),
        ])];
        let groups = Arc::new(Mutex::new(
            UpsertGroups::try_new(source(batches), &schema(), &schema(), ["id"])
                .expect("groups")
                .with_max_keys(2),
        ));
        let mut sizes = Vec::new();
        loop {
            let reader = StatementReader::start(Arc::clone(&groups), schema());
            sizes.push(
                reader
                    .map(|batch| batch.expect("batch").num_rows())
                    .sum::<usize>(),
            );
            if groups.lock().expect("lock").exhausted {
                break;
            }
        }
        assert_eq!(sizes, vec![2, 2, 1]);
    }

    #[test]
    fn a_composite_key_repeats_only_when_every_column_repeats() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
        ]));
        let b = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 1, 2, 1])),
                Arc::new(StringArray::from(vec!["x", "y", "x", "x"])),
            ],
        )
        .expect("batch");
        assert_eq!(
            statement_sizes(vec![b], &["a", "b"]),
            vec![3, 1],
            "(1,x) (1,y) (2,x) share no full key; the second (1,x) starts a statement"
        );
    }

    #[test]
    fn negative_zero_and_every_nan_repeat_a_double_key() {
        let sizes = statement_sizes(
            vec![
                f64_batch(&[Some(0.0), Some(f64::NAN), None]),
                f64_batch(&[Some(-0.0), Some(-f64::NAN), None]),
            ],
            &["id"],
        );
        assert_eq!(
            sizes,
            vec![3, 3],
            "-0.0 repeats 0.0 and -NaN repeats NaN, so the second batch starts a statement"
        );
    }

    #[test]
    fn negative_zero_and_every_nan_repeat_a_float_key() {
        let sizes = statement_sizes(
            vec![
                f32_batch(&[Some(0.0), Some(f32::NAN), None]),
                f32_batch(&[Some(-0.0), Some(-f32::NAN), None]),
            ],
            &["id"],
        );
        assert_eq!(sizes, vec![3, 3]);
    }

    #[test]
    fn the_rows_handed_out_still_carry_the_float_as_written() {
        let got = drive(
            vec![f64_batch(&[Some(0.0)]), f64_batch(&[Some(-0.0)])],
            &["id"],
            |batch| {
                batch
                    .column(0)
                    .as_primitive::<Float64Type>()
                    .value(0)
                    .is_sign_negative()
            },
        );
        assert_eq!(
            got,
            vec![vec![false], vec![true]],
            "only the compared key is canonical"
        );
    }

    #[test]
    fn the_conflict_target_matches_a_column_ignoring_ascii_case() {
        let got = statements(
            vec![batch(&[(Some(1), "a")]), batch(&[(Some(1), "b")])],
            &["ID"],
        );
        assert_eq!(got, vec![rows(&[(Some(1), "a")]), rows(&[(Some(1), "b")])]);
    }

    #[test]
    fn a_conflict_target_that_names_no_column_is_an_error() {
        let err = UpsertGroups::try_new(source(vec![]), &schema(), &schema(), ["nope"])
            .err()
            .expect("an error");
        assert_eq!(
            err.to_string(),
            "Schema error: on_conflict column 'nope' is not a column of the table"
        );
    }

    #[test]
    fn a_conflict_target_matching_two_columns_ignoring_case_is_an_error() {
        let schema = Schema::new(vec![
            Field::new("Id", DataType::Int64, false),
            Field::new("ID", DataType::Int64, false),
        ]);
        let err = UpsertGroups::try_new(source(vec![]), &schema, &schema, ["id"])
            .err()
            .expect("an error");
        assert_eq!(
            err.to_string(),
            "Schema error: on_conflict column 'id' matches more than one column of the table \
             ignoring case"
        );
    }

    #[test]
    fn an_insert_error_ends_the_write_with_that_error() {
        let batches = vec![batch(&[(Some(1), "a")]), batch(&[(Some(1), "b")])];
        let schema = batches[0].schema();
        let mut statements = 0;
        let err = write_statements(source(batches), &schema, &schema, ["id"], |reader| {
            statements += 1;
            let rows = reader.count() as u64;
            if statements == 2 {
                return Err(ArrowError::ExternalError("insert failed".into()));
            }
            Ok(rows)
        })
        .expect_err("the second statement's error");
        assert_eq!(err.to_string(), "External error: insert failed");
        assert_eq!(statements, 2, "no statement runs after the failed one");
    }

    /// An `(id, v)` batch whose `id` column is `ids`, of `ids`' own type.
    fn keyed(ids: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", ids.data_type().clone(), true),
            Field::new("v", DataType::Utf8, true),
        ]));
        let rows = ids.len();
        RecordBatch::try_new(
            schema,
            vec![ids, Arc::new(StringArray::from(vec!["v"; rows]))],
        )
        .expect("batch")
    }

    #[test]
    fn a_date64_key_repeats_when_it_names_the_day_duckdb_stores() {
        // 1 s and 2 s after the epoch are one DATE; -1 ms truncates to that day too;
        // the next day is another key.
        let sizes = statement_sizes(
            vec![
                keyed(Arc::new(Date64Array::from(vec![1_000, 86_400_000]))),
                keyed(Arc::new(Date64Array::from(vec![2_000]))),
                keyed(Arc::new(Date64Array::from(vec![-1]))),
            ],
            &["id"],
        );
        assert_eq!(sizes, vec![2, 1, 1]);
    }

    #[test]
    fn a_nanosecond_time_key_repeats_when_it_names_the_microsecond_duckdb_stores() {
        let sizes = statement_sizes(
            vec![
                keyed(Arc::new(Time64NanosecondArray::from(vec![1_000, 2_000]))),
                keyed(Arc::new(Time64NanosecondArray::from(vec![1_999]))),
            ],
            &["id"],
        );
        assert_eq!(sizes, vec![2, 1]);
    }

    #[test]
    fn a_zoned_nanosecond_timestamp_key_repeats_at_the_microsecond_but_a_plain_one_does_not() {
        let zoned = |values: Vec<i64>| {
            Arc::new(TimestampNanosecondArray::from(values).with_timezone("UTC")) as ArrayRef
        };
        assert_eq!(
            statement_sizes(
                vec![
                    keyed(zoned(vec![1_000, 2_000])),
                    keyed(zoned(vec![1_999])),
                    keyed(zoned(vec![-1_000_001])),
                    keyed(zoned(vec![-1_000_000])),
                ],
                &["id"],
            ),
            vec![2, 2, 1],
            "1_999 ns is the microsecond 1_000 ns holds, so it starts a statement; -1_000_001 ns              truncates toward zero to the microsecond -1_000_000 ns holds, so the latter starts              one too"
        );
        assert_eq!(
            statement_sizes(
                vec![
                    keyed(Arc::new(TimestampNanosecondArray::from(vec![1_000]))),
                    keyed(Arc::new(TimestampNanosecondArray::from(vec![1_999]))),
                ],
                &["id"],
            ),
            vec![2],
            "a TIMESTAMP_NS keeps every nanosecond"
        );
    }

    #[test]
    fn a_dictionary_key_repeats_by_its_value() {
        let dictionary = |values: Vec<&str>| {
            Arc::new(
                values
                    .into_iter()
                    .map(Some)
                    .collect::<DictionaryArray<Int32Type>>(),
            ) as ArrayRef
        };
        let sizes = statement_sizes(
            vec![
                keyed(dictionary(vec!["a", "b"])),
                keyed(dictionary(vec!["b"])),
            ],
            &["id"],
        );
        assert_eq!(sizes, vec![2, 1]);
    }

    #[test]
    fn the_conflict_target_is_read_from_the_written_data_by_the_tables_position() {
        // The table is (id, v); the written data names its columns (k, id) and its
        // `id` is NOT the key: the key is column 0, which the table calls `id`.
        let table_schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Utf8, true),
        ]);
        let written = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("id", DataType::Utf8, true),
        ]));
        let batch = |k: Vec<i64>, id: Vec<&str>| {
            RecordBatch::try_new(
                Arc::clone(&written),
                vec![
                    Arc::new(Int64Array::from(k)),
                    Arc::new(StringArray::from(id)),
                ],
            )
            .expect("batch")
        };
        let mut sizes = Vec::new();
        write_statements(
            source(vec![
                batch(vec![1, 2], vec!["x", "x"]),
                batch(vec![1], vec!["y"]),
            ]),
            &written,
            &table_schema,
            ["id"],
            |reader| {
                let rows: u64 = reader.map(|b| b.expect("batch").num_rows() as u64).sum();
                sizes.push(rows);
                Ok::<u64, ArrowError>(rows)
            },
        )
        .expect("write");
        assert_eq!(
            sizes,
            vec![2, 1],
            "column 0 repeats 1, so the second batch starts a statement; the written \
             column named id (x, x, y) does not decide it"
        );
    }

    #[test]
    fn written_data_with_fewer_columns_than_the_tables_key_position_is_an_error() {
        let table_schema = Schema::new(vec![
            Field::new("v", DataType::Utf8, true),
            Field::new("id", DataType::Int64, false),
        ]);
        let written = Schema::new(vec![Field::new("v", DataType::Utf8, true)]);
        let err = UpsertGroups::try_new(source(vec![]), &written, &table_schema, ["id"])
            .err()
            .expect("an error");
        assert_eq!(
            err.to_string(),
            "Schema error: on_conflict column 'id' is column 2 of the table, but the written \
             data has only 1 columns"
        );
    }

    #[test]
    fn a_run_end_encoded_key_repeats_by_its_decoded_value() {
        let encoded = |run_ends: Vec<i32>, values: Vec<i64>| {
            let values: ArrayRef = Arc::new(Time64NanosecondArray::from(values));
            Arc::new(
                RunArray::<Int32Type>::try_new(&Int32Array::from(run_ends), &values)
                    .expect("run array"),
            ) as ArrayRef
        };
        // (1_000, 1_000, 2_000) then (1_999): 1_999 ns is the microsecond 1_000 ns holds.
        let sizes = statement_sizes(
            vec![
                keyed(encoded(vec![2, 3], vec![1_000, 2_000])),
                keyed(encoded(vec![1], vec![1_999])),
            ],
            &["id"],
        );
        assert_eq!(
            sizes,
            vec![1, 2, 1],
            "the second 1_000 is a repeat and starts a statement, which 2_000 joins; 1_999 ns is              the microsecond 1_000 ns holds and starts the third"
        );
    }

    #[test]
    fn a_key_written_as_a_type_the_insert_converts_inexactly_isolates_every_row() {
        // The table's key is BIGINT; the written data carries it as text, which the
        // insert converts on its own terms, so no two rows share a statement.
        let table_schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Utf8, true),
        ]);
        let written = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("v", DataType::Utf8, true),
        ]));
        let batch = |ids: Vec<&str>| {
            RecordBatch::try_new(
                Arc::clone(&written),
                vec![
                    Arc::new(StringArray::from(ids.clone())),
                    Arc::new(StringArray::from(vec!["v"; ids.len()])),
                ],
            )
            .expect("batch")
        };
        let mut sizes = Vec::new();
        write_statements(
            source(vec![
                batch(vec!["01", "2"]),
                batch(vec!["1"]),
                batch(vec!["x", "y"]),
            ]),
            &written,
            &table_schema,
            ["id"],
            |reader| {
                let rows: u64 = reader.map(|b| b.expect("batch").num_rows() as u64).sum();
                sizes.push(rows);
                Ok::<u64, ArrowError>(rows)
            },
        )
        .expect("write");
        assert_eq!(sizes, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn a_key_written_as_a_narrower_integer_is_compared_as_the_tables_integer() {
        // The table's key is BIGINT; the written data carries it as INTEGER, which
        // the insert widens exactly, so the rows group by value.
        let table_schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Utf8, true),
        ]);
        let written = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("v", DataType::Utf8, true),
        ]));
        let batch = |ids: Vec<i32>| {
            RecordBatch::try_new(
                Arc::clone(&written),
                vec![
                    Arc::new(Int32Array::from(ids.clone())),
                    Arc::new(StringArray::from(vec!["v"; ids.len()])),
                ],
            )
            .expect("batch")
        };
        let mut sizes = Vec::new();
        write_statements(
            source(vec![batch(vec![1, 2]), batch(vec![3]), batch(vec![1])]),
            &written,
            &table_schema,
            ["id"],
            |reader| {
                let rows: u64 = reader.map(|b| b.expect("batch").num_rows() as u64).sum();
                sizes.push(rows);
                Ok::<u64, ArrowError>(rows)
            },
        )
        .expect("write");
        assert_eq!(sizes, vec![3, 1]);
    }

    #[test]
    fn only_exact_conversions_are_trusted() {
        use DataType::{Date32, Date64, Float64, Int32, Int64, Timestamp, UInt64, Utf8, Utf8View};
        assert!(converts_exactly(&Int32, &Int64));
        assert!(converts_exactly(&Utf8View, &Utf8));
        assert!(converts_exactly(&Date32, &Date64));
        assert!(converts_exactly(
            &Timestamp(TimeUnit::Microsecond, None),
            &Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        ));
        assert!(
            !converts_exactly(&Float64, &Int64),
            "arrow truncates, DuckDB rounds"
        );
        assert!(!converts_exactly(&Int64, &Int32), "narrowing");
        assert!(!converts_exactly(&UInt64, &Int64), "half the range");
        assert!(
            !converts_exactly(&Utf8, &Int64),
            "text parsed on the insert's terms"
        );
        assert!(!converts_exactly(&Utf8, &Date32));
        assert!(!converts_exactly(
            &Timestamp(TimeUnit::Nanosecond, None),
            &Timestamp(TimeUnit::Microsecond, None)
        ));
    }
}
