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

use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex, PoisonError};

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch, RecordBatchReader};
use arrow::datatypes::{DataType, Float32Type, Float64Type, Schema, SchemaRef};
use arrow::row::{RowConverter, SortField};
use arrow_schema::ArrowError;
use tokio::sync::mpsc::Receiver;

/// The most distinct keys one statement holds before the next statement starts,
/// which bounds the memory of a write that repeats no key: the set is
/// `u64` hashes, so about 16 MiB at this size.
const MAX_KEYS_PER_STATEMENT: usize = 1 << 20;

/// Writes every batch of `source` as one `insert` per statement, cutting
/// statements at the first row whose key (the `key_columns` of `schema`) an
/// earlier row of the same statement carried. Returns the rows `insert`
/// reported in total.
pub(super) fn write_statements<'a, E>(
    source: Receiver<RecordBatch>,
    schema: &SchemaRef,
    key_columns: impl IntoIterator<Item = &'a str>,
    mut insert: impl FnMut(Box<dyn RecordBatchReader + Send>) -> Result<u64, E>,
) -> Result<u64, E>
where
    E: From<ArrowError>,
{
    let groups = Arc::new(Mutex::new(UpsertGroups::try_new(
        source,
        schema,
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
    key_indices: Vec<usize>,
    converter: RowConverter,
    /// Hashes of the keys the current statement holds.
    seen: HashSet<u64>,
    max_keys: usize,
    /// The rows that start the next statement: a batch the current statement
    /// ended inside, with the hash of every row's key (`None` for a NULL key)
    /// and the row the next statement starts at.
    held: Option<(RecordBatch, Vec<Option<u64>>, usize)>,
    /// The current statement ended at a repeated key; cleared by
    /// [`Self::start_statement`].
    ended: bool,
    /// The source gave its last batch.
    exhausted: bool,
}

impl UpsertGroups {
    /// Groups `source` by the conflict target `key_columns`, matched to `schema`
    /// the way `DuckDB` matches an identifier: exactly, else ignoring ASCII case.
    fn try_new<'a>(
        source: Receiver<RecordBatch>,
        schema: &Schema,
        key_columns: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self, ArrowError> {
        let key_indices = key_columns
            .into_iter()
            .map(|column| key_index(schema, column))
            .collect::<Result<Vec<_>, _>>()?;
        let converter = RowConverter::new(
            key_indices
                .iter()
                .map(|&index| SortField::new(schema.field(index).data_type().clone()))
                .collect(),
        )?;
        Ok(Self {
            source,
            key_indices,
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
    /// before it. A row with a NULL in its key conflicts with nothing, so it is
    /// neither recorded nor a repeat.
    fn first_repeat(&mut self, hashes: &[Option<u64>]) -> Option<usize> {
        hashes.iter().position(|hash| {
            let Some(hash) = hash else {
                return false;
            };
            self.seen.len() >= self.max_keys || !self.seen.insert(*hash)
        })
    }

    /// The hash of each row's key, `None` where a key column is NULL.
    fn key_hashes(&self, batch: &RecordBatch) -> Result<Vec<Option<u64>>, ArrowError> {
        let columns: Vec<ArrayRef> = self
            .key_indices
            .iter()
            .map(|&index| canonical_key_column(batch.column(index)))
            .collect();
        let rows = self.converter.convert_columns(&columns)?;
        let has_nulls = columns.iter().any(|column| column.null_count() > 0);
        Ok((0..batch.num_rows())
            .map(|row| {
                if has_nulls && columns.iter().any(|column| column.is_null(row)) {
                    return None;
                }
                let mut hasher = DefaultHasher::new();
                rows.row(row).as_ref().hash(&mut hasher);
                Some(hasher.finish())
            })
            .collect())
    }
}

/// The index of the field `column` names: the field with exactly that name,
/// else the only one equal to it ignoring ASCII case, which is how `DuckDB`
/// matches the conflict target to a column.
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
            "on_conflict column '{column}' is not a column of the written data"
        ))),
        (Some(_), Some(_)) => Err(ArrowError::SchemaError(format!(
            "on_conflict column '{column}' matches more than one column of the written data \
             ignoring case"
        ))),
    }
}

/// `column` with each float `DuckDB` holds as one key written one way: `-0.0`
/// as `0.0`, and every NaN as the same NaN. `DuckDB` treats `-0.0` as a repeat
/// of a stored `0.0` and every NaN as one key, while their bits, and arrow's
/// row format, tell them apart. The write still carries the values it was
/// given; only the keys compared change. Any other column is returned as is: a
/// `DuckDB` key column is a plain `FLOAT` or `DOUBLE`, never a half float or a
/// float inside a struct or dictionary.
fn canonical_key_column(column: &ArrayRef) -> ArrayRef {
    match column.data_type() {
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
        _ => Arc::clone(column),
    }
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
    use arrow::array::{Float32Array, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{Field, Int64Type};
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
        let rows = write_statements(source(batches), &schema, keys.iter().copied(), |reader| {
            let mut statement = Vec::new();
            let mut rows = 0;
            for batch in reader {
                let batch = batch?;
                rows += batch.num_rows() as u64;
                statement.push(f(&batch));
            }
            statements.push(statement);
            Ok::<u64, ArrowError>(rows)
        })
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
    fn a_null_key_never_repeats_and_never_ends_a_statement() {
        let got = statements(
            vec![
                batch(&[(None, "a"), (Some(1), "b")]),
                batch(&[(None, "c"), (None, "d")]),
            ],
            &["id"],
        );
        assert_eq!(
            got,
            vec![rows(&[
                (None, "a"),
                (Some(1), "b"),
                (None, "c"),
                (None, "d")
            ])]
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
            UpsertGroups::try_new(source(batches), &schema(), ["id"])
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
        let err = UpsertGroups::try_new(source(vec![]), &schema(), ["nope"])
            .err()
            .expect("an error");
        assert_eq!(
            err.to_string(),
            "Schema error: on_conflict column 'nope' is not a column of the written data"
        );
    }

    #[test]
    fn a_conflict_target_matching_two_columns_ignoring_case_is_an_error() {
        let schema = Schema::new(vec![
            Field::new("Id", DataType::Int64, false),
            Field::new("ID", DataType::Int64, false),
        ]);
        let err = UpsertGroups::try_new(source(vec![]), &schema, ["id"])
            .err()
            .expect("an error");
        assert_eq!(
            err.to_string(),
            "Schema error: on_conflict column 'id' matches more than one column of the written \
             data ignoring case"
        );
    }

    #[test]
    fn an_insert_error_ends_the_write_with_that_error() {
        let batches = vec![batch(&[(Some(1), "a")]), batch(&[(Some(1), "b")])];
        let schema = batches[0].schema();
        let mut statements = 0;
        let err = write_statements(source(batches), &schema, ["id"], |reader| {
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
}
