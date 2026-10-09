//! Cuts an upsert write into `INSERT` statements so that `ON CONFLICT DO UPDATE`
//! keeps the last copy of a key the write repeats.
//!
//! `DuckDB` resolves a conflict between two rows of the *same* `INSERT` statement
//! by keeping whichever row it processes first, and its parallel scan of the
//! statement's input decides that order. One statement over a whole write
//! therefore keeps a random copy of a key the write repeats. A conflict between
//! a row and a row an *earlier* statement of the same transaction inserted is
//! resolved as `DO UPDATE` says: the later row replaces the earlier one.
//!
//! So the write is cut at the first row whose key an earlier row of the same
//! statement carried. No key repeats within a statement, and across statements
//! the last copy wins, in arrival order. A write that repeats no key is still
//! one statement, as before. The keys of the current statement are held as
//! hashes; a hash collision only ends a statement early, which costs one more
//! statement and changes no result.

use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex, PoisonError};

use arrow::array::{
    Array, ArrayRef, AsArray, Float32Array, Float64Array, RecordBatch, RecordBatchReader,
};
use arrow::datatypes::{DataType, Float32Type, Float64Type, Schema, SchemaRef};
use arrow::row::{RowConverter, SortField};
use arrow_schema::ArrowError;
use tokio::sync::mpsc::Receiver;

/// The most distinct keys one statement holds before the next statement starts,
/// which bounds the memory of a write that repeats no key: the set is
/// `u64` hashes, so about 16 MiB at this size.
pub(super) const MAX_KEYS_PER_STATEMENT: usize = 1 << 20;

/// Where the batches of a write come from.
pub(super) trait BatchSource: Send {
    /// The next batch, or `None` once the write has no more.
    fn next_batch(&mut self) -> Option<RecordBatch>;
}

impl BatchSource for Receiver<RecordBatch> {
    fn next_batch(&mut self) -> Option<RecordBatch> {
        self.blocking_recv()
    }
}

impl BatchSource for std::vec::IntoIter<RecordBatch> {
    fn next_batch(&mut self) -> Option<RecordBatch> {
        self.next()
    }
}

/// The batches of one write, handed out one statement at a time.
pub(super) struct UpsertGroups<S> {
    source: S,
    key_indices: Vec<usize>,
    converter: RowConverter,
    /// Hashes of the keys the current statement holds.
    seen: HashSet<u64>,
    max_keys: usize,
    /// The rows that start the next statement: the rest of a batch the current
    /// statement ended inside.
    held: Option<RecordBatch>,
    /// The current statement ended at a repeated key; cleared by
    /// [`Self::start_statement`].
    ended: bool,
    /// The source gave its last batch.
    exhausted: bool,
}

impl<S: BatchSource> UpsertGroups<S> {
    /// Groups `source` by the conflict target `key_columns`, matched to `schema`
    /// the way `DuckDB` matches an identifier: exactly, else ignoring ASCII case.
    pub(super) fn try_new<'a>(
        source: S,
        schema: &Schema,
        key_columns: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self, ArrowError> {
        Self::try_with_max_keys(source, schema, key_columns, MAX_KEYS_PER_STATEMENT)
    }

    pub(super) fn try_with_max_keys<'a>(
        source: S,
        schema: &Schema,
        key_columns: impl IntoIterator<Item = &'a str>,
        max_keys: usize,
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
            max_keys: max_keys.max(1),
            held: None,
            ended: false,
            exhausted: false,
        })
    }

    /// Starts the next statement: its first rows are the ones the previous
    /// statement ended at, and it holds no keys yet.
    pub(super) fn start_statement(&mut self) {
        self.seen.clear();
        self.ended = false;
    }

    /// Whether the source has given every batch of the write. Only meaningful
    /// once the current statement has ended.
    pub(super) fn is_exhausted(&self) -> bool {
        self.exhausted && self.held.is_none()
    }

    /// The next rows of the current statement, or `None` when it has ended,
    /// either at a repeated key or at the end of the write.
    pub(super) fn next_in_statement(&mut self) -> Result<Option<RecordBatch>, ArrowError> {
        if self.ended {
            return Ok(None);
        }
        let Some(batch) = self.held.take().or_else(|| self.source.next_batch()) else {
            self.exhausted = true;
            self.ended = true;
            return Ok(None);
        };
        let Some(row) = self.first_repeat(&batch)? else {
            return Ok(Some(batch));
        };
        // The statement ends before `row`; the rest starts the next one. A
        // statement never ends at its own first row: the key set is empty when a
        // statement starts, so `row` is 0 only after this statement took rows
        // from an earlier batch.
        self.held = Some(batch.slice(row, batch.num_rows() - row));
        self.ended = true;
        if row == 0 {
            return Ok(None);
        }
        Ok(Some(batch.slice(0, row)))
    }

    /// The first row of `batch` whose key the current statement already holds
    /// (or that would take it past `max_keys`), recording the keys of the rows
    /// before it. A row with a NULL in its key conflicts with nothing, so it is
    /// neither recorded nor a repeat.
    fn first_repeat(&mut self, batch: &RecordBatch) -> Result<Option<usize>, ArrowError> {
        let columns: Vec<ArrayRef> = self
            .key_indices
            .iter()
            .map(|&index| canonical_key_column(batch.column(index)))
            .collect();
        let rows = self.converter.convert_columns(&columns)?;
        for row in 0..batch.num_rows() {
            if columns.iter().any(|column| column.is_null(row)) {
                continue;
            }
            if self.seen.len() >= self.max_keys {
                return Ok(Some(row));
            }
            let mut hasher = DefaultHasher::new();
            rows.row(row).as_ref().hash(&mut hasher);
            if !self.seen.insert(hasher.finish()) {
                return Ok(Some(row));
            }
        }
        Ok(None)
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
        _ => Err(ArrowError::SchemaError(format!(
            "on_conflict column '{column}' is not a column of the written data"
        ))),
    }
}

/// `column` with each float `DuckDB` holds as one key written one way: `-0.0` as
/// `0.0`, and every NaN as the same NaN. `DuckDB` treats `-0.0` as a repeat of a
/// stored `0.0` and every NaN as one key, while their bits, and arrow's row
/// format, tell them apart. The write still carries the values it was given;
/// only the keys compared change. Any other column is returned as is.
fn canonical_key_column(column: &ArrayRef) -> ArrayRef {
    match column.data_type() {
        DataType::Float32 => {
            let array: &Float32Array = column.as_primitive::<Float32Type>();
            Arc::new(array.unary::<_, Float32Type>(canonical_f32))
        }
        DataType::Float64 => {
            let array: &Float64Array = column.as_primitive::<Float64Type>();
            Arc::new(array.unary::<_, Float64Type>(canonical_f64))
        }
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
pub(super) struct StatementReader<S> {
    groups: Arc<Mutex<UpsertGroups<S>>>,
    schema: SchemaRef,
}

impl<S: BatchSource> StatementReader<S> {
    pub(super) fn new(groups: Arc<Mutex<UpsertGroups<S>>>, schema: SchemaRef) -> Self {
        Self { groups, schema }
    }
}

impl<S: BatchSource> Iterator for StatementReader<S> {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.groups
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .next_in_statement()
            .transpose()
    }
}

impl<S: BatchSource> RecordBatchReader for StatementReader<S> {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::datatypes::Field;

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

    fn float_batch(rows: &[(f64, &str)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Float64, false),
            Field::new("v", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(
                    rows.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
                )),
            ],
        )
        .expect("batch")
    }

    /// Drains `groups` statement by statement, as `write_to_table` does, and
    /// returns each statement's rows as `(id, v)` pairs.
    fn statements<S: BatchSource>(mut groups: UpsertGroups<S>) -> Vec<Vec<(Option<i64>, String)>> {
        let mut all = Vec::new();
        loop {
            groups.start_statement();
            let mut rows = Vec::new();
            while let Some(batch) = groups.next_in_statement().expect("next rows") {
                let ids = batch
                    .column(0)
                    .as_primitive::<arrow::datatypes::Int64Type>();
                let vs = batch.column(1).as_string::<i32>();
                for row in 0..batch.num_rows() {
                    let id = (!ids.is_null(row)).then(|| ids.value(row));
                    rows.push((id, vs.value(row).to_string()));
                }
            }
            all.push(rows);
            if groups.is_exhausted() {
                return all;
            }
        }
    }

    fn groups(
        batches: Vec<RecordBatch>,
        keys: &[&str],
    ) -> UpsertGroups<std::vec::IntoIter<RecordBatch>> {
        let schema = batches.first().map_or_else(schema, RecordBatch::schema);
        UpsertGroups::try_new(batches.into_iter(), &schema, keys.iter().copied()).expect("groups")
    }

    fn rows(pairs: &[(Option<i64>, &str)]) -> Vec<(Option<i64>, String)> {
        pairs
            .iter()
            .map(|(id, v)| (*id, (*v).to_string()))
            .collect()
    }

    #[test]
    fn a_write_that_repeats_no_key_is_one_statement() {
        let got = statements(groups(
            vec![
                batch(&[(Some(1), "a"), (Some(2), "b")]),
                batch(&[(Some(3), "c")]),
            ],
            &["id"],
        ));
        assert_eq!(
            got,
            vec![rows(&[(Some(1), "a"), (Some(2), "b"), (Some(3), "c")])]
        );
    }

    #[test]
    fn a_key_repeated_in_a_later_batch_starts_a_new_statement_at_that_batch() {
        let got = statements(groups(
            vec![
                batch(&[(Some(1), "a"), (Some(2), "b")]),
                batch(&[(Some(1), "c"), (Some(3), "d")]),
            ],
            &["id"],
        ));
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
        let got = statements(groups(
            vec![batch(&[
                (Some(1), "a"),
                (Some(2), "b"),
                (Some(1), "c"),
                (Some(2), "d"),
            ])],
            &["id"],
        ));
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
        let got = statements(groups(
            vec![
                batch(&[(Some(1), "a")]),
                batch(&[(Some(1), "b")]),
                batch(&[(Some(1), "c")]),
            ],
            &["id"],
        ));
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
        let got = statements(groups(
            vec![
                batch(&[(None, "a"), (Some(1), "b")]),
                batch(&[(None, "c"), (None, "d")]),
            ],
            &["id"],
        ));
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
        let got = statements(groups(vec![], &["id"]));
        assert_eq!(got, vec![Vec::new()]);
    }

    #[test]
    fn an_empty_batch_is_passed_through() {
        let got = statements(groups(
            vec![
                batch(&[]),
                batch(&[(Some(1), "a")]),
                batch(&[]),
                batch(&[(Some(1), "b")]),
            ],
            &["id"],
        ));
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
        let groups = UpsertGroups::try_with_max_keys(batches.into_iter(), &schema(), ["id"], 2)
            .expect("groups");
        assert_eq!(
            statements(groups),
            vec![
                rows(&[(Some(1), "a"), (Some(2), "b")]),
                rows(&[(Some(3), "c"), (Some(4), "d")]),
                rows(&[(Some(1), "e")]),
            ]
        );
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
        let mut groups =
            UpsertGroups::try_new(vec![b].into_iter(), &schema, ["a", "b"]).expect("groups");
        groups.start_statement();
        let first = groups.next_in_statement().expect("rows").expect("a batch");
        assert_eq!(first.num_rows(), 3, "(1,x) (1,y) (2,x) share no full key");
        assert!(groups.next_in_statement().expect("rows").is_none());
        groups.start_statement();
        let second = groups.next_in_statement().expect("rows").expect("a batch");
        assert_eq!(second.num_rows(), 1, "(1,x) again");
        assert!(groups.next_in_statement().expect("rows").is_none());
        assert!(groups.is_exhausted());
    }

    #[test]
    fn negative_zero_and_every_nan_repeat_the_float_key_duckdb_holds_them_as() {
        let batches = vec![
            float_batch(&[(0.0, "zero"), (f64::NAN, "nan")]),
            float_batch(&[(-0.0, "negative zero"), (-f64::NAN, "another nan")]),
        ];
        let schema = batches[0].schema();
        let mut groups =
            UpsertGroups::try_new(batches.into_iter(), &schema, ["id"]).expect("groups");
        groups.start_statement();
        let first = groups.next_in_statement().expect("rows").expect("a batch");
        assert_eq!(first.num_rows(), 2);
        assert!(groups.next_in_statement().expect("rows").is_none());
        groups.start_statement();
        let second = groups.next_in_statement().expect("rows").expect("a batch");
        assert_eq!(
            second.num_rows(),
            2,
            "-0.0 repeats 0.0, so the statement ended before it"
        );
        let ids = second.column(0).as_primitive::<Float64Type>();
        assert!(
            ids.value(0).is_sign_negative(),
            "the write still carries -0.0; only the compared key is canonical"
        );
        assert!(groups.next_in_statement().expect("rows").is_none());
        assert!(groups.is_exhausted());
    }

    #[test]
    fn the_conflict_target_matches_a_column_ignoring_ascii_case() {
        let got = statements(groups(
            vec![batch(&[(Some(1), "a")]), batch(&[(Some(1), "b")])],
            &["ID"],
        ));
        assert_eq!(got, vec![rows(&[(Some(1), "a")]), rows(&[(Some(1), "b")])]);
    }

    #[test]
    fn a_conflict_target_that_names_no_column_is_an_error() {
        let err = UpsertGroups::try_new(Vec::<RecordBatch>::new().into_iter(), &schema(), ["nope"])
            .err()
            .expect("an error");
        assert_eq!(
            err.to_string(),
            "Schema error: on_conflict column 'nope' is not a column of the written data"
        );
    }

    #[test]
    fn a_statement_reader_ends_at_the_repeat_and_the_next_one_resumes_there() {
        let batches = vec![batch(&[(Some(1), "a")]), batch(&[(Some(1), "b")])];
        let schema = batches[0].schema();
        let groups = Arc::new(Mutex::new(
            UpsertGroups::try_new(batches.into_iter(), &schema, ["id"]).expect("groups"),
        ));
        let mut statements_read = Vec::new();
        loop {
            groups.lock().expect("lock").start_statement();
            let reader = StatementReader::new(Arc::clone(&groups), Arc::clone(&schema));
            assert_eq!(reader.schema(), schema);
            let rows: usize = reader.map(|b| b.expect("batch").num_rows()).sum();
            statements_read.push(rows);
            if groups.lock().expect("lock").is_exhausted() {
                break;
            }
        }
        assert_eq!(statements_read, vec![1, 1]);
    }
}
