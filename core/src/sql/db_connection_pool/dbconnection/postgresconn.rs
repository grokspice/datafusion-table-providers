use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

use crate::sql::arrow_sql_gen::postgres::rows_to_arrow;
use crate::sql::arrow_sql_gen::postgres::schema::pg_data_type_to_arrow_type;
use crate::sql::arrow_sql_gen::postgres::schema::ParseContext;
use crate::sql::db_connection_pool::postgrespool::ConnectionManager;
use crate::util::handle_unsupported_type_error;
use crate::util::schema::SchemaValidator;
use crate::UnsupportedTypeAction;
use arrow::datatypes::Field;
use arrow::datatypes::Schema;
use arrow::datatypes::SchemaRef;
use arrow_schema::DataType;
use async_stream::stream;
use bb8_postgres::tokio_postgres::types::ToSql;

fn maybe_db_source_err(err: tokio_postgres::Error) -> Box<dyn Error + Send + Sync> {
    if let Some(err) = err.as_db_error() {
        Box::new(err.clone()) as Box<dyn Error + Send + Sync>
    } else {
        Box::new(err)
    }
}

/// A pooled Postgres connection obtained from a [`PostgresConnectionPool`](crate::sql::db_connection_pool::postgrespool::PostgresConnectionPool).
///
/// Dereferences to [`tokio_postgres::Client`](bb8_postgres::tokio_postgres::Client) for executing queries.
// Defined here rather than in `postgrespool` to avoid a type-resolution cycle
// between the two modules (postgrespool imports PostgresConnection, which uses
// this alias).
pub type PostgresPooledConnection = bb8::PooledConnection<'static, ConnectionManager>;
use datafusion::common::TableReference;
use datafusion::error::DataFusionError;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use futures::StreamExt;

use snafu::prelude::*;
use tokio_postgres::Row;

use super::AsyncDbConnection;
use super::DbConnection;
use super::Result;

const SCHEMA_QUERY: &str = r"
WITH custom_type_details AS (
SELECT
t.typname,
t.typtype,
CASE
    WHEN t.typtype = 'e' THEN
        jsonb_build_object(
            'type', 'enum',
            'values', (
                SELECT jsonb_agg(e.enumlabel ORDER BY e.enumsortorder)
                FROM pg_enum e
                WHERE e.enumtypid = t.oid
            )
        )
    WHEN t.typtype = 'c' THEN
        jsonb_build_object(
            'type', 'composite',
            'attributes', (
                SELECT jsonb_agg(
                    jsonb_build_object(
                        'name', a2.attname,
                        'type', pg_catalog.format_type(a2.atttypid, a2.atttypmod)
                    )
                    ORDER BY a2.attnum
                )
                FROM pg_attribute a2
                WHERE a2.attrelid = t.typrelid
                AND a2.attnum > 0
                AND NOT a2.attisdropped
            )
        )
END as type_details
FROM pg_type t
JOIN pg_namespace n ON t.typnamespace = n.oid
WHERE n.nspname = $1
)
SELECT
    cls.relname AS table_name,
    a.attname AS column_name,
    CASE
    -- when an array type is encountered, label as 'array'
    WHEN t.typcategory = 'A' THEN 'array'
    -- if it’s a user-defined enum or composite type then output that specific string
    WHEN t.typtype = 'e' THEN 'enum'
    WHEN t.typtype = 'c' THEN 'composite'
    ELSE pg_catalog.format_type(a.atttypid, a.atttypmod)
    END AS data_type,
    -- A foreign table's NOT NULL is an unverified promise about remote data:
    -- PostgreSQL never enforces it, so the remote can and does return NULL for
    -- such a column. Reporting it non-nullable would let a consumer assume a
    -- null-free column and produce wrong results, so foreign tables are always
    -- reported nullable.
    CASE WHEN a.attnotnull AND cls.relkind <> 'f' THEN 'NO' ELSE 'YES' END AS is_nullable,
    CASE
    WHEN t.typcategory = 'A' THEN
        jsonb_build_object(
        'type', 'array',
        'element_type', (
            SELECT pg_catalog.format_type(et.oid, a.atttypmod)
            FROM pg_type t2
            JOIN pg_type et ON t2.typelem = et.oid
            WHERE t2.oid = a.atttypid
        ),
        -- When the array element is a composite type, carry its attributes so the
        -- array can be resolved to a List<Struct>. Null for non-composite elements.
        'element_details', (
            SELECT jsonb_build_object(
                'type', 'composite',
                'attributes', (
                    SELECT jsonb_agg(
                        jsonb_build_object(
                            'name', a2.attname,
                            'type', pg_catalog.format_type(a2.atttypid, a2.atttypmod)
                        )
                        ORDER BY a2.attnum
                    )
                    FROM pg_attribute a2
                    WHERE a2.attrelid = et.typrelid
                    AND a2.attnum > 0
                    AND NOT a2.attisdropped
                )
            )
            FROM pg_type t2
            JOIN pg_type et ON t2.typelem = et.oid
            WHERE t2.oid = a.atttypid
            AND et.typtype = 'c'
        )
        )
    ELSE custom.type_details
    END AS type_details
FROM pg_class cls
JOIN pg_namespace ns ON cls.relnamespace = ns.oid
JOIN pg_attribute a ON a.attrelid = cls.oid
LEFT JOIN pg_type t ON t.oid = a.atttypid
LEFT JOIN custom_type_details custom ON custom.typname = t.typname
WHERE ns.nspname = $1
    -- Named rather than optional: a caller resolving many relations at once
    -- lists exactly the ones it will use, so the query never describes a
    -- relation nobody asked about.
    AND cls.relname = ANY($2)
    -- covers tables, normal views, materialized views, partitioned tables, &
    -- foreign tables. Foreign tables carry a full local column definition in
    -- pg_attribute, so their schema resolves here rather than falling through
    -- to inferring it from a data query (see `infer_schema_from_data`).
    AND cls.relkind IN ('r','v','m','p','f')
    AND a.attnum > 0
    AND NOT a.attisdropped
ORDER BY cls.relname, a.attnum;
";

// Redshift schema inference uses `SHOW COLUMNS FROM TABLE <db>.<schema>.<table>` rather
// than the `svv_*` catalog views. The catalog views truncate `data_type` at the source
// (`svv_all_columns.data_type` is `varchar(128)`; `svv_external_columns.external_type` is
// `text`, i.e. `varchar(256)` on Redshift), which corrupts deeply nested Spectrum complex
// types (`array<struct<...>>`) — and casting can't recover characters already dropped
// inside the view. `SHOW COLUMNS` returns the full, untruncated type.
//
// `SHOW COLUMNS` is a standalone command: it can't take bind parameters or be wrapped in
// a CTE/subquery, so we interpolate quoted identifiers (see `quote_pg_identifier`) and do
// the `numeric(p,s)` rebuild and `is_nullable` normalization in Rust instead of SQL (see
// `redshift_columns`). It covers local, datashare, and external tables.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostgresVariant {
    Default,
    Redshift,
}

/// A column description normalized across the per-variant schema-inference queries, so
/// `get_schema` can build Arrow fields without caring how the metadata was obtained.
struct ColumnDef {
    name: String,
    /// Fully formatted SQL type string (e.g. `numeric(10,2)`, `array<struct<...>>`).
    data_type: String,
    nullable: bool,
    type_details: Option<serde_json::Value>,
}

/// One `SCHEMA_QUERY` row as a [`ColumnDef`], read by column name so that adding
/// to the projection cannot silently shift a field onto the wrong column.
fn column_def_from_row(row: &Row) -> ColumnDef {
    ColumnDef {
        name: row.get("column_name"),
        // `data_type` is already a formatted type string via
        // `pg_catalog.format_type` (e.g. `numeric(10,2)`).
        data_type: row.get("data_type"),
        nullable: row.get::<_, String>("is_nullable") == "YES",
        type_details: row.get("type_details"),
    }
}

/// Classifies a server from what `SELECT version()` reports.
///
/// Split out from the query so the classification is testable without a server,
/// and so the pool can detect the variant once while validating itself.
#[must_use]
pub fn variant_from_version(version: &str) -> PostgresVariant {
    if version.contains("Redshift") {
        PostgresVariant::Redshift
    } else {
        PostgresVariant::Default
    }
}

/// Quotes a SQL identifier for safe interpolation into a statement that can't use bind
/// parameters (e.g. Redshift `SHOW COLUMNS`), doubling any embedded double quotes.
fn quote_pg_identifier(ident: &str) -> String {
    format!("\"{}\"", ident.replace('"', "\"\""))
}

/// Reads an optional integer column without assuming its wire integer width
/// (`SHOW COLUMNS` may report `numeric_precision`/`numeric_scale` as int2/int4/int8).
fn row_opt_int(row: &Row, column: &str) -> Option<i64> {
    if let Ok(v) = row.try_get::<_, Option<i32>>(column) {
        return v.map(i64::from);
    }
    if let Ok(v) = row.try_get::<_, Option<i64>>(column) {
        return v;
    }
    if let Ok(v) = row.try_get::<_, Option<i16>>(column) {
        return v.map(i64::from);
    }
    None
}

/// Maps an error from the catalog schema query into the right `super::Error`, surfacing a
/// missing relation as [`super::Error::UndefinedTable`].
fn map_schema_query_error(
    e: tokio_postgres::Error,
    table_reference: &TableReference,
) -> super::Error {
    if let Some(db_error) = e.as_db_error() {
        if db_error.code() == &tokio_postgres::error::SqlState::UNDEFINED_TABLE {
            return super::Error::UndefinedTable {
                source: Box::new(db_error.clone()),
                table_name: table_reference.to_string(),
            };
        }
    }
    super::Error::UnableToGetSchema {
        source: maybe_db_source_err(e),
    }
}

const SCHEMAS_QUERY: &str = "
SELECT nspname AS schema_name
FROM pg_namespace
WHERE nspname NOT IN ('pg_catalog', 'information_schema')
  AND nspname !~ '^pg_toast';
";

const TABLES_QUERY: &str = "
SELECT tablename
FROM pg_tables
WHERE schemaname = $1;
";

// Schema/table discovery reads from the unified `svv_all_*` system views so that
// schemas/tables consumed from a datashare (absent from the local `pg_catalog`) and
// external schemas/tables (Redshift Spectrum) are listed alongside the connected
// database's own schemas/tables. `svv_all_schemas`/`svv_all_tables` are AWS's unions of
// the `svv_redshift_*` views with the external schemas/tables. `current_database()`
// scopes results to the connected (datashare consumer) database. (Per-column type
// inference, by contrast, uses `SHOW COLUMNS` — see `redshift_columns` — because the
// `svv_*` views truncate `data_type`.)
const REDSHIFT_SCHEMAS_QUERY: &str = "
SELECT schema_name
FROM svv_all_schemas
WHERE database_name = current_database()
  AND schema_name NOT IN ('pg_catalog', 'information_schema')
  AND schema_name NOT LIKE 'pg_temp_%'
  AND schema_name NOT LIKE 'pg_toast%';
";

const REDSHIFT_TABLES_QUERY: &str = "
SELECT table_name
FROM svv_all_tables
WHERE database_name = current_database()
  AND schema_name = $1;
";

#[derive(Debug, Snafu)]
pub enum PostgresError {
    #[snafu(display(
        "Query execution failed.\n{}\nFor details, refer to the PostgreSQL manual: https://www.postgresql.org/docs/17/index.html",
        format_postgres_query_error(source)
    ))]
    QueryError {
        source: bb8_postgres::tokio_postgres::Error,
    },

    #[snafu(display("Failed to convert query result to Arrow.\n{source}\nReport a bug to request support: https://github.com/datafusion-contrib/datafusion-table-providers/issues"))]
    ConversionError {
        source: crate::sql::arrow_sql_gen::postgres::Error,
    },
}

fn format_postgres_query_error(source: &bb8_postgres::tokio_postgres::Error) -> String {
    let Some(db_error) = source.as_db_error() else {
        return source.to_string();
    };

    let mut rendered = format!("{db_error}\nSQLSTATE: {}", db_error.code().code());

    let context: Vec<_> = [
        ("schema", db_error.schema()),
        ("table", db_error.table()),
        ("column", db_error.column()),
    ]
    .into_iter()
    .filter_map(|(k, v)| v.map(|v| format!("{k}={v}")))
    .collect();

    if !context.is_empty() {
        rendered.push_str("\nCONTEXT: ");
        rendered.push_str(&context.join(", "));
    }

    rendered
}

pub struct PostgresConnection {
    pub conn: PostgresPooledConnection,
    unsupported_type_action: UnsupportedTypeAction,
    /// The server's variant, detected once when the pool was built.
    ///
    /// It describes the server, not the connection, so asking per connection
    /// would repeat a fixed answer: each checkout builds a fresh
    /// `PostgresConnection`, and discovery takes one per table. Resolving it
    /// with the pool -- which already connects to validate itself -- means one
    /// `SELECT version()` for the pool's lifetime instead of one per caller,
    /// with no cell to initialize and no race to coordinate.
    variant: PostgresVariant,
}

impl SchemaValidator for PostgresConnection {
    type Error = super::Error;

    fn is_data_type_supported(data_type: &DataType) -> bool {
        !matches!(data_type, DataType::Map(_, _))
    }

    fn unsupported_type_error(data_type: &DataType, field_name: &str) -> Self::Error {
        super::Error::UnsupportedDataType {
            data_type: data_type.to_string(),
            field_name: field_name.to_string(),
        }
    }
}

impl<'a> DbConnection<PostgresPooledConnection, &'a (dyn ToSql + Sync)> for PostgresConnection {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn as_async(
        &self,
    ) -> Option<&dyn AsyncDbConnection<PostgresPooledConnection, &'a (dyn ToSql + Sync)>> {
        Some(self)
    }
}

#[async_trait::async_trait]
impl<'a> AsyncDbConnection<PostgresPooledConnection, &'a (dyn ToSql + Sync)>
    for PostgresConnection
{
    fn new(conn: PostgresPooledConnection) -> Self {
        PostgresConnection {
            conn,
            unsupported_type_action: UnsupportedTypeAction::default(),
            // Overridden by the pool, which has already detected it. A connection
            // built outside a pool assumes vanilla PostgreSQL.
            variant: PostgresVariant::Default,
        }
    }

    async fn tables(&self, schema: &str) -> Result<Vec<String>, super::Error> {
        let query = match self.variant() {
            PostgresVariant::Default => TABLES_QUERY,
            PostgresVariant::Redshift => REDSHIFT_TABLES_QUERY,
        };

        let rows = self.conn.query(query, &[&schema]).await.map_err(|e| {
            super::Error::UnableToGetTables {
                source: maybe_db_source_err(e),
            }
        })?;

        Ok(rows.iter().map(|r| r.get::<usize, String>(0)).collect())
    }

    async fn schemas(&self) -> Result<Vec<String>, super::Error> {
        let query = match self.variant() {
            PostgresVariant::Default => SCHEMAS_QUERY,
            PostgresVariant::Redshift => REDSHIFT_SCHEMAS_QUERY,
        };

        let rows =
            self.conn
                .query(query, &[])
                .await
                .map_err(|e| super::Error::UnableToGetSchemas {
                    source: maybe_db_source_err(e),
                })?;

        Ok(rows.iter().map(|r| r.get::<usize, String>(0)).collect())
    }

    async fn get_schema(
        &self,
        table_reference: &TableReference,
    ) -> Result<SchemaRef, super::Error> {
        let (variant, columns) = self.query_variant_and_schema(table_reference).await?;

        // Native inference can return zero rows even though the table exists and is
        // queryable — e.g. Redshift datashare objects that aren't represented in the
        // catalog views we query. Fall back to inferring the schema from a sample row.
        if columns.is_empty() {
            tracing::warn!(
                "Native PostgreSQL schema inference returned no data. Inferring schema from table data rows."
            );
            return self.infer_schema_from_data(table_reference).await;
        }

        self.schema_from_columns(columns, variant)
    }

    async fn query_arrow(
        &self,
        sql: &str,
        params: &[&'a (dyn ToSql + Sync)],
        projected_schema: Option<SchemaRef>,
    ) -> Result<SendableRecordBatchStream> {
        // TODO: We should have a way to detect if params have been passed
        // if they haven't we should use .copy_out instead, because it should be much faster
        let streamable = self
            .conn
            .query_raw(sql, params.iter().copied()) // use .query_raw to get access to the underlying RowStream
            .await
            .context(QuerySnafu)?;

        // chunk the stream into groups of rows
        let mut stream = streamable.chunks(4_000).boxed().map(move |rows| {
            let rows = rows
                .into_iter()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context(QuerySnafu)?;
            let rec = rows_to_arrow(rows.as_slice(), &projected_schema).context(ConversionSnafu)?;
            Ok::<_, PostgresError>(rec)
        });

        let Some(first_chunk) = stream.next().await else {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                Arc::new(Schema::empty()),
                stream::empty(),
            )));
        };

        let first_chunk = first_chunk?;
        let schema = first_chunk.schema(); // pull out the schema from the first chunk to use in the DataFusion Stream Adapter

        let output_stream = stream! {
           yield Ok(first_chunk);
           while let Some(batch) = stream.next().await {
                match batch {
                    Ok(batch) => {
                        yield Ok(batch); // we can yield the batch as-is because we've already converted to Arrow in the chunk map
                    }
                    Err(e) => {
                        yield Err(DataFusionError::Execution(format!("Failed to fetch batch: {e}")));
                    }
                }
           }
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            output_stream,
        )))
    }

    async fn execute(&self, sql: &str, params: &[&'a (dyn ToSql + Sync)]) -> Result<u64> {
        Ok(self.conn.execute(sql, params).await?)
    }
}

impl PostgresConnection {
    #[must_use]
    pub fn with_unsupported_type_action(mut self, action: UnsupportedTypeAction) -> Self {
        self.unsupported_type_action = action;
        self
    }

    /// Applies the variant the pool detected when it was built, so every
    /// connection it hands out reports the same server without asking again.
    #[must_use]
    pub fn with_variant(mut self, variant: PostgresVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Builds the Arrow schema for one relation's [`ColumnDef`]s.
    ///
    /// Shared by the single-table and whole-schema paths so the type mapping and
    /// `unsupported_type_action` handling cannot drift between them.
    fn schema_from_columns(
        &self,
        columns: Vec<ColumnDef>,
        variant: PostgresVariant,
    ) -> Result<SchemaRef, super::Error> {
        let mut fields = Vec::new();
        for column in columns {
            let mut context =
                ParseContext::new().with_unsupported_type_action(self.unsupported_type_action);

            if let Some(type_details) = column.type_details {
                context = context.with_type_details(type_details);
            };

            let Ok(arrow_type) =
                pg_data_type_to_arrow_type(&column.data_type, &context, Some(variant))
            else {
                handle_unsupported_type_error(
                    self.unsupported_type_action,
                    super::Error::UnsupportedDataType {
                        data_type: column.data_type.clone(),
                        field_name: column.name.clone(),
                    },
                )?;

                continue;
            };

            fields.push(Field::new(column.name, arrow_type, column.nullable));
        }

        Ok(Arc::new(Schema::new(fields)))
    }

    /// The Arrow schemas of `relations` in `schema_name`, resolved in one round
    /// trip instead of one per table.
    ///
    /// A caller otherwise calls `get_schema` per table, and each call is its own
    /// query. This answers all of them at once, turning `T` round trips into one.
    ///
    /// `relations` is named rather than implied so the query describes exactly
    /// what the caller will use. A catalog's discovered set is narrower than the
    /// namespace -- it excludes inheritance children and relations the role
    /// cannot read, along with anything `include`/`exclude` withhold -- and
    /// resolving the namespace instead would fetch columns that are then
    /// discarded, which for a partitioned table means every leaf.
    ///
    /// The result is deliberately *incomplete rather than wrong*: a relation the
    /// catalog query cannot describe is simply absent, and the caller resolves it
    /// with `get_schema`, which keeps the `infer_schema_from_data` fallback that
    /// Redshift datashare objects rely on. Redshift itself answers with an empty
    /// map, since `SHOW COLUMNS` is per-table and cannot be batched, so every
    /// table falls back to that path.
    ///
    /// Callers must therefore treat a missing entry as "ask per table", never as
    /// "this relation has no columns".
    pub async fn get_schemas_in(
        &self,
        schema_name: &str,
        relations: &[String],
    ) -> Result<HashMap<String, SchemaRef>, super::Error> {
        // Nothing to describe, and `= ANY('{}')` would match nothing anyway.
        if relations.is_empty() {
            return Ok(HashMap::new());
        }

        let variant = self.variant();
        if variant == PostgresVariant::Redshift {
            return Ok(HashMap::new());
        }

        let rows = self
            .conn
            .query(SCHEMA_QUERY, &[&schema_name, &relations])
            .await
            .map_err(|e| super::Error::UnableToGetSchema {
                source: maybe_db_source_err(e),
            })?;

        let mut columns_by_table: HashMap<String, Vec<ColumnDef>> = HashMap::new();
        for row in &rows {
            columns_by_table
                .entry(row.get("table_name"))
                .or_default()
                .push(column_def_from_row(row));
        }

        let mut schemas = HashMap::with_capacity(columns_by_table.len());
        for (table, columns) in columns_by_table {
            schemas.insert(table, self.schema_from_columns(columns, variant)?);
        }

        Ok(schemas)
    }

    /// The server's variant, as detected when the pool was built. Infallible and
    /// free: nothing is queried here.
    #[must_use]
    pub fn variant(&self) -> PostgresVariant {
        self.variant
    }

    /// The server's variant. Kept for callers written against the earlier
    /// signature.
    ///
    /// This no longer queries the server and cannot fail: the variant is
    /// detected once when the pool is built, so this returns the stored field
    /// and the `Result` is always `Ok`. Prefer [`PostgresConnection::variant`],
    /// which says so in its type.
    #[expect(
        clippy::unused_async,
        reason = "kept async so existing `get_variant().await` call sites still compile"
    )]
    pub async fn get_variant(&self) -> Result<PostgresVariant, super::Error> {
        Ok(self.variant)
    }

    async fn query_variant_and_schema(
        &self,
        table_reference: &TableReference,
    ) -> Result<(PostgresVariant, Vec<ColumnDef>), super::Error> {
        let table_name = table_reference.table();
        let schema_name = table_reference.schema().unwrap_or("public");

        let variant = self.variant();

        let columns = match variant {
            PostgresVariant::Default => {
                let only = [table_name];
                let rows = self
                    .conn
                    .query(SCHEMA_QUERY, &[&schema_name, &&only[..]])
                    .await
                    .map_err(|e| map_schema_query_error(e, table_reference))?;

                rows.iter().map(column_def_from_row).collect()
            }
            PostgresVariant::Redshift => {
                self.redshift_columns(table_reference.catalog(), schema_name, table_name)
                    .await?
            }
        };

        Ok((variant, columns))
    }

    /// Infers a Redshift table's columns via `SHOW COLUMNS FROM TABLE`, which (unlike the
    /// `svv_*` catalog views) returns the full, untruncated `data_type` — essential for
    /// deeply nested Spectrum complex types. It can't take bind parameters or be composed,
    /// so identifiers are quoted/interpolated and the `numeric(p,s)` rebuild + nullable
    /// normalization happen here in Rust rather than in SQL.
    async fn redshift_columns(
        &self,
        catalog: Option<&str>,
        schema_name: &str,
        table_name: &str,
    ) -> Result<Vec<ColumnDef>, super::Error> {
        // `SHOW COLUMNS` needs a fully-qualified 3-part name including the database. Use
        // the catalog from the table reference when the caller fully-qualified it;
        // otherwise scope to the connected database (matching the previous
        // `current_database()` filter).
        let database_name: String = match catalog {
            Some(catalog) => catalog.to_string(),
            None => self
                .conn
                .query_one("SELECT current_database()", &[])
                .await
                .map_err(|e| super::Error::UnableToGetSchema {
                    source: maybe_db_source_err(e),
                })?
                .try_get(0)
                .map_err(|e| super::Error::UnableToGetSchema {
                    source: maybe_db_source_err(e),
                })?,
        };

        let sql = format!(
            "SHOW COLUMNS FROM TABLE {}.{}.{}",
            quote_pg_identifier(&database_name),
            quote_pg_identifier(schema_name),
            quote_pg_identifier(table_name),
        );

        let rows = match self.conn.query(&sql, &[]).await {
            Ok(rows) => rows,
            // `SHOW COLUMNS` raises `UNDEFINED_TABLE` for relations it can't resolve in the
            // target database (e.g. cross-database datashare objects). Treat only that as a
            // miss and fall back to data-based inference — matching the catalog path's
            // empty-result behavior — while surfacing any other error (permissions, syntax,
            // connectivity) instead of silently hiding it.
            Err(e)
                if e.as_db_error().map(tokio_postgres::error::DbError::code)
                    == Some(&tokio_postgres::error::SqlState::UNDEFINED_TABLE) =>
            {
                tracing::debug!(
                    "Redshift SHOW COLUMNS: {schema_name}.{table_name} not found in {database_name}; falling back to data inference"
                );
                return Ok(Vec::new());
            }
            Err(e) => {
                return Err(super::Error::UnableToGetSchema {
                    source: maybe_db_source_err(e),
                })
            }
        };

        let columns = rows
            .iter()
            .map(|row| {
                let name: String = row.get("column_name");
                let mut data_type: String = row.get("data_type");

                // `SHOW COLUMNS` reports numeric/decimal as the bare type with precision
                // and scale in separate columns; rebuild the formatted `numeric(p,s)`.
                let lowered = data_type.to_lowercase();
                if lowered == "numeric" || lowered == "decimal" {
                    if let Some(precision) = row_opt_int(row, "numeric_precision") {
                        let scale = row_opt_int(row, "numeric_scale").unwrap_or(0);
                        data_type = format!("{data_type}({precision},{scale})");
                    }
                }

                // Normalize nullability: only an explicit no/false is non-nullable; an
                // empty/absent value (common for external columns) is treated as nullable.
                let nullable = row
                    .try_get::<_, Option<String>>("is_nullable")
                    .ok()
                    .flatten()
                    .map(|v| !matches!(v.to_lowercase().as_str(), "no" | "false"))
                    .unwrap_or(true);

                ColumnDef {
                    name,
                    data_type,
                    nullable,
                    type_details: None,
                }
            })
            .collect();

        Ok(columns)
    }

    /// Fallback schema inference used when native (catalog-based) inference returns no
    /// rows for a table that is nonetheless queryable. Reads a single row and derives the
    /// Arrow schema from its column metadata via `rows_to_arrow`.
    ///
    /// Note: an empty table yields an empty schema, since `rows_to_arrow` reads column
    /// types from the first returned row.
    async fn infer_schema_from_data(
        &self,
        table_reference: &TableReference,
    ) -> Result<SchemaRef, super::Error> {
        let rows = self
            .conn
            .query(&format!("SELECT * FROM {table_reference} LIMIT 1"), &[])
            .await
            .map_err(|e| {
                if let Some(error_source) = e.source() {
                    if let Some(pg_error) =
                        error_source.downcast_ref::<tokio_postgres::error::DbError>()
                    {
                        if pg_error.code() == &tokio_postgres::error::SqlState::UNDEFINED_TABLE {
                            return super::Error::UndefinedTable {
                                source: Box::new(pg_error.clone()),
                                table_name: table_reference.to_string(),
                            };
                        }
                    }
                }
                super::Error::UnableToGetSchema {
                    source: maybe_db_source_err(e),
                }
            })?;

        if rows.is_empty() {
            tracing::warn!(
                "Schema inference from table data rows returned no rows. The schema for table '{table_reference}' is empty.",
            );
            return Ok(Arc::new(Schema::empty()));
        }

        let rec =
            rows_to_arrow(rows.as_slice(), &None).map_err(|e| super::Error::UnableToGetSchema {
                source: Box::new(PostgresError::ConversionError { source: e }),
            })?;

        Ok(rec.schema())
    }
}

#[cfg(test)]
mod variant_tests {
    use super::{variant_from_version, PostgresVariant};

    /// Classification is pure, so it is checked against the real strings each
    /// server reports rather than needing one to be running.
    #[test]
    fn classifies_a_server_from_its_version_string() {
        assert_eq!(
            variant_from_version(
                "PostgreSQL 8.0.2 on i686-pc-linux-gnu, compiled by GCC gcc (GCC) 3.4.2, Redshift 1.0.12345"
            ),
            PostgresVariant::Redshift
        );
        assert_eq!(
            variant_from_version(
                "PostgreSQL 16.2 (Debian 16.2-1.pgdg120+2) on aarch64-unknown-linux-gnu"
            ),
            PostgresVariant::Default
        );
        // Anything unrecognized is treated as vanilla, which is the conservative
        // reading: Redshift-specific query paths are only taken when named.
        assert_eq!(variant_from_version(""), PostgresVariant::Default);
    }
}
