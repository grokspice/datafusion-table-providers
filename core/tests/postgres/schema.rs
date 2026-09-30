use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion::catalog::TableProviderFactory;
use datafusion::common::Constraints;
use datafusion::common::TableReference;
use datafusion::common::ToDFSchema;
use datafusion::logical_expr::CreateExternalTable;
use datafusion::prelude::SessionContext;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use crate::postgres::common;
use crate::postgres::PostgresTableProviderFactory;
use datafusion_table_providers::postgres::PostgresTableFactory;
use datafusion_table_providers::sql::db_connection_pool::dbconnection::postgresconn::{
    PostgresConnection, PostgresVariant,
};
use datafusion_table_providers::sql::db_connection_pool::dbconnection::AsyncDbConnection;
use datafusion_table_providers::sql::db_connection_pool::postgrespool::PostgresConnectionPool;
use datafusion_table_providers::sql::db_connection_pool::DbConnectionPool;
use datafusion_table_providers::util::secrets::to_secret_map;
use datafusion_table_providers::UnsupportedTypeAction;

const COMPLEX_TABLE_SQL: &str = include_str!("scripts/complex_table_pg.sql");

fn get_schema() -> SchemaRef {
    let fields = vec![
        Field::new("id", DataType::Int32, false),
        Field::new("large_id", DataType::Int64, true),
        Field::new("name", DataType::Utf8, true),
        Field::new("age", DataType::Int16, true),
        Field::new("height", DataType::Float64, true),
        Field::new("is_active", DataType::Boolean, true),
        Field::new(
            "created_at",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            true,
        ),
        Field::new("data", DataType::Binary, true),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ];

    Arc::new(Schema::new(fields))
}

#[tokio::test]
async fn test_postgres_schema_inference() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let factory = PostgresTableProviderFactory::new();
    let ctx = SessionContext::new();
    let table_name = "test_table";
    let schema = get_schema();

    let cmd = CreateExternalTable {
        schema: schema.to_dfschema_ref().expect("to df schema"),
        name: table_name.into(),
        locations: vec![],
        file_type: "".to_string(),
        table_partition_cols: vec![],
        if_not_exists: false,
        definition: None,
        order_exprs: vec![],
        unbounded: false,
        options: common::get_pg_params(port),
        constraints: Constraints::default(),
        column_defaults: HashMap::new(),
        temporary: false,
        or_replace: false,
    };
    let _ = factory
        .create(&ctx.state(), &cmd)
        .await
        .expect("table provider created");

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );
    let table_factory = PostgresTableFactory::new(postgres_pool);
    let table_provider = table_factory
        .table_provider(TableReference::bare(table_name))
        .await
        .expect("to create table provider");

    assert_eq!(table_provider.schema(), get_schema());

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

#[tokio::test]
async fn test_postgres_schema_inference_complex_types() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let table_name = "example_table";

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );

    let pg_conn = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");
    for cmd in COMPLEX_TABLE_SQL.split(";") {
        pg_conn
            .conn
            .execute(cmd, &[])
            .await
            .expect("to create table");
    }

    let table_factory = PostgresTableFactory::new(postgres_pool);
    let table_provider = table_factory
        .table_provider(TableReference::bare(table_name))
        .await
        .expect("to create table provider");

    let pretty_schema = format!("{:#?}", table_provider.schema());
    insta::assert_snapshot!(pretty_schema);

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

#[tokio::test]
async fn test_postgres_view_schema_inference() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );
    let pg_conn = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");

    for cmd in COMPLEX_TABLE_SQL.split(";") {
        if cmd.trim().is_empty() {
            continue;
        }
        pg_conn
            .conn
            .execute(cmd, &[])
            .await
            .expect("executing SQL from complex_table.sql");
    }

    let table_factory = PostgresTableFactory::new(postgres_pool.clone());
    let table_provider = table_factory
        .table_provider(TableReference::bare("example_view"))
        .await
        .expect("to create table provider for view");

    let pretty_schema = format!("{:#?}", table_provider.schema());
    insta::assert_snapshot!(pretty_schema);

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

#[tokio::test]
async fn test_postgres_materialized_view_schema_inference() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );
    let pg_conn = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");

    for cmd in COMPLEX_TABLE_SQL.split(";") {
        if cmd.trim().is_empty() {
            continue;
        }
        pg_conn
            .conn
            .execute(cmd, &[])
            .await
            .expect("executing SQL from complex_table.sql");
    }

    let table_factory = PostgresTableFactory::new(postgres_pool);
    let table_provider = table_factory
        .table_provider(TableReference::bare("example_materialized_view"))
        .await
        .expect("to create table provider for materialized view");

    let pretty_schema = format!("{:#?}", table_provider.schema());
    insta::assert_snapshot!(pretty_schema);

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

#[tokio::test]
async fn test_postgres_partitioned_table_schema_inference() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );
    let pg_conn = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");

    let partitioned_table_sql = r#"
        CREATE TABLE partitioned_orders (
            id INTEGER NOT NULL,
            order_date DATE NOT NULL,
            amount NUMERIC(10,2),
            description TEXT
        ) PARTITION BY RANGE (order_date);

        CREATE TABLE partitioned_orders_2024 PARTITION OF partitioned_orders
            FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

        CREATE TABLE partitioned_orders_2025 PARTITION OF partitioned_orders
            FOR VALUES FROM ('2025-01-01') TO ('2026-01-01')
    "#;

    for cmd in partitioned_table_sql.split(";") {
        if cmd.trim().is_empty() {
            continue;
        }
        pg_conn
            .conn
            .execute(cmd, &[])
            .await
            .expect("executing partitioned table SQL");
    }

    let table_factory = PostgresTableFactory::new(postgres_pool);
    let table_provider = table_factory
        .table_provider(TableReference::bare("partitioned_orders"))
        .await
        .expect("to create table provider for partitioned table");

    let expected_fields = vec![
        Field::new("id", DataType::Int32, false),
        Field::new("order_date", DataType::Date32, false),
        Field::new("amount", DataType::Decimal128(10, 2), true),
        Field::new("description", DataType::Utf8, true),
    ];
    let expected_schema = Arc::new(Schema::new(expected_fields));

    assert_eq!(table_provider.schema(), expected_schema);

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

/// A foreign table's schema comes from its local `pg_attribute` definition, so it
/// resolves without querying the relation's data.
///
/// Three properties are asserted together because they share one setup:
///
/// 1. Types resolve identically to an ordinary table (`relkind = 'f'` is covered by
///    the catalog query).
/// 2. An **empty** foreign table still reports its full column set. Data-based
///    inference yields `Schema::empty()` when the sample query returns no rows, so a
///    foreign table that happens to be empty at registration would otherwise register
///    with no columns at all.
/// 3. A column declared `NOT NULL` on the foreign table is reported **nullable**.
///    `PostgreSQL` never enforces that declaration against the remote data -- here
///    `note` is nullable on the remote and holds a NULL -- so treating it as non-null
///    would let a consumer assume a null-free column and return wrong results.
#[tokio::test]
async fn test_postgres_foreign_table_schema_inference() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );
    let pg_conn = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");

    let mut params = common::get_pg_params(port);
    let password = params.remove("pg_pass").expect("pg_pass should be present");

    // `loopback` points the foreign tables back at this same database, so the
    // remote side is real without needing a second container. The port is the
    // container-internal 5432, not the host-mapped one.
    let foreign_table_sql = format!(
        r#"
        CREATE TABLE remote_orders (
            id INTEGER NOT NULL,
            note TEXT,
            amount NUMERIC(10,2),
            order_date DATE,
            tags TEXT[]
        );

        INSERT INTO remote_orders (id, note, amount, order_date, tags)
            VALUES (1, NULL, 12.34, DATE '2024-01-01', ARRAY['a','b']);

        CREATE TABLE remote_empty (id INTEGER NOT NULL, label TEXT);

        CREATE EXTENSION IF NOT EXISTS postgres_fdw;

        CREATE SERVER loopback FOREIGN DATA WRAPPER postgres_fdw
            OPTIONS (host 'localhost', port '5432', dbname 'postgres');

        CREATE USER MAPPING FOR postgres SERVER loopback
            OPTIONS (user 'postgres', password '{password}');

        CREATE FOREIGN TABLE ft_orders (
            id INTEGER NOT NULL,
            note TEXT NOT NULL,
            amount NUMERIC(10,2),
            order_date DATE,
            tags TEXT[]
        ) SERVER loopback OPTIONS (table_name 'remote_orders');

        CREATE FOREIGN TABLE ft_empty (
            id INTEGER NOT NULL,
            label TEXT
        ) SERVER loopback OPTIONS (table_name 'remote_empty')
    "#
    );

    for cmd in foreign_table_sql.split(";") {
        if cmd.trim().is_empty() {
            continue;
        }
        pg_conn
            .conn
            .execute(cmd, &[])
            .await
            .expect("executing foreign table SQL");
    }

    let table_factory = PostgresTableFactory::new(postgres_pool);

    let orders = table_factory
        .table_provider(TableReference::bare("ft_orders"))
        .await
        .expect("to create table provider for foreign table");

    // Every field is nullable, including `id` and `note`, which the foreign table
    // declares NOT NULL.
    let expected_orders = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("note", DataType::Utf8, true),
        Field::new("amount", DataType::Decimal128(10, 2), true),
        Field::new("order_date", DataType::Date32, true),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]));
    assert_eq!(orders.schema(), expected_orders);

    // The remote table is empty; the schema must still be complete.
    let empty = table_factory
        .table_provider(TableReference::bare("ft_empty"))
        .await
        .expect("to create table provider for empty foreign table");

    let expected_empty = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("label", DataType::Utf8, true),
    ]));
    assert_eq!(empty.schema(), expected_empty);

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

/// The pool detects the server's variant when it is built, and every connection
/// it hands out reports that same value.
///
/// That no *further* query happens is structural rather than something to
/// measure: [`PostgresConnection::variant`] is a synchronous accessor over a
/// resolved field, with no connection in reach, so there is no code path left
/// that could ask the server again. Detecting it lazily instead would need a
/// cell to initialize and concurrent cold callers to coordinate, for an answer
/// that cannot change while the pool lives.
///
/// What is worth asserting, then, is that the detection reaches both
/// construction paths -- `connect()` and `connect_direct()` -- since a
/// connection built without it would silently fall back to `Default` and take
/// vanilla query paths against a Redshift server.
#[tokio::test]
async fn test_postgres_variant_is_detected_once_per_pool() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );

    let direct = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");
    assert_eq!(
        direct.variant(),
        PostgresVariant::Default,
        "connect_direct must carry the variant the pool detected"
    );
    // The compatibility method must keep reporting what the accessor does, and
    // must not have reacquired a way to fail.
    assert_eq!(
        direct
            .get_variant()
            .await
            .expect("get_variant no longer queries, so it cannot fail"),
        direct.variant(),
        "get_variant must agree with variant()"
    );

    let pooled = DbConnectionPool::connect(&*postgres_pool)
        .await
        .expect("to connect to postgres");
    let pooled = pooled
        .as_any()
        .downcast_ref::<PostgresConnection>()
        .expect("pool hands out PostgresConnection");
    assert_eq!(
        pooled.variant(),
        PostgresVariant::Default,
        "connect must carry the variant the pool detected"
    );

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

/// Resolving a whole schema in one round trip must produce exactly what
/// resolving each table individually produces.
///
/// This is the property the optimization rests on, so it is asserted directly
/// rather than by sampling: every relation the bulk query returns is compared
/// field-for-field against `get_schema` for the same table. A divergence here
/// would mean a catalog's tables silently change shape depending on which path
/// discovered them.
///
/// The fixture deliberately spans the relation kinds and type shapes that make
/// the two paths most likely to differ: a plain table with arrays and a typmod'd
/// numeric, a view, a materialized view, a partitioned parent, and a foreign
/// table (whose NOT NULL is reported nullable -- see the relkind 'f' handling in
/// `SCHEMA_QUERY`).
#[tokio::test]
async fn test_postgres_bulk_schema_matches_per_table_schema() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );
    let pg_conn = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");

    let password = common::get_pg_params(port)
        .remove("pg_pass")
        .expect("pg_pass should be present");
    let fixture = format!(
        r#"
        CREATE TABLE plain (
            id INTEGER NOT NULL,
            note TEXT,
            amount NUMERIC(10,2),
            when_at TIMESTAMP,
            tags TEXT[]
        );
        INSERT INTO plain (id, note, amount) VALUES (1, 'a', 1.50);

        CREATE VIEW plain_view AS SELECT id, amount FROM plain;
        CREATE MATERIALIZED VIEW plain_matview AS SELECT id, tags FROM plain;

        CREATE TABLE parted (id INTEGER NOT NULL, k DATE NOT NULL)
            PARTITION BY RANGE (k);
        CREATE TABLE parted_2026 PARTITION OF parted
            FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

        CREATE EXTENSION IF NOT EXISTS postgres_fdw;
        CREATE SERVER loop_srv FOREIGN DATA WRAPPER postgres_fdw
            OPTIONS (host 'localhost', port '5432', dbname 'postgres');
        CREATE USER MAPPING FOR postgres SERVER loop_srv
            OPTIONS (user 'postgres', password '{password}');
        CREATE FOREIGN TABLE foreign_plain (id INTEGER NOT NULL, amount NUMERIC(10,2))
            SERVER loop_srv OPTIONS (table_name 'plain')
    "#
    );
    for cmd in fixture.split(';') {
        if cmd.trim().is_empty() {
            continue;
        }
        pg_conn
            .conn
            .execute(cmd, &[])
            .await
            .expect("executing bulk-schema fixture");
    }

    // One of each relation kind the catalog query is expected to describe,
    // including a partition leaf: a leaf is `relkind = 'r'`, so it is describable
    // even though a catalog connector may choose not to register it.
    let requested: Vec<String> = [
        "foreign_plain",
        "parted",
        "parted_2026",
        "plain",
        "plain_matview",
        "plain_view",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let bulk = pg_conn
        .get_schemas_in("public", &requested)
        .await
        .expect("bulk schema resolution");

    // Every requested relation must come back; one the query cannot describe
    // would be silently absent, which is the caller's cue to resolve it alone.
    let mut got: Vec<&String> = bulk.keys().collect();
    got.sort();
    let mut want: Vec<&String> = requested.iter().collect();
    want.sort();
    assert_eq!(
        got, want,
        "every relation kind asked for should be described"
    );

    for (table, bulk_schema) in &bulk {
        let per_table = pg_conn
            .get_schema(&TableReference::partial("public", table.clone()))
            .await
            .unwrap_or_else(|e| panic!("per-table schema for {table}: {e}"));
        assert_eq!(
            bulk_schema, &per_table,
            "bulk and per-table schemas diverged for {table}"
        );
    }

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

/// Bulk and per-table resolution must agree under every `unsupported_type_action`,
/// not just the default.
///
/// `schema_from_columns` is shared by both paths precisely so the action cannot
/// be applied differently depending on which one resolved a table, and a fixture
/// of fully supported columns cannot show that: it exercises no unsupported type
/// at all. `jsonb` is the type the mapping rejects, and each action does
/// something different with it -- `Error` fails the whole schema, `String`
/// substitutes `Utf8`, `Warn` and `Ignore` drop the column -- so a divergence
/// would surface as a different error, a different type, or a missing field.
#[tokio::test]
async fn test_postgres_bulk_and_per_table_agree_on_unsupported_types() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let postgres_pool = PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
        .await
        .expect("unable to create Postgres connection pool");

    postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres")
        .conn
        .execute(
            "CREATE TABLE has_unsupported (id INTEGER NOT NULL, payload JSONB, note TEXT)",
            &[],
        )
        .await
        .expect("to create table");

    for action in [
        UnsupportedTypeAction::Error,
        UnsupportedTypeAction::Warn,
        UnsupportedTypeAction::Ignore,
        UnsupportedTypeAction::String,
    ] {
        let conn = postgres_pool
            .connect_direct()
            .await
            .expect("to connect to postgres")
            .with_unsupported_type_action(action);

        let bulk = conn
            .get_schemas_in("public", &["has_unsupported".to_string()])
            .await;
        let per_table = conn
            .get_schema(&TableReference::partial("public", "has_unsupported"))
            .await;

        match (bulk, per_table) {
            (Ok(bulk), Ok(per_table)) => {
                let bulk_schema = bulk
                    .get("has_unsupported")
                    .unwrap_or_else(|| panic!("{action:?}: table missing from bulk result"));
                assert_eq!(
                    bulk_schema, &per_table,
                    "{action:?}: bulk and per-table schemas diverged"
                );
            }
            // `Error` must fail identically on both paths -- not merely fail.
            // A path naming a different column or type would otherwise pass,
            // which is exactly the divergence the shared conversion prevents.
            (Err(bulk), Err(per_table)) => {
                assert_eq!(
                    action,
                    UnsupportedTypeAction::Error,
                    "{action:?}: only Error should fail schema resolution"
                );
                assert_eq!(
                    bulk.to_string(),
                    per_table.to_string(),
                    "{action:?}: paths failed with different errors"
                );
            }
            (bulk, per_table) => panic!(
                "{action:?}: paths disagreed on whether resolution succeeds \
                 (bulk ok={}, per-table ok={})",
                bulk.is_ok(),
                per_table.is_ok()
            ),
        }
    }

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

/// Detection must not be steerable by anything in the target database.
///
/// `version()` resolves through `search_path`, so a `public.version()` — which a
/// user may define for any reason — would otherwise decide how the pool
/// classifies the server. Misreading a vanilla server as Redshift is not a
/// cosmetic error: the connection then takes `SHOW COLUMNS` and the Redshift
/// catalog queries, which a PostgreSQL server cannot answer.
///
/// The shadow here is exactly what would fool an unqualified lookup, so this
/// fails if the qualification is dropped.
#[tokio::test]
async fn test_postgres_variant_detection_ignores_a_shadowed_version_function() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    {
        let setup_pool = PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create setup pool");
        setup_pool
            .connect_direct()
            .await
            .expect("to connect to postgres")
            .conn
            .batch_execute(
                "CREATE FUNCTION public.version() RETURNS text LANGUAGE sql IMMUTABLE AS \
                   $$ SELECT 'PostgreSQL 8.0.2 on i686-pc-linux-gnu, Redshift 1.0.12345'::text $$; \
                 ALTER DATABASE postgres SET search_path = public, pg_catalog;",
            )
            .await
            .expect("to shadow version()");
    }

    // Built after the shadow exists, so detection runs against it.
    let postgres_pool = PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
        .await
        .expect("unable to create Postgres connection pool");

    let conn = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");
    assert_eq!(
        conn.variant(),
        PostgresVariant::Default,
        "a shadowed version() must not decide the server variant"
    );

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

/// Redshift declines bulk resolution, leaving every table to the per-table path.
///
/// `SHOW COLUMNS` is per-table and cannot be batched, so `get_schemas_in` returns
/// an empty map rather than a partial or column-less one — and the caller must
/// read that as "ask per table". The variant is set directly because detection is
/// no longer steerable from the database (see the shadowing test above), so a
/// PostgreSQL server cannot be made to report as Redshift.
#[tokio::test]
async fn test_postgres_redshift_declines_bulk_schema_resolution() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let postgres_pool = PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
        .await
        .expect("unable to create Postgres connection pool");

    let setup = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");
    setup
        .conn
        .execute("CREATE TABLE bulk_probe (id INTEGER, note TEXT)", &[])
        .await
        .expect("to create table");

    // Vanilla resolves the table in bulk...
    let as_postgres = setup
        .get_schemas_in("public", &["bulk_probe".to_string()])
        .await
        .expect("bulk resolution");
    assert!(
        as_postgres.contains_key("bulk_probe"),
        "PostgreSQL must resolve the table in bulk"
    );

    // ...and the same connection, told it is Redshift, declines entirely.
    let as_redshift = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres")
        .with_variant(PostgresVariant::Redshift)
        .get_schemas_in("public", &["bulk_probe".to_string()])
        .await
        .expect("bulk resolution must succeed, empty");
    assert!(
        as_redshift.is_empty(),
        "Redshift must decline bulk resolution, got {} entries",
        as_redshift.len()
    );

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

/// A provider built from an already-resolved schema must be indistinguishable
/// from one that asked the server itself.
///
/// This is what makes bulk resolution usable: `get_schemas_in` answers a whole
/// namespace in one round trip, but the saving only materializes if the
/// resulting schema can be turned into a provider without a second one. The two
/// constructors must therefore agree on more than the schema -- the dialect and
/// the federation wrapping decide how a table plans, and a provider that
/// federated on one path but not the other would behave differently for no
/// reason the caller could see.
#[tokio::test]
async fn test_postgres_table_provider_with_schema_matches_the_querying_constructor() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );
    let conn = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");
    conn.conn
        .execute(
            "CREATE TABLE precomputed (id INTEGER NOT NULL, note TEXT, amount NUMERIC(10,2), tags TEXT[])",
            &[],
        )
        .await
        .expect("to create table");

    let factory = PostgresTableFactory::new(Arc::clone(&postgres_pool));
    let table = TableReference::partial("public", "precomputed");

    let queried = factory
        .table_provider(table.clone())
        .await
        .expect("provider that resolves its own schema");

    // The schema a bulk resolution would hand the caller.
    let bulk = conn
        .get_schemas_in("public", &["precomputed".to_string()])
        .await
        .expect("bulk schema resolution");
    let precomputed = Arc::clone(
        bulk.get("precomputed")
            .expect("bulk resolution must cover the table"),
    );

    let supplied = factory
        .table_provider_with_schema(table, precomputed)
        .expect("provider from an already-resolved schema");

    assert_eq!(
        supplied.schema(),
        queried.schema(),
        "a supplied schema must produce the same table schema as a queried one"
    );
    assert_eq!(
        format!("{:?}", supplied.table_type()),
        format!("{:?}", queried.table_type()),
        "both constructors must produce the same table type"
    );
    // Federation is what decides whether a scan is pushed down; differing here
    // would change plans, not just types. The concrete provider type is the
    // observable trace of which wrappers were applied.
    assert_eq!(
        (*supplied).type_id(),
        (*queried).type_id(),
        "both constructors must produce the same provider type, so they plan alike"
    );

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}

/// Only the named relations are described, whatever else the namespace holds.
///
/// This is the property that makes bulk resolution unconditionally worth taking.
/// A catalog's discovered set is narrower than the namespace — `pg_inherits`
/// children and relations the role cannot read are excluded before any
/// `include`/`exclude` is applied — so a query scoped to the namespace would
/// return columns for relations the caller has already decided to skip. For a
/// partitioned table that is every leaf, which can cost more than the per-table
/// queries it replaces.
///
/// The fixture makes that concrete: a partitioned parent with leaves, plus an
/// unrelated table, of which only the parent is asked for.
#[tokio::test]
async fn test_postgres_bulk_schema_describes_only_the_named_relations() {
    let port = crate::get_random_port();
    let container = common::start_postgres_docker_container("postgres:latest", port, None)
        .await
        .expect("Postgres container to start");

    let postgres_pool = Arc::new(
        PostgresConnectionPool::new(to_secret_map(common::get_pg_params(port)))
            .await
            .expect("unable to create Postgres connection pool"),
    );
    let conn = postgres_pool
        .connect_direct()
        .await
        .expect("to connect to postgres");

    let fixture = r#"
        CREATE TABLE events (id INTEGER NOT NULL, at DATE NOT NULL)
            PARTITION BY RANGE (at);
        CREATE TABLE events_2025 PARTITION OF events
            FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
        CREATE TABLE events_2026 PARTITION OF events
            FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
        CREATE TABLE unrelated (id INTEGER, note TEXT)
    "#;
    for cmd in fixture.split(';') {
        if cmd.trim().is_empty() {
            continue;
        }
        conn.conn
            .execute(cmd, &[])
            .await
            .expect("executing partitioned fixture");
    }

    // What a catalog would discover here: the parent, not its leaves.
    let discovered = vec!["events".to_string()];
    let schemas = conn
        .get_schemas_in("public", &discovered)
        .await
        .expect("bulk schema resolution");

    let mut got: Vec<&String> = schemas.keys().collect();
    got.sort();
    assert_eq!(
        got,
        vec!["events"],
        "only the named relation should be described; the leaves and the \
         unrelated table were not asked for"
    );

    // Asking for nothing describes nothing, and needs no query to say so.
    let none = conn
        .get_schemas_in("public", &[])
        .await
        .expect("empty request");
    assert!(none.is_empty(), "an empty request should describe nothing");

    // Tear down
    container
        .remove()
        .await
        .expect("to stop postgres container");
}
