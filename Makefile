all:
	cargo build --all-features

.PHONY: test
test:
	cargo test --features clickhouse,duckdb,flight,mysql,postgres,sqlite,adbc -p datafusion-table-providers --lib
	cargo test -p datafusion-table-providers-oracle

.PHONY: lint
lint:
	cargo clippy --all-features

.PHONY: test-integration
test-integration:
	RUST_LOG=$${RUST_LOG:-info} cargo test -p datafusion-table-providers --test integration --no-default-features --features postgres,sqlite,mysql,flight,clickhouse,duckdb,mongodb,adbc -- --nocapture

# Type-checks the integration suite without needing Docker, so `make test`
# running only `--lib` can't let the suite rot uncompiled.
.PHONY: check-integration
check-integration:
	cargo test -p datafusion-table-providers --test integration --no-default-features --features postgres,sqlite,mysql,flight,clickhouse,duckdb,mongodb,adbc --no-run

# The DuckDB integration suite; in release to validate DuckDB behavior using producation configuration
.PHONY: test-integration-duckdb
test-integration-duckdb:
	RUST_LOG=info cargo test --release -p datafusion-table-providers --test integration --no-default-features --features duckdb -- --nocapture --test-threads 1
