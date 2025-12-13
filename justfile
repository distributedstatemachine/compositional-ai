# Compositional AI - Development Commands
# Run `just` to see all available commands

# Default: list all commands
default:
    @just --list

# Run all checks (what CI runs)
ci: fmt-check clippy test docs

# Run tests
test:
    cargo test --workspace --all-features

# Run tests with output
test-verbose:
    cargo test --workspace --all-features -- --nocapture

# Run a specific test
test-one TEST:
    cargo test --workspace {{TEST}} -- --nocapture

# Run clippy lints
clippy:
    cargo clippy --workspace --all-features -- -D warnings

# Run clippy and fix what it can
clippy-fix:
    cargo clippy --workspace --all-features --fix --allow-dirty --allow-staged

# Check formatting
fmt-check:
    cargo fmt --all -- --check

# Format code
fmt:
    cargo fmt --all

# Build docs
docs:
    RUSTDOCFLAGS="-Dwarnings" cargo doc --workspace --no-deps

# Open docs in browser
docs-open:
    cargo doc --workspace --no-deps --open

# Build in release mode
build:
    cargo build --workspace --release

# Build in debug mode
build-debug:
    cargo build --workspace

# Clean build artifacts
clean:
    cargo clean

# Check without building
check:
    cargo check --workspace --all-features

# Watch and run tests on file changes (requires cargo-watch)
watch:
    cargo watch -x 'test --workspace'

# Watch and run clippy on file changes
watch-clippy:
    cargo watch -x 'clippy --workspace --all-features'

# Run a specific demo (once demos crate exists)
demo CMD:
    cargo run -p demos -- {{CMD}}

# === Session-specific commands ===

# Session 1: Verify core setup
session1:
    @echo "Running Session 1 checks..."
    cargo test -p compositional-core
    @echo "✓ Session 1 complete!"

# Run all session tests up to N
session N:
    @echo "Running sessions 1-{{N}}..."
    cargo test --workspace
    @echo "✓ Sessions 1-{{N}} complete!"
