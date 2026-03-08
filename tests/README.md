# Walrus Integration Test Suite

This directory contains the language-level test suite for Walrus.

## Layout

- `tests/language_suite.rs`: fixture runner (asserts exact stdout or expected stderr snippets)
- `tests/fixtures/pass/*.walrus`: programs that must execute successfully
- `tests/fixtures/pass/*.stdout`: expected stdout for each passing program
- Optional `tests/fixtures/pass/*.exitcode`: expected process exit code for a fixture (defaults to success for pass fixtures)
- Optional `*.env` sidecar file: fixture-specific environment variables (dotenv syntax)
- Optional `*.stdin` sidecar file: bytes piped to program stdin before assertion
- `tests/fixtures/fail/*.walrus`: programs that must report an error
- `tests/fixtures/fail/*.stderr`: required stderr snippet for each failing program

## Running

```bash
cargo test --test language_suite
```

This runs the fixture suite in VM mode (`walrus <file>`).
