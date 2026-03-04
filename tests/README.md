# Walrus Integration Test Suite

This directory contains the language-level test suite for Walrus.

## Layout

- `tests/language_suite.rs`: fixture runner (asserts exact stdout or expected stderr snippets)
- `tests/fixtures/pass/*.walrus`: programs that must execute successfully
- `tests/fixtures/pass/*.stdout`: expected stdout for each passing program
- `tests/fixtures/fail/*.walrus`: programs that must report an error
- `tests/fixtures/fail/*.stderr`: required stderr snippet for each failing program
- Optional `*.modes` sidecar file:
  - `vm` -> run only in VM mode
  - `interpreted` -> run only in interpreted mode
  - `both` -> run in both modes (default if omitted)

## Running

```bash
cargo test --test language_suite
```

This runs the fixture suite twice:

- VM mode (`walrus <file>`)
- interpreted mode (`walrus <file> --interpreted`)

Fixtures tagged with mode restrictions only run where they apply.
