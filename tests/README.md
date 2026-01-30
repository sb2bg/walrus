# Walrus Test Suite

This directory contains the comprehensive test suite for the Walrus programming language.

## 💬 Comments Support

Walrus supports both single-line (`//`) and multi-line (`/* */`) comments in `.walrus` files. You can use comments to document your code and explain test behavior.

**Single-line comments:**

```walrus
// This is a single-line comment
let x = 42; // Comment after code
```

**Multi-line comments:**

```walrus
/* This is a multi-line comment
   that spans multiple lines */
let y = 84; /* Inline multi-line comment */
```

## Test Files

The files are constantly evolving as new features are added to Walrus. Each `.walrus` file contains tests for specific language features. As such, no complete list is provided here. Instead, please explore the files in this directory to see the various tests implemented.

## Running Tests

```bash
# Compiled mode
./scripts/run-file.sh tests/fibonacci.walrus

# Interpreted mode
cargo run -- tests/fibonacci.walrus -i
```

## State of the Test Suite

As of now, the testing suite is _extremely_ incomplete and immature. Many language features lack dedicated tests, and existing tests may not cover all edge cases. In addition, there is no automatic test runner or framework in place yet. Correctness is evaluated manually by inspecting output.

Contributions to expand and improve the test suite are highly encouraged!
