# Walrus Test Suite

This directory contains the comprehensive test suite for the Walrus programming language.

## ⚠️ Note on Comments

Walrus does **not** currently support comments in `.walrus` files. All test files are written without comments. Comments describing tests are in this README or in the Rust test code.

## Test Files

### Core Language Features

- **`arithmetic.walrus`** - Tests basic arithmetic operations (+, -, \*, /, %) and operator precedence
- **`variables.walrus`** - Tests variable declaration, assignment, and reassignment
- **`conditionals.walrus`** - Tests if/else statements and comparison operators (==, !=, <, >, <=, >=)
- **`loops.walrus`** - Tests for loops, ranges, nested loops, and while loops
- **`functions.walrus`** - Tests function declaration, calling, parameters, returns, and nested calls
- **`strings.walrus`** - Tests string concatenation, comparison, and variables
- **`dictionaries.walrus`** - Tests dictionary creation, modification, and iteration
- **`lists.walrus`** - Tests list creation, iteration, and mixed-type lists
- **`recursion.walrus`** - Tests recursive functions (factorial, countdown)
- **`scope.walrus`** - Tests variable scoping (global, local, block scope)
- **`fibonacci.walrus`** - Classic Fibonacci implementation as a comprehensive test

### Legacy Tests

- **`test_legacy.walrus`** - Original Fibonacci test
- **`issues.walrus`** - Tests for known issues (dictionary memory, iterator behavior)
- **`test_global_reassign.walrus`** - Tests for global variable reassignment

## Running Tests

### Run all tests with Cargo

```bash
cargo test
```

### Run specific test

```bash
cargo test test_fibonacci
```

### Run tests in verbose mode

```bash
cargo test -- --nocapture
```

### Run only compiled (VM) tests

```bash
cargo test compiled
```

### Run only interpreted tests

```bash
cargo test interpreted
```

### Run a single test file manually

```bash
# Compiled mode
./scripts/run-file.sh tests/fibonacci.walrus

# Interpreted mode
cargo run -- tests/fibonacci.walrus -i
```

## Test Organization

The integration tests in `integration_tests.rs` run each `.walrus` file in two modes:

1. **Compiled Mode** (VM) - Tests run through the bytecode compiler and VM
2. **Interpreted Mode** - Tests run through the tree-walking interpreter

This ensures both execution paths work correctly and produce the same results.

## Adding New Tests

To add a new test:

1. Create a new `.walrus` file in this directory with descriptive comments
2. Add corresponding test functions in `integration_tests.rs`:

   ```rust
   #[test]
   fn test_myfeature_compiled() {
       run_walrus_file("myfeature.walrus", true).expect("My feature test failed");
   }

   #[test]
   fn test_myfeature_interpreted() {
       run_walrus_file("myfeature.walrus", false).expect("My feature test (interpreted) failed");
   }
   ```

## Test Coverage

The test suite covers:

- ✅ Arithmetic operations
- ✅ Variable management
- ✅ Control flow (if/else)
- ✅ Loops (for, while)
- ✅ Functions and recursion
- ✅ Data types (strings, dictionaries, lists)
- ✅ Scoping rules
- ✅ Both VM and interpreted execution modes
