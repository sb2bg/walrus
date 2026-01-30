# Python vs Walrus Benchmark Suite

This directory contains a comprehensive benchmark suite comparing Python and Walrus performance across various computational patterns.

## Running the Benchmarks

### Run all benchmarks:
```bash
cd benchmarks
./run_benchmarks.sh
```

### Run individual benchmarks:
```bash
# Python
python3 01_quicksort.py

# Walrus (from project root)
cargo run --release -- benchmarks/01_quicksort.walrus
# or if already built:
./target/release/walrus benchmarks/01_quicksort.walrus
```

## Benchmark Categories

### Sorting Tests
| File | Description |
|------|-------------|
| `01_quicksort` | Quicksort algorithm on 5000 elements |
| `02_bubble_sort` | O(n²) bubble sort on 1000 elements |

### Recursion Tests
| File | Description |
|------|-------------|
| `03_fibonacci_recursive` | Naive recursive Fibonacci(30) - exponential time |
| `04_fibonacci_iterative` | Iterative Fibonacci - 100k iterations |
| `13_factorial` | Recursive factorial(12) - 100k calls |
| `15_ackermann` | Ackermann(3,7) - extreme recursion stress |

### Memory/GC Tests
| File | Description |
|------|-------------|
| `05_gc_stress` | Creates and discards 50k temporary objects |
| `06_gc_linked_structures` | Creates 1000 linked lists of depth 100 |
| `16_tree_traversal` | Binary tree creation and traversal (depth 15) |

### Iteration Tests
| File | Description |
|------|-------------|
| `07_iteration_heavy` | Count to 1 million in a for loop |
| `08_nested_loops` | Triple nested loops (100³ = 1M iterations) |
| `17_while_loop` | While loop counting to 1 million |

### Data Structure Tests
| File | Description |
|------|-------------|
| `09_list_operations` | List append, access, iteration (10k elements) |
| `10_dict_operations` | Dictionary insert and lookup (10k entries) |
| `11_string_concat` | String concatenation (10k chars) |

### Algorithm Tests
| File | Description |
|------|-------------|
| `12_prime_sieve` | Sieve of Eratosthenes to 50000 |
| `14_matrix_multiply` | 50x50 matrix multiplication |

### Language Feature Tests
| File | Description |
|------|-------------|
| `18_closure_stress` | Function call overhead (50k calls) |
| `19_arithmetic_heavy` | Heavy arithmetic operations (1M iterations) |
| `20_method_dispatch` | Struct/class method calls (100k iterations) |

## Interpreting Results

- **Lower times are better**
- The benchmark script shows which language is faster for each test
- Results may vary based on:
  - Python version (3.11+ has JIT optimizations)
  - System hardware and load
  - Walrus compilation optimizations enabled

## Notes

- Walrus compilation time is included in the benchmark (parsing + bytecode generation)
- Python startup time is also included (interpreter initialization)
- Both languages are running equivalent algorithms for fair comparison
- Memory usage is not directly measured, only GC stress patterns
- Some Walrus syntax differs from Python (e.g., `0..n` vs `range(n)`)

## Syntax Differences

| Feature | Python | Walrus |
|---------|--------|--------|
| Range loop | `for i in range(n):` | `for i in 0..n {` |
| Function def | `def foo(x):` | `fn foo : x {` |
| None/null | `None` | `void` |
| Print | `print(x)` | `println(x)` |
| F-string | `f"x={x}"` | `f"x={x}"` |
