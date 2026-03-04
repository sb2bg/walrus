# JSON Parser Example

A small recursive-descent JSON parser written in Walrus.

## Files

- `main.walrus`: parser implementation + runnable demo
- `sample.json`: sample input

## Run

```bash
./target/debug/walrus examples/json_parser/main.walrus
```

## Notes

- Supports: objects, arrays, strings, booleans, `null`, integers, and decimal numbers.
- Keeps the implementation intentionally small and readable for learning purposes.
