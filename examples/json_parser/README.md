# JSON Parser Example

A small recursive-descent JSON parser written in Walrus.

## Files

- `parser.walrus`: parser implementation
- `main.walrus`: runnable demo that imports the parser module
- `sample.json`: sample input

## Run

```bash
./target/debug/walrus examples/json_parser/main.walrus
```

## Notes

- Supports: objects, arrays, strings, booleans, `null`, integers, and decimal numbers.
- Keeps the implementation intentionally small and readable for learning purposes.
