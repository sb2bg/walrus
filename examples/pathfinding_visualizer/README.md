# Pathfinding Visualizer (A* in Walrus)

A deterministic A* pathfinding demo on an ASCII maze, with replay snapshots.

What it demonstrates:
- Grid algorithms with lists, dicts, loops, and functions
- Priority selection in a plain list-backed open set
- Reconstructing shortest paths from `came_from`
- ASCII rendering of exploration progress and final path
- Error flow with `throw` + `try/catch` for no-path scenarios

## Run

```bash
./target/debug/walrus examples/pathfinding_visualizer/main.walrus
```

Or with Cargo:

```bash
cargo run --bin walrus -- examples/pathfinding_visualizer/main.walrus
```
