# Terminal Roguelike (Walrus VM Demo)

A small simulation-style roguelike written in Walrus.

What it demonstrates:
- Structs and methods (`Player`, `Goblin`, `Dungeon`)
- Procedural dungeon generation with loops + RNG (`std/math`)
- List and dict operations for inventory, enemies, and stats
- Combat and movement on a rendered ASCII map
- Save/load round-trip with file I/O (`std/io`)
- VM exception flow with `try/catch` + `throw` during save parsing

## Run

```bash
./target/debug/walrus examples/terminal_roguelike/main.walrus
```

Or with Cargo:

```bash
cargo run --bin walrus -- examples/terminal_roguelike/main.walrus
```
