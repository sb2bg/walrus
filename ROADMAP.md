# Networking and HTTP Roadmap

## Goals

- Keep the stdlib small enough to learn quickly.
- Make the common server path pleasant to use.
- Prefer battle-tested libraries over hand-rolled protocol code.
- Fix correctness and runtime semantics before adding lots of surface area.

## Design Principles

- Use existing crates where they already solve the problem well.
  - HTTP types/building: `hyper::http`
  - HTTP/1 request parsing: `httparse`
  - Query decoding: `url::form_urlencoded`
- Preserve a simple Walrus-facing API, but do not hide important protocol details.
- Separate low-level transport primitives from higher-level request/response helpers.
- Keep stdlib registration and handlers split by namespace so they stay maintainable.

## Current Assessment

### What is already good

- `std/net` and `std/http` are conceptually small and teachable.
- The example router proves the API can support a nice server model.
- HTTP response building already relies on `hyper::http`, not handwritten string formatting.
- Request parsing already relies on `httparse`, which is the right direction.

### What needed work

1. The native stdlib registry had become too large to maintain comfortably.
2. Async networking used cloned raw `TcpStream`s, which risked concurrent read/write races and weak close semantics.
3. HTTP query parsing was too manual and did not percent-decode.
4. Request values were lossy for repeated headers and query parameters.
5. The Walrus-side HTTP response story was too stringly; handlers had to build raw response dicts manually.
6. The networking surface was missing a small set of high-value socket metadata and controls.
7. The stdlib still needs first-class byte buffers and streaming bodies.

## Implemented In This Pass

### 1. Registry modularization

- Split `src/native_registry.rs` into per-namespace files under `src/native_registry/`.
- Added dedicated modules for:
  - `core`
  - `asyncx`
  - `io`
  - `sys`
  - `net`
  - `http`
  - `math`
  - `json`
- Central registration now lives in `src/native_registry/mod.rs`.

### 2. Safer networking handle ownership

- Replaced raw cloned socket use with shared connection ownership.
- Listener handles now share `Arc<TcpListener>`.
- Stream handles now share `Arc<Mutex<TcpStream>>`.
- Async read/write/request parsing now operate on the shared stream object instead of `try_clone()` copies.
- Closing a stream now attempts `shutdown(Shutdown::Both)` before dropping the handle.

### 3. Higher-value `std/net` additions

Added:

- `net.tcp_peer_addr(stream)`
- `net.tcp_stream_local_addr(stream)`
- `net.tcp_set_read_timeout(stream, ms_or_void)`
- `net.tcp_set_write_timeout(stream, ms_or_void)`
- `net.tcp_set_nodelay(stream, enabled)`
- `net.tcp_shutdown(stream, how)`

These are the small socket controls that materially improve real use without turning `std/net` into a kitchen sink.

### 4. Better HTTP request representation

- Switched query parsing to `url::form_urlencoded` so query strings are decoded correctly.
- Preserved repeated values by exposing:
  - `request["query_pairs"]`
  - `request["header_pairs"]`
- Kept existing dict views for convenience:
  - `request["query_params"]`
  - `request["headers"]`

The dict view is still convenient; the pair view preserves correctness for repeated fields.

### 5. Request/Response layer improvements

Added:

- `http.parse_query_pairs(query)`
- `http.make_response(status, body)`
- `http.make_response_with_headers(status, body, headers)`
- `http.serialize_response(response)`

The goal is to let Walrus handlers work with response objects first, and only serialize at the network boundary.

### 6. Example router cleanup

- Updated the HTTP router example to use the new response object helpers.
- The router now serializes responses explicitly right before `net.tcp_write`.

### 7. Regression coverage

Added VM fixtures covering:

- decoded query pair parsing
- request `query_pairs` and `header_pairs`
- response object serialization
- new socket metadata/control functions

### 8. Bounded blocking I/O executor

- Replaced the old thread-per-operation `spawn_io` path with a bounded blocking worker pool.
- The VM task/channel interface stays the same.
- This adds backpressure instead of creating an unbounded number of OS threads under load.

## Next Priorities

### Short term

1. Add a real `Headers` helper abstraction in Walrus-land.
2. Add byte-oriented networking APIs instead of text-only reads/writes.
3. Add byte or streaming HTTP body support.
4. Support response `header_pairs` inputs directly for repeated response headers such as `Set-Cookie`.

### Medium term

1. Split `src/stdlib.rs` by namespace the same way the native registry was split.
2. Add an HTTP client layer built on a library instead of custom transport code.
3. Add TLS support through an existing crate instead of homegrown handling.
4. Tighten request/response helper ergonomics in the example router and future app framework.

### Long term

1. Decide whether Walrus should expose a proper `bytes` type.
2. Move toward streaming request/response bodies for larger payloads.
3. Consider using a fuller async/networking stack where it fits, instead of gradually rebuilding pieces ourselves.

## Non-Goals For This Pass

- A full HTTP client.
- TLS.
- WebSockets.
- Chunked request bodies.
- Streaming response bodies.
- A production-grade reactor/event loop rewrite.

## Notes

- The current parsing/building path already delegates protocol-heavy work to libraries where possible.
- The remaining custom logic is mostly Walrus value-shaping and small framework glue.
- The next big architectural question is the background I/O execution model, not the HTTP API shape.
