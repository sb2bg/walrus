# HTTP Router Server

A small real HTTP server demo written in Walrus using `std/net` and `std/http`.

What it demonstrates:
- Express-style `App` API (`get`, `post`, `put`, `patch`, `delete`)
- First-class handler functions stored in routes
- Native `std/http.match_route` path matching with `:params` and `*` wildcard
- Native `std/http.read_request` request parsing (line, headers, query, body)
- Native `std/http.response_with_headers` response building

## Files

- `router.walrus`: lightweight router/app library
- `main.walrus`: runnable HTTP server (handles 4 requests, then exits)

## Run

```bash
./target/debug/walrus examples/http_router_sim/main.walrus
```

Then in another terminal:

```bash
curl -i http://127.0.0.1:8081/health
curl -i http://127.0.0.1:8081/users/42
curl -i http://127.0.0.1:8081/assets/js/app.js
curl -i -X POST http://127.0.0.1:8081/echo -d 'hello'
```
