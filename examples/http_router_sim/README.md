# HTTP Server

A real, concurrent HTTP server written in Walrus — Express-style routing with
non-blocking I/O and async connection handling.

## Features

- **Express-style API** — `app.get()`, `app.post()`, `app.put()`, `app.delete()`, `app.all()`
- **Route parameters** — `/users/:id`, `/users/:id/posts/:post_id`
- **Wildcard routes** — `/assets/*`
- **Middleware** — `app.use()` runs handlers before every route
- **Custom error handling** — `app.set_not_found()`, `app.set_error_handler()` with try/catch
- **Concurrent connections** — each request handled as an async task
- **Non-blocking I/O** — TCP accept, read, write all run on background threads
- **Slow endpoints don't block** — `await asyncx.sleep(1000)` yields to other handlers

## Files

- `router.walrus` — Express-style router/app framework (~250 lines)
- `main.walrus` — Example server with routes, middleware, and error handling

## Run

```bash
cargo build && ./target/debug/walrus examples/http_router_sim/main.walrus
```

Then in another terminal (or browser for `http://127.0.0.1:8081/`):

```bash
curl http://127.0.0.1:8081/
curl http://127.0.0.1:8081/health
curl http://127.0.0.1:8081/users/42
curl http://127.0.0.1:8081/users/42/posts/7
curl -X POST http://127.0.0.1:8081/echo -d 'hello walrus'
curl http://127.0.0.1:8081/slow          # takes 1s, doesn't block other requests
curl http://127.0.0.1:8081/assets/js/app.js
curl http://127.0.0.1:8081/nonexistent   # custom 404
```

## Concurrency demo

```bash
# Start the slow request and health check at the same time:
curl http://127.0.0.1:8081/slow &
curl http://127.0.0.1:8081/health
# Both return in ~1s — the health check isn't blocked by the slow endpoint
```
