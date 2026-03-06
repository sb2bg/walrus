# HTTP Benchmark Harness

This benchmark suite measures the Walrus HTTP example against comparison
servers that expose the same routes and payload shapes.

## What it measures

- Requests per second
- Latency percentiles (`p50`, `p95`, `p99`)
- Peak resident memory (`RSS`) sampled while the benchmark is running

## Standardization

- Runs on loopback (`127.0.0.1`)
- Uses `ab` with keep-alive enabled
- Same three scenarios for every target:
  - `GET /health`
  - `GET /users/42/posts/7`
  - `POST /echo` with a fixed JSON payload
- Walrus runs from `target/release/walrus`
- Benchmark servers do not log per-request output

## Targets

- `walrus`: quiet Walrus server using the same `std/http` and `std/net` APIs as the example
- `flask_waitress`: Flask on Waitress
- `fastapi_uvicorn`: FastAPI on Uvicorn if those packages are installed

## Run

```bash
python3 benchmarks/http_bench/run_http_bench.py
```

Optional Python library comparisons:

```bash
python3 -m pip install -r benchmarks/http_bench/requirements-fastapi.txt
python3 benchmarks/http_bench/run_http_bench.py
```

Tune the run:

```bash
python3 benchmarks/http_bench/run_http_bench.py --requests 20000 --concurrency 64
```
