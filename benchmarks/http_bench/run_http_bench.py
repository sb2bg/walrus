from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "benchmarks/http_bench/results"
POST_BODY_FILE = ROOT / "benchmarks/http_bench/echo_payload.json"


@dataclass(frozen=True)
class Scenario:
    name: str
    method: str
    path: str
    content_type: str | None = None
    body_file: Path | None = None


@dataclass(frozen=True)
class Target:
    name: str
    command: list[str]
    port: int
    required_modules: tuple[str, ...] = ()


SCENARIOS = (
    Scenario(name="get_health", method="GET", path="/health"),
    Scenario(name="get_post_route", method="GET", path="/users/42/posts/7"),
    Scenario(
        name="post_echo",
        method="POST",
        path="/echo",
        content_type="application/json",
        body_file=POST_BODY_FILE,
    ),
)

TARGETS = (
    Target(
        name="walrus",
        command=[str(ROOT / "target/release/walrus"), "benchmarks/http_bench/walrus_server.walrus"],
        port=18081,
    ),
    Target(
        name="flask_waitress",
        command=[sys.executable, "benchmarks/http_bench/flask_waitress_server.py", "--port", "18082"],
        port=18082,
        required_modules=("flask", "waitress"),
    ),
    Target(
        name="fastapi_uvicorn",
        command=[sys.executable, "benchmarks/http_bench/fastapi_server.py", "--port", "18083"],
        port=18083,
        required_modules=("fastapi", "uvicorn"),
    ),
)


class RssSampler:
    def __init__(self, pid: int, interval: float = 0.05) -> None:
        self.pid = pid
        self.interval = interval
        self.max_rss_kib = 0
        self.samples_kib: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = read_rss_kib(self.pid)
            if rss is not None:
                self.samples_kib.append(rss)
                self.max_rss_kib = max(self.max_rss_kib, rss)
            time.sleep(self.interval)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def ensure_walrus_release_binary() -> None:
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "walrus"],
        cwd=ROOT,
        check=True,
    )


def wait_for_ready(port: int, timeout_s: float = 15.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            conn.request("GET", "/health")
            response = conn.getresponse()
            response.read()
            conn.close()
            if response.status < 500:
                return
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"server on port {port} did not become ready: {last_error}")


def read_rss_kib(pid: int) -> int | None:
    try:
        output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
    except subprocess.CalledProcessError:
        return None
    output = output.strip()
    if not output:
        return None
    return int(output)


def launch_server(target: Target) -> tuple[subprocess.Popen[str], RssSampler]:
    process = subprocess.Popen(
        target.command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_ready(target.port)
    except Exception:
        process.kill()
        _, stderr = process.communicate(timeout=5)
        raise RuntimeError(f"failed to start {target.name}: {stderr.strip()}")
    sampler = RssSampler(process.pid)
    sampler.start()
    return process, sampler


def stop_server(process: subprocess.Popen[str], sampler: RssSampler) -> str:
    sampler.stop()
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    stderr = ""
    if process.stderr is not None:
        stderr = process.stderr.read()
        process.stderr.close()
    return stderr.strip()


def build_ab_command(port: int, scenario: Scenario, requests: int, concurrency: int) -> list[str]:
    command = [
        "ab",
        "-k",
        "-n",
        str(requests),
        "-c",
        str(concurrency),
        "-s",
        "30",
    ]
    if scenario.method != "GET":
        if scenario.body_file is None or scenario.content_type is None:
            raise ValueError(f"scenario {scenario.name} is missing a body file or content type")
        command.extend(["-p", str(scenario.body_file), "-T", scenario.content_type])
    command.append(f"http://127.0.0.1:{port}{scenario.path}")
    return command


def run_ab(port: int, scenario: Scenario, requests: int, concurrency: int) -> dict[str, Any]:
    command = build_ab_command(port, scenario, requests, concurrency)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ab failed for {scenario.name}: {result.stderr or result.stdout}")
    metrics = parse_ab_output(result.stdout)
    if metrics["failed_requests"] != 0:
        raise RuntimeError(
            f"ab reported failed requests for {scenario.name}: {metrics['failed_requests']}\n{result.stdout}"
        )
    return metrics


def parse_ab_output(output: str) -> dict[str, Any]:
    patterns = {
        "requests_per_second": r"Requests per second:\s+([0-9.]+)",
        "time_per_request_ms": r"Time per request:\s+([0-9.]+)\s+\[ms\] \(mean\)",
        "failed_requests": r"Failed requests:\s+([0-9]+)",
        "document_length_bytes": r"Document Length:\s+([0-9]+) bytes",
    }
    metrics: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match is None:
            raise RuntimeError(f"unable to parse {key} from ab output")
        value = match.group(1)
        metrics[key] = float(value) if "." in value else int(value)

    percentile_matches = re.findall(r"^\s*(50|95|99)%\s+([0-9]+)$", output, flags=re.MULTILINE)
    percentiles = {int(level): int(value) for level, value in percentile_matches}
    for level in (50, 95, 99):
        if level not in percentiles:
            raise RuntimeError(f"unable to parse p{level} latency from ab output")
    metrics["latency_p50_ms"] = percentiles[50]
    metrics["latency_p95_ms"] = percentiles[95]
    metrics["latency_p99_ms"] = percentiles[99]

    total = re.search(r"^Total:\s+([0-9]+)\s+([0-9.]+)\s+[0-9.]+\s+([0-9]+)\s+([0-9]+)$", output, flags=re.MULTILINE)
    if total is not None:
        metrics["total_conn_min_ms"] = int(total.group(1))
        metrics["total_conn_mean_ms"] = float(total.group(2))
        metrics["total_conn_median_ms"] = int(total.group(3))
        metrics["total_conn_max_ms"] = int(total.group(4))

    return metrics


def run_target(target: Target, requests: int, concurrency: int, warmup_requests: int) -> dict[str, Any]:
    process, sampler = launch_server(target)
    result: dict[str, Any] = {"target": target.name}
    try:
        if warmup_requests > 0:
            for scenario in SCENARIOS:
                run_ab(target.port, scenario, warmup_requests, min(concurrency, warmup_requests))

        scenario_results: dict[str, Any] = {}
        for scenario in SCENARIOS:
            scenario_results[scenario.name] = run_ab(target.port, scenario, requests, concurrency)

        idle_rss_kib = sampler.samples_kib[0] if sampler.samples_kib else 0
        result["peak_rss_mib"] = round(sampler.max_rss_kib / 1024, 2)
        result["idle_rss_mib"] = round(idle_rss_kib / 1024, 2)
        result["scenarios"] = scenario_results
    finally:
        stderr = stop_server(process, sampler)
        if stderr:
            result["stderr"] = stderr
    return result


def write_results(results: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "latest.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    return output_path


def format_summary(results: dict[str, Any]) -> str:
    lines = []
    lines.append(
        f"Settings: {results['requests']} requests/scenario, concurrency {results['concurrency']}, warmup {results['warmup_requests']}"
    )
    lines.append("")
    header = (
        "target".ljust(18)
        + "scenario".ljust(18)
        + "rps".rjust(12)
        + "p50".rjust(8)
        + "p95".rjust(8)
        + "p99".rjust(8)
        + "peak rss".rjust(12)
    )
    lines.append(header)
    lines.append("-" * len(header))
    for target in results["targets"]:
        peak = f"{target['peak_rss_mib']:.2f} MiB"
        for scenario_name, metrics in target["scenarios"].items():
            lines.append(
                target["target"].ljust(18)
                + scenario_name.ljust(18)
                + f"{metrics['requests_per_second']:.2f}".rjust(12)
                + f"{metrics['latency_p50_ms']} ms".rjust(8)
                + f"{metrics['latency_p95_ms']} ms".rjust(8)
                + f"{metrics['latency_p99_ms']} ms".rjust(8)
                + peak.rjust(12)
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=10000)
    parser.add_argument("--warmup-requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument(
        "--targets",
        nargs="*",
        choices=[target.name for target in TARGETS],
        help="Restrict the run to a subset of targets.",
    )
    args = parser.parse_args()

    POST_BODY_FILE.write_text('{"message":"hello walrus","count":1}\n')
    ensure_walrus_release_binary()

    selected_names = set(args.targets or [target.name for target in TARGETS])
    selected_targets = []
    skipped = []
    for target in TARGETS:
        if target.name not in selected_names:
            continue
        missing = [name for name in target.required_modules if not module_available(name)]
        if missing:
            skipped.append({"target": target.name, "reason": f"missing modules: {', '.join(missing)}"})
            continue
        selected_targets.append(target)

    if not selected_targets:
        raise SystemExit("no runnable targets were selected")

    results: dict[str, Any] = {
        "requests": args.requests,
        "warmup_requests": args.warmup_requests,
        "concurrency": args.concurrency,
        "targets": [],
        "skipped": skipped,
    }

    for target in selected_targets:
        print(f"benchmarking {target.name}...", flush=True)
        results["targets"].append(
            run_target(
                target,
                requests=args.requests,
                concurrency=args.concurrency,
                warmup_requests=args.warmup_requests,
            )
        )

    output_path = write_results(results)
    print("")
    print(format_summary(results))
    print("")
    print(f"results written to {output_path.relative_to(ROOT)}")
    if skipped:
        print("skipped targets:")
        for item in skipped:
            print(f"  {item['target']}: {item['reason']}")


if __name__ == "__main__":
    main()
