#!/usr/bin/env python3

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
NC = "\033[0m"


@dataclass
class RunResult:
    command: list[str]
    durations: list[float]
    sample_output: str
    returncode: int
    failure_output: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and bool(self.durations)

    @property
    def median(self) -> float:
        return statistics.median(self.durations)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.durations)

    @property
    def best(self) -> float:
        return min(self.durations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Python and Walrus benchmark runtimes with repeated high-resolution timings."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=7,
        help="Measured runs per command (default: 7).",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Warmup runs per command before measuring (default: 1).",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Only run benchmark pairs whose basename contains this substring.",
    )
    parser.add_argument(
        "--walrus-bin",
        type=Path,
        default=None,
        help="Path to walrus binary. Defaults to target/release/walrus.",
    )
    return parser.parse_args()


def format_runs(durations: list[float]) -> str:
    return " ".join(f"{value:.6f}" for value in durations)


def render_output_block(output: str) -> None:
    stripped = output.strip()
    if not stripped:
        print("(no output)")
        return
    print(stripped)


def run_command(command: list[str], cwd: Path, warmups: int, iterations: int) -> RunResult:
    sample_output = ""

    for _ in range(warmups):
        warmup = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
        if warmup.returncode != 0:
            combined = (warmup.stdout or "") + (warmup.stderr or "")
            return RunResult(command, [], "", warmup.returncode, combined.strip())

    durations: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        run = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000_000

        if run.returncode != 0:
            combined = (run.stdout or "") + (run.stderr or "")
            return RunResult(command, durations, sample_output, run.returncode, combined.strip())

        if not sample_output:
            sample_output = ((run.stdout or "") + (run.stderr or "")).strip()

        durations.append(elapsed)

    return RunResult(command, durations, sample_output, 0, "")


def ensure_walrus_binary(repo_root: Path, walrus_bin: Path) -> None:
    if walrus_bin.exists():
        return

    print("Building Walrus in release mode...")
    subprocess.run(["cargo", "build", "--release"], cwd=repo_root, check=True)


def compare(py_result: RunResult, walrus_result: RunResult) -> str:
    if not py_result.ok or not walrus_result.ok:
        return ""

    py_median = py_result.median
    walrus_median = walrus_result.median
    if py_median <= 0 or walrus_median <= 0:
        return ""

    if walrus_median < py_median:
        speedup = py_median / walrus_median
        return f"{GREEN}Walrus is {speedup:.2f}x faster{NC}"

    speedup = walrus_median / py_median
    return f"{RED}Python is {speedup:.2f}x faster{NC}"


def benchmark_pairs(bench_dir: Path, name_filter: str) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    for walrus_file in sorted(bench_dir.glob("*.walrus")):
        base = walrus_file.stem
        if name_filter and name_filter not in base:
            continue
        python_file = bench_dir / f"{base}.py"
        if python_file.exists():
            pairs.append((base, python_file, walrus_file))
    return pairs


def print_result(label: str, result: RunResult) -> None:
    print(f"--- {label} ---")
    if result.ok:
        render_output_block(result.sample_output)
        print(f"Runs (s): {format_runs(result.durations)}")
        print(
            "Median: "
            f"{result.median:.6f}s | Mean: {result.mean:.6f}s | Best: {result.best:.6f}s"
        )
    else:
        render_output_block(result.failure_output)
        print(f"{RED}{label} benchmark failed{NC}")
    print("")


def main() -> int:
    args = parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.warmups < 0:
        raise SystemExit("--warmups cannot be negative")

    bench_dir = Path(__file__).resolve().parent
    repo_root = bench_dir.parent
    walrus_bin = (args.walrus_bin or (repo_root / "target" / "release" / "walrus")).resolve()

    ensure_walrus_binary(repo_root, walrus_bin)

    python_version = subprocess.run(
        ["python3", "--version"], cwd=repo_root, capture_output=True, text=True, check=True
    )
    python_version_output = (python_version.stdout or python_version.stderr).strip()

    pairs = benchmark_pairs(bench_dir, args.filter)

    print("==========================================")
    print("  Python vs Walrus Benchmark Suite")
    print("==========================================")
    print("")
    print(f"Python version: {python_version_output}")
    print(f"Walrus binary: {walrus_bin}")
    print(f"Warmups per command: {args.warmups}")
    print(f"Measured iterations: {args.iterations}")
    if args.filter:
        print(f"Filter: {args.filter}")
    print("")

    if not pairs:
        print(f"{YELLOW}No benchmark pairs matched.{NC}")
        return 0

    for base, python_file, walrus_file in pairs:
        print("==========================================")
        print(f"Benchmark: {base}")
        print("==========================================")
        print("")

        py_result = run_command(
            ["python3", str(python_file)], repo_root, args.warmups, args.iterations
        )
        print_result("Python", py_result)

        walrus_result = run_command(
            [str(walrus_bin), str(walrus_file)], repo_root, args.warmups, args.iterations
        )
        print_result("Walrus", walrus_result)

        comparison = compare(py_result, walrus_result)
        if comparison:
            print(comparison)
            print("")

    print("==========================================")
    print("  Benchmark Complete!")
    print("==========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
