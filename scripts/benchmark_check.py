#!/usr/bin/env python3
"""Benchmark-regression guard for CI.

Runs one of the registered benchmark scripts a handful of times,
takes the median wall-clock seconds, and exits non-zero if the
median is more than ``threshold ×`` the committed baseline.

Usage::

    python scripts/benchmark_check.py --baseline tests/benchmarks/baseline.json
    python scripts/benchmark_check.py --baseline <path> --benchmark example_01
    python scripts/benchmark_check.py --baseline <path> --update       # refresh

Design
======

* **Benchmark registry.**  :data:`BENCHMARKS` maps a short name to the
  script path (resolved relative to the repo root).  Add a new entry +
  a baseline-JSON key to wire up another benchmark; the rest of the
  machinery doesn't change.

* **Median of ``runs`` samples.**  Protects against one-off outliers
  from runner noise.  Default 3 runs; override with ``--runs``.

* **Optional warmup.**  First run can be discarded with ``--warmup``.
  Useful on cold filesystem caches, where the first import of numpy +
  xarray is 400–800 ms slower than subsequent runs.

* **Noise-reduction env vars.**  ``PYTHONHASHSEED=0`` and
  ``PYTHONDONTWRITEBYTECODE=1`` are injected before each subprocess
  so bytecode-write timing and dict-order randomisation don't dominate
  the variance.

* **Separate check / update modes.**  ``--update`` refreshes the
  baseline in-place (bumps ``median_seconds``, records the hostname
  + Python version + git-SHA as metadata).  Without it, the script
  only *reads* the baseline — CI always runs in check mode.

* **Slowdown threshold.**  Fail when ``median > baseline × threshold``
  (default 1.5).  Also emit a *warning* (non-fatal) when
  ``median < baseline × under_threshold`` (default 0.5) — the baseline
  is probably stale.

Exit codes
----------
* ``0`` — benchmark passed (``median ≤ baseline × threshold``).
* ``1`` — benchmark failed (regression).
* ``2`` — argument / configuration error.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Registered benchmarks.  Add entries to run them via ``--benchmark <name>``.
BENCHMARKS: dict[str, Path] = {
    "example_01": REPO_ROOT / "examples" / "01_synthetic_end_to_end.py",
}

#: Env vars injected to reduce run-to-run variance.
_STABLE_ENV = {
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}

logger = logging.getLogger("benchmark_check")


@dataclass
class BenchmarkResult:
    """One benchmark × many samples."""

    name: str
    samples_seconds: list[float]

    @property
    def median(self) -> float:
        return float(statistics.median(self.samples_seconds))

    @property
    def mean(self) -> float:
        return float(statistics.fmean(self.samples_seconds))

    @property
    def best(self) -> float:
        return float(min(self.samples_seconds))


def time_script(script: Path, *, timeout: float) -> float:
    """Run ``script`` under :data:`_STABLE_ENV` and return wall-clock seconds.

    stdout / stderr are captured so a chatty benchmark doesn't spam the
    CI log.  Raises :class:`subprocess.CalledProcessError` if the
    subprocess exits non-zero (benchmark intrinsically broken).
    """
    env = os.environ.copy()
    env.update(_STABLE_ENV)
    # Ensure the in-tree package is importable even if not pip-installed.
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    start = time.perf_counter()
    subprocess.run(
        [sys.executable, str(script)],
        check=True, env=env, capture_output=True, text=True,
        timeout=timeout,
    )
    return time.perf_counter() - start


def run_benchmark(
    name: str,
    *,
    runs: int,
    warmup: bool,
    timeout: float,
) -> BenchmarkResult:
    """Time a registered benchmark ``runs`` times."""
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; known: {sorted(BENCHMARKS)}")
    script = BENCHMARKS[name]
    if not script.is_file():
        raise FileNotFoundError(f"benchmark script missing: {script}")

    if warmup:
        logger.info("warmup run (discarded)")
        time_script(script, timeout=timeout)

    samples: list[float] = []
    for i in range(runs):
        elapsed = time_script(script, timeout=timeout)
        samples.append(elapsed)
        logger.info("  run %d/%d: %.3fs", i + 1, runs, elapsed)
    return BenchmarkResult(name=name, samples_seconds=samples)


def _git_sha() -> str | None:
    """Return the current git SHA (short), or :data:`None` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def load_baseline(path: Path) -> dict:
    """Load the baseline JSON.  Raises :class:`FileNotFoundError` if absent."""
    if not path.is_file():
        raise FileNotFoundError(f"baseline file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_baseline(path: Path, baseline: dict) -> None:
    """Write ``baseline`` to ``path`` with stable 2-space indent + newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(baseline, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def update_baseline_entry(
    baseline: dict,
    *,
    name: str,
    result: BenchmarkResult,
    threshold: float,
) -> dict:
    """Return a new baseline dict with ``name`` refreshed from ``result``."""
    entry = dict(baseline.get(name) or {})
    entry.update({
        "median_seconds": round(result.median, 3),
        "best_seconds": round(result.best, 3),
        "runs": len(result.samples_seconds),
        "slowdown_threshold": threshold,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "git_sha": _git_sha(),
    })
    new = dict(baseline)
    new[name] = entry
    return new


@dataclass
class CheckOutcome:
    """Human-readable + machine-readable result of one check."""

    passed: bool
    median: float
    baseline_seconds: float
    threshold: float
    message: str


def check_regression(
    result: BenchmarkResult,
    *,
    baseline_seconds: float,
    threshold: float,
    under_threshold: float,
) -> CheckOutcome:
    """Compare ``result.median`` to ``baseline_seconds * threshold``.

    Also emits a (non-fatal) note when the median is suspiciously fast.
    """
    median = result.median
    fail_limit = baseline_seconds * threshold
    passed = median <= fail_limit

    header = (
        f"{result.name}: median={median:.3f}s  "
        f"baseline={baseline_seconds:.3f}s  "
        f"threshold={threshold:g}×  "
        f"limit={fail_limit:.3f}s"
    )
    if not passed:
        slowdown = median / baseline_seconds if baseline_seconds > 0 else float("inf")
        message = f"{header}  →  FAIL (slowdown {slowdown:.2f}×)"
    else:
        under_limit = baseline_seconds * under_threshold
        if median < under_limit:
            message = (
                f"{header}  →  PASS, but suspiciously fast "
                f"(median < {under_threshold:g}× baseline; refresh with --update)"
            )
        else:
            message = f"{header}  →  PASS"

    return CheckOutcome(
        passed=passed,
        median=median,
        baseline_seconds=baseline_seconds,
        threshold=threshold,
        message=message,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path,
                        help="Path to the baseline JSON (committed to git).")
    parser.add_argument("--benchmark", default="example_01",
                        choices=sorted(BENCHMARKS),
                        help="Which benchmark to run.")
    parser.add_argument("--runs", type=int, default=3,
                        help="Sample count for the median (default: 3).")
    parser.add_argument("--warmup", action="store_true",
                        help="Discard an initial warmup run before sampling.")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Per-run timeout, seconds (default: 120).")
    parser.add_argument("--threshold", type=float, default=1.5,
                        help="Fail when median > baseline * threshold (default: 1.5).")
    parser.add_argument("--under-threshold", type=float, default=0.5,
                        help="Warn when median < baseline * this (default: 0.5).")
    parser.add_argument("--update", action="store_true",
                        help="Write the measured median back to the baseline JSON "
                             "instead of checking.  Use via workflow_dispatch.")
    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.threshold <= 1.0:
        parser.error("--threshold must be > 1.0 (otherwise identical runs fail)")

    try:
        result = run_benchmark(
            args.benchmark,
            runs=args.runs,
            warmup=args.warmup,
            timeout=args.timeout,
        )
    except KeyError as exc:
        parser.error(str(exc))
        return 2
    except FileNotFoundError as exc:
        logger.error("benchmark script missing: %s", exc)
        return 2
    except subprocess.CalledProcessError as exc:
        logger.error("benchmark subprocess exited %d; stderr tail:\n%s",
                     exc.returncode, (exc.stderr or "")[-400:])
        return 2

    logger.info(
        "samples: median=%.3fs best=%.3fs mean=%.3fs over %d runs",
        result.median, result.best, result.mean, len(result.samples_seconds),
    )

    if args.update:
        baseline = {}
        if args.baseline.is_file():
            baseline = load_baseline(args.baseline)
        new_baseline = update_baseline_entry(
            baseline, name=args.benchmark,
            result=result, threshold=args.threshold,
        )
        save_baseline(args.baseline, new_baseline)
        logger.info(
            "updated %s[%s].median_seconds = %.3f",
            args.baseline, args.benchmark, result.median,
        )
        return 0

    try:
        baseline = load_baseline(args.baseline)
    except FileNotFoundError:
        logger.error(
            "baseline %s missing; run once with --update to create it.",
            args.baseline,
        )
        return 2

    entry = baseline.get(args.benchmark)
    if entry is None or "median_seconds" not in entry:
        logger.error(
            "baseline has no entry for %r; run with --update to seed it.",
            args.benchmark,
        )
        return 2

    outcome = check_regression(
        result,
        baseline_seconds=float(entry["median_seconds"]),
        threshold=args.threshold,
        under_threshold=args.under_threshold,
    )
    logger.info(outcome.message)
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    sys.exit(main())
