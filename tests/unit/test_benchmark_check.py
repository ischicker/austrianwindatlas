"""Unit tests for :mod:`scripts.benchmark_check`.

We test the logic (baseline I/O, threshold check, CLI wiring) without
actually running example 01 — subprocess timing is monkey-patched so
these stay under 1 s and can't be flaky.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "benchmark_check.py"
BASELINE = REPO / "tests" / "benchmarks" / "baseline.json"


def _load_module():
    """Load ``scripts/benchmark_check.py`` as a module.

    Can't just ``import scripts.benchmark_check`` because ``scripts/``
    isn't a package.  :mod:`importlib.util` does the equivalent in
    a few lines.
    """
    spec = importlib.util.spec_from_file_location("benchmark_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_check"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bc():
    return _load_module()


# ---------------------------------------------------------------------------
# BenchmarkResult arithmetic
# ---------------------------------------------------------------------------
def test_benchmark_result_median_and_best(bc):
    r = bc.BenchmarkResult(name="x", samples_seconds=[1.0, 3.0, 2.0])
    assert r.median == pytest.approx(2.0)
    assert r.best == pytest.approx(1.0)
    assert r.mean == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# check_regression — threshold logic
# ---------------------------------------------------------------------------
def test_check_regression_passes_at_baseline(bc):
    r = bc.BenchmarkResult("x", [4.0, 4.0, 4.0])
    out = bc.check_regression(r, baseline_seconds=4.0, threshold=1.5,
                              under_threshold=0.5)
    assert out.passed is True
    assert "PASS" in out.message


def test_check_regression_passes_just_below_limit(bc):
    r = bc.BenchmarkResult("x", [5.9, 6.0, 6.1])
    out = bc.check_regression(r, baseline_seconds=4.0, threshold=1.5,
                              under_threshold=0.5)
    # Limit = 4.0 * 1.5 = 6.0 → median 6.0 ≤ 6.0.
    assert out.passed is True


def test_check_regression_fails_above_limit(bc):
    r = bc.BenchmarkResult("x", [6.5, 7.0, 6.8])
    out = bc.check_regression(r, baseline_seconds=4.0, threshold=1.5,
                              under_threshold=0.5)
    assert out.passed is False
    assert "FAIL" in out.message
    assert "slowdown" in out.message


def test_check_regression_warns_when_suspiciously_fast(bc):
    r = bc.BenchmarkResult("x", [0.5, 0.6, 0.7])
    out = bc.check_regression(r, baseline_seconds=4.0, threshold=1.5,
                              under_threshold=0.5)
    # Median 0.6s << 4.0 * 0.5 = 2.0 → still passes but with a note.
    assert out.passed is True
    assert "suspiciously fast" in out.message


# ---------------------------------------------------------------------------
# baseline I/O round-trip
# ---------------------------------------------------------------------------
def test_committed_baseline_loads_and_has_example_01(bc):
    data = bc.load_baseline(BASELINE)
    assert "example_01" in data
    entry = data["example_01"]
    assert isinstance(entry["median_seconds"], (int, float))
    assert entry["median_seconds"] > 0.0


def test_load_baseline_raises_when_missing(bc, tmp_path):
    with pytest.raises(FileNotFoundError):
        bc.load_baseline(tmp_path / "does-not-exist.json")


def test_save_baseline_writes_formatted_json(bc, tmp_path):
    data = {"example_01": {"median_seconds": 4.5, "runs": 3}}
    out = tmp_path / "baseline.json"
    bc.save_baseline(out, data)
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n")                                # trailing newline
    assert "  \"example_01\"" in text                         # 2-space indent
    # Round-trip.
    assert json.loads(text) == data


def test_update_baseline_entry_preserves_other_benchmarks(bc):
    existing = {
        "example_01": {"median_seconds": 4.5, "notes": "keep me"},
        "example_99": {"median_seconds": 10.0},
    }
    r = bc.BenchmarkResult("example_01", [2.0, 2.1, 2.2])
    new = bc.update_baseline_entry(existing, name="example_01",
                                    result=r, threshold=1.5)
    # Updated entry has the new median plus bookkeeping.
    assert new["example_01"]["median_seconds"] == pytest.approx(2.1)
    assert new["example_01"]["slowdown_threshold"] == 1.5
    assert "updated_at" in new["example_01"]
    assert "python_version" in new["example_01"]
    assert new["example_01"]["host"]      # not empty
    # The 'notes' field from the old entry is overwritten
    # (dict.update semantics); the other benchmark is untouched.
    assert new["example_99"]["median_seconds"] == 10.0


# ---------------------------------------------------------------------------
# run_benchmark error paths
# ---------------------------------------------------------------------------
def test_run_benchmark_rejects_unknown_name(bc):
    with pytest.raises(KeyError, match="unknown benchmark"):
        bc.run_benchmark("nonsense", runs=1, warmup=False, timeout=5)


def test_run_benchmark_runs_the_registered_script(bc, monkeypatch):
    """Patch time_script so we don't actually spawn python."""
    calls: list[Path] = []

    def fake_time_script(script, *, timeout):
        calls.append(script)
        return 1.23

    monkeypatch.setattr(bc, "time_script", fake_time_script)
    result = bc.run_benchmark("example_01", runs=3, warmup=True, timeout=5)
    # warmup + 3 samples = 4 calls.
    assert len(calls) == 4
    # All four hit the registered example 01 path.
    assert calls[0].name == "01_synthetic_end_to_end.py"
    assert result.samples_seconds == [1.23, 1.23, 1.23]
    assert result.median == pytest.approx(1.23)


def test_run_benchmark_without_warmup_takes_runs_samples(bc, monkeypatch):
    calls: list[Path] = []

    def fake_time_script(script, *, timeout):
        calls.append(script)
        return 2.0

    monkeypatch.setattr(bc, "time_script", fake_time_script)
    result = bc.run_benchmark("example_01", runs=5, warmup=False, timeout=5)
    assert len(calls) == 5
    assert len(result.samples_seconds) == 5


# ---------------------------------------------------------------------------
# CLI (main())
# ---------------------------------------------------------------------------
def test_cli_passes_when_runtime_under_limit(bc, monkeypatch, tmp_path):
    """Full CLI exercise: --runs 3 on a fake-fast benchmark → exit 0."""
    monkeypatch.setattr(bc, "time_script", lambda script, timeout: 1.0)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"example_01": {"median_seconds": 4.5}}))

    rc = bc.main(["--baseline", str(baseline), "--runs", "3"])
    assert rc == 0


def test_cli_fails_when_runtime_exceeds_limit(bc, monkeypatch, tmp_path):
    monkeypatch.setattr(bc, "time_script", lambda script, timeout: 10.0)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"example_01": {"median_seconds": 4.5}}))

    rc = bc.main(["--baseline", str(baseline), "--runs", "3"])
    assert rc == 1


def test_cli_update_writes_new_median(bc, monkeypatch, tmp_path):
    monkeypatch.setattr(bc, "time_script", lambda script, timeout: 3.33)
    baseline = tmp_path / "baseline.json"
    # Pre-seed so we can verify other keys survive.
    baseline.write_text(json.dumps({
        "example_01": {"median_seconds": 4.5, "notes": "keep"},
        "example_99": {"median_seconds": 7.0},
    }))

    rc = bc.main(["--baseline", str(baseline), "--runs", "1", "--update"])
    assert rc == 0
    data = json.loads(baseline.read_text())
    assert data["example_01"]["median_seconds"] == pytest.approx(3.33)
    # Other benchmark untouched.
    assert data["example_99"]["median_seconds"] == 7.0


def test_cli_rejects_missing_baseline_in_check_mode(bc, monkeypatch, tmp_path):
    monkeypatch.setattr(bc, "time_script", lambda script, timeout: 1.0)
    rc = bc.main(["--baseline", str(tmp_path / "nope.json")])
    assert rc == 2


def test_cli_rejects_threshold_le_one(bc, monkeypatch, tmp_path):
    monkeypatch.setattr(bc, "time_script", lambda script, timeout: 1.0)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"example_01": {"median_seconds": 4.5}}))
    with pytest.raises(SystemExit) as exc:
        bc.main(["--baseline", str(baseline), "--threshold", "1.0"])
    # argparse errors exit with code 2 via SystemExit.
    assert exc.value.code == 2


def test_cli_update_creates_baseline_if_missing(bc, monkeypatch, tmp_path):
    monkeypatch.setattr(bc, "time_script", lambda script, timeout: 2.5)
    baseline = tmp_path / "sub" / "baseline.json"
    assert not baseline.exists()
    rc = bc.main(["--baseline", str(baseline), "--runs", "1", "--update"])
    assert rc == 0
    assert baseline.is_file()
    data = json.loads(baseline.read_text())
    assert data["example_01"]["median_seconds"] == pytest.approx(2.5)
