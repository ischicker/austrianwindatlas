"""Smoke-run example 04 (klima-v2-1h real-data demo) from CI.

Runs the example in a fresh subprocess with the bundled warm-cache
fixture as ``$XDG_CACHE_HOME`` and ``--no-download``, so no network
is needed.  Mirrors the pattern already in place for example 01.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "04_reproduce_fig3_tawes_open.py"
FIXTURE_CACHE = REPO / "tests" / "fixtures" / "klima_cache"


def _prepare_xdg_cache(tmp_path: Path) -> Path:
    """Copy the git-tracked warm-cache fixture under tmp_path / austrianwindatlas.

    Returns the XDG_CACHE_HOME that the example should use.
    """
    xdg = tmp_path / "xdg"
    pkg_cache = xdg / "austrianwindatlas"
    pkg_cache.mkdir(parents=True)
    for src in FIXTURE_CACHE.glob("klima_v2_1h_*.csv"):
        shutil.copy2(src, pkg_cache / src.name)
    return xdg


def test_fixture_cache_is_present():
    """Catches a regression where the fixture gets .gitignored."""
    hits = list(FIXTURE_CACHE.glob("klima_v2_1h_*.csv"))
    assert hits, f"no fixture CSV under {FIXTURE_CACHE}"


def test_example_04_runs_from_warm_cache(tmp_path):
    assert EXAMPLE.is_file(), f"example script missing at {EXAMPLE}"
    pytest.importorskip("matplotlib")

    xdg = _prepare_xdg_cache(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["XDG_CACHE_HOME"] = str(xdg)

    # cwd=tmp_path so the examples_output/ directory lands under the
    # temp dir (keeps the repo clean).
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE), "--no-download"],
        capture_output=True, text=True, timeout=180,
        env=env, cwd=tmp_path,
    )

    assert proc.returncode == 0, (
        f"example exited {proc.returncode}\n--- stderr (tail) ---\n"
        + "\n".join(proc.stderr.splitlines()[-30:])
    )
    assert "OK." in proc.stderr
    # Figure landed on disk.
    assert (tmp_path / "examples_output" / "04_fig3_tawes_open.pdf").is_file()
    assert (tmp_path / "examples_output" / "04_fig3_tawes_open.png").is_file()


def test_example_04_no_download_fails_on_cold_cache(tmp_path):
    """--no-download without a warm cache must exit non-zero with a clear message."""
    assert EXAMPLE.is_file()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    # Point XDG_CACHE_HOME at an empty dir so the cache is cold.
    env["XDG_CACHE_HOME"] = str(tmp_path / "empty-xdg")

    proc = subprocess.run(
        [sys.executable, str(EXAMPLE), "--no-download"],
        capture_output=True, text=True, timeout=60,
        env=env, cwd=tmp_path,
    )
    assert proc.returncode == 1
    assert "allow_download=False" in proc.stderr or "No cached data" in proc.stderr
