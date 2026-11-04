"""Smoke-run the end-to-end synthetic example from CI.

The example is executed in a fresh Python subprocess so its global
side effects (logging.basicConfig, logger handlers) cannot leak into
other tests.  Runs in a couple of seconds.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "01_synthetic_end_to_end.py"


def test_synthetic_end_to_end_runs_and_exits_zero():
    assert EXAMPLE.is_file(), f"example script missing at {EXAMPLE}"
    env = os.environ.copy()
    # Make sure the subprocess resolves the in-tree package even when
    # pytest is invoked without a prior pip install -e.
    src_path = str(REPO / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing}" if existing else src_path
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        capture_output=True, text=True, timeout=120,
        env=env,
    )
    # The example prints its diagnostics via `logging` to stderr; surface
    # the tail on failure so CI logs are useful.
    assert proc.returncode == 0, (
        f"example exited {proc.returncode}\n--- stderr (tail) ---\n"
        + "\n".join(proc.stderr.splitlines()[-30:])
    )
    assert "OK." in proc.stderr, "example did not reach its 'OK.' marker"
