"""Unit tests for :mod:`austrianwindatlas.stage1.amato` stubs.

The real Amato family can't be redistributed pending upstream-licence
permission.  These tests verify that:

* The stub sub-package imports cleanly (so docs / IDEs can introspect it).
* Every stub raises :class:`AmatoUnavailableError` on invocation,
  carrying a message that points at the upstream repo and
  :file:`docs/amato.md`.
* Status constants are consistent with the on-disk licence doc.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from austrianwindatlas.stage1 import amato
from austrianwindatlas.stage1.amato import (
    AMATO_LICENCE_DOC,
    AMATO_STATUS,
    AMATO_UPSTREAM_URL,
    AmatoSVD,
    AmatoUnavailableError,
    compute_amato_svd,
    reconstruct,
    reconstruct_month,
    svd,
)


# ---------------------------------------------------------------------------
# Importable + introspectable
# ---------------------------------------------------------------------------
def test_amato_submodules_accessible_from_package_root():
    # svd + reconstruct must be reachable as attributes of amato.
    assert amato.svd is svd
    assert amato.reconstruct is reconstruct


def test_amato_reachable_via_stage1_namespace():
    # The docstring promises stage1.amato works; guard against accidental
    # removal from stage1/__init__.py's re-export list.
    from austrianwindatlas import stage1
    assert stage1.amato is amato


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------
def test_status_is_one_of_the_documented_values():
    assert AMATO_STATUS in {"requested", "granted", "blocked"}


def test_upstream_url_points_at_the_federhub_repo():
    # Exact URL kept in sync with docs/amato.md.
    assert AMATO_UPSTREAM_URL == "https://github.com/federhub/ST_DeepLearning"


def test_licence_doc_path_exists_in_repo():
    """The docs/amato.md path should resolve to a real file in the
    repo so the error message's pointer isn't dead."""
    repo_root = Path(__file__).resolve().parents[2]
    doc_path = repo_root / AMATO_LICENCE_DOC
    assert doc_path.is_file(), f"missing: {doc_path}"


def test_doc_mentions_the_amato_upstream_url():
    """Keep the status doc in sync with the module-level constant."""
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / AMATO_LICENCE_DOC).read_text(encoding="utf-8")
    assert AMATO_UPSTREAM_URL in text


# ---------------------------------------------------------------------------
# Error class hierarchy
# ---------------------------------------------------------------------------
def test_amato_unavailable_error_is_a_notimplementederror():
    """Allows users to keep writing ``except NotImplementedError`` but
    react more specifically when they care."""
    assert issubclass(AmatoUnavailableError, NotImplementedError)


def test_amato_unavailable_error_message_mentions_entry_point_and_doc():
    err = AmatoUnavailableError("svd.compute_amato_svd")
    msg = str(err)
    assert "svd.compute_amato_svd" in msg
    assert AMATO_UPSTREAM_URL in msg
    assert AMATO_LICENCE_DOC in msg
    assert "enhanced_mae" in msg          # it suggests the real alternative


def test_amato_unavailable_error_records_entry_point_attribute():
    err = AmatoUnavailableError("reconstruct.reconstruct_month")
    assert err.entry_point == "reconstruct.reconstruct_month"


# ---------------------------------------------------------------------------
# Stubs raise, and only when called (not at import time)
# ---------------------------------------------------------------------------
def test_compute_amato_svd_raises_amato_unavailable():
    with pytest.raises(AmatoUnavailableError) as excinfo:
        compute_amato_svd(np.zeros((10, 5)), n_eofs=3)
    assert "compute_amato_svd" in str(excinfo.value)


def test_compute_amato_svd_also_catchable_as_notimplementederror():
    with pytest.raises(NotImplementedError):
        compute_amato_svd(np.zeros((10, 5)))


def test_amatosvd_constructor_raises():
    with pytest.raises(AmatoUnavailableError) as excinfo:
        AmatoSVD()
    assert "AmatoSVD" in str(excinfo.value)


def test_reconstruct_month_raises_amato_unavailable():
    ts = pd.date_range("2020-01-01", periods=2, freq="h")
    with pytest.raises(AmatoUnavailableError) as excinfo:
        reconstruct_month(
            V_grid=np.zeros((2, 3, 4)),
            sigma=np.ones(2),
            U=np.zeros((2, 2)),
            temporal_mean=np.zeros(2),
            timestamps=ts,
            year=2020, month=1,
        )
    assert "reconstruct_month" in str(excinfo.value)


def test_importing_amato_does_not_raise():
    """Importing the stub must succeed so docs / type checkers / IDEs
    can introspect without blowing up.  This test simply importing and
    accessing attributes is the proof.
    """
    assert hasattr(amato, "AMATO_STATUS")
    assert callable(amato.compute_amato_svd)
    assert callable(amato.reconstruct_month)
