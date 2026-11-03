"""Smoke test: the package imports and the five sub-packages exist."""
from __future__ import annotations

import importlib


def test_top_level_import():
    awa = importlib.import_module("austrianwindatlas")
    assert hasattr(awa, "__version__")
    assert isinstance(awa.__version__, str) and awa.__version__ != ""


def test_subpackages_import():
    for name in ("core", "stage1", "stage2", "validation", "io"):
        importlib.import_module(f"austrianwindatlas.{name}")
