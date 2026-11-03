"""`austrianwindatlas.core.paths` — path resolution against AWA_DATA_ROOT."""
from __future__ import annotations

from pathlib import Path

from austrianwindatlas.core import paths


def test_defaults_to_cwd_awa_data(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    root = paths.data_root()
    assert root == paths.DEFAULT_ROOT
    assert root.name == "awa_data"


def test_env_var_override(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    assert paths.data_root() == tmp_path


def test_resolve_joins_parts(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    out = paths.resolve("stations", "tawes_2020.nc")
    assert out == tmp_path / "stations" / "tawes_2020.nc"


def test_resolve_accepts_path_objects(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    out = paths.resolve(Path("era5"), Path("2020") / "01.nc")
    assert out == tmp_path / "era5" / "2020" / "01.nc"


def test_user_expansion(monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, "~/awa_data_user")
    root = paths.data_root()
    # Resolved path should no longer start with "~"; home-dir expansion
    # went through.
    assert not str(root).startswith("~")
