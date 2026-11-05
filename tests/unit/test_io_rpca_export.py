"""Unit tests for :mod:`austrianwindatlas.io.rpca_export`.

Tests the round-trip against
:class:`austrianwindatlas.stage1.gam_bamlss.RExportInputs` so that the
library's own writer and reader are consistent, plus a hand-crafted
test for the paper's full export layout (extra files).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from austrianwindatlas.io.rpca_export import read_rpca_export
from austrianwindatlas.stage1.gam_bamlss import RExportInputs


def _write_minimal_export(root, year: int = 2017) -> tuple[str, dict]:
    """Write a minimal export via RExportInputs and return (dir, inputs)."""
    rng = np.random.default_rng(0)
    n_stations, n_eofs = 8, 3
    inputs = {
        "station_id": np.arange(100, 100 + n_stations, dtype=np.int64),
        "station_x": rng.uniform(100_000, 200_000, n_stations),
        "station_y": rng.uniform(300_000, 400_000, n_stations),
        "eof_coefficients": rng.standard_normal((n_stations, n_eofs)),
        "spatial_mean": rng.uniform(2.0, 6.0, n_stations),
    }
    rex = RExportInputs(**inputs)
    rex_dir = root / f"year_{year}"
    rex.write(rex_dir)
    return str(rex_dir), inputs


def test_read_rpca_export_round_trip_minimal(tmp_path):
    rex_dir, inputs = _write_minimal_export(tmp_path, year=2018)

    export = read_rpca_export(rex_dir)
    assert export.year == 2018
    assert export.n_stations == 8
    assert export.n_eofs == 3
    np.testing.assert_array_equal(export.station_id, inputs["station_id"])
    np.testing.assert_allclose(export.station_x, inputs["station_x"])
    np.testing.assert_allclose(export.station_y, inputs["station_y"])
    np.testing.assert_allclose(export.eof_coefficients, inputs["eof_coefficients"])
    np.testing.assert_allclose(export.spatial_mean, inputs["spatial_mean"])
    # Optional fields: none written by the minimal writer.
    assert export.singular_values is None
    assert export.temporal_basis is None
    assert export.timestamps is None
    assert export.sparse_flags is None
    assert export.station_lon is None
    assert export.station_lat is None


def test_read_rpca_export_loads_optional_files(tmp_path):
    rex_dir, _ = _write_minimal_export(tmp_path, year=2019)
    rex_path = tmp_path / "year_2019"

    # Simulate paper's full export: add singular values, temporal basis,
    # sparse flags, plus lon/lat/alt on the metadata CSV.
    pd.DataFrame({
        "component": [1, 2, 3],
        "singular_value": [10.0, 5.0, 2.5],
        "variance_explained": [0.5, 0.3, 0.2],
    }).to_csv(rex_path / "singular_values.csv", index=False)

    n_times = 48
    timestamps = pd.date_range("2019-01-01", periods=n_times, freq="h")
    rng = np.random.default_rng(1)
    basis = rng.standard_normal((n_times, 3))
    pd.DataFrame({
        "time_idx": np.arange(n_times),
        "timestamp": timestamps,
        "basis_1": basis[:, 0],
        "basis_2": basis[:, 1],
        "basis_3": basis[:, 2],
    }).to_csv(rex_path / "temporal_basis.csv", index=False)

    sparse_mask = np.zeros(n_times, dtype=bool)
    sparse_mask[[3, 10, 42]] = True
    pd.DataFrame({
        "time_idx": np.arange(n_times),
        "timestamp": timestamps,
        "is_sparse": sparse_mask,
    }).to_csv(rex_path / "sparse_timesteps.csv", index=False)

    # Rewrite station_metadata.csv with lon/lat/alt in a reshuffled row order
    # so we exercise the id-based re-ordering in the reader.
    meta = pd.read_csv(rex_path / "station_metadata.csv")
    meta["lon"] = np.linspace(9.0, 17.0, len(meta))
    meta["lat"] = np.linspace(46.0, 49.0, len(meta))
    meta["alt"] = np.linspace(200.0, 2000.0, len(meta))
    meta_shuffled = meta.sample(frac=1.0, random_state=2).reset_index(drop=True)
    meta_shuffled.to_csv(rex_path / "station_metadata.csv", index=False)

    export = read_rpca_export(rex_dir)
    np.testing.assert_allclose(export.singular_values, [10.0, 5.0, 2.5])
    assert export.n_times == n_times
    np.testing.assert_allclose(export.temporal_basis, basis)
    assert list(export.timestamps[:3]) == list(timestamps[:3])
    np.testing.assert_array_equal(export.sparse_flags, sparse_mask)

    # Reader must re-order lon/lat/alt back to EOF-file station order.
    order = pd.Index(meta_shuffled["station_id"].values).get_indexer(export.station_id)
    np.testing.assert_allclose(export.station_lon, meta_shuffled["lon"].values[order])
    np.testing.assert_allclose(export.station_lat, meta_shuffled["lat"].values[order])
    np.testing.assert_allclose(export.station_alt, meta_shuffled["alt"].values[order])


def test_read_rpca_export_accepts_paper_column_naming(tmp_path):
    """The paper exporter writes 'EOF_1'/'basis_1' etc. — upper-/lower-case mix."""
    rex_dir = tmp_path / "year_2020"
    rex_dir.mkdir()
    rng = np.random.default_rng(3)
    n_stations, n_eofs = 5, 2
    sid = np.arange(200, 200 + n_stations, dtype=np.int64)

    # EOF file with upper-case 'EOF_k' columns (paper style).
    eof_df = pd.DataFrame(
        rng.standard_normal((n_stations, n_eofs)),
        columns=["EOF_1", "EOF_2"],
    )
    eof_df.insert(0, "station_id", sid)
    eof_df.to_csv(rex_dir / "eof_coefficients_training.csv", index=False)

    pd.DataFrame({
        "station_id": sid,
        "x": rng.uniform(100_000, 200_000, n_stations),
        "y": rng.uniform(300_000, 400_000, n_stations),
    }).to_csv(rex_dir / "station_metadata.csv", index=False)

    pd.DataFrame({
        "station_id": sid,
        "x": rng.uniform(100_000, 200_000, n_stations),
        "y": rng.uniform(300_000, 400_000, n_stations),
        "mean_wind": rng.uniform(2.0, 6.0, n_stations),
    }).to_csv(rex_dir / "spatial_mean_field.csv", index=False)

    export = read_rpca_export(rex_dir)
    assert export.n_eofs == 2
    np.testing.assert_allclose(
        export.eof_coefficients, eof_df[["EOF_1", "EOF_2"]].to_numpy()
    )


def test_read_rpca_export_detects_missing_required_file(tmp_path):
    rex_dir = tmp_path / "year_2020"
    rex_dir.mkdir()
    # Only write one of the three required files.
    (rex_dir / "eof_coefficients_training.csv").write_text("station_id,eof_1\n1,0.5\n")
    with pytest.raises(FileNotFoundError, match="required rpca export file missing"):
        read_rpca_export(rex_dir)


def test_read_rpca_export_rejects_non_year_dir(tmp_path):
    bogus = tmp_path / "not_a_year_dir"
    bogus.mkdir()
    with pytest.raises(ValueError, match="year_"):
        read_rpca_export(bogus)


def test_read_rpca_export_rejects_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="rpca export directory not found"):
        read_rpca_export(tmp_path / "year_1999")


def test_read_rpca_export_mismatched_station_ids_raise(tmp_path):
    rex_dir = tmp_path / "year_2021"
    rex_dir.mkdir()
    pd.DataFrame({"station_id": [1, 2], "eof_1": [0.1, 0.2]}).to_csv(
        rex_dir / "eof_coefficients_training.csv", index=False,
    )
    # station_metadata only has station 1 and 99 — station 2 missing.
    pd.DataFrame({"station_id": [1, 99], "x": [100.0, 200.0], "y": [300.0, 400.0]}).to_csv(
        rex_dir / "station_metadata.csv", index=False,
    )
    pd.DataFrame({"station_id": [1, 2], "mean_wind": [3.0, 4.0]}).to_csv(
        rex_dir / "spatial_mean_field.csv", index=False,
    )
    with pytest.raises(ValueError, match="not found in the station-metadata file"):
        read_rpca_export(rex_dir)
