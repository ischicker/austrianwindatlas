"""Unit tests for :mod:`austrianwindatlas.io.windfield`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from austrianwindatlas.io.windfield import (  # noqa: E402
    INCA_GRID,
    GridSpec,
    build_wind_field_dataset,
    write_wind_field_netcdf,
)


def test_inca_grid_shape_and_crs():
    assert INCA_GRID.shape == (401, 701)
    assert INCA_GRID.crs == "EPSG:31287"
    # Paper code requires linspace, not arange — step width should be exactly 1 km.
    x = INCA_GRID.x_axis()
    y = INCA_GRID.y_axis()
    assert x[0] == 19500.0 and x[-1] == 720500.0
    assert y[0] == 219500.0 and y[-1] == 620500.0
    # Nominal "1 km" spacing — linspace over 701/401 points gives
    # ~1001.4 m / ~1002.5 m per step (cells vs cell-centres).  The
    # production code uses the same linspace so these values are
    # the ones the paper's NetCDFs carry.
    np.testing.assert_allclose(np.diff(x), (720500.0 - 19500.0) / 700.0)
    np.testing.assert_allclose(np.diff(y), (620500.0 - 219500.0) / 400.0)


def test_grid_spec_shape_property():
    g = GridSpec(xmin=0, xmax=10, ymin=0, ymax=20, ncols=11, nrows=21, crs="EPSG:3857")
    assert g.shape == (21, 11)
    assert g.x_axis().shape == (11,)
    assert g.y_axis().shape == (21,)


def test_build_wind_field_dataset_small():
    grid = GridSpec(xmin=0.0, xmax=4.0, ymin=0.0, ymax=2.0, ncols=5, nrows=3, crs="EPSG:31287")
    times = pd.date_range("2020-01-01", periods=4, freq="h")
    wind = np.ones((4, 3, 5), dtype=np.float32) * 7.5

    ds = build_wind_field_dataset(
        wind, times, grid=grid,
        method="rpca_reconstruction",
        year=2020, month=1, n_eofs=5,
    )

    assert ds["wind_speed"].dims == ("time", "y", "x")
    assert ds["wind_speed"].shape == (4, 3, 5)
    assert ds.sizes == {"time": 4, "y": 3, "x": 5}
    assert ds.attrs["method"] == "rpca_reconstruction"
    assert ds.attrs["crs"] == "EPSG:31287"
    assert ds.attrs["year"] == 2020
    assert ds.attrs["month"] == 1
    assert ds.attrs["n_eofs"] == 5
    assert ds.attrs["background_field"] == "none"
    assert "created" in ds.attrs


def test_build_wind_field_dataset_extra_attrs_merge():
    grid = GridSpec(0, 1, 0, 1, 2, 2)
    times = pd.date_range("2020-01-01", periods=1, freq="h")
    wind = np.zeros((1, 2, 2))
    ds = build_wind_field_dataset(
        wind, times, grid=grid,
        background_field="era5",
        extra_attrs={"processing_time_seconds": 42.0, "method": "overridden"},
    )
    assert ds.attrs["background_field"] == "era5"
    assert ds.attrs["processing_time_seconds"] == 42.0
    # extra_attrs win over defaults by design
    assert ds.attrs["method"] == "overridden"


def test_build_wind_field_dataset_rejects_wrong_shape():
    grid = GridSpec(0, 4, 0, 2, 5, 3)
    times = pd.date_range("2020-01-01", periods=4, freq="h")
    wind = np.zeros((3, 3, 5))  # wrong time axis
    with pytest.raises(ValueError, match="does not match"):
        build_wind_field_dataset(wind, times, grid=grid)


def test_write_wind_field_netcdf_round_trip(tmp_path):
    grid = GridSpec(0.0, 4.0, 0.0, 2.0, 5, 3)
    times = pd.date_range("2020-07-01", periods=6, freq="h")
    wind = np.linspace(0, 10, 6 * 3 * 5, dtype=np.float32).reshape(6, 3, 5)
    ds = build_wind_field_dataset(
        wind, times, grid=grid,
        year=2020, month=7, n_eofs=12,
    )
    out = tmp_path / "subdir" / "wind_rpca_ked_202007.nc"
    returned = write_wind_field_netcdf(ds, out)

    assert returned == out
    assert out.is_file()
    reopened = xr.open_dataset(out)
    try:
        assert reopened["wind_speed"].shape == (6, 3, 5)
        assert reopened["wind_speed"].encoding.get("dtype") == np.dtype("float32")
        # zlib compression takes effect → file size << uncompressed
        # (6*3*5*4 bytes = 360). Hard to assert tightly, but the attrs
        # must be preserved exactly.
        assert reopened.attrs["method"] == "rpca_reconstruction"
        assert reopened.attrs["year"] == 2020
        assert reopened.attrs["month"] == 7
        assert reopened.attrs["n_eofs"] == 12
        np.testing.assert_allclose(reopened["wind_speed"].values, wind, rtol=0, atol=0)
    finally:
        reopened.close()


def test_write_wind_field_netcdf_creates_parent(tmp_path):
    grid = GridSpec(0, 1, 0, 1, 2, 2)
    times = pd.date_range("2020-01-01", periods=1, freq="h")
    ds = build_wind_field_dataset(np.zeros((1, 2, 2)), times, grid=grid)
    out = tmp_path / "deep" / "nested" / "wind.nc"
    write_wind_field_netcdf(ds, out)
    assert out.is_file()
