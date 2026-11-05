"""Unit tests for :mod:`austrianwindatlas.io.background`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from austrianwindatlas.io.background import (  # noqa: E402
    BackgroundField,
    load_cerra_background,
    load_era5_background,
    regrid_to_inca,
)
from austrianwindatlas.io.windfield import GridSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers to write synthetic ERA5 / CERRA NetCDFs
# ---------------------------------------------------------------------------
def _write_era5(path, year, month, *, times_per_month=6, use_components=True):
    times = pd.date_range(f"{year}-{month:02d}-01", periods=times_per_month, freq="h")
    lat = np.linspace(49.0, 46.0, 5)
    lon = np.linspace(9.0, 18.0, 7)
    rng = np.random.default_rng(0)
    if use_components:
        u = rng.uniform(-5, 5, size=(times_per_month, 5, 7))
        v = rng.uniform(-5, 5, size=(times_per_month, 5, 7))
        ds = xr.Dataset(
            {"u10": (("time", "latitude", "longitude"), u),
             "v10": (("time", "latitude", "longitude"), v)},
            coords={"time": times, "latitude": lat, "longitude": lon},
        )
    else:
        ws = rng.uniform(0, 10, size=(times_per_month, 5, 7))
        ds = xr.Dataset(
            {"wind_speed": (("time", "latitude", "longitude"), ws)},
            coords={"time": times, "latitude": lat, "longitude": lon},
        )
    ds.to_netcdf(path)


def _write_cerra(path, year, month, *, curvilinear=False):
    times = pd.date_range(f"{year}-{month:02d}-01", periods=4, freq="h")
    rng = np.random.default_rng(0)
    if curvilinear:
        lat = np.repeat(np.linspace(46.0, 49.0, 4)[:, None], 6, axis=1) + \
              np.linspace(-0.05, 0.05, 6)[None, :]
        lon = np.linspace(9.0, 18.0, 6)[None, :].repeat(4, axis=0) + \
              np.linspace(-0.1, 0.1, 4)[:, None]
        ws = rng.uniform(0, 10, size=(4, 4, 6))
        ds = xr.Dataset(
            {"wind_speed": (("time", "y", "x"), ws)},
            coords={
                "time": times,
                "latitude": (("y", "x"), lat),
                "longitude": (("y", "x"), lon),
            },
        )
    else:
        lat = np.linspace(49.0, 46.0, 4)
        lon = np.linspace(9.0, 18.0, 6)
        ws = rng.uniform(0, 10, size=(4, 4, 6))
        ds = xr.Dataset(
            {"wind_speed": (("time", "latitude", "longitude"), ws)},
            coords={"time": times, "latitude": lat, "longitude": lon},
        )
    ds.to_netcdf(path)


def _tiny_grid() -> GridSpec:
    """A tiny Austrian-ish grid used for regrid tests."""
    return GridSpec(xmin=100_000.0, xmax=500_000.0, ymin=300_000.0,
                    ymax=500_000.0, ncols=5, nrows=4, crs="EPSG:31287")


# ---------------------------------------------------------------------------
# ERA5 loader
# ---------------------------------------------------------------------------
def test_load_era5_derives_speed_from_components(tmp_path):
    _write_era5(tmp_path / "era5_wind_202001.nc", 2020, 1)
    bg = load_era5_background(tmp_path, 2020, 1)
    assert isinstance(bg, BackgroundField)
    assert bg.source == "era5"
    assert bg.wind_speed.shape == (6, 5, 7)
    # |u| and |v| both in [-5, 5] → speed in [0, 7.07].
    assert bg.wind_speed.min() >= 0.0
    assert bg.wind_speed.max() <= np.sqrt(2) * 5 + 1e-5


def test_load_era5_reads_wind_speed_variable(tmp_path):
    _write_era5(tmp_path / "era5_wind_202002.nc", 2020, 2, use_components=False)
    bg = load_era5_background(tmp_path, 2020, 2)
    assert bg.wind_speed.min() >= 0.0


def test_load_era5_combined_file_slices_to_month(tmp_path):
    """Per-year-combined files must be sliced to only the requested month."""
    # Build a full year combined file (ERA5 naming convention).
    all_times = pd.date_range("2021-01-01", "2021-12-31 23:00", freq="h")
    lat = np.linspace(49.0, 46.0, 3)
    lon = np.linspace(9.0, 18.0, 4)
    rng = np.random.default_rng(0)
    ws = rng.uniform(0, 10, size=(len(all_times), 3, 4))
    ds = xr.Dataset(
        {"wind_speed": (("time", "latitude", "longitude"), ws)},
        coords={"time": all_times, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(tmp_path / "era5_wind_2021_combined.nc")

    bg = load_era5_background(tmp_path, 2021, 7)
    assert bg.wind_speed.shape[0] == (all_times.month == 7).sum()
    assert bg.times.month.unique().tolist() == [7]


def test_load_era5_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="no era5 file"):
        load_era5_background(tmp_path, 1999, 1)


# ---------------------------------------------------------------------------
# CERRA loader
# ---------------------------------------------------------------------------
def test_load_cerra_regular_grid(tmp_path):
    _write_cerra(tmp_path / "cerra_inca_202003.nc", 2020, 3, curvilinear=False)
    bg = load_cerra_background(tmp_path, 2020, 3)
    assert bg.source == "cerra"
    assert bg.lat.ndim == 1
    assert bg.lon.ndim == 1


def test_load_cerra_curvilinear_returns_2d_coords(tmp_path):
    _write_cerra(tmp_path / "cerra_inca_wind_202004.nc", 2020, 4, curvilinear=True)
    bg = load_cerra_background(tmp_path, 2020, 4)
    assert bg.lat.ndim == 2
    assert bg.lon.ndim == 2


def test_load_cerra_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="no cerra file"):
        load_cerra_background(tmp_path, 1999, 1)


# ---------------------------------------------------------------------------
# Regridder
# ---------------------------------------------------------------------------
def test_regrid_regular_grid_preserves_constant_field():
    grid = _tiny_grid()
    src_lat = np.linspace(49.0, 46.0, 10)
    src_lon = np.linspace(9.0, 18.0, 15)
    field = np.full((3, 10, 15), 5.0, dtype=np.float32)
    out = regrid_to_inca(field, src_lat, src_lon, grid=grid)
    assert out.shape == (3, grid.nrows, grid.ncols)
    assert out.dtype == np.float32
    # Constant in, constant out (modulo boundary fill-NaN → 0 — but
    # this source grid covers Austria so every target cell is inside).
    assert out.min() > 4.5
    assert out.max() < 5.5


def test_regrid_curvilinear_runs_and_preserves_constant_interior():
    grid = _tiny_grid()
    src_lat_2d = np.linspace(49.0, 46.0, 8)[:, None].repeat(12, axis=1) + \
                 np.linspace(-0.05, 0.05, 12)[None, :]
    src_lon_2d = np.linspace(9.0, 18.0, 12)[None, :].repeat(8, axis=0) + \
                 np.linspace(-0.1, 0.1, 8)[:, None]
    field = np.full((2, 8, 12), 3.0, dtype=np.float32)
    out = regrid_to_inca(field, src_lat_2d, src_lon_2d, grid=grid)
    assert out.shape == (2, grid.nrows, grid.ncols)
    assert out.min() >= 0.0
    # Constant 3 everywhere in the source → ~3 everywhere in the target
    # (boundary cells may fall back to nearest-neighbour which is also 3).
    assert np.all(np.abs(out - 3.0) < 1e-4)


def test_regrid_rejects_wrong_field_dim():
    grid = _tiny_grid()
    with pytest.raises(ValueError, match="3-D"):
        regrid_to_inca(np.zeros((5, 5)), np.zeros(5), np.zeros(5), grid=grid)


def test_regrid_uses_caller_tgt_latlon_when_provided():
    grid = _tiny_grid()
    src_lat = np.linspace(49.0, 46.0, 6)
    src_lon = np.linspace(9.0, 18.0, 8)
    field = np.ones((1, 6, 8), dtype=np.float32) * 7.0
    # Caller-provided tgt lat/lon of a different size — should respect it.
    tgt_lat = np.full((3, 4), 47.5)
    tgt_lon = np.full((3, 4), 13.5)
    out = regrid_to_inca(field, src_lat, src_lon, grid=grid,
                         tgt_lat=tgt_lat, tgt_lon=tgt_lon)
    # tgt shape wins over grid.shape when tgt_lat is 2-D.
    assert out.shape == (1, 3, 4)
