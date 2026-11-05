"""Unit tests for :mod:`austrianwindatlas.io.netcdf`."""
from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pyproj = pytest.importorskip("pyproj")

from austrianwindatlas.io.netcdf import (  # noqa: E402
    add_latlon_variables,
    compute_latlon_grid,
    ensure_crs_variable,
    epsg_from_dataset,
)


def _make_projected_dataset(nx: int = 11, ny: int = 7) -> xr.Dataset:
    """A small synthetic (y, x) dataset on the Austrian Lambert grid."""
    x = np.linspace(100_000.0, 200_000.0, nx)
    y = np.linspace(300_000.0, 400_000.0, ny)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((ny, nx)).astype(np.float32)
    return xr.Dataset(
        {"wind_speed": (["y", "x"], data)},
        coords={"x": x, "y": y},
        attrs={"crs": "EPSG:31287"},
    )


def test_epsg_from_dataset_prefers_attr():
    ds = _make_projected_dataset()
    assert epsg_from_dataset(ds) == "EPSG:31287"


def test_epsg_from_dataset_falls_back():
    ds = _make_projected_dataset()
    ds.attrs.pop("crs")
    assert epsg_from_dataset(ds, fallback="EPSG:3035") == "EPSG:3035"


def test_epsg_from_dataset_normalises_case_and_spacing():
    ds = _make_projected_dataset()
    ds.attrs["crs"] = "  epsg: 31287  "
    assert epsg_from_dataset(ds) == "EPSG:31287"


def test_compute_latlon_grid_shapes_and_dtype():
    x = np.linspace(100_000.0, 200_000.0, 5)
    y = np.linspace(300_000.0, 400_000.0, 3)
    lat, lon = compute_latlon_grid(x, y, "EPSG:31287")
    assert lat.shape == (3, 5)
    assert lon.shape == (3, 5)
    assert lat.dtype == np.float32
    assert lon.dtype == np.float32


def test_compute_latlon_grid_austria_in_wgs84():
    # The Austrian Lambert grid origin (~100 km east, 300 km north from
    # the EPSG:31287 false origin) is somewhere in Austria — should
    # land inside (~9 .. 18 °E, ~46 .. 50 °N).
    x = np.array([100_000.0, 500_000.0])
    y = np.array([300_000.0, 500_000.0])
    lat, lon = compute_latlon_grid(x, y, "EPSG:31287")
    assert np.all((lon > 9.0) & (lon < 18.0))
    assert np.all((lat > 46.0) & (lat < 50.0))


def test_ensure_crs_variable_attaches_wkt_and_attrs():
    ds = _make_projected_dataset()
    out = ensure_crs_variable(ds, "EPSG:31287")
    assert "crs" in out.variables
    assert "spatial_ref" in out["crs"].attrs
    assert "WGS_1984" in out["crs"].attrs["spatial_ref"] or "MGI" in out["crs"].attrs["spatial_ref"]
    assert out.attrs["crs"].startswith("EPSG:")


def test_add_latlon_variables_adds_lat_lon_and_crs():
    ds = _make_projected_dataset()
    out = add_latlon_variables(ds)
    assert "lat" in out.variables
    assert "lon" in out.variables
    assert out["lat"].dims == ("y", "x")
    assert out["lon"].dims == ("y", "x")
    assert out["lat"].attrs["standard_name"] == "latitude"
    assert out["lon"].attrs["standard_name"] == "longitude"
    assert "crs" in out.variables


def test_add_latlon_variables_skips_when_present_unless_overwrite():
    ds = _make_projected_dataset()
    once = add_latlon_variables(ds)
    # Smash the existing lat/lon so we can detect whether it was touched.
    once = once.assign(lat=once["lat"] * 0.0 + 999.0, lon=once["lon"] * 0.0 + 999.0)
    twice = add_latlon_variables(once, overwrite=False)
    assert float(twice["lat"].values[0, 0]) == 999.0

    thrice = add_latlon_variables(once, overwrite=True)
    assert float(thrice["lat"].values[0, 0]) != 999.0


def test_add_latlon_variables_grid_mapping_only_on_3d_vars():
    ds = _make_projected_dataset(nx=5, ny=4)
    rng = np.random.default_rng(1)
    ds["wind3d"] = (
        ("time", "y", "x"),
        rng.standard_normal((3, 4, 5)).astype(np.float32),
    )
    ds = ds.assign_coords(time=np.arange(3))
    out = add_latlon_variables(ds, add_grid_mapping=True)
    assert out["wind3d"].attrs.get("grid_mapping") == "crs"
    assert "grid_mapping" not in out["wind_speed"].attrs


def test_add_latlon_variables_raises_without_x():
    # Build a dataset without an 'x' coordinate at all.
    y = np.linspace(300_000.0, 400_000.0, 4)
    ds = xr.Dataset(
        {"wind_speed": (["y", "col"], np.zeros((4, 5)))},
        coords={"col": np.arange(5), "y": y},
    )
    with pytest.raises(ValueError, match="no 'x'"):
        add_latlon_variables(ds)
