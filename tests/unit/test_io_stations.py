"""Unit tests for :mod:`austrianwindatlas.io.stations`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from austrianwindatlas.io.stations import (  # noqa: E402
    DEFAULT_WIND_VAR_NAMES,
    apply_quality_control,
    extract_station_metadata,
    load_station_month,
    load_station_series,
    read_station_netcdf,
    resolve_wind_variable,
)


def _make_station_dataset(
    n_times: int = 6,
    n_stations: int = 4,
    *,
    wind_var: str = "ff",
    start: str = "2016-01-01",
    seed: int = 0,
) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=n_times, freq="h")
    wid = np.array([1001, 1002, 1003, 1004])[:n_stations]
    lon = np.array([14.0, 15.0, 16.0, 11.0])[:n_stations]
    lat = np.array([47.0, 48.0, 48.5, 47.5])[:n_stations]
    alt = np.array([200.0, 500.0, 1000.0, 300.0])[:n_stations]
    wind = rng.uniform(0.5, 10.0, size=(n_times, n_stations)).astype(np.float32)
    return xr.Dataset(
        {wind_var: (["time", "w_id"], wind)},
        coords={
            "time": times,
            "w_id": wid,
            "lon": ("w_id", lon),
            "lat": ("w_id", lat),
            "alt": ("w_id", alt),
        },
    )


def _write_to(tmp_path, ds: xr.Dataset, name: str) -> str:
    path = tmp_path / name
    ds.to_netcdf(path)
    return str(path)


def test_resolve_wind_variable_prefers_ff():
    ds = xr.Dataset({"ff": ("x", [1.0]), "wind_speed": ("x", [1.0])})
    assert resolve_wind_variable(ds) == "ff"


def test_resolve_wind_variable_fallback_to_wind_speed():
    ds = xr.Dataset({"wind_speed": ("x", [1.0])})
    assert resolve_wind_variable(ds) == "wind_speed"


def test_resolve_wind_variable_raises_when_none():
    ds = xr.Dataset({"temperature": ("x", [1.0])})
    with pytest.raises(KeyError):
        resolve_wind_variable(ds)


def test_default_wind_var_names_ordered():
    # Preference order: ff beats wind_speed beats ws beats FF
    assert DEFAULT_WIND_VAR_NAMES.index("ff") < DEFAULT_WIND_VAR_NAMES.index("wind_speed")


def test_read_station_netcdf_renames_to_wind_speed(tmp_path):
    ds = _make_station_dataset(wind_var="ff")
    path = _write_to(tmp_path, ds, "wind.nc")
    opened = read_station_netcdf(path)
    assert "wind_speed" in opened.variables
    assert "ff" not in opened.variables


def test_read_station_netcdf_passthrough_when_already_wind_speed(tmp_path):
    ds = _make_station_dataset(wind_var="wind_speed")
    path = _write_to(tmp_path, ds, "wind.nc")
    opened = read_station_netcdf(path)
    assert "wind_speed" in opened.variables


def test_read_station_netcdf_closes_on_failure(tmp_path):
    ds = xr.Dataset({"temperature": ("x", [1.0])})
    path = _write_to(tmp_path, ds, "no_wind.nc")
    with pytest.raises(KeyError):
        read_station_netcdf(path)


def test_load_station_month_finds_combined_pattern(tmp_path):
    ds = _make_station_dataset(start="2017-03-01")
    _write_to(tmp_path, ds, "combined_wind_201703.nc")
    opened = load_station_month(tmp_path, year=2017, month=3)
    assert "wind_speed" in opened.variables
    assert opened.sizes["time"] == 6


def test_load_station_month_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="no station NetCDF found"):
        load_station_month(tmp_path, year=2020, month=1)


def test_load_station_series_concats_months_sorted(tmp_path):
    ds_jan = _make_station_dataset(start="2018-01-01", n_times=3)
    ds_feb = _make_station_dataset(start="2018-02-01", n_times=3, seed=1)
    _write_to(tmp_path, ds_feb, "combined_wind_201802.nc")  # write out of order
    _write_to(tmp_path, ds_jan, "combined_wind_201801.nc")
    merged = load_station_series(tmp_path, "2018-01-01", "2018-02-28")
    assert merged.sizes["time"] == 6
    t = pd.to_datetime(merged.time.values)
    assert (t.sort_values() == t).all()  # sorted after concat
    # Spans Jan + Feb.
    assert t.min().month == 1 and t.max().month == 2


def test_load_station_series_tolerates_missing_months(tmp_path, caplog):
    ds_jan = _make_station_dataset(start="2018-01-01", n_times=3)
    _write_to(tmp_path, ds_jan, "combined_wind_201801.nc")
    # Request Jan + Feb; Feb is missing on disk.
    import logging
    with caplog.at_level(logging.WARNING, logger="austrianwindatlas.io.stations"):
        merged = load_station_series(tmp_path, "2018-01-01", "2018-02-28")
    assert merged.sizes["time"] == 3
    assert any("no station file for" in m for m in caplog.messages)


def test_load_station_series_raises_when_no_months_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="no station files found"):
        load_station_series(tmp_path, "2020-01-01", "2020-03-31")


def test_load_station_series_rejects_inverted_range(tmp_path):
    with pytest.raises(ValueError, match="end .* must be >= start"):
        load_station_series(tmp_path, "2020-03-01", "2020-01-01")


def test_extract_station_metadata_returns_expected_columns():
    ds = _make_station_dataset()
    meta = extract_station_metadata(ds)
    assert list(meta.columns) == ["w_id", "lon", "lat", "alt"]
    assert len(meta) == 4
    assert meta["w_id"].iloc[0] == 1001


def test_extract_station_metadata_missing_alt_is_nan():
    ds = _make_station_dataset().drop_vars("alt")
    meta = extract_station_metadata(ds)
    assert np.isnan(meta["alt"]).all()


def test_extract_station_metadata_averages_wiggly_coords():
    # Some files store (station, time) for lon with tiny jitter; the
    # function should average over the extra dim.
    ds = _make_station_dataset()
    rng = np.random.default_rng(0)
    jitter_lon = ds["lon"].values[:, None] + rng.normal(0, 1e-6, size=(4, 6))
    ds2 = ds.drop_vars("lon")
    ds2["lon"] = (("w_id", "time"), jitter_lon)
    meta = extract_station_metadata(ds2)
    np.testing.assert_allclose(meta["lon"].values, ds["lon"].values, rtol=0, atol=1e-5)


def test_apply_quality_control_flags_out_of_range():
    rng = np.random.default_rng(0)
    arr = rng.uniform(0, 8, size=(20, 4)).astype(float)
    arr[5, 0] = -1.0           # negative
    arr[7, 1] = 99.0           # too large
    arr[10, 2] = np.inf        # non-finite
    arr[13, 3] = np.nan        # already NaN
    result = apply_quality_control(arr, max_speed=50.0, min_completeness=0.0)
    assert result.n_invalid == 4
    assert np.isnan(result.wind_speed[5, 0])
    assert np.isnan(result.wind_speed[7, 1])
    assert np.isnan(result.wind_speed[10, 2])
    assert np.isnan(result.wind_speed[13, 3])


def test_apply_quality_control_valid_stations_mask():
    rng = np.random.default_rng(0)
    arr = rng.uniform(0, 8, size=(10, 3)).astype(float)
    arr[:8, 0] = np.nan   # station 0 is 80 % missing → below 50 %
    arr[:2, 1] = np.nan   # station 1 is 20 % missing → above 50 %
    result = apply_quality_control(arr, max_speed=50.0, min_completeness=0.5)
    assert not result.valid_stations[0]
    assert result.valid_stations[1]
    assert result.valid_stations[2]
    assert result.station_completeness[0] < 0.5
    assert result.station_completeness[1] >= 0.5


def test_apply_quality_control_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        apply_quality_control(np.zeros(5))
