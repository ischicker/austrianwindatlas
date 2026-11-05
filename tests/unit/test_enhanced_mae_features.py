"""Unit tests for :mod:`austrianwindatlas.stage1.enhanced_mae.features`.

Uses tiny synthetic GeoTIFFs (written with rasterio) rather than real
EU-DEM / CORINE files — keeps the tests hermetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from austrianwindatlas.io.windfield import GridSpec  # noqa: E402
from austrianwindatlas.stage1.enhanced_mae.features import (  # noqa: E402
    ALT_SCALE_METERS,
    DEFAULT_COVARIATE_FILES,
    FeatureSpec,
    build_feature_spec_for_covariates,
    sample_features_at_stations,
    sample_features_on_grid,
)


def _tiny_grid() -> GridSpec:
    # 10-km grid, 10 cells on each axis — small enough to sample fully
    # in a test without hitting real memory.
    return GridSpec(xmin=0.0, xmax=10_000.0, ymin=0.0, ymax=10_000.0,
                    ncols=10, nrows=10, crs="EPSG:31287")


def _write_raster(path, grid: GridSpec, data: np.ndarray, nodata: float | None = None):
    """Write a 1-band float32 GeoTIFF matching `grid`."""
    transform = from_origin(grid.xmin, grid.ymax,
                            (grid.xmax - grid.xmin) / grid.ncols,
                            (grid.ymax - grid.ymin) / grid.nrows)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": grid.nrows,
        "width": grid.ncols,
        "transform": transform,
        "crs": grid.crs,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.asarray(data, dtype=np.float32), 1)


# ---------------------------------------------------------------------------
# build_feature_spec_for_covariates
# ---------------------------------------------------------------------------
def test_build_feature_spec_detects_existing_files(tmp_path):
    grid = _tiny_grid()
    _write_raster(tmp_path / "EUDEM_AUSTRIA_1km.tif", grid, np.ones((10, 10)) * 500.0)
    _write_raster(tmp_path / "SLOPE_1km.tif", grid, np.zeros((10, 10)))
    spec = build_feature_spec_for_covariates(tmp_path, grid=grid)
    # elev + slope present on disk → in the names.
    assert "elev" in spec.names
    assert "slope" in spec.names
    # tpi / sin / cos / rough all missing → absent.
    assert "tpi" not in spec.names
    assert "sin" not in spec.names


def test_build_feature_spec_none_dir_has_only_base():
    spec = build_feature_spec_for_covariates(None)
    assert spec.names == ("x_norm", "y_norm", "alt_norm")
    assert spec.covariate_dir is None


def test_build_feature_spec_without_altitude():
    spec = build_feature_spec_for_covariates(None, include_altitude=False)
    assert "alt_norm" not in spec.names


def test_feature_spec_covariate_names_includes_only_rasters():
    spec = FeatureSpec(names=("x_norm", "y_norm", "alt_norm", "elev", "slope"))
    assert spec.covariate_names() == ("elev", "slope")


# ---------------------------------------------------------------------------
# sample_features_at_stations
# ---------------------------------------------------------------------------
def test_sample_features_at_stations_position_normalisation():
    grid = _tiny_grid()
    meta = pd.DataFrame({
        "x": [0.0, 10_000.0, 5_000.0],
        "y": [0.0, 10_000.0, 5_000.0],
        "alt": [0.0, 3_000.0, 1_500.0],
    })
    spec = FeatureSpec(names=("x_norm", "y_norm", "alt_norm"), grid=grid,
                       covariate_dir=None)
    X = sample_features_at_stations(meta, spec)
    assert X.shape == (3, 3)
    np.testing.assert_allclose(X[:, 0], [0.0, 1.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(X[:, 1], [0.0, 1.0, 0.5], atol=1e-6)
    # Altitude in km: 0, 1, 0.5
    np.testing.assert_allclose(X[:, 2], [0.0, 1.0, 0.5], atol=1e-6)


def test_sample_features_at_stations_raster_covariate(tmp_path):
    grid = _tiny_grid()
    # DEM: value i*10 + j at pixel (row=i, col=j).  Row 0 is the TOP
    # because rasterio stores north-up; the function computes
    # pixel = ~transform * (x, y) so station (x=500, y=9500) lands at
    # the top-left cell (0, 0) which carries value 0.
    dem = np.arange(100, dtype=np.float32).reshape(10, 10)
    _write_raster(tmp_path / "EUDEM_AUSTRIA_1km.tif", grid, dem)
    spec = FeatureSpec(
        names=("x_norm", "y_norm", "elev"),
        grid=grid, covariate_dir=tmp_path,
    )
    meta = pd.DataFrame({"x": [500.0, 9_500.0], "y": [9_500.0, 500.0]})
    X = sample_features_at_stations(meta, spec)
    assert X.shape == (2, 3)
    # Second station is at the bottom-right → row=9, col=9 → val 99.
    assert X[1, 2] == pytest.approx(99.0)
    assert X[0, 2] == pytest.approx(0.0)


def test_sample_features_at_stations_fills_nan_coords():
    grid = _tiny_grid()
    meta = pd.DataFrame({
        "x": [np.nan, 10_000.0],
        "y": [5_000.0, np.nan],
        "alt": [500.0, 1_000.0],
    })
    spec = FeatureSpec(names=("x_norm", "y_norm", "alt_norm"), grid=grid,
                       covariate_dir=None)
    X = sample_features_at_stations(meta, spec)
    assert np.isfinite(X).all()


def test_sample_features_at_stations_raises_without_alt_column():
    grid = _tiny_grid()
    meta = pd.DataFrame({"x": [0.0], "y": [0.0]})  # no 'alt' column
    spec = FeatureSpec(names=("x_norm", "y_norm", "alt_norm"), grid=grid,
                       covariate_dir=None)
    with pytest.raises(KeyError, match="alt"):
        sample_features_at_stations(meta, spec)


# ---------------------------------------------------------------------------
# sample_features_on_grid
# ---------------------------------------------------------------------------
def test_sample_features_on_grid_shape_matches_grid():
    grid = _tiny_grid()
    spec = FeatureSpec(names=("x_norm", "y_norm"), grid=grid, covariate_dir=None)
    X = sample_features_on_grid(spec)
    assert X.shape == (100, 2)


def test_sample_features_on_grid_alt_norm_flipped(tmp_path):
    grid = _tiny_grid()
    # DEM: row 0 (TOP of raster) is elevation 1000, row 9 (bottom) is 0.
    dem = np.repeat(np.linspace(1_000.0, 0.0, 10)[:, None], 10, axis=1)
    _write_raster(tmp_path / "EUDEM_AUSTRIA_1km.tif", grid, dem)
    spec = FeatureSpec(
        names=("alt_norm",), grid=grid,
        covariate_dir=tmp_path,
    )
    X = sample_features_on_grid(spec)  # (100, 1)
    # After flipud, row 0 of the returned feature matrix (= y_axis[0] =
    # grid.ymin) should see raster row 9 (elevation 0).
    # The feature matrix is C-ordered over (y, x), so the first 10
    # rows cover y=ymin and should be ≈ 0 / ALT_SCALE_METERS.
    assert X[:10, 0].mean() == pytest.approx(0.0, abs=1e-5)
    assert X[90:, 0].mean() == pytest.approx(1_000.0 / ALT_SCALE_METERS, abs=1e-5)


def test_sample_features_on_grid_raises_on_shape_mismatch(tmp_path):
    grid = _tiny_grid()
    # DEM has wrong shape.
    other = GridSpec(xmin=0.0, xmax=10_000.0, ymin=0.0, ymax=10_000.0,
                     ncols=5, nrows=5, crs="EPSG:31287")
    _write_raster(tmp_path / "EUDEM_AUSTRIA_1km.tif", other, np.zeros((5, 5)))
    spec = FeatureSpec(names=("alt_norm",), grid=grid, covariate_dir=tmp_path)
    with pytest.raises(ValueError, match="DEM shape"):
        sample_features_on_grid(spec)


def test_sample_features_on_grid_rejects_unknown_feature():
    grid = _tiny_grid()
    spec = FeatureSpec(names=("x_norm", "bogus"), grid=grid, covariate_dir=None,
                       covariate_files=dict(DEFAULT_COVARIATE_FILES))
    with pytest.raises(ValueError, match="unknown feature name"):
        sample_features_on_grid(spec)
