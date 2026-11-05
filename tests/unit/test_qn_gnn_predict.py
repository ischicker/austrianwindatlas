"""Unit tests for :mod:`austrianwindatlas.stage1.qn_gnn.predict`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from austrianwindatlas.io.windfield import GridSpec
from austrianwindatlas.stage1.qn_gnn.predict import spatial_bias_map, variance_correct
from austrianwindatlas.stage1.qn_gnn.svd import compute_zscore_svd

torch = pytest.importorskip("torch")

from austrianwindatlas.stage1.enhanced_mae.features import FeatureSpec  # noqa: E402
from austrianwindatlas.stage1.qn_gnn.predict import predict_grid  # noqa: E402
from austrianwindatlas.stage1.qn_gnn.train import TrainConfig, train_gnn  # noqa: E402


def _tiny_grid() -> GridSpec:
    # Metre-scale extents so the spatial_bias_map's 1 000-m minimum
    # distance floor doesn't swamp everything.  (This matches the
    # units the production INCA grid uses.)
    return GridSpec(xmin=0.0, xmax=100_000.0, ymin=0.0, ymax=100_000.0,
                    ncols=6, nrows=5, crs="EPSG:31287")


# ---------------------------------------------------------------------------
# variance_correct
# ---------------------------------------------------------------------------
def test_variance_correct_matches_station_std():
    rng = np.random.default_rng(0)
    V_grid = rng.standard_normal((100, 3)) * 0.1          # tiny variance
    V_stations = rng.standard_normal((20, 3)) * 2.0        # larger variance
    out = variance_correct(V_grid, V_stations)
    np.testing.assert_allclose(out.std(axis=0), V_stations.std(axis=0),
                               rtol=1e-5)


def test_variance_correct_preserves_mean():
    rng = np.random.default_rng(1)
    V_grid = rng.standard_normal((50, 2)) + 10.0
    V_stations = rng.standard_normal((10, 2)) * 5.0
    out = variance_correct(V_grid, V_stations)
    np.testing.assert_allclose(out.mean(axis=0), V_grid.mean(axis=0),
                               rtol=1e-5)


def test_variance_correct_leaves_zero_std_columns_alone():
    V_grid = np.ones((20, 2)) * 3.0                        # std = 0
    V_stations = np.random.default_rng(0).standard_normal((10, 2))
    out = variance_correct(V_grid, V_stations)
    np.testing.assert_allclose(out[:, 0], 3.0)


def test_variance_correct_rejects_eof_mismatch():
    with pytest.raises(ValueError, match="EOF count mismatch"):
        variance_correct(np.zeros((10, 3)), np.zeros((5, 2)))


# ---------------------------------------------------------------------------
# spatial_bias_map
# ---------------------------------------------------------------------------
def test_spatial_bias_map_zero_residuals_zero_map():
    grid = _tiny_grid()
    meta = pd.DataFrame({
        "x": [0.0, 100_000.0], "y": [50_000.0, 50_000.0], "alt": [0.0, 0.0],
    })
    residuals = np.zeros(2)
    bmap = spatial_bias_map(residuals, meta, grid=grid)
    assert bmap.shape == grid.shape
    assert np.all(bmap == 0.0)


def test_spatial_bias_map_station_dominates_nearest_cell():
    grid = _tiny_grid()
    # Stations at the two corners, 100 km apart, with opposite-sign
    # residuals.  The corresponding grid corners should pick up the
    # right sign strongly.
    meta = pd.DataFrame({
        "x": [grid.xmin, grid.xmax],
        "y": [grid.ymin, grid.ymax],
        "alt": [0.0, 0.0],
    })
    residuals = np.array([5.0, -5.0])
    bmap = spatial_bias_map(residuals, meta, grid=grid)
    assert bmap[0, 0] > 2.0          # bottom-left -> positive
    assert bmap[-1, -1] < -2.0       # top-right -> negative


def test_spatial_bias_map_ignores_nan_residuals():
    grid = _tiny_grid()
    meta = pd.DataFrame({
        "x": [50_000.0, 50_000.0, 50_000.0],
        "y": [50_000.0, 50_000.0, 50_000.0],
        "alt": [0.0, 0.0, 0.0],
    })
    residuals = np.array([3.0, np.nan, 7.0])  # middle station ignored
    bmap = spatial_bias_map(residuals, meta, grid=grid)
    # Three colocated stations with residuals [3, nan, 7] → ignore nan,
    # IDW collapses to equal weights on the two finite ones → mean 5.
    assert bmap.mean() == pytest.approx(5.0, abs=1e-4)


def test_spatial_bias_map_elevation_shape_check():
    grid = _tiny_grid()
    meta = pd.DataFrame({"x": [50_000.0], "y": [50_000.0], "alt": [0.0]})
    with pytest.raises(ValueError, match="elevation_grid size"):
        spatial_bias_map(np.array([1.0]), meta, grid=grid,
                         elevation_grid=np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# predict_grid (end-to-end with a tiny trained TerrainGNN)
# ---------------------------------------------------------------------------
def test_predict_grid_end_to_end():
    rng = np.random.default_rng(0)
    n_stations, n_eofs = 20, 2
    # Stations on a small grid area.
    xs = rng.uniform(0.0, 1.0, n_stations)
    ys = rng.uniform(0.0, 1.0, n_stations)
    X = np.column_stack([xs, ys]).astype(np.float32)
    meta = pd.DataFrame({
        "x": xs, "y": ys,
        "alt": rng.uniform(200, 2000, n_stations),
    })

    # Synthetic wind data so compute_zscore_svd + train_gnn have
    # something sensible to work with.
    n_times = 80
    sigma_true = np.array([3.0, 1.5])
    U = rng.standard_normal((n_times, n_eofs))
    V_true = rng.standard_normal((n_stations, n_eofs)) * 0.3
    tmean = rng.standard_normal(n_times)
    z = tmean[:, None] + U @ np.diag(sigma_true) @ V_true.T
    mu_st = 3.0 + rng.standard_normal(n_stations)
    sig_st = 1.0 + 0.3 * rng.random(n_stations)
    wind = z * sig_st[None, :] + mu_st[None, :]

    svd = compute_zscore_svd(wind, n_eofs=n_eofs)
    cfg = TrainConfig(
        epochs=3, patience=5, verbose_every=0, seed=1,
        hidden_dim=8, n_gnn_layers=1, n_mlp_layers=1,
        recon_batch=20, batch_size=16, spatial_val_frac=0.2,
    )
    res = train_gnn(X, svd, meta, wind, cfg)

    grid = _tiny_grid()
    spec = FeatureSpec(names=("x_norm", "y_norm"), grid=grid, covariate_dir=None)
    V_grid = predict_grid(res.model, res.X_scaler, res.V_scaler, spec, svd)
    assert V_grid.shape == (n_eofs, grid.nrows, grid.ncols)
    assert V_grid.dtype == np.float32
    assert np.isfinite(V_grid).all()


def test_predict_grid_rejects_feature_mismatch():
    # Train with 3 features, predict with a 2-feature spec → width
    # mismatch in the scaler must raise before we hit any torch call.
    rng = np.random.default_rng(0)
    n_stations, n_eofs = 20, 2
    xs = rng.uniform(0.0, 100_000.0, n_stations)
    ys = rng.uniform(0.0, 100_000.0, n_stations)
    extra = rng.uniform(-1.0, 1.0, n_stations)
    X = np.column_stack([xs, ys, extra]).astype(np.float32)        # 3 features
    meta = pd.DataFrame({
        "x": xs, "y": ys, "alt": rng.uniform(200, 2000, n_stations),
    })

    wind = rng.uniform(1, 10, size=(60, n_stations)).astype(np.float32)
    svd = compute_zscore_svd(wind, n_eofs=n_eofs)
    cfg = TrainConfig(epochs=1, patience=5, verbose_every=0, seed=0,
                      hidden_dim=8, n_gnn_layers=1, n_mlp_layers=1,
                      recon_batch=10, batch_size=8)
    res = train_gnn(X, svd, meta, wind, cfg)

    grid = _tiny_grid()
    spec = FeatureSpec(names=("x_norm", "y_norm"), grid=grid, covariate_dir=None)  # 2 features
    with pytest.raises(ValueError, match="feature stack width"):
        predict_grid(res.model, res.X_scaler, res.V_scaler, spec, svd)
