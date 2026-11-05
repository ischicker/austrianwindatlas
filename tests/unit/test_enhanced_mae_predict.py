"""Unit tests for :mod:`austrianwindatlas.stage1.enhanced_mae.predict`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from austrianwindatlas.io.windfield import GridSpec
from austrianwindatlas.stage1.enhanced_mae.predict import (
    project_onto_eofs,
    reconstruct_month,
)
from austrianwindatlas.stage1.enhanced_mae.svd import compute_temporal_svd

torch = pytest.importorskip("torch")

from austrianwindatlas.stage1.enhanced_mae.features import FeatureSpec  # noqa: E402
from austrianwindatlas.stage1.enhanced_mae.model import TrainConfig, train_dnn  # noqa: E402
from austrianwindatlas.stage1.enhanced_mae.predict import predict_grid  # noqa: E402


# ---------------------------------------------------------------------------
# project_onto_eofs
# ---------------------------------------------------------------------------
def _toy_svd(seed: int = 0, n_times: int = 100, n_stations: int = 15, k: int = 3):
    """Build a wind matrix whose temporal-SVD factors are exactly known.

    V has zero across-station mean AND orthonormal columns; U has
    orthonormal columns.  Under these conditions
    ``project_onto_eofs(wind, V, sigma)`` must recover ``U`` exactly.
    """
    rng = np.random.default_rng(seed)
    U_raw = rng.standard_normal((n_times, k))
    U, _ = np.linalg.qr(U_raw)
    V_raw = rng.standard_normal((n_stations, k))
    V_centered = V_raw - V_raw.mean(axis=0, keepdims=True)
    V, _ = np.linalg.qr(V_centered)
    sigma = np.sort(rng.uniform(2.0, 10.0, size=k))[::-1]
    temporal_mean = 5.0 + rng.standard_normal(n_times)
    wind = temporal_mean[:, None] + U @ np.diag(sigma) @ V.T
    return wind, U, sigma, V, temporal_mean


def test_project_onto_eofs_recovers_U_and_temporal_mean():
    wind, U_true, sigma_true, V_true, tm_true = _toy_svd()
    U_hat, tm_hat = project_onto_eofs(wind, V_true, sigma_true)
    np.testing.assert_allclose(tm_hat, tm_true, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(U_hat, U_true, rtol=1e-6, atol=1e-6)


def test_project_onto_eofs_tolerates_nan():
    wind, _, sigma_true, V_true, _ = _toy_svd()
    rng = np.random.default_rng(1)
    wind_nan = wind.copy()
    wind_nan[rng.random(wind.shape) < 0.05] = np.nan
    U_hat, tm_hat = project_onto_eofs(wind_nan, V_true, sigma_true)
    assert np.isfinite(U_hat).all()
    assert np.isfinite(tm_hat).all()


def test_project_onto_eofs_shape_check():
    wind = np.zeros((10, 5))
    V = np.zeros((5, 3))
    sigma = np.array([1.0, 2.0])  # wrong length
    with pytest.raises(ValueError, match="sigma must have shape"):
        project_onto_eofs(wind, V, sigma)


def test_project_onto_eofs_rejects_1d_V():
    with pytest.raises(ValueError, match="V_stations must be 2-D"):
        project_onto_eofs(np.zeros((10, 5)), np.zeros(5), np.array([1.0]))


# ---------------------------------------------------------------------------
# reconstruct_month
# ---------------------------------------------------------------------------
def _small_grid() -> GridSpec:
    return GridSpec(xmin=0.0, xmax=4.0, ymin=0.0, ymax=2.0,
                    ncols=5, nrows=3, crs="EPSG:31287")


def test_reconstruct_month_formula_and_shape():
    grid = _small_grid()
    k = 2
    ny, nx = grid.nrows, grid.ncols
    rng = np.random.default_rng(0)
    V_grid = rng.standard_normal((k, ny, nx))
    sigma = np.array([3.0, 1.5])
    # 72 hourly timesteps split across two months (Jan + Feb).
    ts = pd.date_range("2020-01-01", periods=72, freq="h")
    U = rng.standard_normal((72, k))
    tmean = 5.0 + rng.standard_normal(72)

    ds = reconstruct_month(V_grid, sigma, U, tmean, ts,
                           year=2020, month=1, grid=grid, max_wind=50.0)
    assert ds is not None
    assert ds["wind_speed"].dims == ("time", "y", "x")
    # Hours in January 2020 between the start of ts and now: all 72 are
    # in January actually (ts goes from 2020-01-01 to 2020-01-03 23:00).
    assert ds.sizes["time"] == 72

    # Reconstruction formula check against a plain einsum.
    V_flat = V_grid.reshape(k, -1)
    expected_flat = (U * sigma) @ V_flat + tmean[:, None]
    expected = expected_flat.reshape(72, ny, nx).clip(0, 50)
    np.testing.assert_allclose(
        ds["wind_speed"].values, expected.astype(np.float32),
        rtol=0, atol=1e-5,
    )


def test_reconstruct_month_returns_none_for_empty_month():
    grid = _small_grid()
    k = 2
    ny, nx = grid.nrows, grid.ncols
    ts = pd.date_range("2020-01-01", periods=12, freq="h")
    V_grid = np.zeros((k, ny, nx))
    sigma = np.ones(k)
    U = np.zeros((12, k))
    tmean = np.zeros(12)
    ds = reconstruct_month(V_grid, sigma, U, tmean, ts,
                           year=2020, month=3, grid=grid)
    assert ds is None


def test_reconstruct_month_applies_background_blend():
    grid = _small_grid()
    k = 1
    ny, nx = grid.nrows, grid.ncols
    ts = pd.date_range("2020-01-01", periods=4, freq="h")
    # Pure-rPCA would predict ~0 everywhere; background is 10 everywhere.
    V_grid = np.zeros((k, ny, nx))
    sigma = np.ones(k)
    U = np.zeros((4, k))
    tmean = np.zeros(4)
    background = np.ones((4, ny, nx)) * 10.0

    ds = reconstruct_month(
        V_grid, sigma, U, tmean, ts, year=2020, month=1, grid=grid,
        background=background, background_weight=0.3,
    )
    assert ds is not None
    # (1 - 0.3) * 0 + 0.3 * 10 = 3 everywhere.
    np.testing.assert_allclose(ds["wind_speed"].values, 3.0, rtol=0, atol=1e-5)
    assert ds.attrs["background_field"] == "blended"
    assert ds.attrs["background_weight"] == pytest.approx(0.3)


def test_reconstruct_month_clips_above_max_wind():
    grid = _small_grid()
    k = 1
    ny, nx = grid.nrows, grid.ncols
    ts = pd.date_range("2020-01-01", periods=2, freq="h")
    V_grid = np.ones((k, ny, nx)) * 100.0
    sigma = np.ones(k)
    U = np.ones((2, k))
    tmean = np.zeros(2)
    ds = reconstruct_month(V_grid, sigma, U, tmean, ts, year=2020, month=1,
                           grid=grid, max_wind=25.0)
    assert ds["wind_speed"].values.max() <= 25.0


def test_reconstruct_month_rejects_grid_mismatch():
    grid = _small_grid()
    ts = pd.date_range("2020-01-01", periods=1, freq="h")
    # V_grid has wrong (ny, nx) compared to grid.
    V_grid = np.zeros((1, grid.nrows + 1, grid.ncols))
    with pytest.raises(ValueError, match="does not match grid"):
        reconstruct_month(V_grid, np.ones(1),
                          np.zeros((1, 1)), np.zeros(1), ts,
                          year=2020, month=1, grid=grid)


# ---------------------------------------------------------------------------
# predict_grid (end-to-end with a trained toy DNN)
# ---------------------------------------------------------------------------
def test_predict_grid_shape_matches_grid_and_svd():
    # Use a tiny grid, no covariates — predict_grid still works because
    # the feature spec is just (x_norm, y_norm).
    grid = GridSpec(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                    ncols=6, nrows=5, crs="EPSG:31287")
    rng = np.random.default_rng(0)
    # Stations on the grid corners + interior; features = (x_norm, y_norm).
    xs = np.linspace(0.0, 1.0, 10)
    ys = np.linspace(0.0, 1.0, 10)
    station_x, station_y = np.meshgrid(xs, ys)
    station_x = station_x.ravel()
    station_y = station_y.ravel()
    X = np.column_stack([station_x, station_y]).astype(np.float32)

    # Fake wind data so compute_temporal_svd has something to decompose.
    n_times, n_stations, k = 60, X.shape[0], 2
    U = rng.standard_normal((n_times, k))
    V_true = rng.standard_normal((n_stations, k)) * 0.5
    sigma_true = np.array([3.0, 1.5])
    tmean = 5.0 + rng.standard_normal(n_times)
    wind = tmean[:, None] + U @ np.diag(sigma_true) @ V_true.T

    svd = compute_temporal_svd(wind, n_eofs=k)
    cfg = TrainConfig(epochs=5, patience=50, verbose_every=0, seed=0,
                      recon_sample_size=30,
                      hidden_layers=2, hidden_units=16)
    result = train_dnn(X, svd, wind, cfg)

    spec = FeatureSpec(names=("x_norm", "y_norm"), grid=grid, covariate_dir=None)
    V_grid = predict_grid(
        result.model, result.X_scaler, result.V_scaler, spec,
    )
    assert V_grid.shape == (k, grid.nrows, grid.ncols)
    assert np.isfinite(V_grid).all()
    assert V_grid.dtype == np.float32
