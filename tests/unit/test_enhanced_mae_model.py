"""Unit tests for :mod:`austrianwindatlas.stage1.enhanced_mae.model`."""
from __future__ import annotations

import numpy as np
import pytest

from austrianwindatlas.stage1.enhanced_mae.svd import compute_temporal_svd

torch = pytest.importorskip("torch")

from austrianwindatlas.stage1.enhanced_mae.model import (  # noqa: E402
    SpatialPatternDNN,
    TrainConfig,
    TrainResult,
    train_dnn,
)


def _toy_terrain_problem(n_stations: int = 50, n_eofs: int = 3, n_times: int = 400,
                        n_features: int = 6, seed: int = 0):
    """Synthetic (X, wind) pair where V is (noisily) linear in X.

    Lets the trainer actually converge in 30 epochs on a laptop CPU.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_stations, n_features))
    # V is linear-in-X + mild noise → DNN can learn it.
    mixer = rng.standard_normal((n_features, n_eofs)) * 0.5
    V_true = X @ mixer + rng.standard_normal((n_stations, n_eofs)) * 0.05
    sigma_true = np.array([5.0, 3.0, 1.5])[:n_eofs]
    U_true = rng.standard_normal((n_times, n_eofs))
    temporal_mean = 4.0 + rng.standard_normal(n_times)
    wind = temporal_mean[:, None] + U_true @ np.diag(sigma_true) @ V_true.T
    return X.astype(np.float32), wind.astype(np.float32)


def test_spatial_pattern_dnn_forward_shape():
    model = SpatialPatternDNN(input_dim=5, output_dim=8,
                              hidden_layers=2, hidden_units=16)
    x = torch.randn(4, 5)
    y = model(x)
    assert y.shape == (4, 8)


def test_spatial_pattern_dnn_has_batchnorm():
    model = SpatialPatternDNN(input_dim=5, output_dim=8,
                              hidden_layers=2, hidden_units=16)
    names = {type(m).__name__ for m in model.modules()}
    assert "BatchNorm1d" in names
    assert "ReLU" in names


def test_train_dnn_converges_on_toy_problem():
    X, wind = _toy_terrain_problem()
    svd = compute_temporal_svd(wind, n_eofs=3)
    cfg = TrainConfig(
        epochs=30, hidden_layers=3, hidden_units=32,
        lr=0.01, batch_size=8, patience=50, verbose_every=0, seed=42,
        recon_sample_size=50,
    )
    result = train_dnn(X, svd, wind, cfg)
    assert isinstance(result, TrainResult)
    # History populated.
    assert len(result.history) == 30
    # Loss should drop appreciably over training.
    first_loss = result.history[0].coef_loss
    last_loss = result.history[-1].coef_loss
    assert last_loss < first_loss * 0.8, (
        f"coef_loss did not fall enough: first={first_loss:.4f}, last={last_loss:.4f}"
    )


def test_train_dnn_returns_best_epoch_state():
    X, wind = _toy_terrain_problem()
    svd = compute_temporal_svd(wind, n_eofs=3)
    cfg = TrainConfig(epochs=20, patience=50, verbose_every=0, seed=1,
                      recon_sample_size=50)
    result = train_dnn(X, svd, wind, cfg)
    assert result.best_epoch >= 1
    # best_recon_mae must be <= the recon MAE at the final epoch.
    assert result.best_recon_mae <= result.history[-1].recon_mae + 1e-9


def test_train_dnn_with_weighted_mse_does_not_crash():
    X, wind = _toy_terrain_problem()
    svd = compute_temporal_svd(wind, n_eofs=3)
    cfg = TrainConfig(
        epochs=5, patience=50, verbose_every=0,
        loss_type="weighted_mse", weight_method="mean_wind", seed=0,
        recon_sample_size=50,
    )
    result = train_dnn(X, svd, wind, cfg)
    assert np.isfinite(result.best_recon_mae)


def test_train_dnn_rejects_mismatched_X_shape():
    X, wind = _toy_terrain_problem(n_stations=50)
    svd = compute_temporal_svd(wind, n_eofs=3)
    with pytest.raises(ValueError, match="svd describes"):
        # wrong number of rows
        train_dnn(X[:40], svd, wind, TrainConfig(epochs=1, verbose_every=0))
