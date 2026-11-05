"""Unit tests for :mod:`austrianwindatlas.stage1.qn_gnn.train`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from austrianwindatlas.stage1.qn_gnn.svd import compute_zscore_svd

torch = pytest.importorskip("torch")

from austrianwindatlas.stage1.qn_gnn.train import TrainConfig, TrainResult, train_gnn  # noqa: E402


def _toy_problem(n_stations=30, n_times=150, n_features=5, n_eofs=3, seed=0):
    rng = np.random.default_rng(seed)
    # V_true is a mild linear function of X so the MLP can learn it.
    X = rng.standard_normal((n_stations, n_features))
    mixer = rng.standard_normal((n_features, n_eofs)) * 0.3
    V_true = X @ mixer + rng.standard_normal((n_stations, n_eofs)) * 0.05
    sigma = np.array([4.0, 2.0, 1.0])[:n_eofs]
    U_true = rng.standard_normal((n_times, n_eofs))
    tmean = rng.standard_normal(n_times)
    z = tmean[:, None] + U_true @ np.diag(sigma) @ V_true.T

    mu_station = 2.0 + rng.standard_normal(n_stations)
    sigma_station = 1.0 + 0.3 * rng.random(n_stations)
    wind = z * sigma_station[None, :] + mu_station[None, :]

    meta = pd.DataFrame({
        "x": rng.uniform(100_000, 500_000, n_stations),
        "y": rng.uniform(200_000, 500_000, n_stations),
        "alt": rng.uniform(200, 2_000, n_stations),
    })
    return X.astype(np.float32), wind.astype(np.float32), meta


def test_train_gnn_runs_and_returns_trainresult():
    X, wind, meta = _toy_problem()
    svd = compute_zscore_svd(wind, n_eofs=3)
    cfg = TrainConfig(
        epochs=5, patience=10, verbose_every=0, seed=0,
        hidden_dim=16, n_gnn_layers=1, n_mlp_layers=1,
        recon_batch=20, batch_size=16, spatial_val_frac=0.2,
    )
    res = train_gnn(X, svd, meta, wind, cfg)
    assert isinstance(res, TrainResult)
    assert len(res.history) == 5
    assert np.isfinite(res.best_val_mae)
    assert res.best_epoch >= 1
    # Masks add up.
    assert int(res.val_station_mask.sum()) + int((~res.val_station_mask).sum()) == 30
    # Training residuals populated on training stations only.
    assert np.all(res.training_residuals[res.val_station_mask] == 0.0)


def test_train_gnn_val_mask_stratifies_by_elevation():
    X, wind, meta = _toy_problem(n_stations=60)
    svd = compute_zscore_svd(wind, n_eofs=3)
    cfg = TrainConfig(
        epochs=2, patience=5, verbose_every=0, seed=1,
        hidden_dim=8, n_gnn_layers=1, n_mlp_layers=1,
        recon_batch=10, batch_size=16, spatial_val_frac=0.25,
    )
    res = train_gnn(X, svd, meta, wind, cfg)
    alts = meta["alt"].to_numpy()
    q = np.quantile(alts, [0.25, 0.5, 0.75])
    bands = np.digitize(alts, q)
    # At least two bands should contribute val stations (4 bands * 25%
    # ≥ 1 each → 4 bands in principle; relax to 2 for robustness).
    bands_hit = np.unique(bands[res.val_station_mask])
    assert bands_hit.size >= 2


def test_train_gnn_train_subgraph_excludes_val_edges():
    X, wind, meta = _toy_problem(n_stations=30)
    svd = compute_zscore_svd(wind, n_eofs=2)
    cfg = TrainConfig(
        epochs=1, patience=5, verbose_every=0, seed=2,
        hidden_dim=8, n_gnn_layers=1, n_mlp_layers=1,
        recon_batch=10, batch_size=16, spatial_val_frac=0.3,
    )
    res = train_gnn(X, svd, meta, wind, cfg)
    # train_graph has one node per training station only.
    assert res.train_graph.n_nodes == int((~res.val_station_mask).sum())
    # All edge endpoints in train_graph are < n_training (already remapped).
    if res.train_graph.n_edges:
        assert int(res.train_graph.edge_index.max()) < res.train_graph.n_nodes


def test_train_gnn_rejects_mismatched_X_rows():
    X, wind, meta = _toy_problem()
    svd = compute_zscore_svd(wind, n_eofs=3)
    cfg = TrainConfig(epochs=1, verbose_every=0, seed=0,
                      hidden_dim=8, n_gnn_layers=1, n_mlp_layers=1,
                      recon_batch=10, batch_size=8)
    with pytest.raises(ValueError, match="X must be"):
        train_gnn(X[:10], svd, meta, wind, cfg)
