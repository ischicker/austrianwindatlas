"""Unit tests for :mod:`austrianwindatlas.stage1.enhanced_mae.losses`."""
from __future__ import annotations

import numpy as np
import pytest

from austrianwindatlas.stage1.enhanced_mae.losses import (
    LossFunction,
    compute_station_weights,
)

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# compute_station_weights (pure numpy)
# ---------------------------------------------------------------------------
def test_station_weights_mean_wind_shape_and_range():
    rng = np.random.default_rng(0)
    wind = rng.uniform(1, 10, size=(100, 8))
    w = compute_station_weights(wind, "mean_wind")
    assert w.shape == (8,)
    assert w.dtype == np.float32
    # Linear rescale pins the min to 0.5 and the max to 2.0.
    assert w.min() == pytest.approx(0.5, abs=1e-5)
    assert w.max() == pytest.approx(2.0, abs=1e-5)


def test_station_weights_variance_tracks_std():
    # Build columns with obvious std-ordering.
    rng = np.random.default_rng(1)
    base = rng.standard_normal((200, 4))
    scales = np.array([0.1, 1.0, 5.0, 10.0])
    wind = base * scales
    w = compute_station_weights(wind, "variance")
    # Most-variable column should get the highest weight.
    assert np.argmax(w) == 3
    assert np.argmin(w) == 0
    assert w.min() == pytest.approx(0.5, abs=1e-5)
    assert w.max() == pytest.approx(2.0, abs=1e-5)


def test_station_weights_quantile_is_binary_mask():
    means = np.linspace(1, 10, 20)[None, :]  # one timestep
    wind = np.repeat(means, 10, axis=0)
    w = compute_station_weights(wind, "quantile")
    # Top 25 % → weight 2; rest → weight 1.
    assert set(np.unique(w)) == {1.0, 2.0}
    assert np.sum(w == 2.0) == 5     # exactly 5 of 20 stations in top quartile
    # And those 5 are indeed the five windiest.
    high_w_idx = np.where(w == 2.0)[0]
    np.testing.assert_array_equal(np.sort(high_w_idx), np.arange(15, 20))


def test_station_weights_none_is_ones():
    wind = np.ones((5, 7))
    w = compute_station_weights(wind, "none")
    np.testing.assert_array_equal(w, np.ones(7, dtype=np.float32))


def test_station_weights_handles_nan():
    rng = np.random.default_rng(2)
    wind = rng.uniform(1, 10, size=(50, 4))
    wind[:10, 2] = np.nan
    w_mean = compute_station_weights(wind, "mean_wind")
    assert np.all(np.isfinite(w_mean))
    w_var = compute_station_weights(wind, "variance")
    assert np.all(np.isfinite(w_var))


def test_station_weights_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown weight method"):
        compute_station_weights(np.zeros((3, 3)), "bogus")  # type: ignore[arg-type]


def test_station_weights_rejects_1d_input():
    with pytest.raises(ValueError, match="2-D"):
        compute_station_weights(np.zeros(5), "mean_wind")


# ---------------------------------------------------------------------------
# LossFunction (torch)
# ---------------------------------------------------------------------------
def test_lossfunction_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown loss_type"):
        LossFunction("bogus")  # type: ignore[arg-type]


def test_lossfunction_mse_matches_torch_builtin():
    import torch.nn.functional as F
    rng = np.random.default_rng(3)
    pred = torch.tensor(rng.standard_normal((10, 4)), dtype=torch.float32)
    target = torch.tensor(rng.standard_normal((10, 4)), dtype=torch.float32)
    loss_fn = LossFunction("mse")
    expected = F.mse_loss(pred, target)
    assert torch.allclose(loss_fn(pred, target), expected)


def test_lossfunction_mae_matches_torch_builtin():
    import torch.nn.functional as F
    rng = np.random.default_rng(4)
    pred = torch.tensor(rng.standard_normal((10, 4)), dtype=torch.float32)
    target = torch.tensor(rng.standard_normal((10, 4)), dtype=torch.float32)
    expected = F.l1_loss(pred, target)
    assert torch.allclose(LossFunction("mae")(pred, target), expected)


def test_lossfunction_asymmetric_penalises_underestimation_more():
    # Build a case where pred < target (under-estimation) by a fixed margin.
    target = torch.ones(100, 1) * 2.0
    pred_under = torch.ones(100, 1) * 1.0  # under by 1
    pred_over = torch.ones(100, 1) * 3.0   # over by 1

    # alpha = 0.8 → under weighted 0.8, over weighted 0.2.
    loss_fn = LossFunction("asymmetric", alpha=0.8)
    under_loss = float(loss_fn(pred_under, target).item())
    over_loss = float(loss_fn(pred_over, target).item())
    # 4× the penalty for an under-estimation of the same magnitude.
    assert under_loss == pytest.approx(0.8)
    assert over_loss == pytest.approx(0.2)


def test_lossfunction_focal_penalises_large_errors_more():
    target = torch.zeros(100, 1)
    pred_small = torch.ones(100, 1) * 0.1  # small error
    pred_large = torch.ones(100, 1) * 2.0  # large error
    loss_fn = LossFunction("focal", gamma=2.0)
    small = float(loss_fn(pred_small, target).item())
    large = float(loss_fn(pred_large, target).item())
    # Large-error loss grows faster than linear due to (1+mae)^gamma factor.
    assert large / small > (2.0 / 0.1)


def test_lossfunction_weighted_mse_uses_station_weights():
    rng = np.random.default_rng(5)
    pred = torch.tensor(rng.standard_normal((10, 4)), dtype=torch.float32)
    target = torch.tensor(rng.standard_normal((10, 4)), dtype=torch.float32)
    # 10 stations, weights 1.0 everywhere except station 0 which gets 10.0
    w = torch.ones(10)
    w[0] = 10.0
    loss_fn = LossFunction("weighted_mse", station_weights=w)
    idx = torch.arange(10)
    out = loss_fn(pred, target, idx)

    # Match the paper's formula explicitly.
    expected = (w[idx].unsqueeze(1) * (pred - target).pow(2)).mean()
    assert torch.allclose(out, expected)


def test_lossfunction_weighted_fallbacks_without_indices_or_weights():
    import torch.nn.functional as F
    rng = np.random.default_rng(6)
    pred = torch.tensor(rng.standard_normal((10, 4)), dtype=torch.float32)
    target = torch.tensor(rng.standard_normal((10, 4)), dtype=torch.float32)

    # No indices → plain MSE.
    loss_fn = LossFunction("weighted_mse", station_weights=torch.ones(10))
    assert torch.allclose(loss_fn(pred, target, None), F.mse_loss(pred, target))

    # No weights → plain MAE.
    loss_fn = LossFunction("weighted_mae")
    assert torch.allclose(loss_fn(pred, target, torch.arange(10)), F.l1_loss(pred, target))
