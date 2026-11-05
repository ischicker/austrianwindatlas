"""Unit tests for :mod:`austrianwindatlas.stage1.qn_gnn.losses`."""
from __future__ import annotations

import numpy as np
import pytest

from austrianwindatlas.stage1.qn_gnn.losses import high_wind_time_weights

torch = pytest.importorskip("torch")

from austrianwindatlas.stage1.qn_gnn.losses import (  # noqa: E402
    directional_loss,
    physical_recon_loss,
    spatial_smoothness,
    variance_penalty,
)


# ---------------------------------------------------------------------------
# directional_loss
# ---------------------------------------------------------------------------
def test_directional_mse_matches_builtin():
    import torch.nn.functional as F
    rng = np.random.default_rng(0)
    pred = torch.tensor(rng.standard_normal((10, 4)), dtype=torch.float32)
    tgt = torch.tensor(rng.standard_normal((10, 4)), dtype=torch.float32)
    assert torch.allclose(
        directional_loss(pred, tgt, loss_type="mse"),
        F.mse_loss(pred, tgt),
    )


def test_directional_asymmetric_penalises_underestimation():
    target = torch.ones(50, 1) * 2.0
    pred_under = torch.ones(50, 1) * 1.0      # under by 1
    pred_over = torch.ones(50, 1) * 3.0       # over by 1

    def _loss(pred):
        return directional_loss(pred, target, loss_type="asymmetric", alpha=0.8)

    assert float(_loss(pred_under).item()) == pytest.approx(0.8)
    assert float(_loss(pred_over).item()) == pytest.approx(0.2)


def test_directional_loss_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown loss_type"):
        directional_loss(torch.zeros(1, 1), torch.zeros(1, 1), loss_type="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# variance_penalty
# ---------------------------------------------------------------------------
def test_variance_penalty_zero_when_stds_match():
    rng = np.random.default_rng(1)
    pred = torch.tensor(rng.standard_normal((100, 3)) * 2.0)
    target = pred * 1.0                    # same std
    assert float(variance_penalty(pred, target).item()) == pytest.approx(0.0, abs=1e-6)


def test_variance_penalty_grows_quadratically():
    rng = np.random.default_rng(2)
    target = torch.tensor(rng.standard_normal((100, 3)))  # std ~ 1
    pred_low = target * 0.5
    pred_high = target * 2.0
    # Ratios 0.5 and 2.0 → (ratio - 1)^2 is 0.25 and 1.0.
    assert float(variance_penalty(pred_low, target).item()) == pytest.approx(0.25, rel=0.1)
    assert float(variance_penalty(pred_high, target).item()) == pytest.approx(1.0, rel=0.1)


# ---------------------------------------------------------------------------
# physical_recon_loss
# ---------------------------------------------------------------------------
def test_physical_recon_zero_when_V_matches_truth():
    """If V_pred == V_true and U, sigma, mu, sigma_tr, temporal_mean are
    consistent, the physical recon loss should be ~ zero.
    """
    rng = np.random.default_rng(0)
    n_tr, n_eofs, n_times = 10, 3, 20

    U_raw = rng.standard_normal((n_times, n_eofs))
    U, _ = np.linalg.qr(U_raw)
    V_raw = rng.standard_normal((n_tr, n_eofs))
    V_cent = V_raw - V_raw.mean(axis=0, keepdims=True)
    V, _ = np.linalg.qr(V_cent)
    sigma = np.array([4.0, 2.0, 1.0])
    tmean = rng.standard_normal(n_times)
    mu_tr = 3.0 + rng.standard_normal(n_tr)
    sigma_tr = 1.0 + 0.3 * rng.random(n_tr)

    recon_z = tmean[:, None] + (U * sigma) @ V.T
    wind_phys = recon_z * sigma_tr[None, :] + mu_tr[None, :]

    loss = physical_recon_loss(
        V_pred=torch.tensor(V, dtype=torch.float32),
        U_batch=torch.tensor(U, dtype=torch.float32),
        sigma=torch.tensor(sigma, dtype=torch.float32),
        temporal_mean_batch=torch.tensor(tmean, dtype=torch.float32),
        wind_phys_batch=torch.tensor(wind_phys, dtype=torch.float32),
        mu_tr=torch.tensor(mu_tr, dtype=torch.float32),
        sigma_tr=torch.tensor(sigma_tr, dtype=torch.float32),
        loss_type="mae", alpha=0.5, max_wind=100.0,
    )
    # Tiny loss — the QR-orthonormal V is only *approximately* zero-station-
    # mean after centering, so the physical-space loss carries a sub-cm/s
    # residual.  Well below the 0.1 m/s scale that actually matters.
    assert float(loss.item()) < 1e-2


def test_physical_recon_time_weights_rescale():
    # Simple case: constant V_pred wrong by 1 m/s everywhere.
    n_tr, n_eofs, n_times = 4, 2, 6
    V = torch.zeros(n_tr, n_eofs)
    U = torch.zeros(n_times, n_eofs)
    sigma = torch.ones(n_eofs)
    tmean = torch.zeros(n_times)
    mu_tr = torch.zeros(n_tr)
    sigma_tr = torch.ones(n_tr)
    wind_phys = torch.ones(n_times, n_tr)        # obs = 1

    loss_uniform = physical_recon_loss(
        V, U, sigma, tmean, wind_phys, mu_tr, sigma_tr,
        loss_type="mae", alpha=0.5,
    )
    # With weights = [1,1,1,1,1,0] normalised to mean 1:
    weights = torch.tensor([1.2, 1.2, 1.2, 1.2, 1.2, 0.0])
    loss_weighted = physical_recon_loss(
        V, U, sigma, tmean, wind_phys, mu_tr, sigma_tr,
        loss_type="mae", alpha=0.5, time_weights=weights,
    )
    # Every element contributes |1 - 0| = 1; uniform mean = 1.
    assert float(loss_uniform.item()) == pytest.approx(1.0)
    # Weighted: 5 rows contribute 1.2, 1 row contributes 0 → mean = 1.0.
    assert float(loss_weighted.item()) == pytest.approx(1.0, abs=1e-5)


def test_physical_recon_treats_nan_obs_as_climatology():
    # When obs is NaN, substitute mu_tr → diff is zero → that element
    # contributes nothing on top of the baseline.
    n_tr, n_eofs, n_times = 3, 1, 2
    V = torch.zeros(n_tr, n_eofs)
    U = torch.zeros(n_times, n_eofs)
    sigma = torch.ones(n_eofs)
    tmean = torch.zeros(n_times)
    mu_tr = torch.tensor([5.0, 5.0, 5.0])
    sigma_tr = torch.ones(n_tr)
    # Obs: all finite at 5 → diff zero; then make one NaN.
    obs = torch.full((n_times, n_tr), 5.0)
    obs[0, 1] = float("nan")
    loss = physical_recon_loss(
        V, U, sigma, tmean, obs, mu_tr, sigma_tr,
        loss_type="mae", alpha=0.5,
    )
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# spatial_smoothness
# ---------------------------------------------------------------------------
def test_spatial_smoothness_zero_on_constant_V():
    V = torch.ones(5, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_attr = torch.tensor([0.1, 0.2, 0.3, 0.4])
    assert float(spatial_smoothness(V, edge_index, edge_attr).item()) == pytest.approx(0.0)


def test_spatial_smoothness_weighting_prefers_closer_neighbours():
    # Two edges with equal V-differences but different distances.
    V = torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    # Edge 0 → 1 (distance 0.1) and edge 0 → 2 (distance 10.0): same
    # diff, but the closer edge contributes much more to the loss.
    e0 = torch.tensor([[0], [1]])
    e1 = torch.tensor([[0], [2]])
    close = float(spatial_smoothness(V, e0, torch.tensor([0.1])).item())
    far = float(spatial_smoothness(V, e1, torch.tensor([10.0])).item())
    assert close > 50 * far


# ---------------------------------------------------------------------------
# high_wind_time_weights (pure numpy)
# ---------------------------------------------------------------------------
def test_high_wind_time_weights_mean_is_one():
    rng = np.random.default_rng(0)
    wind = rng.uniform(0, 20, size=(100, 10))
    w = high_wind_time_weights(wind)
    assert w.mean() == pytest.approx(1.0, rel=1e-5)
    assert w.dtype == np.float32


def test_high_wind_time_weights_above_median_higher():
    wind = np.linspace(0, 20, 50)[:, None].repeat(5, axis=1)  # monotone increase
    w = high_wind_time_weights(wind, percentile=50.0, scale=2.0)
    # The top-half timesteps should carry more weight than the bottom.
    assert w[-10:].mean() > w[:10].mean()


def test_high_wind_time_weights_rejects_1d():
    with pytest.raises(ValueError, match="2-D"):
        high_wind_time_weights(np.zeros(5))
