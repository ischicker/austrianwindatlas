"""Loss-function helpers for the QN-GNN trainer.

Four pieces, used together inside :func:`austrianwindatlas.stage1.qn_gnn.train.train_gnn`:

* :func:`directional_loss` — MSE / MAE / asymmetric-MAE on the scaled
  ``V`` targets.
* :func:`variance_penalty` — bounds the model from collapsing V to
  near-constant.
* :func:`physical_recon_loss` — the paper's key innovation: evaluate
  the reconstruction loss in physical ``m/s`` (after inverting the
  per-station z-score) rather than in EOF space.  High-wind
  time-weighting optionally applied so storm timesteps dominate the
  signal.
* :func:`spatial_smoothness` — graph-weighted smoothness on V across
  neighbouring stations.

All functions are PyTorch; torch is imported lazily inside each
callable so the module loads without torch (same pattern as
Enhanced-MAE's :mod:`losses`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from numpy.typing import NDArray

__all__ = [
    "DirectionalLossKind",
    "directional_loss",
    "variance_penalty",
    "physical_recon_loss",
    "spatial_smoothness",
    "high_wind_time_weights",
]

DirectionalLossKind = Literal["mse", "mae", "asymmetric"]


def directional_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: DirectionalLossKind = "asymmetric",
    alpha: float = 0.65,
) -> torch.Tensor:
    """MSE / MAE / asymmetric MAE between predicted and target V.

    For ``asymmetric``, ``alpha`` weights under-estimation
    (``target > pred``) and ``1 - alpha`` weights over-estimation.
    ``alpha > 0.5`` penalises under-prediction more, which is the
    regime used in Enhanced-MAE and in QN-GNN.
    """
    import torch
    import torch.nn.functional as F

    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "mae":
        return F.l1_loss(pred, target)
    if loss_type == "asymmetric":
        diff = target - pred
        w = torch.where(diff > 0, alpha, 1.0 - alpha)
        return (w * diff.abs()).mean()
    raise ValueError(f"unknown loss_type {loss_type!r}")


def variance_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Penalise ``std(pred) / std(target) != 1``.

    The naive MSE objective is often minimised by shrinking the
    prediction towards its mean.  This term resists that.
    """

    target_std = target.std().clamp(min=1e-6)
    return (pred.std() / target_std - 1.0) ** 2


def physical_recon_loss(
    V_pred: torch.Tensor,            # (n_tr, n_eofs)
    U_batch: torch.Tensor,           # (B, n_eofs)
    sigma: torch.Tensor,             # (n_eofs,)
    temporal_mean_batch: torch.Tensor,   # (B,)
    wind_phys_batch: torch.Tensor,   # (B, n_tr) physical m/s, may hold NaN
    mu_tr: torch.Tensor,             # (n_tr,)   per-station z-score mean
    sigma_tr: torch.Tensor,          # (n_tr,)   per-station z-score std
    *,
    loss_type: DirectionalLossKind = "asymmetric",
    alpha: float = 0.65,
    max_wind: float = 50.0,
    time_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruction loss evaluated in physical m/s.

    Workflow::

        recon_z     = temporal_mean_batch + (U * sigma) @ V_pred.T
        recon_phys  = recon_z * sigma_tr + mu_tr        (linear inverse z-score)
        recon_phys  = clip(recon_phys, 0, max_wind)
        loss        = element-wise mae / mse / asymmetric-mae  of  (obs - recon_phys)
                       optionally reweighted per timestep by ``time_weights``.

    The point of evaluating here in m/s rather than in EOF space is
    that the paper's verification metric is RMSE in m/s on 54
    held-out stations; training against the same scale closes the
    train/verify gap by ~0.5 in Pearson-r (documented in the source
    code comments as the v5 → v6 improvement).
    """
    import torch

    # (B, n_eofs) → broadcast; then einsum against V_pred.
    US = U_batch * sigma
    recon_z = temporal_mean_batch.unsqueeze(1) + US @ V_pred.T       # (B, n_tr)

    recon_phys = recon_z * sigma_tr.unsqueeze(0) + mu_tr.unsqueeze(0)
    recon_phys = recon_phys.clamp(min=0.0, max=max_wind)

    # Where the observation is NaN, substitute the station climatology
    # (mu_tr) so the element contributes zero to the loss after the
    # reduction.  This is the paper's documented behaviour.
    ref = torch.where(
        torch.isfinite(wind_phys_batch),
        wind_phys_batch,
        mu_tr.unsqueeze(0).expand_as(wind_phys_batch),
    )

    diff = ref - recon_phys
    if loss_type == "mse":
        elem = diff.pow(2)
    elif loss_type == "mae":
        elem = diff.abs()
    elif loss_type == "asymmetric":
        w = torch.where(diff > 0, alpha, 1.0 - alpha)
        elem = w * diff.abs()
    else:  # pragma: no cover
        raise ValueError(f"unknown loss_type {loss_type!r}")

    if time_weights is not None:
        elem = elem * time_weights.unsqueeze(1)
    return elem.mean()


def spatial_smoothness(
    V_pred: torch.Tensor,            # (n_tr, n_eofs)
    edge_index: torch.Tensor,        # (2, E)  int64
    edge_attr: torch.Tensor,         # (E,) or (E, 1)  normalised distances in [0, 1]
    *,
    eps: float = 0.01,
) -> torch.Tensor:
    """Graph-weighted ``||V_i - V_j||^2`` summed over edges.

    Weight is ``1 / (d_ij + eps)`` with ``d_ij`` the normalised
    terrain-barrier distance from
    :func:`austrianwindatlas.stage1.qn_gnn.graph.build_station_graph`.
    Closer / less-barriered neighbours contribute more, forcing a
    terrain-coherent V field.
    """
    src, tgt = edge_index[0], edge_index[1]
    diff = V_pred[src] - V_pred[tgt]
    ea = edge_attr.squeeze(-1) if edge_attr.ndim == 2 else edge_attr
    w = 1.0 / (ea + eps)
    return (w.unsqueeze(1) * diff.pow(2)).mean()


def high_wind_time_weights(
    wind_data: NDArray[np.floating],
    *,
    percentile: float = 50.0,
    scale: float = 2.0,
    min_std: float = 0.1,
) -> NDArray[np.float32]:
    """Return per-timestep weights that upweight stormy periods.

    Formula::

        w_t = 1 + scale * max(0, wind_mean_t - P_{percentile}) / global_std
        w_t  /=  mean_t(w_t)                 (so the mean weight is 1)

    with ``wind_mean_t`` the mean across stations at timestep ``t``.

    The output is ``float32`` so it feeds straight into a torch tensor
    on the training device.  Paper default: ``percentile=50``,
    ``scale=2.0`` (upweight above-median timesteps).
    """
    arr = np.asarray(wind_data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"wind_data must be 2-D; got shape {arr.shape!r}")
    mean_t = np.nanmean(arr, axis=1)
    p = np.nanpercentile(mean_t, percentile)
    std = max(float(np.nanstd(mean_t)), min_std)
    w = 1.0 + scale * np.maximum(0.0, mean_t - p) / std
    w = w / w.mean()
    return w.astype(np.float32)
