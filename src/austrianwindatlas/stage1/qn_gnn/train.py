"""Training loop for the QN-GNN TerrainGNN.

Key features — all faithful to ``dl_qn_gnn.py``'s ``train_gnn_v6``:

* **Spatial validation split** — a stratified-by-elevation subset of
  stations is held out of both the training graph AND the training
  losses.  Early stopping monitors per-epoch reconstruction MAE on
  the spatial holdout; this measures "can the model interpolate to
  unseen locations?" which is exactly what the 54 permanently
  held-out verification stations in the paper measure.

* **Physical-space reconstruction loss** — see
  :func:`austrianwindatlas.stage1.qn_gnn.losses.physical_recon_loss`.

* **GNN + MLP blending** — both heads predict V; the final target is
  ``(1 - w) * V_gnn + w * V_mlp``.  Grad flows into both so they
  agree, giving the MLP a usable grid-prediction head even though it
  never sees the graph at inference.

* **Spatial smoothness** regularisation on V across training-subgraph
  edges.

The trainer returns a dataclass :class:`TrainResult` with the model,
the two scalers, the training residuals (useful for the spatial bias
correction at inference time), the training edge subgraph and a
history of per-epoch metrics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .graph import StationGraph, build_station_graph
from .losses import (
    DirectionalLossKind,
    directional_loss,
    high_wind_time_weights,
    physical_recon_loss,
    spatial_smoothness,
    variance_penalty,
)
from .model import build_terrain_gnn
from .svd import ZScoreSVD

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import torch.nn as nn
    from numpy.typing import NDArray
    from sklearn.preprocessing import StandardScaler

__all__ = [
    "TrainConfig",
    "EpochRecord",
    "TrainResult",
    "train_gnn",
]

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Hyperparameters for :func:`train_gnn`.

    All defaults match the paper's final (v6) QN-GNN run.
    """

    epochs: int = 400
    lr: float = 1e-3
    patience: int = 60
    hidden_dim: int = 128
    n_gnn_layers: int = 6
    n_mlp_layers: int = 4
    dropout: float = 0.20
    weight_decay: float = 5e-4
    grad_clip: float = 1.0

    mlp_weight: float = 0.5                 # blend GNN + MLP targets
    loss_type: DirectionalLossKind = "asymmetric"
    alpha: float = 0.65
    lambda_var: float = 0.02                # variance_penalty coefficient
    lambda_recon: float = 0.15              # physical_recon_loss coefficient
    lambda_smooth: float = 0.05             # spatial_smoothness coefficient

    recon_batch: int = 128
    batch_size: int = 256

    spatial_val_frac: float = 0.15
    spatial_val_seed: int = 42

    device: str = "cpu"
    seed: int | None = None
    verbose_every: int = 20


@dataclass
class EpochRecord:
    """One-row diagnostic snapshot per epoch."""
    epoch: int
    train_loss: float
    val_mae_phys: float
    lr: float


@dataclass
class TrainResult:
    """Return value of :func:`train_gnn`.

    Attributes
    ----------
    model
        Trained :class:`torch.nn.Module` restored to the best-val-MAE
        epoch, in ``eval`` mode.
    X_scaler, V_scaler
        Fitted :class:`sklearn.preprocessing.StandardScaler`s.
    training_residuals
        ``(n_stations,)`` mean residual (predicted - observed) in
        physical m/s on the *training* stations after the final
        prediction.  Used downstream by the spatial bias-correction
        map in :func:`austrianwindatlas.stage1.qn_gnn.predict.predict_grid`.
    graph
        The full station graph (training + validation stations).
    train_graph
        Sub-graph with only the training-station edges — same object
        passed into the smoothness regulariser.
    val_station_mask
        ``(n_stations,)`` bool, True for the held-out validation
        stations (same numbering as the input ``meta_df``).
    history
        Per-epoch diagnostics.
    best_val_mae
        Lowest validation MAE achieved.
    best_epoch
        1-based epoch index at which ``best_val_mae`` was hit.
    """

    model: nn.Module
    X_scaler: StandardScaler
    V_scaler: StandardScaler
    training_residuals: NDArray[np.floating]
    graph: StationGraph
    train_graph: StationGraph
    val_station_mask: NDArray[np.bool_]
    history: list[EpochRecord] = field(default_factory=list)
    best_val_mae: float = float("inf")
    best_epoch: int = -1


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------
def _stratified_elevation_split(
    elevations: NDArray[np.floating],
    val_frac: float,
    seed: int,
) -> NDArray[np.bool_]:
    """Stratify stations into elevation bands and hold ``val_frac`` out of each.

    Returns a boolean mask of shape ``(n_stations,)`` — True for the
    validation stations.  Four elevation bands (quartiles) are used;
    that's deep enough that each terrain regime (plains, hills,
    mid-mountain, high-alps) is represented in the holdout.
    """
    n = elevations.size
    val_n = max(1, int(round(val_frac * n)))
    rng = np.random.default_rng(seed)

    # Quartile edges, with a fallback when < 4 unique elevations.
    finite = elevations[np.isfinite(elevations)]
    if finite.size < 4:
        picks = rng.choice(n, size=min(val_n, n), replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[picks] = True
        return mask
    q = np.quantile(finite, [0.25, 0.5, 0.75])
    bands = np.digitize(elevations, q)                         # 0..3
    mask = np.zeros(n, dtype=bool)
    for b in range(4):
        band_idx = np.where(bands == b)[0]
        if band_idx.size == 0:
            continue
        k = max(1, int(round(val_frac * band_idx.size)))
        picks = rng.choice(band_idx, size=min(k, band_idx.size), replace=False)
        mask[picks] = True
    return mask


def _train_subgraph(graph: StationGraph, train_mask: NDArray[np.bool_]) -> StationGraph:
    """Keep only edges whose endpoints are both training stations.

    Node indices are *re-numbered* to the compact training-only space
    (so ``edge_index.max() < train_mask.sum()``).  The caller is
    responsible for mapping arrays from the full station space into
    the training-only space before passing them to the GNN.
    """
    old_to_new = -np.ones(graph.n_nodes, dtype=np.int64)
    kept_nodes = np.where(train_mask)[0]
    old_to_new[kept_nodes] = np.arange(kept_nodes.size)

    ei = graph.edge_index
    keep = train_mask[ei[0]] & train_mask[ei[1]]
    new_ei = np.stack([old_to_new[ei[0][keep]], old_to_new[ei[1][keep]]], axis=0)
    new_attr = graph.edge_attr[keep]
    return StationGraph(
        edge_index=new_ei.astype(np.int64),
        edge_attr=new_attr.astype(np.float32),
        n_nodes=int(kept_nodes.size),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def train_gnn(
    X: NDArray[np.floating],
    svd: ZScoreSVD,
    meta_df: pd.DataFrame,
    wind_data: NDArray[np.floating],
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train the TerrainGNN with spatial validation holdout.

    Parameters
    ----------
    X
        ``(n_stations, n_features)`` terrain feature matrix.
    svd
        Result of :func:`~austrianwindatlas.stage1.qn_gnn.svd.compute_zscore_svd`
        on ``wind_data``.  Columns of ``X`` must align with the stations.
    meta_df
        Station metadata — must carry at least ``x``, ``y``, ``alt``
        columns.  Used for graph construction and elevation
        stratification.
    wind_data
        ``(n_times, n_stations)`` raw wind matrix (m/s, NaN-tolerant).
        Used for the physical reconstruction loss and the high-wind
        temporal weighting.
    config
        :class:`TrainConfig`; falls back to defaults when ``None``.

    Returns
    -------
    TrainResult
    """
    import torch
    from sklearn.preprocessing import StandardScaler

    cfg = config if config is not None else TrainConfig()
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != svd.n_stations:
        raise ValueError(
            f"X must be (n_stations={svd.n_stations}, n_features); got shape {X.shape!r}"
        )

    device = torch.device(cfg.device)
    n_stations, n_features = X.shape
    n_eofs = svd.n_eofs

    # ----- Build the full graph ------------------------------------------
    coords = meta_df[["x", "y"]].to_numpy(dtype=float).copy()
    valid_c = np.isfinite(coords).all(axis=1)
    coords[~valid_c] = coords[valid_c].mean(axis=0)
    elevations = (
        np.nan_to_num(meta_df["alt"].to_numpy(dtype=float), nan=500.0)
        if "alt" in meta_df.columns else None
    )

    full_graph = build_station_graph(coords, elevations=elevations)
    edge_max = max(float(full_graph.edge_attr.max() or 0.0), 1e-9)
    normalised_attr = full_graph.edge_attr.astype(np.float32) / edge_max
    normalised_graph = StationGraph(
        edge_index=full_graph.edge_index,
        edge_attr=normalised_attr,
        n_nodes=full_graph.n_nodes,
    )

    # ----- Spatial holdout ------------------------------------------------
    val_mask = _stratified_elevation_split(
        elevations if elevations is not None else np.zeros(n_stations),
        cfg.spatial_val_frac,
        cfg.spatial_val_seed,
    )
    train_mask = ~val_mask
    train_graph = _train_subgraph(normalised_graph, train_mask)
    tr_idx = np.where(train_mask)[0]
    n_tr = int(tr_idx.size)
    logger.info("train_gnn: %d training stations, %d validation", n_tr, int(val_mask.sum()))

    # ----- Scalers (on training subset only) -----------------------------
    X_scaler = StandardScaler().fit(X[tr_idx])
    V_scaler = StandardScaler().fit(svd.V[tr_idx])
    X_scaled_full = X_scaler.transform(X)              # (n_sta, n_feat)
    V_scaled_tr = V_scaler.transform(svd.V[tr_idx])    # (n_tr, n_eofs)

    # ----- Per-timestep high-wind weights --------------------------------
    time_weights_np = high_wind_time_weights(wind_data)
    time_weights = torch.as_tensor(time_weights_np, dtype=torch.float32, device=device)

    # ----- Tensors --------------------------------------------------------
    X_tr = torch.as_tensor(X_scaled_full[tr_idx], dtype=torch.float32, device=device)
    X_val = torch.as_tensor(X_scaled_full[val_mask], dtype=torch.float32, device=device)
    V_target_tr = torch.as_tensor(V_scaled_tr, dtype=torch.float32, device=device)
    ei_t = torch.as_tensor(train_graph.edge_index, dtype=torch.long, device=device)
    ea_t = torch.as_tensor(train_graph.edge_attr, dtype=torch.float32, device=device)
    sigma_t = torch.as_tensor(svd.sigma, dtype=torch.float32, device=device)
    U_all = torch.as_tensor(svd.U, dtype=torch.float32, device=device)
    tm_all = torch.as_tensor(svd.temporal_mean, dtype=torch.float32, device=device)

    mu_tr = torch.as_tensor(svd.qt.means_[tr_idx], dtype=torch.float32, device=device)
    sigma_tr = torch.as_tensor(svd.qt.stds_[tr_idx], dtype=torch.float32, device=device)
    mu_val = torch.as_tensor(svd.qt.means_[val_mask], dtype=torch.float32, device=device)
    sigma_val = torch.as_tensor(svd.qt.stds_[val_mask], dtype=torch.float32, device=device)

    # Physical wind matrices for the recon loss and validation.
    wind_phys_all_np = np.where(np.isfinite(wind_data), wind_data, np.nan).astype(np.float32)
    wind_phys_tr = torch.as_tensor(wind_phys_all_np[:, tr_idx], device=device)
    wind_phys_val = torch.as_tensor(wind_phys_all_np[:, val_mask], device=device)

    V_mean_t = torch.as_tensor(V_scaler.mean_, dtype=torch.float32, device=device)
    V_std_t = torch.as_tensor(V_scaler.scale_, dtype=torch.float32, device=device)

    # ----- Model + optimiser ---------------------------------------------
    model = build_terrain_gnn(
        n_features, n_eofs,
        hidden_dim=cfg.hidden_dim,
        n_gnn_layers=cfg.n_gnn_layers,
        n_mlp_layers=cfg.n_mlp_layers,
        dropout=cfg.dropout,
    ).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    best_val = float("inf")
    best_state: dict | None = None
    best_epoch = -1
    wait = 0
    history: list[EpochRecord] = []
    n_times = svd.n_times

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        # ---- Step 1: V-space loss on the training stations -------------
        V_gnn = model.forward_gnn(X_tr, ei_t, ea_t)              # (n_tr, n_eofs)
        V_mlp = model.forward_mlp(X_tr)                          # (n_tr, n_eofs)
        V_pred_scaled = (1.0 - cfg.mlp_weight) * V_gnn + cfg.mlp_weight * V_mlp

        v_loss = directional_loss(
            V_pred_scaled, V_target_tr,
            loss_type=cfg.loss_type, alpha=cfg.alpha,
        )
        v_var = variance_penalty(V_pred_scaled, V_target_tr)

        # ---- Step 2: physical recon loss on a random timestep batch ----
        batch_idx = torch.randint(0, n_times, (cfg.recon_batch,), device=device)
        V_pred_unscaled = V_pred_scaled * V_std_t + V_mean_t
        v_recon = physical_recon_loss(
            V_pred=V_pred_unscaled,
            U_batch=U_all[batch_idx],
            sigma=sigma_t,
            temporal_mean_batch=tm_all[batch_idx],
            wind_phys_batch=wind_phys_tr[batch_idx],
            mu_tr=mu_tr, sigma_tr=sigma_tr,
            loss_type=cfg.loss_type, alpha=cfg.alpha,
            time_weights=time_weights[batch_idx],
        )

        # ---- Step 3: spatial smoothness on training subgraph edges ----
        if train_graph.n_edges > 0:
            v_smooth = spatial_smoothness(V_pred_scaled, ei_t, ea_t)
        else:
            v_smooth = torch.zeros((), device=device)

        loss = (
            v_loss
            + cfg.lambda_var * v_var
            + cfg.lambda_recon * v_recon
            + cfg.lambda_smooth * v_smooth
        )
        if not torch.isfinite(loss):
            logger.warning("non-finite training loss at epoch %d; skipping", epoch)
            continue

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optim.step()

        # ---- Validation: physical-space MAE at held-out stations ------
        model.eval()
        with torch.no_grad():
            V_val_pred_scaled = model.forward_mlp(X_val)
            V_val_unscaled = V_val_pred_scaled * V_std_t + V_mean_t

            sample_idx = torch.randint(0, n_times, (min(cfg.recon_batch, n_times),),
                                       device=device)
            US = U_all[sample_idx] * sigma_t
            recon_z = tm_all[sample_idx].unsqueeze(1) + US @ V_val_unscaled.T
            recon_phys = (recon_z * sigma_val.unsqueeze(0)
                          + mu_val.unsqueeze(0)).clamp(min=0.0, max=50.0)
            ref = wind_phys_val[sample_idx]
            diff = ref - recon_phys
            finite = torch.isfinite(diff)
            val_mae = float(diff[finite].abs().mean().item()) if finite.any() else float("inf")

        current_lr = float(optim.param_groups[0]["lr"])
        history.append(EpochRecord(
            epoch=epoch, train_loss=float(loss.item()),
            val_mae_phys=val_mae, lr=current_lr,
        ))

        if epoch == 1 or (cfg.verbose_every and epoch % cfg.verbose_every == 0):
            logger.info(
                "  ep %3d: train=%.4f  val_mae=%.3f  lr=%.6f",
                epoch, float(loss.item()), val_mae, current_lr,
            )

        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                logger.info("early stop at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ----- Final pass: training residuals in physical m/s ---------------
    with torch.no_grad():
        V_final_scaled = model.forward_mlp(X_tr)
        V_final = (V_final_scaled * V_std_t + V_mean_t).cpu().numpy()
    recon_norm = (svd.U * svd.sigma) @ V_final.T                 # (n_times, n_tr)
    recon_norm += svd.temporal_mean[:, np.newaxis]
    recon_phys = (
        recon_norm * svd.qt.stds_[tr_idx][np.newaxis, :]
        + svd.qt.means_[tr_idx][np.newaxis, :]
    ).clip(0.0, 50.0)
    obs_tr = wind_phys_all_np[:, tr_idx].astype(float)
    residuals_per_station = np.nanmean(recon_phys - obs_tr, axis=0)
    training_residuals = np.zeros(n_stations, dtype=float)
    training_residuals[tr_idx] = residuals_per_station

    return TrainResult(
        model=model,
        X_scaler=X_scaler,
        V_scaler=V_scaler,
        training_residuals=training_residuals,
        graph=normalised_graph,
        train_graph=train_graph,
        val_station_mask=val_mask,
        history=history,
        best_val_mae=best_val,
        best_epoch=best_epoch,
    )
