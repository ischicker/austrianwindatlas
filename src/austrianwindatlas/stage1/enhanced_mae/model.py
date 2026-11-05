"""PyTorch model + training loop for the Enhanced-MAE / Enhanced-WMSE DNN.

Trains :class:`SpatialPatternDNN` to predict the station-side EOF
loadings :math:`V` from terrain features — the "spatial-patterns"
head of the paper's Enhanced-Amato architecture.

Usage
-----

.. code-block:: python

    from austrianwindatlas.stage1.enhanced_mae.svd import compute_temporal_svd
    from austrianwindatlas.stage1.enhanced_mae.losses import compute_station_weights
    from austrianwindatlas.stage1.enhanced_mae.model import TrainConfig, train_dnn

    svd = compute_temporal_svd(wind_data, n_eofs=80)
    weights = compute_station_weights(wind_data, method="mean_wind")
    cfg = TrainConfig(loss_type="weighted_mae", weight_method="mean_wind", epochs=200)
    result = train_dnn(X_features, svd, wind_data, cfg)

``result.model``, ``result.X_scaler`` and ``result.V_scaler`` are then
fed to :func:`austrianwindatlas.stage1.enhanced_mae.predict.predict_grid`
to get the gridded ``V`` cube at inference time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .losses import LossFunction, LossKind, WeightMethod, compute_station_weights
from .svd import TemporalSVD, fill_nans_with_station_mean

if TYPE_CHECKING:  # pragma: no cover
    import torch
    import torch.nn as nn
    from numpy.typing import NDArray
    from sklearn.preprocessing import StandardScaler

__all__ = [
    "SpatialPatternDNN",
    "TrainConfig",
    "TrainResult",
    "EpochRecord",
    "train_dnn",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _spatial_pattern_dnn(
    input_dim: int,
    output_dim: int,
    *,
    hidden_layers: int = 6,
    hidden_units: int = 128,
    dropout: float = 0.1,
) -> nn.Module:
    """Build the Enhanced-Amato SpatialPatternDNN (private factory).

    Architecture: ``hidden_layers`` × [Linear → BatchNorm1d → ReLU
    (+ optional Dropout)] blocks, followed by a Linear head of size
    ``output_dim``.  Dropout is skipped on the last hidden block, as
    in the paper.

    The factory is exposed through the public :class:`SpatialPatternDNN`
    class below (thin subclass so the class name shows up in model
    summaries).
    """
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev = input_dim
    for i in range(hidden_layers):
        layers.append(nn.Linear(prev, hidden_units))
        layers.append(nn.BatchNorm1d(hidden_units))
        layers.append(nn.ReLU())
        if i < hidden_layers - 1:
            layers.append(nn.Dropout(dropout))
        prev = hidden_units
    hidden = nn.Sequential(*layers)
    head = nn.Linear(hidden_units, output_dim)

    class SpatialPatternDNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = hidden
            self.output = head

        def forward(self, x):  # noqa: ANN001 — tensor-in tensor-out
            return self.output(self.hidden(x))

    return SpatialPatternDNN()


class SpatialPatternDNN:  # pragma: no cover — thin facade only
    """Public constructor name; see :func:`_spatial_pattern_dnn` for the
    actual nn.Module factory.

    Instantiating this class returns the underlying ``nn.Module`` so
    checkpoints load with ``model.load_state_dict(...)`` as usual.
    """

    def __new__(
        cls,
        input_dim: int,
        output_dim: int,
        *,
        hidden_layers: int = 6,
        hidden_units: int = 128,
        dropout: float = 0.1,
    ):  # type: ignore[override]
        return _spatial_pattern_dnn(
            input_dim, output_dim,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            dropout=dropout,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Training hyperparameters for :func:`train_dnn`.

    Defaults are the ones used in the Enhanced-MAE variant that ships
    the paper's best-scoring non-background runs.  Change ``loss_type``
    to ``"mae"`` for Enhanced-WMSE-style training.
    """

    epochs: int = 300
    lr: float = 0.005
    batch_size: int = 32
    patience: int = 40  # early-stop on best `recon_mae`
    hidden_layers: int = 6
    hidden_units: int = 128
    dropout: float = 0.1
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    loss_type: LossKind = "mse"
    alpha: float = 0.7
    gamma: float = 2.0
    weight_method: WeightMethod = "none"
    device: str = "cpu"
    recon_sample_size: int = 1000
    verbose_every: int = 20
    seed: int | None = None


@dataclass
class EpochRecord:
    """Per-epoch training diagnostics (returned in ``TrainResult.history``)."""
    epoch: int
    coef_loss: float
    recon_mae: float
    mean_r: float
    lr: float


@dataclass
class TrainResult:
    """Return value of :func:`train_dnn`.

    Attributes
    ----------
    model
        The trained PyTorch module (in eval mode), already restored to
        the best-recon-mae state.
    X_scaler, V_scaler
        Fitted :class:`sklearn.preprocessing.StandardScaler`s for the
        feature matrix and the target EOF loadings respectively.
    history
        Per-epoch diagnostics.
    best_recon_mae
        The lowest validation reconstruction MAE seen during training.
    best_epoch
        Epoch index (1-based) at which ``best_recon_mae`` was hit.
    """

    model: nn.Module
    X_scaler: StandardScaler
    V_scaler: StandardScaler
    history: list[EpochRecord] = field(default_factory=list)
    best_recon_mae: float = float("inf")
    best_epoch: int = -1


def train_dnn(
    X: NDArray[np.floating],
    svd: TemporalSVD,
    wind_data: NDArray[np.floating],
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train the SpatialPatternDNN to predict ``V`` from ``X``.

    Parameters
    ----------
    X
        ``(n_stations, n_features)`` terrain feature matrix.
    svd
        TemporalSVD from :func:`~austrianwindatlas.stage1.enhanced_mae.svd.compute_temporal_svd`.
        Target is ``svd.V`` (shape ``(n_stations, n_eofs)``).
    wind_data
        The raw ``(n_times, n_stations)`` wind matrix.  Used only to
        (a) compute station weights for ``weighted_*`` losses, and
        (b) evaluate the reconstruction MAE on a sampled slice during
        training (early-stop signal).
    config
        :class:`TrainConfig`; falls back to defaults when None.

    Returns
    -------
    TrainResult
    """
    import torch
    import torch.nn.functional as F
    from sklearn.preprocessing import StandardScaler

    cfg = config if config is not None else TrainConfig()

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_stations, n_features); got shape {X.shape!r}")
    if X.shape[0] != svd.n_stations:
        raise ValueError(
            f"X has {X.shape[0]} rows but svd describes {svd.n_stations} stations"
        )

    device = torch.device(cfg.device)
    n_stations, n_features = X.shape
    n_eofs = svd.n_eofs
    logger.info(
        "train_dnn: stations=%d, features=%d, eofs=%d, loss=%s, weights=%s, device=%s",
        n_stations, n_features, n_eofs, cfg.loss_type, cfg.weight_method, cfg.device,
    )

    # ----- Scalers --------------------------------------------------------
    X_scaler = StandardScaler().fit(X)
    V_scaler = StandardScaler().fit(svd.V)
    X_scaled = X_scaler.transform(X)
    V_scaled = V_scaler.transform(svd.V)

    # ----- Station weights ------------------------------------------------
    station_weights_tensor: torch.Tensor | None = None
    if cfg.weight_method != "none":
        weights_np = compute_station_weights(wind_data, method=cfg.weight_method)
        station_weights_tensor = torch.as_tensor(weights_np, dtype=torch.float32, device=device)

    loss_fn = LossFunction(
        loss_type=cfg.loss_type,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        station_weights=station_weights_tensor,
    )

    # ----- Tensors --------------------------------------------------------
    X_t = torch.as_tensor(X_scaled, dtype=torch.float32, device=device)
    V_t = torch.as_tensor(V_scaled, dtype=torch.float32, device=device)

    # Reconstruction-MAE sample (fixed across epochs so the signal is comparable).
    wind_filled = fill_nans_with_station_mean(wind_data)
    sample_n = min(cfg.recon_sample_size, svd.n_times)
    sample_idx = np.random.default_rng(cfg.seed).choice(svd.n_times, size=sample_n, replace=False)
    U_sample = torch.as_tensor(svd.U[sample_idx], dtype=torch.float32, device=device)
    S_t = torch.as_tensor(svd.sigma, dtype=torch.float32, device=device)
    tmean_sample = torch.as_tensor(svd.temporal_mean[sample_idx], dtype=torch.float32, device=device)
    W_sample = torch.as_tensor(wind_filled[sample_idx], dtype=torch.float32, device=device)

    V_mean_t = torch.as_tensor(V_scaler.mean_, dtype=torch.float32, device=device)
    V_std_t = torch.as_tensor(V_scaler.scale_, dtype=torch.float32, device=device)

    # ----- Model + optimiser ---------------------------------------------
    model = _spatial_pattern_dnn(
        n_features, n_eofs,
        hidden_layers=cfg.hidden_layers,
        hidden_units=cfg.hidden_units,
        dropout=cfg.dropout,
    ).to(device)

    optim = torch.optim.NAdam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=10)

    best_mae = float("inf")
    best_state: dict | None = None
    best_epoch = -1
    wait = 0
    history: list[EpochRecord] = []

    # ----- Training loop --------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n_stations, device=device)
        for i in range(0, n_stations, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            if idx.numel() < 2:
                # Avoid BatchNorm1d exploding on batches with < 2 samples.
                continue
            optim.zero_grad()
            pred = model(X_t[idx])
            loss = loss_fn(pred, V_t[idx], idx)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optim.step()

        model.eval()
        with torch.no_grad():
            V_pred_scaled = model(X_t)
            coef_loss = float(loss_fn(V_pred_scaled, V_t, None).item())

            V_pred_raw = V_pred_scaled * V_std_t + V_mean_t
            V_np = V_pred_raw.cpu().numpy()

            k_for_r = min(10, n_eofs)
            corrs = np.array([
                np.corrcoef(svd.V[:, k], V_np[:, k])[0, 1]
                for k in range(k_for_r)
            ])
            mean_r = float(np.nanmean(corrs))

            recon = tmean_sample.unsqueeze(1) + (U_sample * S_t) @ V_pred_raw.T
            recon = torch.clamp(recon, min=0.0)
            recon_mae = float(F.l1_loss(recon, W_sample).item())

        sched.step(recon_mae)
        current_lr = float(optim.param_groups[0]["lr"])
        history.append(EpochRecord(
            epoch=epoch, coef_loss=coef_loss, recon_mae=recon_mae,
            mean_r=mean_r, lr=current_lr,
        ))

        if epoch == 1 or (cfg.verbose_every and epoch % cfg.verbose_every == 0):
            logger.info(
                "  ep %3d: coef_loss=%.4f  recon_mae=%.3f  r=%.3f  lr=%.6f",
                epoch, coef_loss, recon_mae, mean_r, current_lr,
            )

        if recon_mae < best_mae:
            best_mae = recon_mae
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

    return TrainResult(
        model=model,
        X_scaler=X_scaler,
        V_scaler=V_scaler,
        history=history,
        best_recon_mae=best_mae,
        best_epoch=best_epoch,
    )
