"""Temporal-CNN training loop + checkpoint save / load.

Keeps the paper's hyperparameters (AdamW + weight_decay=1e-4,
CosineAnnealingWarmRestarts with T_0=40 / T_mult=2, patience=30,
grad-norm clip at 1.0, 48 h window, 30 EOFs).  Train/val split is
*temporal* (last ``val_frac`` of timesteps), so the signal is
generalisation to unseen periods — complementary to the GNN's
*spatial* holdout.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .data import UResidualDataset
from .model import build_temporal_cnn

if TYPE_CHECKING:  # pragma: no cover
    import torch.nn as nn
    from numpy.typing import NDArray

__all__ = [
    "TrainConfig",
    "EpochRecord",
    "TrainResult",
    "train_temporal_cnn",
    "save_cnn",
    "load_cnn",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Hyperparameters for :func:`train_temporal_cnn`.  Paper defaults."""

    n_eofs: int = 30
    window: int = 48
    epochs: int = 200
    batch_size: int = 512
    lr: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 30
    dropout: float = 0.20
    proj_dim: int = 64
    grad_clip: float = 1.0
    cosine_t0: int = 40
    cosine_tmult: int = 2
    cosine_eta_min: float = 1e-6
    val_frac: float = 0.15

    # Augmentation knobs — passed through to UResidualDataset.
    p_fail: float = 0.08
    p_ice_event: float = 0.04
    p_ice_fail: float = 0.30

    device: str = "cpu"
    seed: int | None = None
    verbose_every: int = 10


@dataclass
class EpochRecord:
    """Per-epoch diagnostic row."""
    epoch: int
    train_loss: float
    val_loss: float
    lr: float


@dataclass
class TrainResult:
    """Return value of :func:`train_temporal_cnn`."""
    model: nn.Module
    n_stations: int
    history: list[EpochRecord] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def train_temporal_cnn(
    wind_phys: NDArray[np.floating],
    U_residual: NDArray[np.floating],
    *,
    holdout_mask: NDArray[np.bool_] | None = None,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train a :class:`TemporalCNN` on ``(wind_window → U_residual)`` pairs.

    Parameters
    ----------
    wind_phys
        ``(n_times, n_stations)`` observed wind in m/s.  NaN-tolerant.
    U_residual
        ``(n_times, n_eofs)`` supervision target — typically
        ``US_true(t) - US_GNN(t)`` (both scaled by sigma so values are
        O(1)); see :func:`austrianwindatlas.stage1.qn_gnn.temporal_cnn.data.compute_U_true`.
    holdout_mask
        Optional ``(n_stations,)`` bool mask marking the GNN's
        spatial-validation stations.  Held-out stations are
        permanently zeroed in both observations and availability so
        the CNN can't "cheat" by reading off a validation station.
    config
        :class:`TrainConfig`; falls back to defaults when ``None``.

    Returns
    -------
    TrainResult with the best-val-loss checkpoint loaded and in eval mode.
    """
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    cfg = config if config is not None else TrainConfig()
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    wind_phys = np.asarray(wind_phys, dtype=np.float32)
    U_residual = np.asarray(U_residual, dtype=np.float32)
    if wind_phys.ndim != 2:
        raise ValueError(f"wind_phys must be 2-D; got shape {wind_phys.shape!r}")
    if U_residual.shape[0] != wind_phys.shape[0]:
        raise ValueError(
            f"wind_phys and U_residual must share n_times; "
            f"got {wind_phys.shape[0]} vs {U_residual.shape[0]}"
        )

    n_times, n_stations = wind_phys.shape
    if U_residual.shape[1] != cfg.n_eofs:
        raise ValueError(
            f"U_residual.shape[1] ({U_residual.shape[1]}) != cfg.n_eofs ({cfg.n_eofs})"
        )
    device = torch.device(cfg.device)

    logger.info(
        "train_temporal_cnn: n_times=%d, n_stations=%d, n_eofs=%d, window=%dh",
        n_times, n_stations, cfg.n_eofs, cfg.window,
    )

    # ----- Temporal train/val split -------------------------------------
    split = int(n_times * (1.0 - cfg.val_frac))
    split = max(split, cfg.window + 1)
    if split >= n_times:
        raise ValueError(
            f"val_frac={cfg.val_frac} leaves no training data; raise n_times or lower val_frac"
        )

    rng_train = np.random.default_rng(cfg.seed)
    ds_train = UResidualDataset(
        wind_phys[:split], U_residual[:split],
        window=cfg.window, holdout_mask=holdout_mask, augment=True,
        p_fail=cfg.p_fail, p_ice_event=cfg.p_ice_event, p_ice_fail=cfg.p_ice_fail,
        rng=rng_train,
    )
    val_start = max(0, split - cfg.window)
    ds_val = UResidualDataset(
        wind_phys[val_start:], U_residual[val_start:],
        window=cfg.window, holdout_mask=holdout_mask, augment=False,
    )

    loader_train = torch.utils.data.DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    loader_val = torch.utils.data.DataLoader(
        ds_val, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # ----- Model + optimiser --------------------------------------------
    model = build_temporal_cnn(
        n_stations, cfg.n_eofs,
        window=cfg.window, dropout=cfg.dropout, proj_dim=cfg.proj_dim,
    ).to(device)
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingWarmRestarts(
        optim, T_0=cfg.cosine_t0, T_mult=cfg.cosine_tmult, eta_min=cfg.cosine_eta_min,
    )

    best_val = float("inf")
    best_state: dict | None = None
    best_epoch = -1
    wait = 0
    history: list[EpochRecord] = []

    for epoch in range(1, cfg.epochs + 1):
        # ---- Train ----------------------------------------------------
        model.train()
        train_losses: list[float] = []
        for obs_b, avail_b, tgt_b in loader_train:
            obs_b = obs_b.to(device)
            avail_b = avail_b.to(device)
            tgt_b = tgt_b.to(device)

            optim.zero_grad()
            pred = model(obs_b, avail_b)
            loss = F.mse_loss(pred, tgt_b)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optim.step()
            train_losses.append(float(loss.item()))

        sched.step(epoch - 1)

        # ---- Validation ----------------------------------------------
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for obs_b, avail_b, tgt_b in loader_val:
                obs_b = obs_b.to(device)
                avail_b = avail_b.to(device)
                tgt_b = tgt_b.to(device)
                pred = model(obs_b, avail_b)
                val_losses.append(float(F.mse_loss(pred, tgt_b).item()))

        tr = float(np.mean(train_losses)) if train_losses else float("nan")
        vl = float(np.mean(val_losses)) if val_losses else float("nan")
        current_lr = float(optim.param_groups[0]["lr"])
        history.append(EpochRecord(
            epoch=epoch, train_loss=tr, val_loss=vl, lr=current_lr,
        ))

        if epoch == 1 or (cfg.verbose_every and epoch % cfg.verbose_every == 0):
            logger.info(
                "  ep %3d: train=%.5f  val=%.5f  lr=%.6f",
                epoch, tr, vl, current_lr,
            )

        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                logger.info("early stop at epoch %d (best_val=%.5f)", epoch, best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return TrainResult(
        model=model,
        n_stations=n_stations,
        history=history,
        best_val_loss=best_val,
        best_epoch=best_epoch,
    )


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def _default_meta(
    model: nn.Module,
    n_stations: int,
    config: TrainConfig,
) -> dict:
    """Minimum metadata needed to rebuild the model for inference."""
    return {
        "n_stations": int(n_stations),
        "n_eofs": int(config.n_eofs),
        "window": int(config.window),
        "proj_dim": int(config.proj_dim),
    }


def save_cnn(
    result: TrainResult,
    output_dir: str | Path,
    *,
    config: TrainConfig | None = None,
    extra_meta: dict | None = None,
) -> Path:
    """Save the trained CNN to ``output_dir``.

    Writes:

    * ``temporal_cnn.pt`` — ``state_dict`` of the best-val-loss model.
    * ``temporal_cnn_meta.json`` — the minimum metadata needed to
      rebuild the model at inference time (``n_stations``, ``n_eofs``,
      ``window``, ``proj_dim``, plus optional ``extra_meta``).

    Returns the output directory as a :class:`Path`.
    """
    import torch

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = config if config is not None else TrainConfig()
    meta = _default_meta(result.model, result.n_stations, cfg)
    meta["best_val_loss"] = float(result.best_val_loss)
    meta["best_epoch"] = int(result.best_epoch)
    if extra_meta:
        meta.update(extra_meta)

    torch.save(result.model.state_dict(), out / "temporal_cnn.pt")
    (out / "temporal_cnn_meta.json").write_text(json.dumps(meta, indent=2))
    return out


def load_cnn(
    cnn_dir: str | Path,
    *,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """Load the checkpoint that :func:`save_cnn` produced.

    Returns ``(model, meta)`` with ``model`` already in ``eval`` mode
    on ``device``.
    """
    import torch

    root = Path(cnn_dir)
    meta_path = root / "temporal_cnn_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing metadata: {meta_path}")
    meta = json.loads(meta_path.read_text())
    required = {"n_stations", "n_eofs", "window", "proj_dim"}
    missing = required - set(meta)
    if missing:
        raise ValueError(f"metadata missing keys: {sorted(missing)}")

    model = build_temporal_cnn(
        int(meta["n_stations"]), int(meta["n_eofs"]),
        window=int(meta["window"]), dropout=0.0, proj_dim=int(meta["proj_dim"]),
    )
    state = torch.load(root / "temporal_cnn.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    return model, meta
