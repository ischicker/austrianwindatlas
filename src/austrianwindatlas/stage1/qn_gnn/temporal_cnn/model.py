"""1-D Temporal-CNN that predicts a per-timestep ``U``-correction.

Stage-2 of the QN-GNN+TCNN pipeline.  The GNN learns a *static*
terrain→V mapping; its reconstruction has zero temporal skill at
high-wind events because V is constant in time.  This CNN sees a
48 h sliding window of station observations and predicts a
per-timestep ``U``-residual that re-weights the 30 spatial patterns
for the current synoptic state.

Architecture (faithful to ``dl_temporal_cnn_train.py``)::

    input:  (B, W, n_stations+1)       obs (NaN→0) || availability-mask
    station_proj:  Linear(n_stations+1, 64)  applied per timestep
    reshape to (B, 64, W) channels-first
    conv1:  Conv1d(64, 128, kernel=8, pad=4)  +  BN + ReLU + Dropout
    conv2:  Conv1d(128, 128, kernel=4, stride=2, pad=1) + BN + ReLU + Dropout
    conv3:  Conv1d(128, 64,  kernel=4, stride=2, pad=1) + BN + ReLU + Dropout
    AdaptiveAvgPool1d(1) → (B, 64)
    head:   Linear(64, n_eofs)  zero-initialised
    output: (B, n_eofs)                  — ΔU correction

The station projection is what makes this cheap: naive would feed
``n_stations`` ≈ 280 channels to every Conv1d (236 × 128 × kernel
FLOPs per sample); projecting first to 64 cuts the FLOP count ~15×
on CPU.

The head starts zero so that at epoch 0 the model produces ΔU = 0 —
the pipeline gracefully degrades to pure-GNN reconstruction when the
CNN is under-trained.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch
    import torch.nn as nn

__all__ = ["TemporalCNN", "build_temporal_cnn"]


def build_temporal_cnn(
    n_stations: int,
    n_eofs: int = 30,
    *,
    window: int = 48,
    dropout: float = 0.20,
    proj_dim: int = 64,
) -> nn.Module:
    """Instantiate a Temporal CNN with the paper's defaults.

    Parameters
    ----------
    n_stations
        Station count in the training set.  The model's station
        projection is a ``Linear(n_stations + 1, proj_dim)`` so the
        count must match at inference time.
    n_eofs
        Output dimension (number of EOF coefficients to correct).
        Paper default: 30.
    window
        Sliding-window length in hours.  Paper default: 48.
    dropout
        Applied after each ReLU.  Paper default: 0.20.
    proj_dim
        Width of the station-projection head.  Paper default: 64.

    Returns
    -------
    An ``nn.Module`` whose forward signature is
    ``forward(obs, avail) -> delta_U``, with

    * ``obs``:   ``(B, W, n_stations)``, physical wind with NaN → 0
    * ``avail``: ``(B, W, n_stations)``, 1=valid, 0=missing
    * ``delta_U``: ``(B, n_eofs)`` — the Δ to add to the GNN's U.
    """
    import torch
    import torch.nn as nn

    class TemporalCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_stations = n_stations
            self.n_eofs = n_eofs
            self.window = window
            self.proj_dim = proj_dim

            self.station_proj = nn.Sequential(
                nn.Linear(n_stations + 1, proj_dim, bias=False),
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
            )

            self.conv1 = nn.Sequential(
                nn.Conv1d(proj_dim, 128, kernel_size=8, padding=4, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.conv3 = nn.Sequential(
                nn.Conv1d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)

            self.head = nn.Linear(64, n_eofs)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(
            self,
            obs: torch.Tensor,
            avail: torch.Tensor,
        ) -> torch.Tensor:
            B, W, n_sta = obs.shape
            if n_sta != self.n_stations:
                raise ValueError(
                    f"obs has {n_sta} stations but model was built with "
                    f"{self.n_stations}"
                )
            if self.window != W:
                raise ValueError(
                    f"obs window length {W} does not match configured window {self.window}"
                )
            if avail.shape != obs.shape:
                raise ValueError(
                    f"avail shape {tuple(avail.shape)} must equal obs shape {tuple(obs.shape)}"
                )

            # Concat a "global availability" channel (first-station
            # availability is used as an any-station scalar by the paper).
            x = torch.cat([obs * avail, avail[..., :1]], dim=-1)   # (B, W, n_sta+1)

            # Per-timestep projection.
            x = x.reshape(B * W, -1)
            x = self.station_proj(x)                                # (B*W, proj_dim)
            x = x.reshape(B, W, self.proj_dim)

            # Channels-first for Conv1d.
            x = x.permute(0, 2, 1)                                  # (B, proj_dim, W)
            x = self.conv1(x)[..., : self.window]                   # trim causal pad
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.pool(x).squeeze(-1)                            # (B, 64)
            return self.head(x)                                     # (B, n_eofs)

    return TemporalCNN()


class TemporalCNN:  # pragma: no cover — facade only
    """Public constructor name for :func:`build_temporal_cnn`.

    ``TemporalCNN(...)`` is identical to ``build_temporal_cnn(...)``;
    keeps the summary name readable in checkpoints.
    """

    def __new__(
        cls,
        n_stations: int,
        n_eofs: int = 30,
        *,
        window: int = 48,
        dropout: float = 0.20,
        proj_dim: int = 64,
    ):  # type: ignore[override]
        return build_temporal_cnn(
            n_stations, n_eofs,
            window=window, dropout=dropout, proj_dim=proj_dim,
        )
