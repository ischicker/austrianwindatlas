"""Data plumbing for the Temporal CNN: supervision targets, dataset,
station-failure augmentation.

Three deliverables:

* :func:`sample_availability` — simulate a realistic station-availability
  mask: independent 8 % per-station sensor failures plus a 4 %
  correlated icing event that drops ~30 % of stations.  Used inside
  :class:`UResidualDataset` at training time.
* :func:`compute_U_true` — project physical wind observations onto
  the trained z-score / temporal-SVD basis to obtain the "ground truth"
  U(t) that the CNN ultimately corrects towards.  Returns ``U * sigma``
  (scaled so targets are O(1) instead of O(sigma_k)).
* :class:`UResidualDataset` — sliding-window torch dataset built on
  :func:`numpy.lib.stride_tricks.sliding_window_view` for zero-copy
  access.  Serves ``(obs_window, availability, U_residual_target)``
  triples; applies per-sample station-failure augmentation at
  ``__getitem__`` time so each epoch sees different masking.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..zscore import StationWiseZScoreTransformer

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = [
    "sample_availability",
    "compute_U_true",
    "UResidualDataset",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability / failure simulation
# ---------------------------------------------------------------------------
def sample_availability(
    n_stations: int,
    *,
    p_fail: float = 0.08,
    p_ice_event: float = 0.04,
    p_ice_fail: float = 0.30,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float32]:
    """Return a ``(n_stations,)`` availability mask of ``1.0`` / ``0.0``.

    Layer 1 — independent sensor faults.  Each station has
    ``p_fail`` probability of being missing in the returned mask.

    Layer 2 — correlated icing event.  With probability ``p_ice_event``
    an *event* fires that knocks out ``p_ice_fail`` of stations on top
    of the layer-1 failures.

    Parameters
    ----------
    rng
        Deterministic sampling source (useful in tests).  Defaults to
        :func:`numpy.random.default_rng`.
    """
    rng = rng if rng is not None else np.random.default_rng()
    avail = (rng.random(n_stations) > p_fail).astype(np.float32)
    if rng.random() < p_ice_event:
        ice = (rng.random(n_stations) > p_ice_fail).astype(np.float32)
        avail = avail * ice
    return avail


# ---------------------------------------------------------------------------
# U_true projection (supervision signal)
# ---------------------------------------------------------------------------
def compute_U_true(
    wind_obs: NDArray[np.floating],
    qt: StationWiseZScoreTransformer | None,
    temporal_mean: NDArray[np.floating],
    sigma: NDArray[np.floating],
    V_stations: NDArray[np.floating],
    *,
    n_eofs: int | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Project held-out wind observations onto the trained EOF basis.

    Pipeline::

        wind_norm      = qt.transform(wind_obs)               (per-station z-score)
        temporal_mean  = mean over stations at each timestep
        Z              = wind_norm - temporal_mean
        U_true         = Z @ V_stations / sigma               (shape (n_times, n_eofs))
        US_true        = U_true * sigma                       (same shape; O(1) scale)

    Returning ``US_true`` rather than ``U_true`` matches the paper's
    choice: it makes the CNN's targets O(1), avoiding the degenerate
    "tiny gradients" regime the raw ``U * sigma_last`` values would
    give.

    Parameters
    ----------
    wind_obs
        ``(n_times, n_stations)`` physical wind in m/s.  NaN-tolerant.
    qt
        Fitted :class:`StationWiseZScoreTransformer` from the training
        SVD.  If :data:`None`, a global z-score is applied instead
        (fallback matches the paper's behaviour when the transformer
        is unavailable at inference time).
    temporal_mean
        Kept for API parity with the paper code; **not** used
        directly — the function recomputes the per-timestep mean from
        ``wind_norm`` because that is the consistent choice when
        ``wind_obs`` is a held-out year.  Pass :data:`None` or any
        ``(n_times,)`` array.
    sigma
        ``(n_eofs_max,)`` singular values from the training SVD.
    V_stations
        ``(n_stations, n_eofs_max)`` right singular vectors.
    n_eofs
        Truncation; defaults to ``sigma.size``.

    Returns
    -------
    (US_true, temporal_mean_hat)
        ``US_true`` is ``(n_times, n_eofs)``, dtype ``float32``;
        ``temporal_mean_hat`` is ``(n_times,)``, dtype ``float32``.
    """
    del temporal_mean  # intentionally unused; see docstring

    arr = np.asarray(wind_obs, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    V_stations = np.asarray(V_stations, dtype=float)
    k = int(n_eofs if n_eofs is not None else sigma.size)
    if V_stations.shape[1] < k:
        raise ValueError(
            f"V_stations has {V_stations.shape[1]} EOFs; requested {k}"
        )

    if qt is None or qt.means_ is None:
        # Fallback: global z-score — matches the paper's safeguard when
        # the persisted qt isn't available.
        mu = float(np.nanmean(arr))
        sd = float(np.nanstd(arr))
        if sd < 1e-6:
            sd = 1.0
        wind_norm = np.where(np.isfinite(arr), (arr - mu) / sd, 0.0)
    else:
        wind_norm = qt.transform(arr)
        wind_norm = np.where(np.isfinite(wind_norm), wind_norm, 0.0)

    temporal_mean_hat = wind_norm.mean(axis=1)
    Z = wind_norm - temporal_mean_hat[:, np.newaxis]
    sigma_k = sigma[:k]
    U_true = (Z @ V_stations[:, :k]) / np.maximum(sigma_k, 1e-10)
    US_true = (U_true * sigma_k).astype(np.float32)
    return US_true, temporal_mean_hat.astype(np.float32)


# ---------------------------------------------------------------------------
# Sliding-window dataset
# ---------------------------------------------------------------------------
class UResidualDataset:
    """Sliding-window dataset over ``(wind_phys, U_residual)``.

    Pre-computes a :func:`numpy.lib.stride_tricks.sliding_window_view`
    over the input wind matrix so that ``__getitem__`` does one small
    copy instead of a Python loop.  Station-failure augmentation is
    applied per sample at ``__getitem__`` time (stochastic), so each
    training epoch sees different masking patterns — important for
    robustness against real sensor dropout.

    The *target* array (``U_residual``) must align with the **end** of
    each window: index ``i`` of the dataset returns
    ``(wind_phys[i : i + window], U_residual[i + window - 1])``.

    Call :meth:`get_torch_item` or iterate directly to get
    ``(obs_tensor, avail_tensor, target_tensor)`` triples.
    """

    def __init__(
        self,
        wind_phys: NDArray[np.floating],
        U_residual: NDArray[np.floating],
        *,
        window: int = 48,
        holdout_mask: NDArray[np.bool_] | None = None,
        augment: bool = True,
        p_fail: float = 0.08,
        p_ice_event: float = 0.04,
        p_ice_fail: float = 0.30,
        rng: np.random.Generator | None = None,
    ) -> None:
        from numpy.lib.stride_tricks import sliding_window_view

        wind = np.asarray(wind_phys, dtype=np.float32)
        if wind.ndim != 2:
            raise ValueError(f"wind_phys must be 2-D; got shape {wind.shape!r}")
        if U_residual.ndim != 2:
            raise ValueError(
                f"U_residual must be 2-D (n_times, n_eofs); got shape {U_residual.shape!r}"
            )
        if U_residual.shape[0] != wind.shape[0]:
            raise ValueError(
                f"wind_phys and U_residual must share n_times; "
                f"got {wind.shape[0]} vs {U_residual.shape[0]}"
            )
        if window < 1 or window > wind.shape[0]:
            raise ValueError(
                f"window must be in [1, n_times={wind.shape[0]}]; got {window}"
            )

        self.U_resid = np.asarray(U_residual, dtype=np.float32)
        self.window = int(window)
        self.augment = bool(augment)
        self.p_fail = float(p_fail)
        self.p_ice_event = float(p_ice_event)
        self.p_ice_fail = float(p_ice_fail)
        self.n_stations = int(wind.shape[1])
        self._rng = rng if rng is not None else np.random.default_rng()

        self.holdout = (
            holdout_mask.astype(bool)
            if holdout_mask is not None
            else np.zeros(self.n_stations, dtype=bool)
        )
        if self.holdout.shape != (self.n_stations,):
            raise ValueError(
                f"holdout_mask must have shape ({self.n_stations},); "
                f"got {self.holdout.shape!r}"
            )

        avail_base = np.isfinite(wind).astype(np.float32)
        wind_clean = np.where(np.isfinite(wind), wind, 0.0)

        if self.holdout.any():
            avail_base[:, self.holdout] = 0.0
            wind_clean[:, self.holdout] = 0.0

        wind_view = sliding_window_view(wind_clean, self.window, axis=0)
        avail_view = sliding_window_view(avail_base, self.window, axis=0)
        # sliding_window_view result is (n_valid, n_sta, W) → transpose.
        self.wind_wins = wind_view.transpose(0, 2, 1)
        self.avail_wins = avail_view.transpose(0, 2, 1)

        # Target aligns with window END.
        self.U_resid_aligned = self.U_resid[self.window - 1:]
        if len(self.wind_wins) != len(self.U_resid_aligned):
            raise AssertionError(
                "internal mismatch: windows vs targets "
                f"({len(self.wind_wins)} vs {len(self.U_resid_aligned)})"
            )

    def __len__(self) -> int:
        return int(len(self.wind_wins))

    def __getitem__(self, idx: int):
        obs_raw = self.wind_wins[idx].copy()
        avail = self.avail_wins[idx].copy()

        if self.augment:
            fa = (self._rng.random(self.n_stations) > self.p_fail).astype(np.float32)
            if self._rng.random() < self.p_ice_event:
                ice = (self._rng.random(self.n_stations) > self.p_ice_fail).astype(np.float32)
                fa = fa * ice
            avail = avail * fa[np.newaxis, :]
            obs_raw = obs_raw * fa[np.newaxis, :]

        import torch
        return (
            torch.from_numpy(obs_raw),
            torch.from_numpy(avail),
            torch.from_numpy(self.U_resid_aligned[idx]),
        )
