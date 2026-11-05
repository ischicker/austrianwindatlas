"""Per-station z-score normalisation for the QN-GNN pipeline.

Earlier revisions of the paper's QN-GNN used a non-linear per-station
quantile transformer (hence the "QN" name).  That transformer broke
down at *unseen* grid cells at inference time — the calibrator there
produced a different ``mu(elev)``, ``sigma(elev)`` than the one the
model was trained against, so the inverse mapping was wrong.

The shipped version (v6 of the source code → the canonical paper
method) replaces QN with a strictly linear z-score:

.. math::

    z(t, s) = \\frac{w(t, s) - \\mu_s}{\\sigma_s}
    \\qquad
    w(t, s) = z(t, s) \\cdot \\sigma_s + \\mu_s

The inverse is linear in ``z``, so the reconstruction loss can be
written directly in physical m/s and the per-station ``(mu, sigma)``
at an unseen cell can be interpolated from terrain (the
:mod:`~austrianwindatlas.stage1.qn_gnn` calibrator does this at
inference time).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = ["StationWiseZScoreTransformer"]

logger = logging.getLogger(__name__)


@dataclass
class StationWiseZScoreTransformer:
    """Fit per-station mean / std; transform and inverse-transform.

    Attributes
    ----------
    means_
        ``(n_stations,)`` — per-station climatological mean after fit.
    stds_
        ``(n_stations,)`` — per-station climatological std after fit;
        clamped to ``>= min_std`` to avoid division by zero on constant
        stations.
    station_ids
        Ids of the stations fit was called with, in column order.  Lets
        callers verify alignment at transform time.
    """

    means_: NDArray[np.floating] | None = None
    stds_: NDArray[np.floating] | None = None
    station_ids: NDArray | None = None
    min_std: float = 0.01

    def fit(
        self,
        wind_data: NDArray[np.floating],
        station_ids: NDArray | None = None,
    ) -> StationWiseZScoreTransformer:
        """Learn ``(mu, sigma)`` per station from a wind matrix.

        Parameters
        ----------
        wind_data
            ``(n_times, n_stations)``, in m/s.  NaNs ignored.
        station_ids
            Optional ``(n_stations,)`` identifier array stored for
            later alignment checks.

        Returns
        -------
        self (for chaining).
        """
        arr = np.asarray(wind_data, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"wind_data must be 2-D (time, station); got shape {arr.shape!r}")

        with np.errstate(all="ignore"):
            means = np.nanmean(arr, axis=0)
            stds = np.nanstd(arr, axis=0)

        # Fill all-NaN columns with overall mean / unit std so transform
        # doesn't propagate NaN.  Rare in practice — only stations that
        # were present in the file but had no finite data at all.
        global_mean = float(np.nanmean(arr)) if np.isfinite(np.nanmean(arr)) else 0.0
        means = np.where(np.isfinite(means), means, global_mean)
        stds = np.where(np.isfinite(stds), stds, 1.0)
        stds = np.where(stds < self.min_std, self.min_std, stds)

        self.means_ = means
        self.stds_ = stds
        self.station_ids = (
            np.asarray(station_ids) if station_ids is not None else None
        )
        return self

    def transform(self, wind_data: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply ``(w - mu) / sigma`` per column."""
        self._check_fit()
        arr = np.asarray(wind_data, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != self.means_.size:
            raise ValueError(
                f"wind_data shape {arr.shape!r} does not match fitted "
                f"n_stations {self.means_.size}"
            )
        return (arr - self.means_[np.newaxis, :]) / self.stds_[np.newaxis, :]

    def inverse_transform(
        self,
        z_data: NDArray[np.floating],
        *,
        clip_sigma: float = 7.0,
    ) -> NDArray[np.floating]:
        """Apply ``z * sigma + mu`` per column with optional ``|z|`` clip.

        The clip defends against the DNN emitting huge outliers that
        would produce physically nonsensical winds.  ``7.0`` matches
        the paper's default.
        """
        self._check_fit()
        arr = np.asarray(z_data, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != self.means_.size:
            raise ValueError(
                f"z_data shape {arr.shape!r} does not match fitted "
                f"n_stations {self.means_.size}"
            )
        z_clipped = np.clip(arr, -clip_sigma, clip_sigma)
        return z_clipped * self.stds_[np.newaxis, :] + self.means_[np.newaxis, :]

    def _check_fit(self) -> None:
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("transformer has not been fit yet")
