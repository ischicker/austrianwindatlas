"""SVD of z-score-normalised wind data for the QN-GNN pipeline.

Distinction from :mod:`austrianwindatlas.stage1.enhanced_mae.svd`:

* Enhanced-MAE: *temporal* centering on the raw m/s matrix.
* QN-GNN: *z-score* normalisation per station first, *then* temporal
  centering on the z-score matrix.

The z-score layer makes station-wise variance uniform so the SVD is
not dominated by windy high-elevation stations (where ``sigma`` is
large).  V now encodes dimensionless *shape* patterns that are
easier for a terrain-aware DNN to predict.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .zscore import StationWiseZScoreTransformer

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = ["ZScoreSVD", "compute_zscore_svd"]

logger = logging.getLogger(__name__)


@dataclass
class ZScoreSVD:
    """Output of :func:`compute_zscore_svd`.

    Attributes
    ----------
    U, sigma, V
        Shapes ``(n_times, n_eofs)``, ``(n_eofs,)``, ``(n_stations, n_eofs)``.
    temporal_mean
        ``(n_times,)`` — mean across stations of the z-score matrix.
    qt
        The fitted :class:`StationWiseZScoreTransformer`.  Carried
        along so callers can transform fresh data with the same
        ``(mu, sigma)`` that trained the SVD.
    """

    U: NDArray[np.floating]
    sigma: NDArray[np.floating]
    V: NDArray[np.floating]
    temporal_mean: NDArray[np.floating]
    qt: StationWiseZScoreTransformer
    n_times: int
    n_stations: int
    n_eofs: int

    @property
    def variance_explained(self) -> NDArray[np.floating]:
        s2 = self.sigma ** 2
        return s2.cumsum() / (s2.sum() + 1e-30)


def compute_zscore_svd(
    wind_data: NDArray[np.floating],
    station_ids: NDArray | None = None,
    *,
    n_eofs: int = 30,
) -> ZScoreSVD:
    """Fit a z-score transformer, apply it, temporally centre, SVD, truncate.

    NaN handling is deliberately different from the Enhanced-MAE
    path: ``ZScoreSVD`` fills post-normalisation NaNs with ``0``
    (which, after z-scoring, equals the station climatology).  This
    reproduces the paper's behaviour bit-for-bit.

    Parameters
    ----------
    wind_data
        ``(n_times, n_stations)``, in m/s, NaN-tolerant.
    station_ids
        Optional ids passed through to the z-score transformer for
        later alignment checks.
    n_eofs
        Max EOFs to retain; effectively capped at ``min(sigma.size, n_eofs)``.

    Returns
    -------
    ZScoreSVD
    """
    if n_eofs < 1:
        raise ValueError("n_eofs must be >= 1")

    arr = np.asarray(wind_data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"wind_data must be 2-D; got shape {arr.shape!r}")

    qt = StationWiseZScoreTransformer().fit(arr, station_ids)
    wind_norm = qt.transform(arr)
    # Paper fills post-normalisation NaN with 0 — i.e. "this station
    # behaves at its climatology at this timestep".  Documented choice.
    wind_norm = np.where(np.isfinite(wind_norm), wind_norm, 0.0)

    temporal_mean = wind_norm.mean(axis=1)
    Z = wind_norm - temporal_mean[:, np.newaxis]

    U_full, sigma_full, Vt_full = np.linalg.svd(Z, full_matrices=False)
    k = int(min(n_eofs, sigma_full.size))
    return ZScoreSVD(
        U=U_full[:, :k],
        sigma=sigma_full[:k],
        V=Vt_full[:k, :].T,
        temporal_mean=temporal_mean,
        qt=qt,
        n_times=int(arr.shape[0]),
        n_stations=int(arr.shape[1]),
        n_eofs=k,
    )
