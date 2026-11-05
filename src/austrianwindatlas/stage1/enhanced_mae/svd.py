"""Temporal-centering SVD for the Enhanced-MAE / Enhanced-WMSE pipeline.

Pure NumPy.  The key difference from the rPCA path in
:mod:`austrianwindatlas.core.rpca` is the **centering direction**:

* rPCA centres per-station (subtracts the per-station climatology),
  so ``V`` captures *temporal covariance* patterns.
* Enhanced-MAE centres per-timestep (subtracts the mean across
  stations at each moment), so ``V`` captures *spatial deviation*
  patterns — i.e. "how station :math:`s` differs from the
  area-average at any given moment".  This IS terrain-driven
  (exposed stations always above the field mean, sheltered ones
  always below), so a DNN taking terrain covariates as input and
  predicting ``V`` as output is well-posed.

The reconstruction formula is correspondingly::

    wind(t, s) = temporal_mean(t) + sum_k U(t, k) * sigma_k * V(s, k)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = [
    "TemporalSVD",
    "compute_temporal_svd",
    "fill_nans_with_station_mean",
    "verify_reconstruction",
]


@dataclass
class TemporalSVD:
    """Output of :func:`compute_temporal_svd`.

    Attributes
    ----------
    U
        ``(n_times, n_eofs)`` — left singular vectors of the
        temporally-centred data.
    sigma
        ``(n_eofs,)`` — singular values (not squared).
    V
        ``(n_stations, n_eofs)`` — right singular vectors, i.e. the
        station-side spatial patterns that the DNN learns to predict.
    temporal_mean
        ``(n_times,)`` — mean across stations at each timestep.
    n_times, n_stations, n_eofs
        Convenience dimensions.
    """

    U: NDArray[np.floating]
    sigma: NDArray[np.floating]
    V: NDArray[np.floating]
    temporal_mean: NDArray[np.floating]
    n_times: int
    n_stations: int
    n_eofs: int

    @property
    def variance_explained(self) -> NDArray[np.floating]:
        """Cumulative fraction of variance captured by the retained EOFs."""
        s2 = self.sigma ** 2
        return s2.cumsum() / (s2.sum() + 1e-30)


def fill_nans_with_station_mean(
    wind_data: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return a copy of ``wind_data`` with NaNs filled column-wise.

    Each column's NaNs are replaced with that column's ``nanmean``.
    Columns that are entirely NaN (or empty after the fill) use the
    overall ``nanmean`` as a fallback.  Shape is preserved; output is
    guaranteed to contain no ``NaN`` or ``Inf``.

    Raises
    ------
    ValueError
        If ``wind_data`` is entirely NaN (no fallback value exists).
    """
    arr = np.asarray(wind_data, dtype=float).copy()
    if arr.ndim != 2:
        raise ValueError(f"wind_data must be 2-D (time, station); got shape {arr.shape!r}")

    station_means = np.nanmean(arr, axis=0)
    global_mean = np.nanmean(arr)
    if not np.isfinite(global_mean):
        raise ValueError("wind_data is entirely NaN; no fill value available")

    station_means = np.where(np.isnan(station_means), global_mean, station_means)

    for j in range(arr.shape[1]):
        mask = np.isnan(arr[:, j])
        if mask.any():
            arr[mask, j] = station_means[j]

    # Belt-and-braces: any residual non-finite (e.g. +/-Inf from the input)
    # gets the global fallback.
    bad = ~np.isfinite(arr)
    if bad.any():
        arr[bad] = global_mean
    return arr


def compute_temporal_svd(
    wind_data: NDArray[np.floating],
    n_eofs: int = 80,
) -> TemporalSVD:
    """Fill NaNs column-wise, temporally centre, SVD, and truncate.

    Parameters
    ----------
    wind_data
        ``(n_times, n_stations)``, in m/s.  May contain NaN.
    n_eofs
        Maximum number of EOFs to retain; effective rank is
        ``min(n_eofs, min(n_times, n_stations))``.

    Returns
    -------
    TemporalSVD
    """
    if n_eofs < 1:
        raise ValueError("n_eofs must be >= 1")

    filled = fill_nans_with_station_mean(wind_data)

    temporal_mean = filled.mean(axis=1)                 # (n_times,)
    Z = filled - temporal_mean[:, np.newaxis]           # (n_times, n_stations)

    U_full, sigma_full, Vt_full = np.linalg.svd(Z, full_matrices=False)
    V_full = Vt_full.T                                  # (n_stations, n_eofs_max)

    k = int(min(n_eofs, sigma_full.size))
    return TemporalSVD(
        U=U_full[:, :k],
        sigma=sigma_full[:k],
        V=V_full[:, :k],
        temporal_mean=temporal_mean,
        n_times=int(filled.shape[0]),
        n_stations=int(filled.shape[1]),
        n_eofs=k,
    )


def verify_reconstruction(
    svd: TemporalSVD,
    wind_data: NDArray[np.floating],
    *,
    n_sample: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Compute RMSE / Pearson-r between the SVD reconstruction and the
    NaN-filled input, on a random sample of timesteps.

    Parameters
    ----------
    svd
        Result of :func:`compute_temporal_svd` on ``wind_data``.
    wind_data
        The same array that produced ``svd`` (may still contain NaN;
        this function fills them the same way as :func:`compute_temporal_svd`
        before scoring).
    n_sample
        Draw this many timesteps without replacement for scoring (the
        effective sample size is ``min(n_sample, svd.n_times)``).
    rng
        Optional :class:`numpy.random.Generator` for deterministic
        sampling in tests.

    Returns
    -------
    dict
        ``{"rmse": float, "r": float}`` over the sampled slice.
    """
    rng = rng if rng is not None else np.random.default_rng()
    n = min(int(n_sample), svd.n_times)
    idx = rng.choice(svd.n_times, size=n, replace=False)

    recon = (
        svd.temporal_mean[idx, np.newaxis]
        + (svd.U[idx] * svd.sigma) @ svd.V.T
    )
    orig = fill_nans_with_station_mean(wind_data)[idx]

    rmse = float(np.sqrt(np.mean((recon - orig) ** 2)))
    r = float(np.corrcoef(recon.ravel(), orig.ravel())[0, 1])
    return {"rmse": rmse, "r": r}
