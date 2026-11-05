"""Elevation-aware calibrators that map z-score reconstructions back to m/s.

The QN-GNN pipeline runs entirely in z-score space (per-station
``(mu, sigma)`` from :class:`StationWiseZScoreTransformer`).  At
inference time we need to invert that mapping *on the grid* — i.e.
for every cell interpolate a plausible ``(mu, sigma)`` from the
station climatology and apply ``wind = z * sigma + mu``.

Two calibrators ship:

* :class:`SpatialWindCalibrator` — linear-in-z, with ``(mu, sigma)``
  elevation-interpolated from the station set.  Simple, operationally
  proven.
* :class:`AsymmetricCalibrator` — stretches the positive tail of
  ``z`` before applying ``sigma * z + mu``, so strong-wind events get
  an additional non-linear boost.  Paper default: ``stretch_power=1.15``.

Plus a small :class:`ElevationBiasCorrector` that packages the
"bias learned on calibration years binned by elevation" workflow —
same thing the paper applies as the final step before writing
wind-field NetCDFs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from numpy.typing import NDArray

__all__ = [
    "SpatialWindCalibrator",
    "AsymmetricCalibrator",
    "ElevationBiasCorrector",
]

logger = logging.getLogger(__name__)

#: Elevation bin edges used by :class:`ElevationBiasCorrector`.
_DEFAULT_ELEV_BINS: tuple[float, ...] = (0.0, 300.0, 600.0, 1000.0, 1500.0, 4500.0)


def _station_alts(meta_df: pd.DataFrame) -> NDArray[np.float64]:
    """Return a NaN-filled ``alt`` column, falling back to 500 m where
    missing.  Matches the paper's defensive default."""
    if "alt" in meta_df.columns:
        return np.nan_to_num(meta_df["alt"].to_numpy(dtype=float), nan=500.0)
    return np.full(len(meta_df), 500.0, dtype=float)


# ---------------------------------------------------------------------------
# Spatial calibrator (linear z → m/s)
# ---------------------------------------------------------------------------
class SpatialWindCalibrator:
    """Elevation-aware ``z → m/s`` calibrator.

    Fit from station metadata + the observed wind matrix.  Internally
    builds a monotone ``alt → (mu, sigma)`` interpolator:

    * ``mu(alt)`` = station climatological mean interpolated at ``alt``.
    * ``sigma(alt)`` = station climatological std interpolated at ``alt``.

    Stations with NaN coordinates / stats are filled from the
    rest of the network (same behaviour as the paper's calibrator
    constructor).

    Transforms:

    * :meth:`transform_grid` — ``(T, ny, nx)`` z + ``(ny, nx)`` DEM
      → ``(T, ny, nx)`` physical wind.
    * :meth:`transform_stations` — ``(T, n_stations)`` z + per-station
      altitudes → ``(T, n_stations)`` physical wind (used for the
      bias-correction workflow).

    Both clip the final wind at ``[0, 1.5 * global_max]`` where
    ``global_max`` is the 99.9-th percentile of the fit data, so a
    runaway DNN output can't silently emit 200 m/s cells.
    """

    def __init__(
        self,
        meta_df: pd.DataFrame,
        wind_data: NDArray[np.floating],
        *,
        scale_factor: float = 1.0,
    ) -> None:
        from scipy.interpolate import interp1d

        alts = _station_alts(meta_df)
        with np.errstate(all="ignore"):
            st_mean = np.nanmean(wind_data, axis=0)
            st_std = np.nanstd(wind_data, axis=0)

        if np.isnan(st_mean).all():
            # Degenerate case: no finite data at all.  Fall back to a
            # physically plausible monotone function of elevation.
            st_mean = 1.2 + 0.0009 * alts
            st_std = 0.8 + 0.0003 * alts
        else:
            for arr in (st_mean, st_std):
                m = np.isnan(arr)
                if not m.any():
                    continue
                v = ~m
                order = np.argsort(alts[v])
                arr[m] = np.interp(alts[m], alts[v][order], arr[v][order])

        st_std = np.maximum(st_std, 0.1)
        all_finite = wind_data[np.isfinite(wind_data)]
        self.global_max = float(np.percentile(all_finite, 99.9)) if all_finite.size else 30.0
        if not np.isfinite(self.global_max):
            self.global_max = 30.0

        order = np.argsort(alts)
        self._mean_i = interp1d(
            alts[order], st_mean[order],
            bounds_error=False, fill_value=(st_mean[order][0], st_mean[order][-1]),
        )
        self._std_i = interp1d(
            alts[order], st_std[order],
            bounds_error=False, fill_value=(st_std[order][0], st_std[order][-1]),
        )
        self.scale_factor = float(scale_factor)
        self._clip_cap = self.global_max * 1.5

    def _mu_sigma(
        self,
        elevation: NDArray[np.floating],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        flat = np.asarray(elevation, dtype=float).ravel()
        mu = np.asarray(self._mean_i(flat), dtype=float).reshape(elevation.shape)
        sigma = np.asarray(self._std_i(flat), dtype=float).reshape(elevation.shape)
        return mu, sigma

    def transform_grid(
        self,
        z: NDArray[np.floating],
        elevation: NDArray[np.floating],
        *,
        clip_sigma: float = 7.0,
    ) -> NDArray[np.float32]:
        """``z[t, y, x] → physical_wind[t, y, x]`` with per-cell ``(mu, sigma)``."""
        z = np.asarray(z, dtype=float)
        if z.ndim != 3:
            raise ValueError(f"z must be 3-D (T, ny, nx); got shape {z.shape!r}")
        if elevation.shape != z.shape[1:]:
            raise ValueError(
                f"elevation shape {elevation.shape!r} does not match z[1:] {z.shape[1:]!r}"
            )
        mu, sigma = self._mu_sigma(elevation)
        zc = np.clip(z, -clip_sigma, clip_sigma)
        wind = mu[np.newaxis] + sigma[np.newaxis] * zc
        return np.clip(wind * self.scale_factor, 0.0, self._clip_cap).astype(np.float32)

    def transform_stations(
        self,
        z: NDArray[np.floating],
        station_alts: NDArray[np.floating],
        *,
        clip_sigma: float = 7.0,
    ) -> NDArray[np.float32]:
        """``z[t, s] → physical_wind[t, s]`` using one ``(mu, sigma)`` per station."""
        z = np.asarray(z, dtype=float)
        if z.ndim != 2:
            raise ValueError(f"z must be 2-D (T, n_stations); got shape {z.shape!r}")
        station_alts = np.asarray(station_alts, dtype=float)
        if station_alts.shape != (z.shape[1],):
            raise ValueError(
                f"station_alts shape {station_alts.shape!r} does not match z.shape[1] {z.shape[1]}"
            )
        mu = np.asarray(self._mean_i(station_alts), dtype=float)
        sigma = np.asarray(self._std_i(station_alts), dtype=float)
        zc = np.clip(z, -clip_sigma, clip_sigma)
        wind = mu[np.newaxis, :] + sigma[np.newaxis, :] * zc
        return np.clip(wind * self.scale_factor, 0.0, self._clip_cap).astype(np.float32)


# ---------------------------------------------------------------------------
# Asymmetric calibrator (positive-tail stretch)
# ---------------------------------------------------------------------------
class AsymmetricCalibrator:
    """Like :class:`SpatialWindCalibrator` but with a positive-tail stretch.

    Applies ``z'(t, y, x) = max(z, 0) ** stretch_power + min(z, 0)``
    before the linear ``mu + sigma * z'`` step.  A ``stretch_power > 1``
    inflates strong winds, helping the reconstruction match the paper's
    verification on high-wind events.  Paper default: ``1.15``.
    """

    def __init__(
        self,
        meta_df: pd.DataFrame,
        wind_data: NDArray[np.floating],
        *,
        stretch_power: float = 1.15,
        scale_factor: float = 1.0,
    ) -> None:
        self.stretch_power = float(stretch_power)
        self.scale_factor = float(scale_factor)
        self._linear = SpatialWindCalibrator(meta_df, wind_data, scale_factor=1.0)

    @property
    def global_max(self) -> float:
        return self._linear.global_max

    def transform_grid(
        self,
        z: NDArray[np.floating],
        elevation: NDArray[np.floating],
        *,
        clip_sigma: float = 7.0,
    ) -> NDArray[np.float32]:
        z = np.asarray(z, dtype=float)
        if z.ndim != 3:
            raise ValueError(f"z must be 3-D (T, ny, nx); got shape {z.shape!r}")
        if elevation.shape != z.shape[1:]:
            raise ValueError(
                f"elevation shape {elevation.shape!r} does not match z[1:] {z.shape[1:]!r}"
            )
        mu, sigma = self._linear._mu_sigma(elevation)
        rc = np.clip(z, -clip_sigma, clip_sigma)
        stretched = np.power(np.maximum(rc, 0), self.stretch_power) + np.minimum(rc, 0)
        wind = (mu[np.newaxis] + sigma[np.newaxis] * stretched) * self.scale_factor
        return np.clip(wind, 0.0, self._linear._clip_cap).astype(np.float32)


# ---------------------------------------------------------------------------
# Elevation-stratified bias correction
# ---------------------------------------------------------------------------
@dataclass
class ElevationBiasCorrector:
    """Elevation-binned additive correction learned on calibration-year data.

    Workflow (what the paper does):

    1. Reconstruct physical winds on a set of calibration years.
    2. Compute per-station bias ``mean_t(recon - obs)`` for stations
       with enough valid observations.
    3. Bin stations by elevation; take the mean bias per bin.
    4. Apply :math:`-\\overline{\\mathrm{bias}}_b` to each grid cell based on
       its elevation (linear interpolation between bin midpoints).

    This class packages the fit + apply halves of that workflow.
    """

    bin_edges: tuple[float, ...] = _DEFAULT_ELEV_BINS
    bin_corrections: NDArray[np.float32] | None = None
    min_valid_stations: int = 3
    _bin_mids: NDArray[np.floating] = dc_field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._bin_mids = 0.5 * (
            np.asarray(self.bin_edges[:-1], dtype=float)
            + np.asarray(self.bin_edges[1:], dtype=float)
        )

    def fit(
        self,
        per_station_bias: NDArray[np.floating],
        station_alts: NDArray[np.floating],
    ) -> ElevationBiasCorrector:
        """Compute the per-bin mean bias (ignoring NaN stations)."""
        bias = np.asarray(per_station_bias, dtype=float)
        alts = np.asarray(station_alts, dtype=float)
        if bias.shape != alts.shape:
            raise ValueError(
                f"per_station_bias and station_alts must share shape; "
                f"got {bias.shape!r} and {alts.shape!r}"
            )

        n_bins = len(self.bin_edges) - 1
        corrections = np.zeros(n_bins, dtype=np.float32)
        for b in range(n_bins):
            lo, hi = self.bin_edges[b], self.bin_edges[b + 1]
            m = (alts >= lo) & (alts < hi) & np.isfinite(bias)
            if m.sum() >= self.min_valid_stations:
                # Minus: the correction must cancel the observed bias.
                corrections[b] = -float(np.mean(bias[m]))
        self.bin_corrections = corrections
        return self

    def apply(
        self,
        wind: NDArray[np.floating],
        elevation: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        """Add the elevation-interpolated correction to ``wind`` and clip at 0.

        ``wind`` can be ``(ny, nx)``, ``(T, ny, nx)``, or any shape
        that broadcasts against ``elevation`` — the function flattens
        then reshapes.
        """
        if self.bin_corrections is None:
            raise RuntimeError("fit the corrector before calling apply()")
        from scipy.interpolate import interp1d

        elev = np.asarray(elevation, dtype=float)
        fn = interp1d(
            self._bin_mids, self.bin_corrections,
            bounds_error=False,
            fill_value=(float(self.bin_corrections[0]), float(self.bin_corrections[-1])),
        )
        corr = np.asarray(fn(elev.ravel()), dtype=np.float32).reshape(elev.shape)
        return np.clip(wind + corr[np.newaxis] if wind.ndim == 3 else wind + corr, 0.0, None).astype(
            np.float32
        )
