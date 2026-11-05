"""Grid prediction + per-month reconstruction for Enhanced-MAE.

Three deliverables:

* :func:`predict_grid` runs the trained SpatialPatternDNN over the
  full target grid to produce ``V_grid`` of shape
  ``(n_eofs, ny, nx)``.
* :func:`project_onto_eofs` projects a held-out year's wind matrix
  onto the station-side EOF basis via :math:`U = Z V / \\sigma`
  (same formula the paper uses for the re-projection step at
  inference).
* :func:`reconstruct_month` turns ``V_grid`` + ``(U, sigma, temporal_mean)``
  + timestamps into a monthly wind-field :class:`xarray.Dataset`
  ready for :func:`austrianwindatlas.io.windfield.write_wind_field_netcdf`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ...io.windfield import INCA_GRID, GridSpec, build_wind_field_dataset
from .features import FeatureSpec, sample_features_on_grid
from .svd import fill_nans_with_station_mean

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import torch.nn as nn
    import xarray as xr
    from numpy.typing import NDArray
    from sklearn.preprocessing import StandardScaler

__all__ = [
    "predict_grid",
    "project_onto_eofs",
    "reconstruct_month",
]

logger = logging.getLogger(__name__)


def predict_grid(
    model: nn.Module,
    X_scaler: StandardScaler,
    V_scaler: StandardScaler,
    feature_spec: FeatureSpec,
    *,
    device: str = "cpu",
    altitude_source: str | None = None,
) -> NDArray[np.float32]:
    """Run the trained SpatialPatternDNN on the full target grid.

    Parameters
    ----------
    model
        Trained :class:`torch.nn.Module` (eval mode).
    X_scaler, V_scaler
        Scalers fit during training (see :class:`~austrianwindatlas.stage1.enhanced_mae.model.TrainResult`).
    feature_spec
        Describes the feature stack to sample on the grid.  Must match
        the spec used at training time; otherwise the DNN input
        dimension doesn't line up and PyTorch will raise.
    device
        Torch device string.
    altitude_source
        Optional DEM GeoTIFF override for the ``alt_norm`` channel —
        see :func:`austrianwindatlas.stage1.enhanced_mae.features.sample_features_on_grid`.

    Returns
    -------
    ndarray of shape ``(n_eofs, ny, nx)``, dtype ``float32``.
    """
    import torch

    X_grid = sample_features_on_grid(feature_spec, altitude_source=altitude_source)
    if X_grid.shape[1] != X_scaler.n_features_in_:
        raise ValueError(
            f"feature matrix width {X_grid.shape[1]} does not match X_scaler.n_features_in_ "
            f"{X_scaler.n_features_in_}; make sure feature_spec matches training."
        )

    X_scaled = X_scaler.transform(X_grid)

    model.eval()
    with torch.no_grad():
        V_pred_scaled = model(torch.as_tensor(X_scaled, dtype=torch.float32, device=device))
        V_pred_raw = V_pred_scaled.cpu().numpy()

    V_pred_raw = V_scaler.inverse_transform(V_pred_raw)
    grid = feature_spec.grid
    n_eofs = V_pred_raw.shape[1]
    V_grid = V_pred_raw.reshape(grid.nrows, grid.ncols, n_eofs).transpose(2, 0, 1)
    return V_grid.astype(np.float32)


def project_onto_eofs(
    wind_data: NDArray[np.floating],
    V_stations: NDArray[np.floating],
    sigma: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Project a wind matrix onto a trained temporal-SVD basis.

    Implements the paper's held-out-year re-projection:

    1. ``wind_filled``  ← NaN-fill ``wind_data`` column-wise
       (:func:`austrianwindatlas.stage1.enhanced_mae.svd.fill_nans_with_station_mean`).
    2. ``temporal_mean(t) = mean_s wind_filled(t, s)``.
    3. ``Z(t, s) = wind_filled - temporal_mean(t)``.
    4. ``U(t, k) = (Z @ V)(t, k) / sigma_k``.

    Parameters
    ----------
    wind_data
        ``(n_times, n_stations)``, may contain NaN.
    V_stations
        ``(n_stations, n_eofs)``; station-side loadings from training.
    sigma
        ``(n_eofs,)``; singular values from training.

    Returns
    -------
    (U, temporal_mean)
        ``U`` is ``(n_times, n_eofs)`` and ``temporal_mean`` is
        ``(n_times,)``.
    """
    V_stations = np.asarray(V_stations, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if V_stations.ndim != 2:
        raise ValueError(f"V_stations must be 2-D; got shape {V_stations.shape!r}")
    if sigma.ndim != 1 or sigma.size != V_stations.shape[1]:
        raise ValueError(
            f"sigma must have shape (n_eofs,) matching V_stations.shape[1]; "
            f"got sigma.shape={sigma.shape!r}, V_stations.shape={V_stations.shape!r}"
        )

    filled = fill_nans_with_station_mean(wind_data)
    temporal_mean = filled.mean(axis=1)
    Z = filled - temporal_mean[:, np.newaxis]
    # (n_times, n_stations) @ (n_stations, n_eofs) / (n_eofs,) → (n_times, n_eofs)
    U = (Z @ V_stations) / sigma
    return U, temporal_mean


def reconstruct_month(
    V_grid: NDArray[np.floating],
    sigma: NDArray[np.floating],
    U: NDArray[np.floating],
    temporal_mean: NDArray[np.floating],
    timestamps: pd.DatetimeIndex,
    *,
    year: int,
    month: int,
    grid: GridSpec = INCA_GRID,
    background: NDArray[np.floating] | None = None,
    background_weight: float = 0.0,
    max_wind: float = 50.0,
    method_name: str = "dl_enhanced",
) -> xr.Dataset | None:
    """Build one monthly wind-field :class:`xr.Dataset` from the EOF basis.

    Formula::

        wind(t, y, x) = temporal_mean(t) + (U(t) * sigma) @ V_grid^T

    with optional background blending::

        wind = (1 - w) * wind + w * background       (per-timestep)

    Parameters
    ----------
    V_grid
        ``(n_eofs, ny, nx)``; output of :func:`predict_grid`.
    sigma
        ``(n_eofs,)`` singular values.
    U, temporal_mean
        Outputs of :func:`project_onto_eofs`, shape ``(n_times, n_eofs)``
        and ``(n_times,)``.  Must cover the month requested.
    timestamps
        DatetimeIndex aligned with the first axis of ``U``.
    year, month
        Target month to slice.
    grid
        Target grid spec — defaults to :data:`~austrianwindatlas.io.windfield.INCA_GRID`.
    background
        Optional ``(n_times, ny, nx)`` background field (ERA5 / CERRA
        regridded to ``grid``).  When provided and
        ``background_weight > 0``, blended in per timestep.
    background_weight
        ``[0, 1]``.
    max_wind
        Upper clip for the reconstructed field (m/s).
    method_name
        Written into the output dataset's ``method`` attribute.

    Returns
    -------
    :class:`xr.Dataset` with ``wind_speed`` variable on
    ``(time, y, x)``, or :data:`None` if the month has zero timesteps.
    """
    import pandas as pd

    ts = pd.DatetimeIndex(timestamps)
    mask = (ts.year == year) & (ts.month == month)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None

    month_ts = ts[idx]
    U_m = np.asarray(U, dtype=float)[idx]
    tmean_m = np.asarray(temporal_mean, dtype=float)[idx]

    V_grid = np.asarray(V_grid, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n_eofs, ny, nx = V_grid.shape
    if (ny, nx) != (grid.nrows, grid.ncols):
        raise ValueError(
            f"V_grid shape (ny, nx)=({ny}, {nx}) does not match grid "
            f"({grid.nrows}, {grid.ncols})"
        )
    if sigma.size != n_eofs:
        raise ValueError(f"sigma has {sigma.size} values but V_grid has {n_eofs} EOFs")
    if U_m.shape[1] != n_eofs:
        raise ValueError(f"U has {U_m.shape[1]} EOFs but V_grid has {n_eofs}")

    V_flat = V_grid.reshape(n_eofs, -1)        # (n_eofs, ny*nx)
    recon_flat = (U_m * sigma) @ V_flat        # (n_times, ny*nx)
    recon_flat += tmean_m[:, np.newaxis]
    wind = recon_flat.reshape(idx.size, ny, nx).astype(np.float32)

    if background is not None and background_weight > 0.0:
        bg = np.asarray(background, dtype=np.float32)
        if bg.shape[1:] != (ny, nx):
            raise ValueError(
                f"background shape {bg.shape!r} does not match (n_times, ny, nx)"
            )
        n_bg = min(bg.shape[0], wind.shape[0])
        wind[:n_bg] = (1.0 - background_weight) * wind[:n_bg] + background_weight * bg[:n_bg]

    np.clip(wind, 0.0, max_wind, out=wind)

    return build_wind_field_dataset(
        wind, month_ts, grid=grid,
        method=method_name, year=year, month=month, n_eofs=n_eofs,
        background_field="none" if background is None else "blended",
        extra_attrs={
            "centering": "temporal",
            "background_weight": float(background_weight),
        },
    )
