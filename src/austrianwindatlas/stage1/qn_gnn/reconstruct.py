"""End-to-end reconstruction for the QN-GNN+TCNN pipeline.

Given a trained GNN (:class:`TerrainGNN`), its SVD basis
(:class:`ZScoreSVD`), the per-grid V cube from
:func:`predict_grid`, an optional TemporalCNN refinement, and an
elevation DEM, this module produces physical-m/s wind fields
timestep by timestep.

Two deliverables:

* :func:`compute_cnn_U` — sliding-window CNN rollout over a wind
  observation matrix, returning the per-timestep ``ΔU`` correction
  (shape ``(n_times, n_eofs)``).
* :func:`reconstruct_month` — single-month reconstruction that
  combines ``V_grid``, ``U_gnn``, optional ``ΔU``, calibrator and
  (optionally) a background blend into an :class:`xarray.Dataset`
  ready for :func:`austrianwindatlas.io.windfield.write_wind_field_netcdf`.

The CLI entry point (``main()`` in the paper's script) is deliberately
**not** ported — it's argparse plumbing, not a reusable library
surface.  Call :func:`reconstruct_month` from a short driver script
instead.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Union

import numpy as np

from ...io.windfield import INCA_GRID, GridSpec, build_wind_field_dataset
from .calibrator import AsymmetricCalibrator, SpatialWindCalibrator

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import torch.nn as nn
    import xarray as xr
    from numpy.typing import NDArray

__all__ = [
    "CalibratorLike",
    "CNNMode",
    "compute_cnn_U",
    "reconstruct_month",
]

logger = logging.getLogger(__name__)

#: Narrower type alias for the two calibrators that expose ``transform_grid``.
#: Use ``typing.Union`` (not ``X | Y``) because this alias is evaluated at
#: module-import time; ``X | Y`` on arbitrary classes only became a runtime
#: construct in Python 3.10, and some of our dev hosts still run tests on 3.9.
CalibratorLike = Union[SpatialWindCalibrator, AsymmetricCalibrator]  # noqa: UP007

#: How the CNN prediction combines with the GNN's ``U``.
CNNMode = Literal["additive", "replace"]


# ---------------------------------------------------------------------------
# CNN rollout
# ---------------------------------------------------------------------------
def compute_cnn_U(
    cnn_model: nn.Module,
    wind_phys: NDArray[np.floating],
    sigma: NDArray[np.floating],
    *,
    window: int,
    batch_size: int = 1024,
    device: str = "cpu",
) -> NDArray[np.float32]:
    """Run the TemporalCNN over a station wind matrix.

    Pads the first ``window - 1`` rows with zeros so timestep 0 still
    gets a full window, then evaluates the model in batches.  The
    model returns ``ΔU * sigma`` (scaled targets, as in training); we
    divide by ``sigma`` to recover ``ΔU`` for downstream reconstruction.

    Parameters
    ----------
    cnn_model
        Trained :class:`austrianwindatlas.stage1.qn_gnn.temporal_cnn.TemporalCNN`.
    wind_phys
        ``(n_times, n_stations)`` physical wind.  NaN handled: masked
        out of the obs AND the availability channel (same convention
        as training).
    sigma
        ``(n_eofs,)`` singular values from the training SVD.  The CNN
        was trained on ``ΔU * sigma`` targets, so we divide here to
        bring ``ΔU`` back to the "raw" scale that the reconstruction
        formula expects.
    window
        Sliding-window length — must match the CNN's training window
        (the model's ``forward`` checks this and raises otherwise).
    batch_size
        Number of windows per forward pass.
    device
        Torch device string.

    Returns
    -------
    ``(n_times, n_eofs)`` float32 ``ΔU`` array.
    """
    import torch

    wind_phys = np.asarray(wind_phys, dtype=np.float32)
    if wind_phys.ndim != 2:
        raise ValueError(f"wind_phys must be 2-D; got shape {wind_phys.shape!r}")

    sigma = np.asarray(sigma, dtype=np.float32)
    if sigma.ndim != 1:
        raise ValueError(f"sigma must be 1-D; got shape {sigma.shape!r}")
    n_eofs = sigma.size
    n_times, n_stations = wind_phys.shape

    pad = np.zeros((window - 1, n_stations), dtype=np.float32)
    wp = np.vstack([pad, wind_phys])
    delta_US = np.zeros((n_times, n_eofs), dtype=np.float32)

    cnn_model.eval()
    with torch.no_grad():
        for start in range(0, n_times, batch_size):
            end = min(start + batch_size, n_times)
            windows = np.stack([wp[t : t + window] for t in range(start, end)], axis=0)
            avail = np.isfinite(windows).astype(np.float32)
            obs = np.where(np.isfinite(windows), windows, 0.0)
            pred = cnn_model(
                torch.from_numpy(obs).to(device),
                torch.from_numpy(avail).to(device),
            ).cpu().numpy()
            delta_US[start:end] = pred

    delta_US = np.where(np.isfinite(delta_US), delta_US, 0.0)
    return (delta_US / np.maximum(sigma, 1e-10)).astype(np.float32)


# ---------------------------------------------------------------------------
# Single-month reconstruction
# ---------------------------------------------------------------------------
def _combine_U(
    U_gnn: NDArray[np.floating],
    delta_U: NDArray[np.floating] | None,
    mode: CNNMode,
) -> NDArray[np.floating]:
    if delta_U is None:
        return np.asarray(U_gnn, dtype=np.float64)
    if mode == "additive":
        return np.asarray(U_gnn, dtype=np.float64) + np.asarray(delta_U, dtype=np.float64)
    if mode == "replace":
        return np.asarray(delta_U, dtype=np.float64)
    raise ValueError(f"unknown CNN mode {mode!r}; expected 'additive' or 'replace'")


def reconstruct_month(
    V_grid: NDArray[np.floating],
    sigma: NDArray[np.floating],
    U_gnn: NDArray[np.floating],
    temporal_mean: NDArray[np.floating],
    timestamps: pd.DatetimeIndex,
    *,
    elevation_grid: NDArray[np.floating],
    calibrator: CalibratorLike,
    year: int,
    month: int,
    grid: GridSpec = INCA_GRID,
    delta_U: NDArray[np.floating] | None = None,
    cnn_mode: CNNMode = "additive",
    background: NDArray[np.floating] | None = None,
    background_weight: float = 0.0,
    background_source: str = "none",
    max_wind: float = 50.0,
    method_name: str = "qn_gnn_cnn",
    clip_sigma: float = 7.0,
) -> xr.Dataset | None:
    """Reconstruct a single month's wind field as an :class:`xr.Dataset`.

    Formula::

        recon_z(t, y, x) = temporal_mean(t) + (U(t) * sigma) @ V_grid^T
        U(t)             = U_gnn(t)                     (CNN disabled)
                          | U_gnn(t) + delta_U(t)       (cnn_mode="additive")
                          | delta_U(t)                  (cnn_mode="replace")
        recon_phys       = calibrator.transform_grid(recon_z, elevation_grid)
        recon_phys       = (1 - w) * recon_phys + w * background        (optional)
        recon_phys       = clip(recon_phys, 0, max_wind)

    Parameters
    ----------
    V_grid
        ``(n_eofs, ny, nx)`` from :func:`austrianwindatlas.stage1.qn_gnn.predict.predict_grid`.
    sigma
        ``(n_eofs,)`` singular values.
    U_gnn, temporal_mean
        ``(n_times, n_eofs)`` and ``(n_times,)`` — the GNN's time
        coefficients and the z-score temporal-mean series.
    timestamps
        :class:`pandas.DatetimeIndex` aligned with the first axis of
        ``U_gnn``.
    elevation_grid
        ``(ny, nx)`` DEM in metres, used by the calibrator.
    calibrator
        :class:`SpatialWindCalibrator` or :class:`AsymmetricCalibrator`.
    year, month
        Target month to slice.
    grid
        Projected :class:`GridSpec`; defaults to INCA.
    delta_U
        Optional ``(n_times, n_eofs)`` CNN correction; combined with
        ``U_gnn`` per ``cnn_mode``.
    cnn_mode
        ``"additive"`` (default) or ``"replace"``.
    background
        Optional ``(n_times, ny, nx)`` ERA5 / CERRA regridded field —
        apply :func:`austrianwindatlas.io.background.regrid_to_inca`
        upstream to get it onto the target grid.
    background_weight
        Blend factor in ``[0, 1]``.  0 → pure DL, 1 → pure background.
    background_source
        Tag written into the output NetCDF attrs
        (``"era5"`` / ``"cerra"`` / ``"blended"``).
    max_wind
        Upper clip.
    method_name
        Written into the output dataset's ``method`` attribute.
    clip_sigma
        Forwarded to the calibrator; clamps |z| before the linear
        inverse z-score step.

    Returns
    -------
    :class:`xr.Dataset` with ``wind_speed(time, y, x)``, or
    :data:`None` if the requested month has zero timesteps.
    """
    import pandas as pd

    ts = pd.DatetimeIndex(timestamps)
    mask = (ts.year == year) & (ts.month == month)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    month_ts = ts[idx]

    U = _combine_U(U_gnn, delta_U, cnn_mode)[idx]            # (B, n_eofs)
    tm = np.asarray(temporal_mean, dtype=np.float64)[idx]    # (B,)

    V_grid = np.asarray(V_grid, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n_eofs, ny, nx = V_grid.shape
    if (ny, nx) != (grid.nrows, grid.ncols):
        raise ValueError(
            f"V_grid (ny, nx)=({ny}, {nx}) does not match grid ({grid.nrows}, {grid.ncols})"
        )
    if sigma.size != n_eofs:
        raise ValueError(f"sigma has {sigma.size} values but V_grid has {n_eofs} EOFs")
    if U.shape[1] != n_eofs:
        raise ValueError(f"U has {U.shape[1]} EOFs but V_grid has {n_eofs}")
    if elevation_grid.shape != (ny, nx):
        raise ValueError(
            f"elevation_grid shape {elevation_grid.shape!r} does not match grid"
        )

    V_flat = V_grid.reshape(n_eofs, -1)                      # (n_eofs, ny*nx)
    recon_z_flat = (U * sigma) @ V_flat                      # (B, ny*nx)
    recon_z_flat += tm[:, np.newaxis]
    recon_z = recon_z_flat.reshape(idx.size, ny, nx)

    wind = calibrator.transform_grid(recon_z, elevation_grid, clip_sigma=clip_sigma)

    actual_bg = "none"
    if background is not None and background_weight > 0.0:
        bg = np.asarray(background, dtype=np.float32)
        if bg.shape[1:] != (ny, nx):
            raise ValueError(
                f"background shape {bg.shape!r} does not match (n_times, ny, nx)"
            )
        n_bg = min(bg.shape[0], wind.shape[0])
        wind[:n_bg] = (1.0 - background_weight) * wind[:n_bg] + background_weight * bg[:n_bg]
        actual_bg = background_source or "blended"

    np.clip(wind, 0.0, max_wind, out=wind)

    return build_wind_field_dataset(
        wind, month_ts, grid=grid,
        method=method_name, year=year, month=month, n_eofs=n_eofs,
        background_field=actual_bg,
        extra_attrs={
            "centering": "zscore_temporal",
            "background_weight": float(background_weight),
            "cnn_mode": cnn_mode if delta_U is not None else "none",
        },
    )
