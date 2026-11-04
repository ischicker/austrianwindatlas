"""Reconstruct hourly wind-speed fields from Stage-1 rPCA components.

The decomposition produced by :class:`austrianwindatlas.core.rpca.RobustPCA`
and the per-mode gridded loadings produced by
:mod:`austrianwindatlas.stage1.rpca_eof` (or
:mod:`austrianwindatlas.stage1.gam_bamlss`) combine here into the
hourly 10-m wind field

.. math::

    u(t, x) \\;=\\; \\overline{u}(x) \\;+\\; \\sum_{k=1}^{K} U(t, k) \\,
                   V_{\\mathrm{grid}}(k, x)

where

* :math:`\\overline{u}(x)` is the gridded climatological mean
  (``mean_grid``),
* :math:`V_{\\mathrm{grid}}(k, x) = V_{k}(x)\\,\\sigma_{k}` is the
  rPCA spatial loading of mode *k* scaled by its singular value,
  interpolated onto the target grid, and
* :math:`U(t, k)` is the per-mode temporal coefficient for timestep
  *t*, either lifted from the rPCA training basis (when the target
  year was part of the training window) or re-projected from the
  target-year observations (held-out case).

With an optional reanalysis background :math:`b(t, x)` the paper's
pipeline replaces :math:`\\overline{u}(x)` with :math:`b(t, x)` and
interprets the sum over *k* as an anomaly.  Both modes are exposed
here.

The module is pure numerics: no NetCDF or YAML. The I/O that drives
it (loading rPCA exports, monthly reanalysis regridding, writing
monthly NetCDFs) migrates with the ``io/`` module in a later PR.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_CLIP",
    "impute_station_nans",
    "project_onto_eofs",
    "reconstruct_timestep",
    "reconstruct_series",
]

#: Default (lower, upper) clip applied to the final wind-speed field,
#: in m/s.  These are physical bounds — the atlas does not emit
#: negative wind speeds and clips the top at 50 m/s to catch
#: pathological reconstructions.  Matches the production pipeline's
#: ``MIN_WIND_SPEED`` / ``MAX_WIND_SPEED`` config.
DEFAULT_CLIP: tuple[float, float] = (0.0, 50.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def impute_station_nans(
    observations: NDArray[np.floating],
    spatial_mean: NDArray[np.floating],
    *,
    mode: Literal["mean", "zero"] = "mean",
) -> NDArray[np.floating]:
    """Fill NaN entries in a ``(n_times, n_stations)`` observation
    matrix.

    Parameters
    ----------
    observations : (n_times, n_stations) ndarray
        Per-station hourly observations; may contain NaN.
    spatial_mean : (n_stations,) ndarray
        Per-station climatological mean used as the imputation value
        when ``mode="mean"``.  Must be finite.
    mode : {"mean", "zero"}, default "mean"
        ``"mean"`` fills each column's NaNs with the corresponding
        ``spatial_mean`` entry.  ``"zero"`` fills with 0, which is
        the right choice when ``observations`` is an anomaly
        (observation − background) rather than a raw value.

    Returns
    -------
    (n_times, n_stations) ndarray
        A copy of ``observations`` with NaNs replaced.
    """
    if observations.ndim != 2:
        raise ValueError(
            f"observations must be 2-D (time, stations); got shape {observations.shape!r}"
        )
    n_stations = observations.shape[1]
    if spatial_mean.shape != (n_stations,):
        raise ValueError(
            f"spatial_mean must have length {n_stations}; got {spatial_mean.shape!r}"
        )
    if mode not in {"mean", "zero"}:
        raise ValueError(f"mode must be 'mean' or 'zero'; got {mode!r}")

    out = observations.astype(float, copy=True)
    mask = np.isnan(out)
    if not mask.any():
        return out

    if mode == "zero":
        out[mask] = 0.0
        return out

    if not np.all(np.isfinite(spatial_mean)):
        raise ValueError("spatial_mean must be finite when mode='mean'")
    # Broadcast the per-station mean across rows.
    out = np.where(mask, np.broadcast_to(spatial_mean, out.shape), out)
    return out


# ---------------------------------------------------------------------------
# Projection: observations -> per-mode temporal coefficients
# ---------------------------------------------------------------------------
def project_onto_eofs(
    observations: NDArray[np.floating],
    *,
    spatial_mean: NDArray[np.floating],
    v_sigma: NDArray[np.floating],
    singular_values: NDArray[np.floating],
    k: int | None = None,
    anomaly: bool = False,
) -> NDArray[np.floating]:
    r"""Project held-out-year observations onto the trained EOF basis.

    The rPCA export stores :math:`V\\sigma` (``v_sigma``, shape
    :math:`n_{\\mathrm{stations}} \\times K_\\mathrm{export}`) and the
    singular values :math:`\\sigma` (``singular_values``, shape
    :math:`K_\\mathrm{export}`).  The temporal coefficients for a
    new observation matrix :math:`X(t, s)` are then

    .. math::

        U(t, k) \\;=\\; \\sum_{s} \\bigl(X(t, s) - \\overline{u}(s)\\bigr)
                       \\;\\frac{V\\sigma_{s,k}}{\\sigma_{k}^{2}}

    which is :math:`(X - \\overline{u}) V \\sigma^{-1}` in matrix
    form (since :math:`V\\sigma \\cdot \\sigma^{-2} = V \\sigma^{-1}`).
    This is the "project onto trained EOFs" formula used by the
    paper's leave-one-year-out verification.

    Parameters
    ----------
    observations : (n_times, n_stations) ndarray
        Hourly station observations for the target year.  Must be
        free of NaN; call :func:`impute_station_nans` first if
        necessary.
    spatial_mean : (n_stations,) ndarray
        Per-station climatological mean from the training window.
        Ignored when ``anomaly=True`` (anomalies are already
        centered on the background).
    v_sigma : (n_stations, K_export) ndarray
        :math:`V \\sigma` at the training stations, in the order
        matching ``observations``' column axis and ``spatial_mean``.
    singular_values : (K_export,) ndarray
        rPCA singular values, same order as the columns of ``v_sigma``.
    k : int, optional
        Truncate to the first ``k`` modes before projecting.  This is
        required when the gridded loadings are shorter than the
        export (``K_grid < K_export``) — otherwise :math:`U` would be
        the right length for the wrong :math:`V`.  Defaults to all
        ``K_export`` modes.
    anomaly : bool, default False
        If ``True``, skip the ``- spatial_mean`` step on the premise
        that ``observations`` is already a (observation − background)
        anomaly.

    Returns
    -------
    (n_times, k) ndarray
        :math:`U(t, k)` for every input timestep.
    """
    if observations.ndim != 2:
        raise ValueError(
            f"observations must be 2-D; got shape {observations.shape!r}"
        )
    n_times, n_stations = observations.shape
    if spatial_mean.shape != (n_stations,):
        raise ValueError(
            f"spatial_mean must have length {n_stations}; got {spatial_mean.shape!r}"
        )
    if v_sigma.ndim != 2 or v_sigma.shape[0] != n_stations:
        raise ValueError(
            f"v_sigma must be (n_stations={n_stations}, K); got {v_sigma.shape!r}"
        )
    k_export = v_sigma.shape[1]
    if singular_values.shape != (k_export,):
        raise ValueError(
            f"singular_values must have length {k_export} (matching v_sigma cols); "
            f"got {singular_values.shape!r}"
        )

    if k is None:
        k_use = k_export
    else:
        if not (1 <= k <= k_export):
            raise ValueError(
                f"k must be in [1, {k_export}] (K_export); got {k!r}"
            )
        k_use = k

    if np.any(~np.isfinite(observations)):
        raise ValueError(
            "observations contains NaN or Inf; call impute_station_nans first"
        )

    v_sigma_trunc = v_sigma[:, :k_use]
    sigma_sq = singular_values[:k_use] ** 2
    if np.any(sigma_sq == 0):
        raise ValueError("singular values must be strictly positive; got a zero")

    centered = observations if anomaly else observations - spatial_mean[np.newaxis, :]

    # U = (X_centered @ V_sigma) / sigma^2 — broadcasts over modes.
    return (centered @ v_sigma_trunc) / sigma_sq


# ---------------------------------------------------------------------------
# Reconstruction: temporal coefficients + spatial loadings -> wind grid
# ---------------------------------------------------------------------------
def reconstruct_timestep(
    temporal_coefs: NDArray[np.floating],
    v_grid: NDArray[np.floating],
    *,
    mean_grid: NDArray[np.floating] | None = None,
    background_grid: NDArray[np.floating] | None = None,
    clip: tuple[float, float] | None = DEFAULT_CLIP,
    dtype: np.dtype = np.float32,
) -> NDArray[np.floating]:
    r"""Reconstruct the gridded wind field at a single timestep.

    Computes

    .. math::

        u(x) \\;=\\; a(x) \\;+\\; \\sum_{k=1}^{K} U_{k} \\, V_{\\mathrm{grid}}(k, x)

    where the anchor :math:`a(x)` is ``mean_grid`` (pure-rPCA mode)
    or ``background_grid`` (background-blended mode).  Exactly one
    of the two anchor arguments must be supplied; passing both or
    neither is an error.

    Parameters
    ----------
    temporal_coefs : (K,) ndarray
        :math:`U_{k}` — the per-mode temporal coefficient for this
        timestep.
    v_grid : (K, n_rows, n_cols) ndarray
        :math:`V_{\\mathrm{grid}}(k, y, x)` — the per-mode spatial
        loading interpolated to the target grid.  Must have the
        same leading dimension as ``temporal_coefs``.
    mean_grid : (n_rows, n_cols) ndarray, optional
        Gridded climatological mean :math:`\\overline{u}(x)`.
    background_grid : (n_rows, n_cols) ndarray, optional
        Reanalysis background :math:`b(x)` at this timestep;
        supplied in place of ``mean_grid`` when the atlas is run in
        background-blended mode.
    clip : (lower, upper), optional
        Physical clip in m/s.  Pass ``None`` to disable clipping
        (useful when you want the raw sum for diagnostics).  Default
        is :data:`DEFAULT_CLIP` = ``(0.0, 50.0)``.
    dtype : numpy dtype, default ``np.float32``
        Output precision.  The production atlas writes float32 to
        keep monthly NetCDFs manageable.

    Returns
    -------
    (n_rows, n_cols) ndarray
    """
    anchor = _resolve_anchor(mean_grid, background_grid)
    _validate_temporal_coefs(temporal_coefs, v_grid)

    # L[y, x] = sum_k U[k] * V_grid[k, y, x].
    L = np.einsum("k,kyx->yx", temporal_coefs, v_grid)

    field = anchor + L
    if clip is not None:
        field = np.clip(field, clip[0], clip[1])
    return field.astype(dtype, copy=False)


def reconstruct_series(
    temporal_coefs: NDArray[np.floating],
    v_grid: NDArray[np.floating],
    *,
    mean_grid: NDArray[np.floating] | None = None,
    background_series: NDArray[np.floating] | None = None,
    clip: tuple[float, float] | None = DEFAULT_CLIP,
    dtype: np.dtype = np.float32,
) -> NDArray[np.floating]:
    r"""Reconstruct the gridded wind field for every timestep at once.

    Vectorised counterpart of :func:`reconstruct_timestep`.  Accepts
    either a single ``mean_grid`` (reused for every timestep) or a
    full ``background_series`` with shape ``(n_times, n_rows, n_cols)``.
    Exactly one of the two must be supplied.

    Parameters
    ----------
    temporal_coefs : (n_times, K) ndarray
    v_grid : (K, n_rows, n_cols) ndarray
    mean_grid : (n_rows, n_cols) ndarray, optional
    background_series : (n_times, n_rows, n_cols) ndarray, optional
    clip : (lower, upper), optional
    dtype : numpy dtype, default ``np.float32``

    Returns
    -------
    (n_times, n_rows, n_cols) ndarray
    """
    if temporal_coefs.ndim != 2:
        raise ValueError(
            f"temporal_coefs must be 2-D (time, K); got shape {temporal_coefs.shape!r}"
        )
    n_times, _ = temporal_coefs.shape
    _validate_v_grid_and_coefs(temporal_coefs, v_grid)

    if (mean_grid is None) == (background_series is None):
        raise ValueError(
            "exactly one of mean_grid or background_series must be supplied"
        )

    if background_series is not None:
        if background_series.shape != (n_times, v_grid.shape[1], v_grid.shape[2]):
            raise ValueError(
                "background_series must have shape "
                f"({n_times}, {v_grid.shape[1]}, {v_grid.shape[2]}); "
                f"got {background_series.shape!r}"
            )
    else:
        assert mean_grid is not None  # for the type checker
        if mean_grid.shape != (v_grid.shape[1], v_grid.shape[2]):
            raise ValueError(
                "mean_grid must have shape "
                f"({v_grid.shape[1]}, {v_grid.shape[2]}); got {mean_grid.shape!r}"
            )

    # L[t, y, x] = sum_k U[t, k] * V_grid[k, y, x].
    L = np.einsum("tk,kyx->tyx", temporal_coefs, v_grid)
    anchor = mean_grid if background_series is None else background_series
    field = anchor + L
    if clip is not None:
        field = np.clip(field, clip[0], clip[1])
    return field.astype(dtype, copy=False)


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------
def _resolve_anchor(
    mean_grid: NDArray[np.floating] | None,
    background_grid: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
    if (mean_grid is None) == (background_grid is None):
        raise ValueError(
            "exactly one of mean_grid or background_grid must be supplied"
        )
    anchor = mean_grid if background_grid is None else background_grid
    assert anchor is not None
    if anchor.ndim != 2:
        raise ValueError(
            f"anchor grid must be 2-D (y, x); got shape {anchor.shape!r}"
        )
    return anchor


def _validate_temporal_coefs(
    temporal_coefs: NDArray[np.floating],
    v_grid: NDArray[np.floating],
) -> None:
    if temporal_coefs.ndim != 1:
        raise ValueError(
            f"temporal_coefs must be 1-D for reconstruct_timestep; "
            f"got shape {temporal_coefs.shape!r}"
        )
    _validate_v_grid_and_coefs(temporal_coefs, v_grid)


def _validate_v_grid_and_coefs(
    temporal_coefs: NDArray[np.floating],
    v_grid: NDArray[np.floating],
) -> None:
    if v_grid.ndim != 3:
        raise ValueError(
            f"v_grid must be 3-D (K, y, x); got shape {v_grid.shape!r}"
        )
    k_coefs = temporal_coefs.shape[-1]
    if v_grid.shape[0] != k_coefs:
        raise ValueError(
            f"K mismatch: temporal_coefs last axis is {k_coefs}, "
            f"v_grid first axis is {v_grid.shape[0]}"
        )
