"""Readers + regridder for ERA5 / CERRA background wind fields.

These are the *reanalysis* products the paper uses as physics-based
backgrounds when blending reconstructions.  They are deliberately in
the ``io`` sub-package (not in ``stage1.qn_gnn.*``) because Stage-2
hub-height extrapolation also consumes them as a reference.

Three deliverables:

* :func:`load_era5_background` — read an ERA5 monthly NetCDF.  Auto-
  detects either ``(u10, v10)`` with speed derived or a pre-computed
  ``wind_speed``, handles both per-month and per-year-combined file
  naming, surfaces 1-D ``latitude``/``longitude`` coordinate axes.
* :func:`load_cerra_background` — same for CERRA.  CERRA's native
  grid is curvilinear so the loader returns 2-D ``lat``/``lon`` when
  that's what the file carries.
* :func:`regrid_to_inca` — barycentric regrid of ``(T, ny, nx)``
  field to the INCA grid.  Regular 1-D source grids use
  :class:`scipy.interpolate.RegularGridInterpolator` (one call per
  timestep); 2-D curvilinear sources build a Delaunay triangulation
  *once* and reuse barycentric weights across all timesteps — orders
  of magnitude faster than per-timestep :func:`scipy.interpolate.griddata`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .windfield import INCA_GRID, GridSpec

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = [
    "BackgroundField",
    "DEFAULT_ERA5_PATTERNS",
    "DEFAULT_CERRA_PATTERNS",
    "load_era5_background",
    "load_cerra_background",
    "regrid_to_inca",
]

logger = logging.getLogger(__name__)

#: Filename patterns tried by :func:`load_era5_background`, in order.
DEFAULT_ERA5_PATTERNS: tuple[str, ...] = (
    "era5_wind_{year:04d}{month:02d}.nc",
    "era5_wind_{year:04d}_combined.nc",
)

#: Filename patterns tried by :func:`load_cerra_background`, in order.
DEFAULT_CERRA_PATTERNS: tuple[str, ...] = (
    "cerra_inca_{year:04d}{month:02d}.nc",
    "cerra_inca_wind_{year:04d}{month:02d}.nc",
)


@dataclass(frozen=True)
class BackgroundField:
    """Return value of :func:`load_era5_background` / :func:`load_cerra_background`.

    Attributes
    ----------
    wind_speed
        ``(n_times, ny_src, nx_src)`` scalar wind speed in m/s.
    times
        :class:`pandas.DatetimeIndex` aligned with the first axis.
    lat, lon
        Coordinate arrays.  1-D for regular grids, 2-D for curvilinear
        (CERRA native).  Always ``float64``.
    source
        Human-readable tag (``"era5"`` / ``"cerra"``).  Carried through
        for the writer to surface in NetCDF attrs.
    """

    wind_speed: NDArray[np.floating]
    times: pd.DatetimeIndex
    lat: NDArray[np.floating]
    lon: NDArray[np.floating]
    source: str


# ---------------------------------------------------------------------------
# Shared loader helpers
# ---------------------------------------------------------------------------
_WIND_VAR_CANDIDATES: tuple[str, ...] = ("wind_speed", "si10", "ws10")


def _resolve_wind_speed(ds) -> NDArray[np.floating] | None:
    """Return a ``(T, H, W)`` wind-speed array, derived from ``(u10, v10)``
    if needed, or ``None`` if the dataset has no wind variable at all.
    """
    for name in _WIND_VAR_CANDIDATES:
        if name in ds:
            return ds[name].values
    if "u10" in ds and "v10" in ds:
        return np.sqrt(ds["u10"].values ** 2 + ds["v10"].values ** 2)
    return None


def _resolve_time(ds) -> pd.DatetimeIndex:
    """Pick either ``valid_time`` (new ERA5 convention) or ``time``."""
    key = "valid_time" if "valid_time" in ds.dims else "time"
    return pd.to_datetime(ds[key].values)


def _resolve_latlon(ds) -> tuple[NDArray | None, NDArray | None]:
    for lat_key in ("latitude", "lat"):
        if lat_key in ds.coords:
            lat = ds[lat_key].values
            break
    else:
        return None, None
    for lon_key in ("longitude", "lon"):
        if lon_key in ds.coords:
            lon = ds[lon_key].values
            break
    else:
        return None, None
    return lat, lon


def _load_background(
    root: str | Path,
    year: int,
    month: int,
    patterns: tuple[str, ...],
    source: str,
) -> BackgroundField:
    """Shared file-probing logic for ERA5 and CERRA.

    Raises :class:`FileNotFoundError` when no candidate exists.
    """
    import xarray as xr

    root = Path(root)
    tried: list[str] = []
    for pat in patterns:
        candidate = root / pat.format(year=year, month=month)
        tried.append(str(candidate))
        if not candidate.exists():
            continue

        with xr.open_dataset(candidate) as ds:
            ws = _resolve_wind_speed(ds)
            if ws is None:
                continue
            times = _resolve_time(ds)
            lat, lon = _resolve_latlon(ds)
            if lat is None or lon is None:
                raise ValueError(
                    f"{candidate.name}: missing latitude/longitude coordinates"
                )

            # Per-year-combined files: slice to the requested month.
            if candidate.name.endswith(f"{year}_combined.nc"):
                mask = times.month == month
                ws = ws[mask]
                times = times[mask]

            logger.info("loaded %s %s: shape=%s times=%s", source, candidate.name, ws.shape, len(times))
            return BackgroundField(
                wind_speed=np.asarray(ws, dtype=np.float32),
                times=times,
                lat=np.asarray(lat, dtype=np.float64),
                lon=np.asarray(lon, dtype=np.float64),
                source=source,
            )

    raise FileNotFoundError(
        f"no {source} file found for {year:04d}-{month:02d} under {root}; tried {tried}"
    )


def load_era5_background(
    era5_dir: str | Path,
    year: int,
    month: int,
    *,
    patterns: tuple[str, ...] = DEFAULT_ERA5_PATTERNS,
) -> BackgroundField:
    """Load an ERA5 monthly wind-speed NetCDF.

    Accepts files named ``era5_wind_YYYYMM.nc`` or
    ``era5_wind_YYYY_combined.nc`` by default; the latter is sliced to
    the requested month.  Wind speed is derived from ``(u10, v10)``
    if no ``wind_speed`` / ``si10`` variable is present.

    Raises :class:`FileNotFoundError` if no candidate matches.
    """
    return _load_background(era5_dir, year, month, patterns, source="era5")


def load_cerra_background(
    cerra_dir: str | Path,
    year: int,
    month: int,
    *,
    patterns: tuple[str, ...] = DEFAULT_CERRA_PATTERNS,
) -> BackgroundField:
    """Load a CERRA monthly wind-speed NetCDF.

    CERRA's native grid is curvilinear — the returned ``lat``/``lon``
    are 2-D in that case.  :func:`regrid_to_inca` handles both
    orientations.
    """
    return _load_background(cerra_dir, year, month, patterns, source="cerra")


# ---------------------------------------------------------------------------
# Regridding
# ---------------------------------------------------------------------------
def _ensure_tgt_latlon(
    grid: GridSpec,
    tgt_lat: NDArray | None,
    tgt_lon: NDArray | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a 2-D ``(lat, lon)`` grid, computing from EPSG if missing."""
    if tgt_lat is not None and tgt_lon is not None:
        return np.asarray(tgt_lat, dtype=np.float64), np.asarray(tgt_lon, dtype=np.float64)

    from .netcdf import compute_latlon_grid

    lat, lon = compute_latlon_grid(grid.x_axis(), grid.y_axis(), grid.crs)
    return lat.astype(np.float64), lon.astype(np.float64)


def regrid_to_inca(
    field: NDArray[np.floating],
    src_lat: NDArray[np.floating],
    src_lon: NDArray[np.floating],
    *,
    grid: GridSpec = INCA_GRID,
    tgt_lat: NDArray | None = None,
    tgt_lon: NDArray | None = None,
) -> NDArray[np.float32]:
    """Regrid a ``(T, H, W)`` field onto the INCA grid.

    Parameters
    ----------
    field
        ``(n_times, ny_src, nx_src)`` scalar field (m/s).
    src_lat, src_lon
        Source grid coordinates.  1-D arrays → regular lat/lon grid,
        handled with :class:`scipy.interpolate.RegularGridInterpolator`.
        2-D arrays → curvilinear grid; a Delaunay triangulation is built
        once and reused with barycentric weights across timesteps.
    grid
        Target projected :class:`austrianwindatlas.io.windfield.GridSpec`.
        Defaults to :data:`~austrianwindatlas.io.windfield.INCA_GRID`.
    tgt_lat, tgt_lon
        Pre-computed 2-D ``(nrows, ncols)`` latitude / longitude of the
        target grid.  If :data:`None`, derived from ``grid.crs``.

    Returns
    -------
    ``(n_times, nrows, ncols)`` ``float32`` regridded field.  Points
    outside the source domain for the curvilinear path fall back to
    nearest-neighbour; for the regular path they get the
    :class:`RegularGridInterpolator` boundary behaviour (NaN →
    :func:`numpy.nan_to_num` 0.0).
    """
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 3:
        raise ValueError(f"field must be 3-D (T, H, W); got shape {field.shape!r}")
    src_lat = np.asarray(src_lat)
    src_lon = np.asarray(src_lon)
    if src_lat.ndim not in (1, 2) or src_lon.ndim not in (1, 2):
        raise ValueError(
            f"src_lat / src_lon must be 1-D or 2-D; got {src_lat.ndim}-D and {src_lon.ndim}-D"
        )

    tgt_lat_2d, tgt_lon_2d = _ensure_tgt_latlon(grid, tgt_lat, tgt_lon)
    ny, nx = tgt_lat_2d.shape
    tgt_pts = np.column_stack([tgt_lat_2d.ravel(), tgt_lon_2d.ravel()])
    n_times = field.shape[0]
    out = np.zeros((n_times, ny, nx), dtype=np.float32)

    if src_lat.ndim == 1 and src_lon.ndim == 1:
        from scipy.interpolate import RegularGridInterpolator

        asc = src_lat[0] < src_lat[-1]
        lat_axis = src_lat if asc else src_lat[::-1]
        for t in range(n_times):
            frame = field[t] if asc else field[t, ::-1, :]
            interp = RegularGridInterpolator(
                (lat_axis, src_lon), frame,
                method="linear", bounds_error=False, fill_value=np.nan,
            )
            out[t] = np.nan_to_num(interp(tgt_pts), nan=0.0).reshape(ny, nx).astype(np.float32)
        return out

    from scipy.spatial import Delaunay, cKDTree

    src_pts = np.column_stack([src_lat.ravel(), src_lon.ravel()])
    tri = Delaunay(src_pts)
    simplex = tri.find_simplex(tgt_pts)
    safe = np.clip(simplex, 0, None)
    vertices = tri.simplices[safe]
    bary_T = tri.transform[safe, :2]
    delta = tgt_pts - tri.transform[safe, 2]
    weights = np.einsum("njk,nk->nj", bary_T, delta)
    weights = np.column_stack([weights, 1.0 - weights.sum(axis=1)])
    outside = simplex < 0

    nn_idx = None
    if outside.any():
        tree = cKDTree(src_pts)
        _, nn_idx = tree.query(tgt_pts[outside])

    for t in range(n_times):
        vals = field[t].ravel()
        interior = np.einsum("nj,nj->n", vals[vertices], weights).astype(np.float32)
        if outside.any() and nn_idx is not None:
            interior[outside] = vals[nn_idx]
        out[t] = interior.reshape(ny, nx)
    return out
