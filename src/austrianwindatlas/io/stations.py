"""Readers for TAWES / ``combined_wind_YYYYMM.nc`` station NetCDFs.

The paper's RPCA pipeline consumes monthly station NetCDFs produced
by GeoSphere's internal station-combining job (variable name ``ff``
or ``wind_speed``; station dimension ``w_id``; coord variables
``lon``, ``lat``, ``alt``).  The same format is published through the
open-data portal at
<https://data.hub.geosphere.at/dataset/klima-v2-10min> — hence the
library-level readers below.

The module is intentionally small.  Heavy lifting (resampling to
hourly means, time-zone handling, …) lives in the production CLI and
is out of scope here: this module is only concerned with reading and
quality-flagging the raw files.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr
    from numpy.typing import NDArray

__all__ = [
    "DEFAULT_WIND_VAR_NAMES",
    "DEFAULT_MONTHLY_PATTERNS",
    "StationQCResult",
    "apply_quality_control",
    "extract_station_metadata",
    "read_station_netcdf",
    "load_station_month",
    "load_station_series",
    "resolve_wind_variable",
]

logger = logging.getLogger(__name__)

#: Variable names the readers accept, in order of preference.  The
#: paper's combined-wind files use ``ff``; the open-data klima-v2
#: product uses ``wind_speed``; older local exports use ``ws`` / ``FF``.
DEFAULT_WIND_VAR_NAMES: tuple[str, ...] = ("ff", "wind_speed", "ws", "FF")

#: Filename patterns tried when loading a single month from a directory.
DEFAULT_MONTHLY_PATTERNS: tuple[str, ...] = (
    "combined_wind_{year:04d}{month:02d}.nc",
    "wind_{year:04d}{month:02d}.nc",
    "combined_{year:04d}{month:02d}.nc",
)


def resolve_wind_variable(
    ds: xr.Dataset,
    candidates: tuple[str, ...] = DEFAULT_WIND_VAR_NAMES,
) -> str:
    """Return the name of the wind-speed variable in ``ds``.

    Raises :class:`KeyError` if none of ``candidates`` is present.
    """
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(
        f"none of {candidates!r} are present in the dataset; "
        f"saw variables {sorted(ds.variables)!r}"
    )


def read_station_netcdf(
    path: str | Path,
    *,
    wind_var: str | None = None,
) -> xr.Dataset:
    """Open a single station NetCDF and return the raw :class:`xr.Dataset`.

    The wind-speed variable is renamed to ``wind_speed`` if it was
    found under another alias (``ff`` / ``ws`` / ``FF``) — downstream
    code can rely on that name.
    """
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        name = wind_var or resolve_wind_variable(ds)
    except KeyError:
        ds.close()
        raise
    if name != "wind_speed":
        ds = ds.rename({name: "wind_speed"})
    return ds


def load_station_month(
    station_data_dir: str | Path,
    year: int,
    month: int,
    *,
    patterns: tuple[str, ...] = DEFAULT_MONTHLY_PATTERNS,
    wind_var: str | None = None,
) -> xr.Dataset:
    """Locate the station NetCDF for ``(year, month)`` under ``station_data_dir``
    (using :data:`DEFAULT_MONTHLY_PATTERNS`) and read it.

    Raises :class:`FileNotFoundError` if no candidate matches.
    """
    root = Path(station_data_dir)
    tried = []
    for pattern in patterns:
        candidate = root / pattern.format(year=year, month=month)
        tried.append(str(candidate))
        if candidate.exists():
            return read_station_netcdf(candidate, wind_var=wind_var)
    raise FileNotFoundError(
        f"no station NetCDF found for {year:04d}-{month:02d} under {root}; tried {tried}"
    )


def load_station_series(
    station_data_dir: str | Path,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    *,
    patterns: tuple[str, ...] = DEFAULT_MONTHLY_PATTERNS,
    wind_var: str | None = None,
) -> xr.Dataset:
    """Load every monthly station file in ``[start, end]`` and concatenate on ``time``.

    Months that don't exist on disk are silently skipped (this mirrors
    the paper code which tolerates missing months); a warning is
    emitted so the behaviour is visible in logs.  Raises
    :class:`FileNotFoundError` if *no* month in the range was found.
    """
    import xarray as xr

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError(f"end ({end_ts}) must be >= start ({start_ts})")

    months = pd.period_range(start_ts, end_ts, freq="M")
    datasets: list[xr.Dataset] = []
    for period in months:
        try:
            ds = load_station_month(
                station_data_dir,
                year=period.year,
                month=period.month,
                patterns=patterns,
                wind_var=wind_var,
            )
            datasets.append(ds)
        except FileNotFoundError:
            logger.warning("no station file for %04d-%02d; skipping", period.year, period.month)

    if not datasets:
        raise FileNotFoundError(
            f"no station files found under {station_data_dir} for {start_ts}..{end_ts}"
        )
    combined = xr.concat(datasets, dim="time")
    return combined.sortby("time")


def extract_station_metadata(ds: xr.Dataset, *, id_var: str = "w_id") -> pd.DataFrame:
    """Return a DataFrame with ``w_id``, ``lon``, ``lat``, ``alt`` columns.

    ``alt`` is filled with :data:`numpy.nan` if the dataset lacks it
    (older open-data files omit elevation).
    """
    if id_var not in ds.variables and id_var not in ds.coords:
        raise KeyError(f"dataset has no station-id variable {id_var!r}")
    station_ids = ds[id_var].values
    if station_ids.ndim > 1:
        station_ids = station_ids.flatten()
    n_stations = int(len(station_ids))

    def _one_d(name: str) -> NDArray[np.floating]:
        if name not in ds.variables and name not in ds.coords:
            return np.full(n_stations, np.nan, dtype=float)
        arr = np.asarray(ds[name].values, dtype=float)
        if arr.ndim == 1 and arr.size == n_stations:
            return arr
        if arr.ndim == 2:
            # Some files store (station, time) with a tiny time-wiggle;
            # average along the extra axis.
            if arr.shape[0] == n_stations:
                return np.nanmean(arr, axis=1)
            if arr.shape[1] == n_stations:
                return np.nanmean(arr, axis=0)
        if arr.ndim == 1 and arr.size > n_stations:
            return arr[:n_stations]
        return np.full(n_stations, np.nan, dtype=float)

    return pd.DataFrame({
        "w_id": station_ids,
        "lon": _one_d("lon"),
        "lat": _one_d("lat"),
        "alt": _one_d("alt"),
    })


@dataclass(frozen=True)
class StationQCResult:
    """Return value of :func:`apply_quality_control`.

    Attributes
    ----------
    wind_speed
        The input array with invalid entries replaced by NaN.  Shape
        unchanged.
    n_invalid
        Count of entries flagged in this pass.
    station_completeness
        Per-station fraction of finite entries after QC, shape
        ``(n_stations,)``.
    valid_stations
        Boolean mask (True where ``completeness >= min_completeness``).
    """
    wind_speed: NDArray[np.floating]
    n_invalid: int
    station_completeness: NDArray[np.floating]
    valid_stations: NDArray[np.bool_]


def apply_quality_control(
    wind_speed: NDArray[np.floating],
    *,
    max_speed: float = 50.0,
    min_completeness: float = 0.5,
) -> StationQCResult:
    """Flag physically-implausible wind speeds and derive per-station
    completeness.

    Parameters
    ----------
    wind_speed
        Array of shape ``(n_times, n_stations)`` in m/s.
    max_speed
        Upper bound; anything greater is NaN-masked.  Also masks
        negative values and non-finite inputs.
    min_completeness
        Minimum fraction of finite entries per station for the station
        to be flagged usable.
    """
    arr = np.asarray(wind_speed, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"wind_speed must be 2-D (time, station); got shape {arr.shape!r}")

    invalid = (arr < 0.0) | (arr > max_speed) | ~np.isfinite(arr)
    n_invalid = int(invalid.sum())

    clean = arr.copy()
    clean[invalid] = np.nan

    n_times = clean.shape[0]
    completeness = 1.0 - np.sum(np.isnan(clean), axis=0) / n_times
    valid = completeness >= min_completeness

    return StationQCResult(
        wind_speed=clean,
        n_invalid=n_invalid,
        station_completeness=completeness,
        valid_stations=valid,
    )
