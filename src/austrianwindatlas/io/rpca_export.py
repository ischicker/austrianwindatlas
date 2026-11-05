"""Reader for the CSV-based rPCA export directory layout.

Writing the layout is handled by :class:`austrianwindatlas.stage1.gam_bamlss.RExportInputs`
— which the R scripts consume.  This module is the Python-side reader
for the same layout, so the full
*decompose → (R interpolate) → (Python reconstruct)* pipeline can be
driven without a single R dependency on the reconstruction leg.

Directory layout (as written by
:meth:`austrianwindatlas.stage1.gam_bamlss.RExportInputs.write` and by
the paper's ``rpca_decomposition.py``)::

    year_YYYY/
        eof_coefficients_training.csv  # station_id, eof_1, ..., eof_K
        station_metadata.csv           # station_id, x, y [, lon, lat, alt, w_id]
        spatial_mean_field.csv         # station_id, mean_wind [, x, y]
        singular_values.csv            # (optional) component, singular_value, ...
        temporal_basis.csv             # (optional) time_idx, timestamp, basis_1, ...
        sparse_timesteps.csv           # (optional) time_idx, timestamp, is_sparse

Columns in ``[brackets]`` are present in the production exports but
not required by the R scripts, hence optional for the reader.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = [
    "RPCAExport",
    "read_rpca_export",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RPCAExport:
    """In-memory view of an ``rpca_export/year_YYYY/`` directory.

    ``eof_coefficients`` has shape ``(n_stations, n_eofs)`` and already
    absorbs the singular values (it is :math:`V \\Sigma` in the
    :math:`L = U \\Sigma V^T` decomposition, i.e. what the R scripts
    treat as the station-side loadings).  ``singular_values`` keeps the
    raw :math:`\\sigma_k` for callers that want to re-separate the
    temporal basis.

    All optional fields are :data:`None` when the corresponding file
    was not present (common when the directory was produced by the
    light-weight :class:`austrianwindatlas.stage1.gam_bamlss.RExportInputs`
    writer rather than the paper's full exporter).
    """

    year: int
    station_id: NDArray[np.integer] | NDArray[np.str_]
    station_x: NDArray[np.floating]
    station_y: NDArray[np.floating]
    eof_coefficients: NDArray[np.floating]  # (n_stations, n_eofs)
    spatial_mean: NDArray[np.floating]      # (n_stations,)
    singular_values: NDArray[np.floating] | None = None  # (n_eofs,)
    temporal_basis: NDArray[np.floating] | None = None   # (n_times, n_eofs)
    timestamps: pd.DatetimeIndex | None = None           # (n_times,)
    sparse_flags: NDArray[np.bool_] | None = None        # (n_times,)
    station_lon: NDArray[np.floating] | None = None
    station_lat: NDArray[np.floating] | None = None
    station_alt: NDArray[np.floating] | None = None

    @property
    def n_stations(self) -> int:
        return int(len(self.station_id))

    @property
    def n_eofs(self) -> int:
        return int(self.eof_coefficients.shape[1])

    @property
    def n_times(self) -> int:
        if self.temporal_basis is None:
            return 0
        return int(self.temporal_basis.shape[0])


def _parse_year_from(name: str) -> int:
    """Parse the ``year_<YYYY>`` suffix the R scripts also parse from
    the directory name.
    """
    if "year_" not in name:
        raise ValueError(f"directory name {name!r} is not of the form 'year_YYYY'")
    digits = name.split("year_", 1)[1]
    if not digits[:4].isdigit():
        raise ValueError(f"directory name {name!r} has no 4-digit year after 'year_'")
    return int(digits[:4])


def _read_eof_coefficients(path: Path) -> tuple[NDArray, NDArray[np.floating]]:
    df = pd.read_csv(path)
    eof_cols = [c for c in df.columns if c.lower().startswith("eof_")]
    if not eof_cols:
        raise ValueError(f"{path.name}: no 'eof_*' columns found (columns={list(df.columns)!r})")
    eof_cols.sort(key=lambda c: int(c.split("_", 1)[1]))

    if "station_id" in df.columns:
        station_id = df["station_id"].values
    elif "w_id" in df.columns:
        station_id = df["w_id"].values
    else:
        raise ValueError(
            f"{path.name}: no 'station_id' or 'w_id' column (columns={list(df.columns)!r})"
        )
    return station_id, df[eof_cols].to_numpy(dtype=float)


def _read_spatial_mean(path: Path, station_id: NDArray) -> NDArray[np.floating]:
    df = pd.read_csv(path)
    if "mean_wind" in df.columns:
        mean_col = "mean_wind"
    elif "spatial_mean" in df.columns:
        mean_col = "spatial_mean"
    else:
        # Fall back on the last column (paper code also does this).
        mean_col = df.columns[-1]
        logger.warning(
            "%s: no 'mean_wind' column; falling back on last column %r",
            path.name, mean_col,
        )
    mean_values = df[mean_col].to_numpy(dtype=float)

    # Optional: align to station_id order if an id column is present.
    id_col = "station_id" if "station_id" in df.columns else "w_id" if "w_id" in df.columns else None
    if id_col is not None:
        order = pd.Index(df[id_col].values).get_indexer(station_id)
        if np.any(order == -1):
            raise ValueError(
                f"{path.name}: {int((order == -1).sum())} station id(s) from EOF file "
                "not found in the spatial-mean file"
            )
        mean_values = mean_values[order]
    return mean_values


def _read_station_metadata(
    path: Path,
    station_id: NDArray,
) -> tuple[NDArray[np.floating], NDArray[np.floating],
           NDArray[np.floating] | None, NDArray[np.floating] | None,
           NDArray[np.floating] | None]:
    df = pd.read_csv(path)
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(
            f"{path.name}: expected 'x' and 'y' columns (columns={list(df.columns)!r})"
        )

    id_col = "station_id" if "station_id" in df.columns else "w_id" if "w_id" in df.columns else None
    if id_col is None:
        if len(df) != len(station_id):
            raise ValueError(
                f"{path.name}: no id column and row count {len(df)} "
                f"!= n_stations {len(station_id)}"
            )
        order = np.arange(len(df))
    else:
        order = pd.Index(df[id_col].values).get_indexer(station_id)
        if np.any(order == -1):
            raise ValueError(
                f"{path.name}: {int((order == -1).sum())} station id(s) from EOF file "
                "not found in the station-metadata file"
            )

    station_x = df["x"].to_numpy(dtype=float)[order]
    station_y = df["y"].to_numpy(dtype=float)[order]
    lon = df["lon"].to_numpy(dtype=float)[order] if "lon" in df.columns else None
    lat = df["lat"].to_numpy(dtype=float)[order] if "lat" in df.columns else None
    alt = df["alt"].to_numpy(dtype=float)[order] if "alt" in df.columns else None
    return station_x, station_y, lon, lat, alt


def read_rpca_export(rpca_dir: str | Path) -> RPCAExport:
    """Read an ``rpca_export/year_YYYY/`` directory into an :class:`RPCAExport`.

    The three required files
    (``eof_coefficients_training.csv``, ``station_metadata.csv``,
    ``spatial_mean_field.csv``) must be present; the three optional
    files are loaded if they exist.
    """
    root = Path(rpca_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"rpca export directory not found: {root}")

    year = _parse_year_from(root.name)

    eof_path = root / "eof_coefficients_training.csv"
    meta_path = root / "station_metadata.csv"
    mean_path = root / "spatial_mean_field.csv"
    for p in (eof_path, meta_path, mean_path):
        if not p.is_file():
            raise FileNotFoundError(f"required rpca export file missing: {p}")

    station_id, eof_coefficients = _read_eof_coefficients(eof_path)
    station_x, station_y, lon, lat, alt = _read_station_metadata(meta_path, station_id)
    spatial_mean = _read_spatial_mean(mean_path, station_id)

    singular_values: NDArray | None = None
    sv_path = root / "singular_values.csv"
    if sv_path.is_file():
        sv_df = pd.read_csv(sv_path)
        sv_col = "singular_value" if "singular_value" in sv_df.columns else sv_df.columns[-1]
        singular_values = sv_df[sv_col].to_numpy(dtype=float)[: eof_coefficients.shape[1]]

    temporal_basis: NDArray | None = None
    timestamps: pd.DatetimeIndex | None = None
    basis_path = root / "temporal_basis.csv"
    if basis_path.is_file():
        basis_df = pd.read_csv(basis_path)
        basis_cols = [c for c in basis_df.columns if c.startswith("basis_")]
        basis_cols.sort(key=lambda c: int(c.split("_", 1)[1]))
        if basis_cols:
            temporal_basis = basis_df[basis_cols].to_numpy(dtype=float)
        ts_col = next((c for c in ("timestamp", "time", "datetime") if c in basis_df.columns), None)
        if ts_col is not None:
            timestamps = pd.to_datetime(basis_df[ts_col])

    sparse_flags: NDArray | None = None
    sparse_path = root / "sparse_timesteps.csv"
    if sparse_path.is_file():
        sparse_df = pd.read_csv(sparse_path)
        if "is_sparse" in sparse_df.columns:
            sparse_flags = sparse_df["is_sparse"].to_numpy(dtype=bool)

    return RPCAExport(
        year=year,
        station_id=station_id,
        station_x=station_x,
        station_y=station_y,
        eof_coefficients=eof_coefficients,
        spatial_mean=spatial_mean,
        singular_values=singular_values,
        temporal_basis=temporal_basis,
        timestamps=timestamps,
        sparse_flags=sparse_flags,
        station_lon=lon,
        station_lat=lat,
        station_alt=alt,
    )
