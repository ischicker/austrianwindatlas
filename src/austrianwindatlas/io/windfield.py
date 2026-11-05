"""Writer for reconstructed wind-field NetCDFs.

Produces files that are interoperable with the paper's production
pipeline — ``wind_rpca_<method>_<YYYYMM>.nc`` on the INCA 1 km grid
(EPSG:31287, 701 × 401 cells, hourly) — but is also usable for
arbitrary projected grids by overriding :data:`INCA_GRID`.

The writer deliberately mirrors the encoding used in
``rpca_reconstruct.py``: ``float32`` + zlib level 5 on the
``wind_speed`` variable and standard CF-ish attributes (``method``,
``year``, ``month``, ``n_eofs``, ``crs``, ``created``).  This keeps
files from this library bit-compatible with files from the production
run for verification purposes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr
    from numpy.typing import NDArray

__all__ = [
    "GridSpec",
    "INCA_GRID",
    "build_wind_field_dataset",
    "write_wind_field_netcdf",
]


@dataclass(frozen=True)
class GridSpec:
    """Projected regular-grid specification.

    ``x`` / ``y`` axes are produced with :func:`numpy.linspace` — the
    paper's working rule (see the production ``CLAUDE.md``) uses
    ``linspace`` not ``arange`` to avoid a half-cell drift at the edge.
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    ncols: int
    nrows: int
    crs: str = "EPSG:31287"

    def x_axis(self) -> NDArray[np.float64]:
        return np.linspace(self.xmin, self.xmax, self.ncols)

    def y_axis(self) -> NDArray[np.float64]:
        return np.linspace(self.ymin, self.ymax, self.nrows)

    @property
    def shape(self) -> tuple[int, int]:
        """``(nrows, ncols)`` — the ``(y, x)`` shape of a 2-D field."""
        return (self.nrows, self.ncols)


#: The Austrian INCA / windatlas grid.  1 km cells on MGI Austria
#: Lambert (EPSG:31287), 701 × 401 cells spanning x ∈ [19500, 720500] m
#: and y ∈ [219500, 620500] m.  Matches the configuration shipped with
#: the paper's ``rpca_reconstruct.py``.
INCA_GRID = GridSpec(
    xmin=19500.0,
    xmax=720500.0,
    ymin=219500.0,
    ymax=620500.0,
    ncols=701,
    nrows=401,
    crs="EPSG:31287",
)


def build_wind_field_dataset(
    wind_grid: NDArray[np.floating],
    timestamps: pd.DatetimeIndex | NDArray,
    *,
    grid: GridSpec = INCA_GRID,
    method: str = "rpca_reconstruction",
    year: int | None = None,
    month: int | None = None,
    n_eofs: int | None = None,
    background_field: str = "none",
    extra_attrs: dict[str, Any] | None = None,
    var_name: str = "wind_speed",
) -> xr.Dataset:
    """Wrap a ``(n_times, n_rows, n_cols)`` wind array in an :class:`xr.Dataset`.

    Coordinates:
        ``time`` — ``timestamps`` as-is
        ``y``   — ``grid.y_axis()``
        ``x``   — ``grid.x_axis()``

    Attributes always set:
        ``method``, ``background_field``, ``crs``, ``created``.
        ``year`` / ``month`` / ``n_eofs`` when provided.

    Raises
    ------
    ValueError
        If ``wind_grid`` shape does not match ``(len(timestamps), grid.nrows, grid.ncols)``.
    """
    import xarray as xr

    arr = np.asarray(wind_grid)
    ts = pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps)))
    expected = (len(ts), grid.nrows, grid.ncols)
    if arr.shape != expected:
        raise ValueError(
            f"wind_grid shape {arr.shape!r} does not match (n_times, nrows, ncols)={expected!r}"
        )

    attrs: dict[str, Any] = {
        "method": method,
        "background_field": background_field,
        "crs": grid.crs,
        "created": pd.Timestamp.utcnow().isoformat(),
    }
    if year is not None:
        attrs["year"] = int(year)
    if month is not None:
        attrs["month"] = int(month)
    if n_eofs is not None:
        attrs["n_eofs"] = int(n_eofs)
    if extra_attrs:
        attrs.update(extra_attrs)

    return xr.Dataset(
        {var_name: (["time", "y", "x"], arr)},
        coords={
            "time": ts.values,
            "y": grid.y_axis(),
            "x": grid.x_axis(),
        },
        attrs=attrs,
    )


def write_wind_field_netcdf(
    ds: xr.Dataset,
    path: str | Path,
    *,
    var_name: str = "wind_speed",
    dtype: str = "float32",
    zlib: bool = True,
    complevel: int = 5,
) -> Path:
    """Write ``ds`` to ``path`` with the paper's default encoding.

    ``path``'s parent directory is created on demand.  Returns the
    resolved :class:`Path`.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    encoding = {var_name: {"zlib": zlib, "complevel": complevel, "dtype": dtype}}
    ds.to_netcdf(out, encoding=encoding)
    return out
