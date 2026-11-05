"""Low-level NetCDF helpers.

The one non-trivial helper here is :func:`add_latlon_variables`: it
takes a projected (``x``, ``y``) :class:`xarray.Dataset`, transforms
the 1-D coordinate axes to WGS84, and attaches 2-D ``lat`` / ``lon``
variables plus a CF-style scalar ``crs`` variable.  It is the library
port of the ``EXTRAPOLATION/add_lat_lon_to_nc.py`` utility used to
decorate the paper's reconstructed wind fields before publication.

The helpers are intentionally thin wrappers around :mod:`xarray` and
:mod:`pyproj` — they do no I/O themselves; use :mod:`austrianwindatlas.io.windfield`
to write the result to disk.
"""
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr
    from numpy.typing import NDArray

__all__ = [
    "compute_latlon_grid",
    "ensure_crs_variable",
    "add_latlon_variables",
    "epsg_from_dataset",
]


def epsg_from_dataset(ds: xr.Dataset, fallback: str = "EPSG:31287") -> str:
    """Resolve the source CRS of ``ds`` to a normalised ``EPSG:XXXXX`` string.

    Prefers the global ``crs`` attribute (``":crs = EPSG:31287"`` as
    written by the paper's reconstruction code).  Falls back to the
    caller-supplied default otherwise.
    """
    crs_attr = ds.attrs.get("crs")
    if isinstance(crs_attr, str) and "EPSG" in crs_attr.upper():
        return crs_attr.upper().replace(" ", "")
    return fallback.upper().replace(" ", "")


def compute_latlon_grid(
    x_1d: NDArray[np.floating],
    y_1d: NDArray[np.floating],
    src_epsg: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Project a 1-D ``(x, y)`` grid to WGS84 and return 2-D ``(lat, lon)``.

    Output shapes are both ``(len(y_1d), len(x_1d))``.  ``src_epsg`` is
    anything :func:`pyproj.CRS.from_user_input` accepts (e.g.
    ``"EPSG:31287"``).
    """
    from pyproj import CRS, Transformer

    xx, yy = np.meshgrid(np.asarray(x_1d), np.asarray(y_1d))

    crs_src = CRS.from_user_input(src_epsg)
    crs_dst = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

    lon, lat = transformer.transform(xx, yy)
    return lat.astype(np.float32), lon.astype(np.float32)


def ensure_crs_variable(ds: xr.Dataset, src_epsg: str) -> xr.Dataset:
    """Attach a CF-style scalar ``crs`` variable to ``ds`` (in place-ish).

    The returned dataset is a shallow copy of ``ds`` with a ``crs``
    variable carrying ``spatial_ref`` / ``crs_wkt`` attributes derived
    from ``src_epsg``.  Robust-than-just-global because GDAL /
    rioxarray look for the variable.
    """
    import xarray as xr
    from pyproj import CRS

    crs_src = CRS.from_user_input(src_epsg)
    wkt = crs_src.to_wkt()

    ds = ds.copy()
    if "crs" not in ds.variables:
        ds["crs"] = xr.DataArray(np.int8(0), attrs={})
    ds["crs"].attrs.update({
        "long_name": "CRS definition",
        "spatial_ref": wkt,
        "crs_wkt": wkt,
    })
    # Not every CRS exposes a CF grid-mapping — safe to skip if it doesn't.
    with contextlib.suppress(Exception):
        ds["crs"].attrs.update(crs_src.to_cf())
    ds.attrs["crs"] = crs_src.to_string()
    return ds


def add_latlon_variables(
    ds: xr.Dataset,
    src_epsg: str | None = None,
    *,
    overwrite: bool = False,
    add_grid_mapping: bool = False,
    fallback_epsg: str = "EPSG:31287",
) -> xr.Dataset:
    """Decorate a projected ``(y, x)`` dataset with WGS84 ``lat``/``lon``.

    Parameters
    ----------
    ds
        Dataset with 1-D ``x`` and ``y`` coordinates (or variables).
    src_epsg
        EPSG string of ``ds``'s projected grid.  If :data:`None`, falls
        back to :func:`epsg_from_dataset`.
    overwrite
        If :data:`True`, overwrite existing ``lat``/``lon`` variables.
    add_grid_mapping
        If :data:`True`, attach ``grid_mapping = "crs"`` attribute to
        every 3-D ``(time, y, x)`` data variable.  Lets GDAL / QGIS
        reproject the dataset without extra hints.
    fallback_epsg
        Used only when ``src_epsg`` is :data:`None` AND the dataset
        carries no ``crs`` attribute.

    Returns
    -------
    xarray.Dataset
        A copy of ``ds`` with ``lat``, ``lon``, and ``crs`` variables.
    """
    import xarray as xr  # noqa: F401 — needed for the DataArray creation below

    if "x" not in ds.coords and "x" not in ds.variables:
        raise ValueError("dataset has no 'x' coordinate or variable")
    if "y" not in ds.coords and "y" not in ds.variables:
        raise ValueError("dataset has no 'y' coordinate or variable")

    if src_epsg is None:
        src_epsg = epsg_from_dataset(ds, fallback=fallback_epsg)

    needs_lat = overwrite or ("lat" not in ds.variables and "latitude" not in ds.variables)
    needs_lon = overwrite or ("lon" not in ds.variables and "longitude" not in ds.variables)

    out = ds.copy()
    if needs_lat or needs_lon:
        lat, lon = compute_latlon_grid(out["x"].values, out["y"].values, src_epsg)
        import xarray as _xr
        out["lat"] = _xr.DataArray(
            lat, dims=("y", "x"),
            attrs={
                "units": "degrees_north",
                "standard_name": "latitude",
                "long_name": "latitude",
            },
        )
        out["lon"] = _xr.DataArray(
            lon, dims=("y", "x"),
            attrs={
                "units": "degrees_east",
                "standard_name": "longitude",
                "long_name": "longitude",
            },
        )

    out = ensure_crs_variable(out, src_epsg)

    if add_grid_mapping:
        for v in out.data_vars:
            if out[v].ndim == 3 and tuple(out[v].dims)[:3] == ("time", "y", "x"):
                out[v].attrs["grid_mapping"] = "crs"

    return out
