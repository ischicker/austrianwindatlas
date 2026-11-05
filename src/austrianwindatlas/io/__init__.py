"""I/O helpers: station NetCDFs, rPCA CSV exports, reconstructed wind fields.

See the submodule docstrings for details.  Public API is re-exported
here for convenience.
"""
from __future__ import annotations

from .netcdf import (
    add_latlon_variables,
    compute_latlon_grid,
    ensure_crs_variable,
    epsg_from_dataset,
)
from .rpca_export import RPCAExport, read_rpca_export
from .stations import (
    DEFAULT_MONTHLY_PATTERNS,
    DEFAULT_WIND_VAR_NAMES,
    StationQCResult,
    apply_quality_control,
    extract_station_metadata,
    load_station_month,
    load_station_series,
    read_station_netcdf,
    resolve_wind_variable,
)
from .windfield import INCA_GRID, GridSpec, build_wind_field_dataset, write_wind_field_netcdf

__all__ = [
    # netcdf
    "add_latlon_variables",
    "compute_latlon_grid",
    "ensure_crs_variable",
    "epsg_from_dataset",
    # rpca_export
    "RPCAExport",
    "read_rpca_export",
    # stations
    "DEFAULT_MONTHLY_PATTERNS",
    "DEFAULT_WIND_VAR_NAMES",
    "StationQCResult",
    "apply_quality_control",
    "extract_station_metadata",
    "load_station_month",
    "load_station_series",
    "read_station_netcdf",
    "resolve_wind_variable",
    # windfield
    "INCA_GRID",
    "GridSpec",
    "build_wind_field_dataset",
    "write_wind_field_netcdf",
]
