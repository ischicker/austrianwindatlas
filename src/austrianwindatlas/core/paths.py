"""Path resolution for the Austrian Windatlas library.

Every I/O helper routes through :func:`resolve` so that method modules
never hard-code absolute paths. The anchor is the ``AWA_DATA_ROOT``
environment variable (default: ``./awa_data``). The expected layout
below the root mirrors the file layout the paper's pipelines use —
documented in ``docs/data.md`` — but is advisory, not enforced:

::

    $AWA_DATA_ROOT/
    ├── stations/            # TAWES open-data NetCDF per month
    ├── dem/                 # EU-DEM v1.1 at 1 km
    ├── corine/              # CORINE 2018 roughness z0
    ├── era5/                # ERA5 10 m wind, hourly
    ├── cerra/               # CERRA 10 m wind, hourly
    ├── newa/                # NEWA multi-height WS NetCDFs
    ├── scada/               # per-turbine SCADA (user-provided; not
    │                        #  redistributed with the library)
    └── results/             # outputs, caches, diagnostics

The library will never *create* paths under the root; it only
reads from and resolves against it. Users point ``AWA_DATA_ROOT``
anywhere they like.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

__all__ = [
    "ENV_VAR",
    "DEFAULT_ROOT",
    "data_root",
    "resolve",
]

#: Name of the environment variable the library consults.
ENV_VAR: Final[str] = "AWA_DATA_ROOT"

#: Fallback used when ``AWA_DATA_ROOT`` is unset. Relative to the
#: current working directory; the user is expected to override this
#: for any serious run.
DEFAULT_ROOT: Final[Path] = Path.cwd() / "awa_data"


def data_root() -> Path:
    """Return the current data root as a :class:`pathlib.Path`.

    Reads the ``AWA_DATA_ROOT`` environment variable fresh on each
    call so that tests can monkeypatch it between cases.

    Returns
    -------
    Path
        The resolved data root. May not exist on disk; callers that
        need existence should check themselves — the library does not
        auto-create directories.
    """
    raw = os.environ.get(ENV_VAR)
    if raw is None or raw.strip() == "":
        return DEFAULT_ROOT
    return Path(raw).expanduser()


def resolve(*parts: str | os.PathLike) -> Path:
    """Join path parts under the data root.

    Example
    -------
    >>> import os
    >>> os.environ["AWA_DATA_ROOT"] = "/tmp/awa"
    >>> str(resolve("stations", "tawes_2020.nc"))
    '/tmp/awa/stations/tawes_2020.nc'
    """
    return data_root().joinpath(*map(os.fspath, parts))
