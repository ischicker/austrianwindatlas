"""Stub: reconstruction step of the Amato Stage-1 family.

See :mod:`austrianwindatlas.stage1.amato` for the full licence
narrative.  :func:`reconstruct_month` raises
:class:`austrianwindatlas.stage1.amato.AmatoUnavailableError` when
invoked.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import pandas as pd
    import xarray as xr
    from numpy.typing import NDArray

__all__ = ["reconstruct_month"]


def reconstruct_month(
    V_grid: NDArray[np.floating],
    sigma: NDArray[np.floating],
    U: NDArray[np.floating],
    temporal_mean: NDArray[np.floating],
    timestamps: pd.DatetimeIndex,
    *,
    year: int,
    month: int,
    **kwargs: object,
) -> xr.Dataset | None:
    """Stub for the Amato monthly reconstruction.

    Will materialise an :class:`xarray.Dataset` of shape
    ``(time, y, x)`` once the upstream-licence situation resolves.
    Signature kept for API stability.

    Raises
    ------
    AmatoUnavailableError
        Always — see :mod:`austrianwindatlas.stage1.amato`.
    """
    del V_grid, sigma, U, temporal_mean, timestamps, year, month, kwargs
    from . import AmatoUnavailableError

    raise AmatoUnavailableError("reconstruct.reconstruct_month")
