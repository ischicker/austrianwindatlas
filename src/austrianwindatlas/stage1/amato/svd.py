"""Stub: SVD step of the Amato Stage-1 family.

See :mod:`austrianwindatlas.stage1.amato` for the full licence
narrative.  Every callable in this module raises
:class:`austrianwindatlas.stage1.amato.AmatoUnavailableError` when
invoked.  Instantiating :class:`AmatoSVD` also raises — carrying the
dataclass skeleton lets type-checkers and docs tools see a surface
area, but it can't be constructed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from numpy.typing import NDArray

__all__ = ["AmatoSVD", "compute_amato_svd"]


class AmatoSVD:
    """Stub for the Amato-style SVD result.

    Constructor raises :class:`AmatoUnavailableError` — the real
    dataclass fields will be documented here once permission is
    granted upstream.  See :mod:`austrianwindatlas.stage1.amato`.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001, ARG002 - stub
        # Local import to dodge the circular ``amato → svd → amato``.
        from . import AmatoUnavailableError

        raise AmatoUnavailableError("svd.AmatoSVD")


def compute_amato_svd(
    wind_data: NDArray[np.floating],
    n_eofs: int = 80,
) -> AmatoSVD:
    """Stub for the Amato SVD computation.

    Will take a ``(n_times, n_stations)`` wind-speed matrix and
    return an :class:`AmatoSVD` once the upstream-licence situation
    resolves.  Signature kept for API stability so callers can be
    written against the typed surface already.

    Raises
    ------
    AmatoUnavailableError
        Always — see :mod:`austrianwindatlas.stage1.amato`.
    """
    del wind_data, n_eofs  # mute the "unused-arg" checker for the stub
    from . import AmatoUnavailableError

    raise AmatoUnavailableError("svd.compute_amato_svd")
