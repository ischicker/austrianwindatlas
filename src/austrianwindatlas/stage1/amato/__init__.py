"""Amato (2020) spatio-temporal DL family — **stub pending upstream licence**.

The reference implementation at <https://github.com/federhub/ST_DeepLearning>
does not declare a licence.  Default copyright is "all rights reserved
by the author", so this library cannot redistribute a port — not even
a modified fork — until explicit permission is granted.

Status (:data:`AMATO_STATUS`):
    * ``requested`` — permission request has been sent upstream.
    * ``granted`` — if / when permission arrives, the stubs are
      replaced with a real port, ``LICENSE.upstream`` is added
      inside this directory, and this constant flips to ``granted``.
    * ``blocked`` — set only if the request was explicitly refused.

Programmatic introspection::

    from austrianwindatlas.stage1 import amato
    print(amato.AMATO_STATUS)                # "requested"
    print(amato.AMATO_UPSTREAM_URL)          # upstream repo URL

Attempting to invoke any Amato function raises :class:`AmatoUnavailableError`
(a subclass of :class:`NotImplementedError`) with a clear pointer at
the upstream repository and :doc:`docs/amato.md`.

What to use instead
-------------------

:mod:`austrianwindatlas.stage1.enhanced_mae` is an *independent*
implementation in the same architectural family, written from
scratch, MIT-licensed, and already ships out of the box.  It is the
variant that appears in the paper's multi-criteria ranking.
"""
from __future__ import annotations

from . import reconstruct, svd
from .reconstruct import reconstruct_month
from .svd import AmatoSVD, compute_amato_svd

__all__ = [
    # Status constants
    "AMATO_STATUS",
    "AMATO_UPSTREAM_URL",
    "AMATO_LICENCE_DOC",
    "AmatoUnavailableError",
    # Sub-modules re-exported for discoverability
    "svd",
    "reconstruct",
    # Stubs (raise AmatoUnavailableError)
    "compute_amato_svd",
    "AmatoSVD",
    "reconstruct_month",
]

#: Current permission state.  One of ``"requested"`` / ``"granted"``
#: / ``"blocked"``.  Flip to ``"granted"`` when the stubs are
#: replaced with the real implementation.
AMATO_STATUS: str = "requested"

#: The upstream reference implementation.
AMATO_UPSTREAM_URL: str = "https://github.com/federhub/ST_DeepLearning"

#: The in-library docs page explaining the licence situation.
AMATO_LICENCE_DOC: str = "docs/amato.md"


class AmatoUnavailableError(NotImplementedError):
    """Raised when a user calls into a stubbed Amato entry point.

    Subclass of :class:`NotImplementedError` so generic
    ``except NotImplementedError`` branches continue to work, but
    callers that want to react specifically to the licence situation
    can ``except AmatoUnavailableError`` to e.g. fall back to
    Enhanced-MAE automatically.
    """

    def __init__(self, entry_point: str) -> None:
        msg = (
            f"austrianwindatlas.stage1.amato.{entry_point} is stubbed "
            f"pending upstream-licence permission.\n"
            f"\n"
            f"  Upstream reference implementation:\n"
            f"    {AMATO_UPSTREAM_URL}\n"
            f"\n"
            f"  Current status: {AMATO_STATUS}\n"
            f"\n"
            f"  See {AMATO_LICENCE_DOC} for the full narrative, and\n"
            f"  consider using austrianwindatlas.stage1.enhanced_mae, which\n"
            f"  is an independent MIT-licensed implementation in the same\n"
            f"  architectural family and ships out of the box."
        )
        super().__init__(msg)
        self.entry_point = entry_point
