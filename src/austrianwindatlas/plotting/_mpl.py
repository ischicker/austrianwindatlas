"""Shared matplotlib bootstrap for :mod:`austrianwindatlas.plotting`.

The whole plotting sub-package treats matplotlib as a soft dependency
behind the ``viz`` extra: importing ``austrianwindatlas.plotting``
must succeed on a minimal install (no matplotlib / cartopy), and any
*call* into a plotting function raises :class:`PlottingUnavailableError`
with a clean message pointing at the ``viz`` extra.

Every plotting function calls :func:`require_matplotlib` at the top to
guarantee the import error surfaces consistently.
"""
from __future__ import annotations

__all__ = ["PlottingUnavailableError", "require_matplotlib", "ensure_axes"]


class PlottingUnavailableError(ImportError):
    """Raised when a plotting function is called without matplotlib installed.

    Subclass of :class:`ImportError` so standard ``except ImportError``
    branches continue to work, but callers that want to react specifically
    to the missing-viz-extra condition can match on this type.
    """

    def __init__(self, cause: Exception | None = None) -> None:
        msg = (
            "matplotlib is required for austrianwindatlas.plotting but is "
            "not installed.\n"
            "\n"
            "  Install the viz extra:\n"
            "      pip install 'austrianwindatlas[viz]'\n"
            "\n"
            "  Or install matplotlib directly:\n"
            "      pip install matplotlib"
        )
        super().__init__(msg)
        self.__cause__ = cause


def require_matplotlib():
    """Import :mod:`matplotlib.pyplot` and return it, or raise :class:`PlottingUnavailableError`.

    Centralising this gives every public plotter identical error
    surfacing — users don't have to guess which import went wrong.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover — tested via its absence
        raise PlottingUnavailableError(exc) from exc
    return plt


def ensure_axes(ax):  # noqa: ANN001 — ``ax`` is matplotlib.axes.Axes | None
    """Return ``(fig, ax)`` from an optional axes handle.

    If ``ax is None``, creates a new figure+axes; otherwise returns the
    existing axes and its figure.  Mirrors the sklearn / seaborn pattern.
    """
    plt = require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax
    return ax.figure, ax
