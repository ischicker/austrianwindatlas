"""Wind-speed histogram with Weibull PDF overlay.

Fits a Weibull to the sample via
:func:`austrianwindatlas.validation.energy.fit_weibull` (or accepts a
pre-computed :class:`WeibullParameters`), then overlays the
analytical PDF ``f(v; k, A) = (k/A)(v/A)^(k-1) exp(-(v/A)^k)`` on
a density histogram.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..validation.energy import WeibullParameters, fit_weibull
from ._mpl import ensure_axes, require_matplotlib

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = ["plot_weibull_fit"]


def _weibull_pdf(v: NDArray[np.floating], k: float, A: float) -> NDArray[np.floating]:
    """Weibull PDF.  Returns 0 for ``v <= 0``."""
    out = np.zeros_like(v, dtype=float)
    pos = v > 0
    vp = v[pos] / A
    out[pos] = (k / A) * (vp ** (k - 1.0)) * np.exp(-(vp ** k))
    return out


def plot_weibull_fit(
    wind_speed: NDArray[np.floating],
    *,
    params: WeibullParameters | None = None,
    n_bins: int = 40,
    max_speed: float | None = None,
    title: str | None = None,
    hist_color: str = "#4477AA",
    pdf_color: str = "#CC6677",
    ax: Axes | None = None,
) -> tuple[Axes, WeibullParameters]:
    """Histogram of wind speeds + analytical Weibull PDF overlay.

    Parameters
    ----------
    wind_speed
        1-D series (m/s).  NaN entries are dropped.
    params
        Pre-computed :class:`WeibullParameters`.  When :data:`None`,
        fitted on ``wind_speed`` via
        :func:`austrianwindatlas.validation.energy.fit_weibull`.
    n_bins
        Histogram bin count.
    max_speed
        Upper x-axis limit.  Defaults to the 99.5-th percentile of
        the sample (keeps a single outlier from stretching the plot).
    title
        Axes title.
    hist_color, pdf_color
        Matplotlib colour specs.
    ax
        Optional matplotlib Axes.  :data:`None` creates one.

    Returns
    -------
    (Axes, WeibullParameters)
        The axes and the (fitted or supplied) parameters, so callers
        can grab ``k`` / ``A`` for a legend or table.
    """
    require_matplotlib()
    fig, ax = ensure_axes(ax)

    ws = np.asarray(wind_speed, dtype=float).ravel()
    ws = ws[np.isfinite(ws)]
    if ws.size == 0:
        raise ValueError("wind_speed has no finite entries")

    if params is None:
        params = fit_weibull(ws)
    if not (np.isfinite(params.k) and np.isfinite(params.A)):
        raise ValueError(
            f"Weibull fit returned NaN parameters (k={params.k!r}, A={params.A!r}); "
            "likely too few positive samples in wind_speed"
        )

    if max_speed is None:
        max_speed = float(np.quantile(ws[ws > 0], 0.995)) if (ws > 0).any() else float(ws.max())

    bins = np.linspace(0.0, max_speed, n_bins + 1)
    ax.hist(
        ws, bins=bins, density=True,
        color=hist_color, alpha=0.6, edgecolor="white",
        linewidth=0.5, label="Observations",
    )

    v = np.linspace(0.0, max_speed, 400)
    ax.plot(
        v, _weibull_pdf(v, params.k, params.A),
        color=pdf_color, linewidth=2.0,
        label=f"Weibull (k={params.k:.2f}, A={params.A:.2f} m/s)",
    )

    ax.set_xlim(0.0, max_speed)
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax, params
