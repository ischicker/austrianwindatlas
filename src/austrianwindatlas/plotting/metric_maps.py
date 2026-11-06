"""Spatial scatter plots of per-station verification metrics.

Given a per-station metric (RMSE, bias, Pearson r, …) at a list of
projected ``(x, y)`` coordinates, :func:`plot_station_metric` produces
a scatter plot with a diverging colormap centred on zero for bias-like
metrics, or a sequential one otherwise.

Deliberately generic: no hard-coded Austria bounding box, no Cartopy
requirement.  Callers who want a country border overlay provide their
own ``background`` callable that gets the Axes as its only argument.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ._mpl import ensure_axes, require_matplotlib

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from numpy.typing import NDArray

__all__ = ["plot_station_metric"]


#: Metric names that follow a "signed around zero" convention and
#: benefit from a diverging colormap centred on 0 (``RdBu_r`` is the
#: paper default).  Matches bias-like naming so most callers don't
#: need to touch ``diverging`` by hand.
_DIVERGING_HINTS: tuple[str, ...] = ("bias", "error", "diff", "residual")


def _should_be_diverging(metric_name: str | None) -> bool:
    if metric_name is None:
        return False
    low = metric_name.lower()
    return any(h in low for h in _DIVERGING_HINTS)


def plot_station_metric(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    values: NDArray[np.floating],
    *,
    metric_name: str | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | Colormap | None = None,
    diverging: bool | None = None,
    marker_size: float = 60.0,
    background: Callable[[Axes], None] | None = None,
    xlabel: str = "Easting",
    ylabel: str = "Northing",
    ax: Axes | None = None,
) -> Axes:
    """Scatter per-station ``values`` at projected coordinates.

    Parameters
    ----------
    x, y
        1-D arrays of the same length — station coordinates in the
        projected CRS of your choice (e.g. metres for EPSG:31287).
    values
        1-D array aligned with ``x`` / ``y``.  NaN entries are
        silently skipped so the colormap isn't dominated by unscored
        stations.
    metric_name
        If supplied, auto-selects ``diverging`` by matching
        ``_DIVERGING_HINTS`` ("bias", "error", "diff", "residual").
        Also becomes the colorbar label.
    title
        Axes title.
    vmin, vmax
        Colormap limits.  If both :data:`None`, inferred from the
        2–98 % quantile of ``values`` (robust to outlier stations).
        If ``diverging`` resolves to :data:`True`, the limits are made
        symmetric around zero.
    cmap
        Matplotlib colormap.  Defaults to ``"RdBu_r"`` when diverging,
        ``"viridis"`` otherwise.
    diverging
        Explicit override of the auto-detection.  :data:`None` (default)
        → auto from ``metric_name``.
    marker_size
        Passed to :meth:`Axes.scatter` as ``s``.
    background
        Optional callable ``f(ax) -> None`` to draw a basemap / border
        / hillshade *before* the scatter.  Users can plug cartopy or
        rasterio into this without the library importing them.
    xlabel, ylabel
        Axis labels.
    ax
        Optional matplotlib Axes.  :data:`None` creates one.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    require_matplotlib()
    fig, ax = ensure_axes(ax)

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if not (x.shape == y.shape == values.shape):
        raise ValueError(
            f"x, y and values must share shape; got "
            f"{x.shape!r}, {y.shape!r}, {values.shape!r}"
        )

    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("values contains no finite entries")

    is_div = diverging if diverging is not None else _should_be_diverging(metric_name)

    if vmin is None or vmax is None:
        q_lo = float(np.quantile(values[finite], 0.02))
        q_hi = float(np.quantile(values[finite], 0.98))
        if vmin is None:
            vmin = q_lo
        if vmax is None:
            vmax = q_hi
    if is_div:
        bound = max(abs(float(vmin)), abs(float(vmax)))
        vmin, vmax = -bound, bound

    effective_cmap = cmap
    if effective_cmap is None:
        effective_cmap = "RdBu_r" if is_div else "viridis"

    if background is not None:
        background(ax)

    sc = ax.scatter(
        x[finite], y[finite], c=values[finite],
        cmap=effective_cmap, vmin=vmin, vmax=vmax,
        s=marker_size, edgecolors="white", linewidths=0.8,
    )

    cbar_label = metric_name.upper() if metric_name else ""
    cbar = fig.colorbar(sc, ax=ax, label=cbar_label)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")

    return ax
