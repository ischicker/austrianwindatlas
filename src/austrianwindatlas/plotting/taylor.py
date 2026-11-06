"""Taylor diagram plotter.

A single polar plot that encodes three statistics at once:

* angular position = correlation (``r`` → ``arccos(r)``),
* radial position  = normalised standard-deviation ratio
  ``std(pred) / std(obs)``,
* reference point at ``(θ=0, r=1)`` marks the observation.

Perfect prediction ⇒ a dot right on top of the reference.  Models to
the left of the reference (``r > 0.9``) and with ``std_ratio ≈ 1``
are the ones to pick.

Interface
---------

:func:`plot_taylor_diagram` accepts a plain list of
:class:`TaylorEntry` tuples.  No pandas requirement, no hard-coded
colours, no file writing — the caller picks the colour per method and
``fig.savefig(...)`` themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._mpl import require_matplotlib

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

__all__ = [
    "TaylorEntry",
    "plot_taylor_diagram",
]


@dataclass
class TaylorEntry:
    """One method's position on a Taylor diagram.

    Attributes
    ----------
    label
        Legend entry (e.g. ``"KED+ERA5"``).
    correlation
        Pearson r against observations.  NaN entries are silently
        skipped by :func:`plot_taylor_diagram`.
    std_ratio
        ``std(pred) / std(obs)``.  1.0 means the model has the same
        variability as the observation.
    color
        Matplotlib colour spec (optional).  Defaults to black.
    marker
        Matplotlib marker code (optional).  Defaults to ``"o"``.
    """

    label: str
    correlation: float
    std_ratio: float
    color: str = "#333333"
    marker: str = "o"


def _reference_arcs(ax: Axes, r_values: tuple[float, ...], max_std_ratio: float) -> None:
    """Draw the dashed reference arcs at fixed r values."""
    for r in r_values:
        theta = np.arccos(r)
        ax.plot([theta, theta], [0.0, max_std_ratio], color="gray",
                linewidth=0.5, alpha=0.5)
        ax.text(theta, max_std_ratio * 1.03, f"{r:.2f}",
                ha="center", fontsize=7)


def plot_taylor_diagram(
    entries: list[TaylorEntry],
    *,
    title: str | None = None,
    max_std_ratio: float = 2.0,
    reference_r: tuple[float, ...] = (0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
    marker_size: float = 10.0,
    show_legend: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Plot a Taylor diagram on a polar axes.

    Parameters
    ----------
    entries
        One :class:`TaylorEntry` per model.  Entries with NaN
        ``correlation`` or ``std_ratio`` are silently skipped — useful
        when the metric aggregator returned NaN for some methods.
    title
        Figure title.  Leave :data:`None` for no title.
    max_std_ratio
        Upper radial limit.  Keeps the plot bounded when one method is
        pathologically over-dispersive.
    reference_r
        Correlation values at which dashed reference arcs are drawn.
    marker_size
        Passed to :meth:`matplotlib.axes.Axes.plot`.
    show_legend
        If :data:`True`, places a legend outside the plot on the right.
    ax
        Optional polar axes to draw into.  When :data:`None` (default)
        a new figure + polar axes is created.  **Must** be a polar axes
        when supplied — the plot is nonsense otherwise.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        The polar axes.  Grab the figure via ``ax.figure`` if you need
        to save it.
    """
    plt = require_matplotlib()

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
    else:
        if getattr(ax, "name", None) != "polar":
            raise ValueError(
                "Taylor diagram requires a polar axes; "
                "create one with plt.subplot(..., projection='polar')."
            )

    ax.set_thetamin(0)
    ax.set_thetamax(90)

    # Reference "observation" point at (theta=0, r=1).
    ax.plot(0.0, 1.0, "ko", markersize=marker_size * 1.2, label="Observation")

    # Reference arcs.
    _reference_arcs(ax, reference_r, max_std_ratio)

    for entry in entries:
        r = float(entry.correlation)
        sigma = float(entry.std_ratio)
        if not (np.isfinite(r) and np.isfinite(sigma)):
            continue
        theta = float(np.arccos(np.clip(r, -1.0, 1.0)))
        ax.plot(
            theta, sigma,
            marker=entry.marker, color=entry.color,
            markersize=marker_size, markeredgecolor="black",
            markeredgewidth=0.5, label=entry.label, linestyle="",
        )

    ax.set_ylim(0.0, max_std_ratio)
    ax.set_rlabel_position(135)       # keep radial tick labels out of the data

    if title:
        ax.set_title(title, fontsize=12, pad=20)

    if show_legend:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.0),
            fontsize=8,
        )

    return ax
