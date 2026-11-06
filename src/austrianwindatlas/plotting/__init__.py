"""Optional matplotlib-based plotters for verification figures.

This sub-package is gated behind the ``viz`` extra of the library
(matplotlib + cartopy); without it, importing
``austrianwindatlas.plotting`` still succeeds, but any plotting
function raises :class:`PlottingUnavailableError` with a clear
message pointing at ``pip install 'austrianwindatlas[viz]'``.

Every plotter follows the sklearn / seaborn pattern:

* Accepts an optional ``ax=None`` parameter.
* Creates a new figure + axes when ``ax`` is :data:`None`.
* Returns the :class:`matplotlib.axes.Axes` it drew into.
* Does **not** touch disk — callers ``fig.savefig(...)`` themselves.
* Does **not** hard-code colours, figure sizes or titles — all
  stylistic choices are kwargs.

Current coverage:

* :func:`plot_taylor_diagram` — Taylor diagrams of correlation vs
  std-ratio for a list of methods.
* :func:`plot_station_metric` — scatter per-station metric
  (RMSE / bias / r) at projected ``(x, y)`` coordinates.
* :func:`plot_diurnal_cycle` — observed vs predicted diurnal cycle
  with ±1σ observational band.
* :func:`plot_weibull_fit` — wind-speed histogram + analytical
  Weibull PDF overlay.

These cover the four figure types the paper uses most often.  Paper-
specific figure assemblies (full six-panel verification layouts,
hillshade overlays, etc.) are intentionally not ported — compose
them from these primitives in your own driver script.
"""
from __future__ import annotations

from ._mpl import PlottingUnavailableError
from .diurnal import diurnal_cycle, plot_diurnal_cycle
from .metric_maps import plot_station_metric
from .taylor import TaylorEntry, plot_taylor_diagram
from .weibull import plot_weibull_fit

__all__ = [
    "PlottingUnavailableError",
    # taylor
    "TaylorEntry",
    "plot_taylor_diagram",
    # metric_maps
    "plot_station_metric",
    # diurnal
    "diurnal_cycle",
    "plot_diurnal_cycle",
    # weibull
    "plot_weibull_fit",
]
