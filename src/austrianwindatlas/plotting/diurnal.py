"""Diurnal-cycle comparison between obs and one-or-more predictions.

Takes a wind-speed series + its timestamps, groups by hour-of-day,
plots the mean (with ±1 σ shading) against one or more predictions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._mpl import ensure_axes, require_matplotlib

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = ["diurnal_cycle", "plot_diurnal_cycle"]


def diurnal_cycle(
    times: pd.DatetimeIndex | NDArray,
    wind_speed: NDArray[np.floating],
) -> pd.DataFrame:
    """Return a per-hour-of-day summary ``(hour, mean, std, count)``.

    Parameters
    ----------
    times
        Timestamps aligned with ``wind_speed`` — anything
        :func:`pandas.to_datetime` accepts.
    wind_speed
        1-D wind-speed series (m/s).  NaN-tolerant.

    Returns
    -------
    :class:`pandas.DataFrame`
        24 rows (hours 0–23) with columns ``hour``, ``mean``,
        ``std``, ``count``.  Hours with zero finite samples still
        appear but with NaN mean/std and ``count=0``.
    """
    ws = np.asarray(wind_speed, dtype=float).ravel()
    ts = pd.DatetimeIndex(pd.to_datetime(np.asarray(times)))
    if ws.shape[0] != ts.shape[0]:
        raise ValueError(
            f"times and wind_speed must share length; got "
            f"{ts.shape[0]} vs {ws.shape[0]}"
        )

    frame = pd.DataFrame({"hour": ts.hour, "ws": ws})
    out = frame.groupby("hour")["ws"].agg(
        mean=lambda s: float(np.nanmean(s)) if np.any(np.isfinite(s)) else np.nan,
        std=lambda s: float(np.nanstd(s)) if np.any(np.isfinite(s)) else np.nan,
        count=lambda s: int(np.sum(np.isfinite(s))),
    ).reset_index()

    # Ensure all 24 hours are represented (dataset might miss hours).
    full = pd.DataFrame({"hour": np.arange(24)})
    return full.merge(out, on="hour", how="left").assign(
        count=lambda d: d["count"].fillna(0).astype(int),
    )


def plot_diurnal_cycle(
    obs_times: pd.DatetimeIndex | NDArray,
    obs_wind: NDArray[np.floating],
    predictions: dict[str, tuple[pd.DatetimeIndex | NDArray, NDArray[np.floating]]],
    *,
    title: str | None = None,
    show_obs_band: bool = True,
    colors: dict[str, str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot the observed diurnal cycle plus one line per prediction.

    Parameters
    ----------
    obs_times, obs_wind
        Observations — timestamps and wind-speed series.
    predictions
        ``{label: (times, wind_speed)}``.  Each entry gets its own
        line.  Use the same timestamps as ``obs_times`` if you have
        paired observations.
    title
        Axes title.
    show_obs_band
        If :data:`True` (default), shade ±1 σ around the observed
        mean so model lines are compared against observational
        spread, not just the mean.
    colors
        ``{label: matplotlib-color}``.  Any missing label gets the
        default matplotlib color cycle.
    ax
        Optional matplotlib Axes.  :data:`None` creates one.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    """
    require_matplotlib()
    fig, ax = ensure_axes(ax)

    obs_df = diurnal_cycle(obs_times, obs_wind)
    hours = obs_df["hour"].to_numpy()
    obs_mean = obs_df["mean"].to_numpy()
    obs_std = obs_df["std"].to_numpy()

    ax.plot(hours, obs_mean, color="black", linewidth=2.0, label="Observations",
            marker="o", markersize=5)
    if show_obs_band:
        finite = np.isfinite(obs_mean) & np.isfinite(obs_std)
        if np.any(finite):
            ax.fill_between(
                hours[finite],
                (obs_mean - obs_std)[finite],
                (obs_mean + obs_std)[finite],
                color="black", alpha=0.12,
                label=r"Obs $\pm 1\sigma$",
            )

    colors = colors or {}
    for label, (times_p, ws_p) in predictions.items():
        pred_df = diurnal_cycle(times_p, ws_p)
        kwargs = {"label": label, "linewidth": 1.5, "marker": "s", "markersize": 4}
        if label in colors:
            kwargs["color"] = colors[label]
        ax.plot(pred_df["hour"], pred_df["mean"], **kwargs)

    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xlim(-0.5, 23.5)
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Wind speed [m/s]")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    return ax
