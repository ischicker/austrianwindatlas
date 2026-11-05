"""Wind-energy diagnostics for Stage-2 validation.

Four pure-NumPy functions:

* :func:`fit_weibull` — Weibull shape / scale from mean and std
  (moment-matching).  Returns ``(k, A)``.
* :func:`wind_power_density` — ``0.5 * rho * mean(ws^3)``.  Units:
  W / m².  Uses the air-density default 1.225 kg/m³ at sea level.
* :func:`capacity_factor` — Fraction of rated output a turbine with
  cut-in / rated / cut-out thresholds would produce.  Uses a cubic
  power curve between cut-in and rated, flat between rated and
  cut-out, zero outside.
* :func:`detect_ramp_events` — Timeseries ramp-event detector.
  Returns a list of :class:`RampEvent` dataclasses with ``start_time``,
  ``end_time``, ``delta`` etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = [
    "DEFAULT_AIR_DENSITY",
    "WeibullParameters",
    "RampEvent",
    "fit_weibull",
    "wind_power_density",
    "capacity_factor",
    "detect_ramp_events",
]

#: Standard sea-level air density (kg/m³).  Paper default.
DEFAULT_AIR_DENSITY: float = 1.225

#: Minimum valid samples for the Weibull / power-density / capacity
#: diagnostics.  Below this they return NaN.
_DEFAULT_MIN_SAMPLES_WEIBULL: int = 100
_DEFAULT_MIN_SAMPLES_PD: int = 10
_DEFAULT_MIN_SAMPLES_CF: int = 100


@dataclass(frozen=True)
class WeibullParameters:
    """Moment-matched Weibull shape (``k``) and scale (``A``)."""
    k: float
    A: float


def fit_weibull(
    wind_speed: NDArray[np.floating],
    *,
    min_samples: int = _DEFAULT_MIN_SAMPLES_WEIBULL,
) -> WeibullParameters:
    """Moment-matched Weibull fit.

    Shape parameter from the Justus 1978 approximation:

    .. math::

        k = (\\sigma / \\mu)^{-1.086}

    clipped to ``[1, 4]`` to keep the ``gamma(1 + 1/k)`` factor
    sensible.  Scale parameter:

    .. math::

        A = \\mu / \\Gamma(1 + 1/k).

    Returns ``WeibullParameters(nan, nan)`` if fewer than
    ``min_samples`` positive, finite samples remain.
    """
    from math import gamma

    ws = np.asarray(wind_speed, dtype=float).ravel()
    ws = ws[np.isfinite(ws) & (ws > 0)]
    if ws.size < int(min_samples):
        return WeibullParameters(k=float("nan"), A=float("nan"))

    mean_ws = float(np.mean(ws))
    std_ws = float(np.std(ws))
    if std_ws <= 0.0 or mean_ws <= 0.0:
        return WeibullParameters(k=float("nan"), A=float("nan"))

    k_raw = (std_ws / mean_ws) ** -1.086
    k = float(np.clip(k_raw, 1.0, 4.0))
    scale = mean_ws / gamma(1.0 + 1.0 / k)
    return WeibullParameters(k=k, A=float(scale))


def wind_power_density(
    wind_speed: NDArray[np.floating],
    *,
    rho: float = DEFAULT_AIR_DENSITY,
    min_samples: int = _DEFAULT_MIN_SAMPLES_PD,
) -> float:
    """``0.5 * rho * mean(ws ** 3)`` in W / m²."""
    ws = np.asarray(wind_speed, dtype=float).ravel()
    ws = ws[np.isfinite(ws)]
    if ws.size < int(min_samples):
        return float("nan")
    return float(0.5 * rho * np.mean(ws ** 3))


def capacity_factor(
    wind_speed: NDArray[np.floating],
    *,
    cut_in: float = 3.0,
    rated: float = 12.0,
    cut_out: float = 25.0,
    min_samples: int = _DEFAULT_MIN_SAMPLES_CF,
) -> float:
    """Fraction of rated output a typical turbine would produce.

    Power curve:

    * ``ws < cut_in`` or ``ws > cut_out`` → 0.
    * ``cut_in <= ws < rated`` → ``((ws - cut_in) / (rated - cut_in)) ** 3``.
    * ``rated <= ws <= cut_out`` → 1.

    Matches the paper's IEC-Class-II assumptions
    (``cut_in=3``, ``rated=12``, ``cut_out=25`` m/s).
    """
    if not (cut_in < rated <= cut_out):
        raise ValueError(
            f"invalid power curve: cut_in={cut_in}, rated={rated}, cut_out={cut_out}; "
            "must satisfy cut_in < rated <= cut_out"
        )

    ws = np.asarray(wind_speed, dtype=float).ravel()
    ws = ws[np.isfinite(ws)]
    if ws.size < int(min_samples):
        return float("nan")

    p = np.zeros_like(ws)
    ramp = (ws >= cut_in) & (ws < rated)
    flat = (ws >= rated) & (ws <= cut_out)
    p[ramp] = ((ws[ramp] - cut_in) / (rated - cut_in)) ** 3
    p[flat] = 1.0
    return float(np.mean(p))


# ---------------------------------------------------------------------------
# Ramp events
# ---------------------------------------------------------------------------
RampKind = Literal["ramp_up", "ramp_down"]


@dataclass(frozen=True)
class RampEvent:
    """Single wind ramp.

    Attributes
    ----------
    start_index, end_index
        0-based integer indices into the input arrays (inclusive).
    start_time, end_time
        Corresponding timestamps (any type the caller passes in — the
        detector doesn't interpret them).
    start_wind, end_wind
        Wind speed at the edges.
    delta
        ``end_wind - start_wind`` (signed).
    kind
        ``"ramp_up"`` when ``delta > 0``, ``"ramp_down"`` otherwise.
    """
    start_index: int
    end_index: int
    start_time: object
    end_time: object
    start_wind: float
    end_wind: float
    delta: float
    kind: RampKind


def detect_ramp_events(
    wind_speed: NDArray[np.floating],
    times: NDArray,
    *,
    threshold: float = 5.0,
    window_hours: int = 3,
) -> list[RampEvent]:
    """Detect ramps of ``|delta_wind| >= threshold`` across ``window_hours`` steps.

    Pure Python iteration; fine for the paper's monthly / yearly
    station timeseries.  Returns an empty list when the input is too
    short for a single window.

    Parameters
    ----------
    wind_speed
        1-D wind-speed series.
    times
        1-D array of timestamps aligned with ``wind_speed``.  Any
        dtype works (``pandas.DatetimeIndex``, :class:`numpy.datetime64`,
        :class:`int` …).
    threshold
        Minimum absolute change in m/s to flag as a ramp.
    window_hours
        Number of timesteps over which to measure the delta.
    """
    if window_hours < 1:
        raise ValueError(f"window_hours must be >= 1; got {window_hours!r}")

    ws = np.asarray(wind_speed, dtype=float).ravel()
    tt = np.asarray(times).ravel()
    if ws.size != tt.size:
        raise ValueError(
            f"wind_speed and times must share length; got {ws.size} vs {tt.size}"
        )
    if ws.size <= window_hours:
        return []

    out: list[RampEvent] = []
    for i in range(ws.size - window_hours):
        s = ws[i]
        e = ws[i + window_hours]
        if not (np.isfinite(s) and np.isfinite(e)):
            continue
        delta = float(e - s)
        if abs(delta) < float(threshold):
            continue
        out.append(RampEvent(
            start_index=int(i),
            end_index=int(i + window_hours),
            start_time=tt[i],
            end_time=tt[i + window_hours],
            start_wind=float(s),
            end_wind=float(e),
            delta=delta,
            kind="ramp_up" if delta > 0 else "ramp_down",
        ))
    return out
