"""Verification metrics + aggregation for wind-field predictions.

The validation harness ships three sub-modules, all pure
NumPy/SciPy — no matplotlib, no NetCDF I/O; those live in the
``examples/`` tree.

* :mod:`.metrics` — individual metric functions (RMSE, MAE, bias,
  Pearson/Spearman, R², skill-score, KS, threshold metrics, quantile
  bias).
* :mod:`.energy` — wind-energy diagnostics (Weibull fit, wind power
  density, capacity factor, ramp-event detector).
* :mod:`.aggregate` — :class:`StationMetrics` (per-station) and
  :class:`AggregatedMetrics` (across stations), plus the drivers
  :func:`station_metrics` and :func:`aggregate_across_stations`.

The pieces are deliberately decoupled: callers can score *any* two
arrays with the metric functions, hand a long list of per-station
results to :func:`aggregate_across_stations`, and skip the energy
diagnostics entirely for non-wind applications.
"""
from __future__ import annotations

from . import aggregate, energy, metrics
from .aggregate import (
    AggregatedMetrics,
    StationMetrics,
    aggregate_across_stations,
    station_metric_field_names,
    station_metrics,
)
from .energy import (
    DEFAULT_AIR_DENSITY,
    RampEvent,
    WeibullParameters,
    capacity_factor,
    detect_ramp_events,
    fit_weibull,
    wind_power_density,
)
from .metrics import (
    DEFAULT_MIN_PAIRS,
    DEFAULT_QUANTILES,
    DEFAULT_THRESHOLDS,
    CorrelationKind,
    ThresholdMetrics,
    bias,
    critical_success_index,
    false_alarm_rate,
    frequency_bias,
    hit_rate,
    ks_statistic,
    mae,
    pearson_r,
    quantile_bias,
    r_squared,
    rmse,
    skill_score,
    spearman_r,
    threshold_metrics,
)

__all__ = [
    # Sub-modules re-exported for discoverability.
    "metrics",
    "energy",
    "aggregate",
    # metrics
    "DEFAULT_MIN_PAIRS",
    "DEFAULT_THRESHOLDS",
    "DEFAULT_QUANTILES",
    "CorrelationKind",
    "ThresholdMetrics",
    "rmse",
    "mae",
    "bias",
    "pearson_r",
    "spearman_r",
    "r_squared",
    "skill_score",
    "quantile_bias",
    "ks_statistic",
    "hit_rate",
    "false_alarm_rate",
    "critical_success_index",
    "frequency_bias",
    "threshold_metrics",
    # energy
    "DEFAULT_AIR_DENSITY",
    "WeibullParameters",
    "RampEvent",
    "fit_weibull",
    "wind_power_density",
    "capacity_factor",
    "detect_ramp_events",
    # aggregate
    "StationMetrics",
    "AggregatedMetrics",
    "station_metrics",
    "aggregate_across_stations",
    "station_metric_field_names",
]
