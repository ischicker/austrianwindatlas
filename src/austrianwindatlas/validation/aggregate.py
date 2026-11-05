"""Per-station metric aggregation.

:func:`station_metrics` runs every metric from
:mod:`austrianwindatlas.validation.metrics` on a single ``(obs, pred)``
pair and packages the result in :class:`StationMetrics`.
:func:`aggregate_across_stations` reduces a list of those to one
:class:`AggregatedMetrics` that carries both the "mean of per-station
scores" and the "pooled over all pairs" variants — the paper reports
both because they answer different questions (an outlier station
affects them differently).

Both the per-station and aggregated dataclasses can be turned into a
:class:`pandas.DataFrame` for easy CSV export via
:meth:`StationMetrics.asdict` / :meth:`AggregatedMetrics.asdict`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING

import numpy as np

from .metrics import (
    DEFAULT_MIN_PAIRS,
    DEFAULT_QUANTILES,
    DEFAULT_THRESHOLDS,
    bias,
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

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = [
    "StationMetrics",
    "AggregatedMetrics",
    "station_metrics",
    "aggregate_across_stations",
]


# ---------------------------------------------------------------------------
# Per-station result
# ---------------------------------------------------------------------------
@dataclass
class StationMetrics:
    """All verification metrics for one station × year.

    Fields default to :data:`numpy.nan` so callers can build a frame
    from partial results without special-casing missing values.

    The threshold metrics are flattened — one set of
    ``(hit_rate_<t>, false_alarm_rate_<t>, critical_success_index_<t>,
    frequency_bias_<t>)`` per element of :data:`DEFAULT_THRESHOLDS`.
    Same for ``quantile_bias_q<q>`` across :data:`DEFAULT_QUANTILES`.
    """

    station_id: object = None
    n_pairs: int = 0

    rmse: float = float("nan")
    mae: float = float("nan")
    bias: float = float("nan")
    pearson_r: float = float("nan")
    spearman_r: float = float("nan")
    r_squared: float = float("nan")
    ks_statistic: float = float("nan")

    skill_vs_climatology: float = float("nan")
    skill_vs_persistence: float = float("nan")

    # Quantile biases (populated dynamically in station_metrics).
    # Field defaults here only cover the three paper defaults; extra
    # quantiles land as plain attrs if the caller passes custom ones.
    quantile_bias_q50: float = float("nan")
    quantile_bias_q90: float = float("nan")
    quantile_bias_q99: float = float("nan")

    # Threshold metrics at 5 / 10 / 15 m/s.
    hit_rate_5: float = float("nan")
    false_alarm_rate_5: float = float("nan")
    critical_success_index_5: float = float("nan")
    frequency_bias_5: float = float("nan")
    hit_rate_10: float = float("nan")
    false_alarm_rate_10: float = float("nan")
    critical_success_index_10: float = float("nan")
    frequency_bias_10: float = float("nan")
    hit_rate_15: float = float("nan")
    false_alarm_rate_15: float = float("nan")
    critical_success_index_15: float = float("nan")
    frequency_bias_15: float = float("nan")

    def asdict(self) -> dict[str, object]:
        """Return a flat dict suitable for ``pd.DataFrame.from_records``."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-station computation
# ---------------------------------------------------------------------------
def _nan_if_short(arr: NDArray[np.floating], min_pairs: int) -> bool:
    return arr.size < int(min_pairs)


def _climatology_and_persistence(
    obs: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Build climatology and 1-step persistence references for skill scores.

    Climatology = constant ``nanmean(obs)``.  Persistence = ``obs`` lagged
    one step (first element → NaN).  Both have the same shape as ``obs``.
    """
    obs = np.asarray(obs, dtype=float).ravel()
    clim = np.full_like(obs, float(np.nanmean(obs)) if np.any(np.isfinite(obs)) else float("nan"))
    pers = np.empty_like(obs)
    pers[0] = float("nan")
    if obs.size > 1:
        pers[1:] = obs[:-1]
    return clim, pers


def station_metrics(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    *,
    station_id: object = None,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> StationMetrics:
    """Compute every metric for a single station × year.

    Matches ``compute_all_station_metrics`` in the paper's verification
    framework: RMSE, MAE, bias, Pearson / Spearman, R², skill vs
    climatology & persistence, KS statistic, quantile biases, and
    four categorical metrics at each threshold.

    Returns a :class:`StationMetrics` with NaN fields for anything
    that can't be computed given the paired-finite sample size.
    """
    o = np.asarray(obs, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    if o.shape != p.shape:
        raise ValueError(f"obs and pred must share shape; got {o.shape!r} vs {p.shape!r}")

    mask = np.isfinite(o) & np.isfinite(p)
    result = StationMetrics(
        station_id=station_id,
        n_pairs=int(mask.sum()),
    )
    if _nan_if_short(o[mask], min_pairs):
        return result

    # Basic + correlation + variance-explained.
    result.rmse = rmse(o, p, min_pairs=min_pairs)
    result.mae = mae(o, p, min_pairs=min_pairs)
    result.bias = bias(o, p, min_pairs=min_pairs)
    result.pearson_r = pearson_r(o, p, min_pairs=min_pairs)
    result.spearman_r = spearman_r(o, p, min_pairs=min_pairs)
    result.r_squared = r_squared(o, p, min_pairs=min_pairs)
    result.ks_statistic = ks_statistic(o, p, min_pairs=min_pairs)

    # Skill scores vs climatology / persistence.
    clim, pers = _climatology_and_persistence(o)
    result.skill_vs_climatology = skill_score(o, p, clim, min_pairs=min_pairs)
    result.skill_vs_persistence = skill_score(o, p, pers, min_pairs=min_pairs)

    # Quantile biases.
    for q in quantiles:
        value = quantile_bias(o, p, q, min_pairs=min_pairs)
        attr = f"quantile_bias_q{int(round(q * 100))}"
        # Set even if the attr isn't one of the pre-declared ones;
        # dataclass still allows arbitrary attribute assignment.
        setattr(result, attr, value)

    # Threshold metrics.
    for t in thresholds:
        tm = threshold_metrics(o, p, t, min_pairs=min_pairs)
        key = int(round(t))
        setattr(result, f"hit_rate_{key}", tm.hit_rate)
        setattr(result, f"false_alarm_rate_{key}", tm.false_alarm_rate)
        setattr(result, f"critical_success_index_{key}", tm.critical_success_index)
        setattr(result, f"frequency_bias_{key}", tm.frequency_bias)

    return result


# ---------------------------------------------------------------------------
# Across-station aggregation
# ---------------------------------------------------------------------------
@dataclass
class AggregatedMetrics:
    """Summary across many :class:`StationMetrics`.

    Reports each per-station metric in two flavours:

    * ``mean_<name>`` — unweighted mean over the per-station scores.
    * ``median_<name>`` — the robust alternative.
    * ``pooled_<name>`` — recomputed from concatenated ``(obs, pred)``
      pairs across stations; equivalent to weighting by station sample
      count.

    Only the metrics that the pooled variant makes sense for carry a
    ``pooled_*`` field (threshold counts, quantile biases and basic
    errors — not the skill scores, which depend on a per-station
    reference).

    ``n_stations`` and ``n_pairs`` give the sample size.
    """

    method_name: str
    n_stations: int = 0
    n_pairs: int = 0

    mean_rmse: float = float("nan")
    median_rmse: float = float("nan")
    std_rmse: float = float("nan")
    mean_mae: float = float("nan")
    mean_bias: float = float("nan")
    mean_pearson_r: float = float("nan")
    mean_spearman_r: float = float("nan")
    mean_r_squared: float = float("nan")
    mean_ks_statistic: float = float("nan")

    pooled_rmse: float = float("nan")
    pooled_mae: float = float("nan")
    pooled_bias: float = float("nan")
    pooled_pearson_r: float = float("nan")

    mean_critical_success_index_5: float = float("nan")
    mean_critical_success_index_10: float = float("nan")
    mean_critical_success_index_15: float = float("nan")
    mean_hit_rate_5: float = float("nan")
    mean_hit_rate_10: float = float("nan")
    mean_hit_rate_15: float = float("nan")
    mean_false_alarm_rate_5: float = float("nan")
    mean_false_alarm_rate_10: float = float("nan")
    mean_false_alarm_rate_15: float = float("nan")
    mean_frequency_bias_5: float = float("nan")
    mean_frequency_bias_10: float = float("nan")
    mean_frequency_bias_15: float = float("nan")

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _nanmedian(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def _nanstd(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanstd(arr))


def aggregate_across_stations(
    station_results: list[StationMetrics],
    *,
    method_name: str,
    pooled_obs: NDArray[np.floating] | None = None,
    pooled_pred: NDArray[np.floating] | None = None,
) -> AggregatedMetrics:
    """Reduce a list of per-station results to one :class:`AggregatedMetrics`.

    ``pooled_obs`` / ``pooled_pred`` (optional) are concatenations of the
    raw ``(obs, pred)`` arrays across stations.  When provided they
    drive the ``pooled_*`` fields; otherwise those stay NaN.
    """
    if not station_results:
        return AggregatedMetrics(method_name=method_name)

    def _collect(name: str) -> list[float]:
        return [float(getattr(r, name)) for r in station_results]

    agg = AggregatedMetrics(
        method_name=method_name,
        n_stations=len(station_results),
        n_pairs=int(sum(r.n_pairs for r in station_results)),
    )

    for name in ("mae", "bias", "pearson_r", "spearman_r", "r_squared", "ks_statistic"):
        setattr(agg, f"mean_{name}", _nanmean(_collect(name)))

    rmse_values = _collect("rmse")
    agg.mean_rmse = _nanmean(rmse_values)
    agg.median_rmse = _nanmedian(rmse_values)
    agg.std_rmse = _nanstd(rmse_values)

    for key in (5, 10, 15):
        for family in ("hit_rate", "false_alarm_rate",
                       "critical_success_index", "frequency_bias"):
            setattr(agg, f"mean_{family}_{key}",
                    _nanmean(_collect(f"{family}_{key}")))

    if pooled_obs is not None and pooled_pred is not None:
        agg.pooled_rmse = rmse(pooled_obs, pooled_pred)
        agg.pooled_mae = mae(pooled_obs, pooled_pred)
        agg.pooled_bias = bias(pooled_obs, pooled_pred)
        agg.pooled_pearson_r = pearson_r(pooled_obs, pooled_pred)

    return agg


# Compatibility helper — surface the full field list for downstream
# consumers that want to turn a list[StationMetrics] into a DataFrame.
def station_metric_field_names() -> tuple[str, ...]:
    """Return the ordered tuple of :class:`StationMetrics` field names."""
    return tuple(f.name for f in dc_fields(StationMetrics))
