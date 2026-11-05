"""Pure verification metrics.

Every function here accepts two 1-D arrays — observations and
predictions — and returns a single float.  NaN entries in either
input are ignored (standard practice for station-vs-grid
verification where some timesteps lack observations).  All functions
return :data:`numpy.nan` when fewer than ``min_pairs`` valid entries
remain, so callers can safely average across stations without
NaN-propagation surprises.

The set covers:

* Basic errors — :func:`rmse`, :func:`mae`, :func:`bias`.
* Correlation — :func:`pearson_r`, :func:`spearman_r`.
* Variance-explained — :func:`r_squared`.
* Skill score against a reference — :func:`skill_score` (``1 - MSE_pred / MSE_ref``).
* Threshold metrics — :func:`hit_rate`, :func:`false_alarm_rate`,
  :func:`critical_success_index`, :func:`frequency_bias`, or get all
  four from :func:`threshold_metrics`.
* Quantile bias — :func:`quantile_bias`.
* Kolmogorov–Smirnov statistic — :func:`ks_statistic`.

Thresholds/quantiles use the paper's defaults (5 / 10 / 15 m/s;
0.50 / 0.90 / 0.99) but each function parameterises so callers can
roll their own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = [
    "DEFAULT_MIN_PAIRS",
    "DEFAULT_THRESHOLDS",
    "DEFAULT_QUANTILES",
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
]

#: Minimum valid pairs required by every metric before returning a
#: non-NaN value.  Matches the paper's ``min_obs=10`` threshold used in
#: ``verification_framework.compute_*``.
DEFAULT_MIN_PAIRS: int = 10

#: Wind-speed thresholds for the categorical metrics (m/s).  Matches
#: the paper's 5 / 10 / 15 default.
DEFAULT_THRESHOLDS: tuple[float, ...] = (5.0, 10.0, 15.0)

#: Quantiles for :func:`quantile_bias` (fractions, not percentiles).
DEFAULT_QUANTILES: tuple[float, ...] = (0.50, 0.90, 0.99)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _pairwise_finite(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return ``(obs, pred)`` filtered to pairs where both are finite."""
    obs = np.asarray(obs, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    if obs.shape != pred.shape:
        raise ValueError(
            f"obs and pred must share shape; got {obs.shape!r} vs {pred.shape!r}"
        )
    mask = np.isfinite(obs) & np.isfinite(pred)
    return obs[mask], pred[mask]


def _enough(n: int, min_pairs: int) -> bool:
    return n >= int(min_pairs)


# ---------------------------------------------------------------------------
# Basic errors
# ---------------------------------------------------------------------------
def rmse(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    *,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> float:
    """Root-mean-square error ``sqrt(mean((obs - pred) ** 2))``."""
    o, p = _pairwise_finite(obs, pred)
    if not _enough(o.size, min_pairs):
        return float("nan")
    return float(np.sqrt(np.mean((o - p) ** 2)))


def mae(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    *,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> float:
    """Mean absolute error."""
    o, p = _pairwise_finite(obs, pred)
    if not _enough(o.size, min_pairs):
        return float("nan")
    return float(np.mean(np.abs(o - p)))


def bias(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    *,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> float:
    """Mean of ``pred - obs`` (positive → over-prediction)."""
    o, p = _pairwise_finite(obs, pred)
    if not _enough(o.size, min_pairs):
        return float("nan")
    return float(np.mean(p - o))


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------
def pearson_r(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    *,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> float:
    """Pearson correlation coefficient."""
    o, p = _pairwise_finite(obs, pred)
    if not _enough(o.size, min_pairs):
        return float("nan")
    if o.std() == 0.0 or p.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(o, p)[0, 1])


def spearman_r(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    *,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> float:
    """Spearman rank correlation.  Uses :mod:`scipy.stats`.

    Returns NaN if either array has zero variance after the finite
    filter (so a constant prediction doesn't produce ``+0.0`` by
    accident).
    """
    from scipy.stats import spearmanr

    o, p = _pairwise_finite(obs, pred)
    if not _enough(o.size, min_pairs):
        return float("nan")
    if o.std() == 0.0 or p.std() == 0.0:
        return float("nan")
    r, _ = spearmanr(o, p)
    return float(r)


# ---------------------------------------------------------------------------
# Variance-explained & skill
# ---------------------------------------------------------------------------
def r_squared(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    *,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> float:
    """Coefficient of determination ``1 - SS_res / SS_tot``.

    Returns NaN when ``SS_tot == 0`` (constant observations).
    """
    o, p = _pairwise_finite(obs, pred)
    if not _enough(o.size, min_pairs):
        return float("nan")
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    ss_res = float(np.sum((o - p) ** 2))
    return 1.0 - ss_res / ss_tot


def skill_score(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    ref: NDArray[np.floating],
    *,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> float:
    """``1 - MSE_pred / MSE_ref``.  Positive values beat the reference.

    ``ref`` is a baseline prediction (e.g. climatology, persistence).
    NaN entries in any of the three inputs are dropped before the ratio.
    Returns NaN when ``MSE_ref == 0``.
    """
    obs = np.asarray(obs, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    if not (obs.shape == pred.shape == ref.shape):
        raise ValueError(
            f"obs, pred and ref must share shape; got "
            f"{obs.shape!r}, {pred.shape!r}, {ref.shape!r}"
        )
    mask = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(ref)
    if not _enough(int(mask.sum()), min_pairs):
        return float("nan")
    mse_pred = float(np.mean((obs[mask] - pred[mask]) ** 2))
    mse_ref = float(np.mean((obs[mask] - ref[mask]) ** 2))
    if mse_ref == 0.0:
        return float("nan")
    return 1.0 - mse_pred / mse_ref


# ---------------------------------------------------------------------------
# Quantile bias & KS
# ---------------------------------------------------------------------------
def quantile_bias(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    quantile: float,
    *,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> float:
    """``quantile(pred) - quantile(obs)`` at the given quantile fraction.

    ``quantile`` is in ``[0, 1]``.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1]; got {quantile!r}")
    o, p = _pairwise_finite(obs, pred)
    if not _enough(o.size, min_pairs):
        return float("nan")
    return float(np.quantile(p, quantile) - np.quantile(o, quantile))


def ks_statistic(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    *,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> float:
    """Two-sample Kolmogorov–Smirnov distance between obs and pred CDFs."""
    from scipy.stats import ks_2samp

    o, p = _pairwise_finite(obs, pred)
    if not _enough(o.size, min_pairs):
        return float("nan")
    return float(ks_2samp(o, p).statistic)


# ---------------------------------------------------------------------------
# Threshold (categorical) metrics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThresholdMetrics:
    """Four categorical metrics evaluated at a single threshold.

    All values in ``[0, 1]`` except ``frequency_bias`` which is
    unbounded.  ``hit_rate`` is also called POD (probability of
    detection); ``csi`` is the critical success index.
    """
    hit_rate: float
    false_alarm_rate: float
    critical_success_index: float
    frequency_bias: float


def _contingency(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    threshold: float,
) -> tuple[int, int, int, int]:
    """Return ``(hits, misses, false_alarms, correct_negatives)``.

    Assumes inputs are already finite-filtered (caller's responsibility).
    """
    obs_event = obs >= threshold
    pred_event = pred >= threshold
    hits = int(np.sum(obs_event & pred_event))
    misses = int(np.sum(obs_event & ~pred_event))
    false_alarms = int(np.sum(~obs_event & pred_event))
    correct_negatives = int(np.sum(~obs_event & ~pred_event))
    return hits, misses, false_alarms, correct_negatives


def hit_rate(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    threshold: float,
    *,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> float:
    """``hits / (hits + misses)`` — probability of detection."""
    o, p = _pairwise_finite(obs, pred)
    if not _enough(o.size, min_pairs):
        return float("nan")
    h, m, _, _ = _contingency(o, p, threshold)
    denom = h + m
    return float(h) / denom if denom > 0 else float("nan")


def false_alarm_rate(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    threshold: float,
    *,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> float:
    """``false_alarms / (false_alarms + correct_negatives)``."""
    o, p = _pairwise_finite(obs, pred)
    if not _enough(o.size, min_pairs):
        return float("nan")
    _, _, fa, cn = _contingency(o, p, threshold)
    denom = fa + cn
    return float(fa) / denom if denom > 0 else float("nan")


def critical_success_index(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    threshold: float,
    *,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> float:
    """``hits / (hits + misses + false_alarms)``."""
    o, p = _pairwise_finite(obs, pred)
    if not _enough(o.size, min_pairs):
        return float("nan")
    h, m, fa, _ = _contingency(o, p, threshold)
    denom = h + m + fa
    return float(h) / denom if denom > 0 else float("nan")


def frequency_bias(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    threshold: float,
    *,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> float:
    """``(hits + false_alarms) / (hits + misses)``.

    ``> 1`` → model over-forecasts events; ``< 1`` → under-forecasts.
    """
    o, p = _pairwise_finite(obs, pred)
    if not _enough(o.size, min_pairs):
        return float("nan")
    h, m, fa, _ = _contingency(o, p, threshold)
    denom = h + m
    return (h + fa) / denom if denom > 0 else float("nan")


def threshold_metrics(
    obs: NDArray[np.floating],
    pred: NDArray[np.floating],
    threshold: float,
    *,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> ThresholdMetrics:
    """Compute all four categorical metrics in one pass over the arrays."""
    o, p = _pairwise_finite(obs, pred)
    if not _enough(o.size, min_pairs):
        return ThresholdMetrics(
            hit_rate=float("nan"),
            false_alarm_rate=float("nan"),
            critical_success_index=float("nan"),
            frequency_bias=float("nan"),
        )
    h, m, fa, cn = _contingency(o, p, threshold)
    hits_misses = h + m
    hr = float(h) / hits_misses if hits_misses > 0 else float("nan")
    far_denom = fa + cn
    far = float(fa) / far_denom if far_denom > 0 else float("nan")
    csi_denom = h + m + fa
    csi = float(h) / csi_denom if csi_denom > 0 else float("nan")
    fb = (h + fa) / hits_misses if hits_misses > 0 else float("nan")
    return ThresholdMetrics(
        hit_rate=hr,
        false_alarm_rate=far,
        critical_success_index=csi,
        frequency_bias=fb,
    )


#: Re-exported in __init__ so callers can do ``from austrianwindatlas.validation import CorrelationKind``.
CorrelationKind = Literal["pearson", "spearman"]
