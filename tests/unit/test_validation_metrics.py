"""Unit tests for :mod:`austrianwindatlas.validation.metrics`."""
from __future__ import annotations

import numpy as np
import pytest

from austrianwindatlas.validation.metrics import (
    DEFAULT_MIN_PAIRS,
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


# ---------------------------------------------------------------------------
# Basic errors
# ---------------------------------------------------------------------------
def test_rmse_simple_case():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.1, 2.1, 2.9, 4.2])
    expected = float(np.sqrt(np.mean((obs - pred) ** 2)))
    assert rmse(obs, pred, min_pairs=1) == pytest.approx(expected, rel=1e-10)


def test_rmse_ignores_nan_pairs():
    obs = np.array([1.0, np.nan, 3.0, 4.0])
    pred = np.array([1.0, 2.0, np.nan, 4.0])
    # Only (0, 0) and (3, 3) are finite pairs → RMSE = 0.
    assert rmse(obs, pred, min_pairs=1) == 0.0


def test_rmse_returns_nan_below_min_pairs():
    obs = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 3.0])
    assert np.isnan(rmse(obs, pred, min_pairs=10))


def test_rmse_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="share shape"):
        rmse(np.zeros(5), np.zeros(6))


def test_mae_simple_case():
    obs = np.array([0.0, 1.0, 2.0])
    pred = np.array([1.0, 1.0, 4.0])
    # |diffs| = [1, 0, 2] → mean 1.0.
    assert mae(obs, pred, min_pairs=1) == pytest.approx(1.0)


def test_bias_sign_convention():
    obs = np.ones(20)
    pred = np.ones(20) * 1.5
    # pred > obs → positive bias.
    assert bias(obs, pred) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------
def test_pearson_r_perfect_correlation():
    obs = np.linspace(0, 10, 50)
    assert pearson_r(obs, obs * 2 + 3) == pytest.approx(1.0)


def test_pearson_r_anticorrelation():
    obs = np.linspace(0, 10, 50)
    assert pearson_r(obs, -obs) == pytest.approx(-1.0)


def test_pearson_r_zero_variance_returns_nan():
    obs = np.linspace(0, 10, 50)
    pred = np.full_like(obs, 5.0)
    assert np.isnan(pearson_r(obs, pred))


def test_spearman_r_handles_monotone_nonlinear():
    obs = np.linspace(0, 10, 50)
    pred = obs ** 3           # monotone but not linear
    # Spearman = 1 for monotone-increasing, Pearson < 1.
    assert spearman_r(obs, pred) == pytest.approx(1.0, abs=1e-10)
    assert pearson_r(obs, pred) < 0.99


# ---------------------------------------------------------------------------
# R² and skill score
# ---------------------------------------------------------------------------
def test_r_squared_perfect_prediction():
    obs = np.random.default_rng(0).standard_normal(100)
    assert r_squared(obs, obs) == pytest.approx(1.0)


def test_r_squared_constant_obs_returns_nan():
    obs = np.ones(50)
    pred = np.linspace(0, 1, 50)
    # SS_tot = 0 → undefined.
    assert np.isnan(r_squared(obs, pred))


def test_skill_score_zero_when_pred_equals_ref():
    rng = np.random.default_rng(0)
    obs = rng.standard_normal(50)
    pred = rng.standard_normal(50)
    assert skill_score(obs, pred, pred) == 0.0


def test_skill_score_positive_when_pred_better_than_ref():
    rng = np.random.default_rng(1)
    obs = rng.standard_normal(100)
    pred = obs + 0.1 * rng.standard_normal(100)        # close to obs
    ref = obs + 2.0 * rng.standard_normal(100)         # far from obs
    assert skill_score(obs, pred, ref) > 0.5


def test_skill_score_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="share shape"):
        skill_score(np.zeros(5), np.zeros(5), np.zeros(6))


# ---------------------------------------------------------------------------
# Quantile bias
# ---------------------------------------------------------------------------
def test_quantile_bias_matches_formula():
    obs = np.linspace(0, 100, 1000)
    pred = obs + 5.0                                # uniformly over-predict
    # At any quantile, bias ≈ +5.
    assert quantile_bias(obs, pred, 0.5) == pytest.approx(5.0, rel=1e-6)
    assert quantile_bias(obs, pred, 0.99) == pytest.approx(5.0, rel=1e-6)


def test_quantile_bias_rejects_out_of_range():
    with pytest.raises(ValueError, match="quantile"):
        quantile_bias(np.zeros(100), np.zeros(100), 1.5)


# ---------------------------------------------------------------------------
# KS
# ---------------------------------------------------------------------------
def test_ks_statistic_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    obs = rng.standard_normal(5_000)
    assert ks_statistic(obs, obs) == 0.0


def test_ks_statistic_positive_for_shifted_distribution():
    rng = np.random.default_rng(0)
    obs = rng.standard_normal(1_000)
    pred = obs + 1.0                                # mean-shifted
    assert ks_statistic(obs, pred) > 0.2


# ---------------------------------------------------------------------------
# Threshold metrics
# ---------------------------------------------------------------------------
def test_threshold_metrics_perfect_forecast():
    rng = np.random.default_rng(0)
    obs = rng.uniform(0, 20, size=500)
    tm = threshold_metrics(obs, obs, threshold=10.0)
    assert isinstance(tm, ThresholdMetrics)
    assert tm.hit_rate == 1.0
    assert tm.false_alarm_rate == 0.0
    assert tm.critical_success_index == 1.0
    assert tm.frequency_bias == pytest.approx(1.0)


def test_threshold_metrics_zero_events():
    """No observed events → hit_rate is NaN, freq_bias is NaN."""
    obs = np.ones(200) * 2.0
    pred = np.ones(200) * 2.0
    tm = threshold_metrics(obs, pred, threshold=10.0)
    assert np.isnan(tm.hit_rate)
    assert np.isnan(tm.frequency_bias)
    # But far / csi are well-defined (0 and NaN respectively).
    assert tm.false_alarm_rate == 0.0


def test_hit_rate_matches_individual_call():
    rng = np.random.default_rng(0)
    obs = rng.uniform(0, 20, size=500)
    pred = obs + rng.standard_normal(500) * 2.0
    tm = threshold_metrics(obs, pred, 10.0)
    assert hit_rate(obs, pred, 10.0) == pytest.approx(tm.hit_rate)


def test_false_alarm_rate_matches_individual_call():
    rng = np.random.default_rng(1)
    obs = rng.uniform(0, 20, size=500)
    pred = obs + rng.standard_normal(500) * 2.0
    tm = threshold_metrics(obs, pred, 10.0)
    assert false_alarm_rate(obs, pred, 10.0) == pytest.approx(tm.false_alarm_rate)


def test_csi_matches_individual_call():
    rng = np.random.default_rng(2)
    obs = rng.uniform(0, 20, size=500)
    pred = obs + rng.standard_normal(500) * 2.0
    tm = threshold_metrics(obs, pred, 10.0)
    assert critical_success_index(obs, pred, 10.0) == pytest.approx(tm.critical_success_index)


def test_frequency_bias_matches_individual_call():
    rng = np.random.default_rng(3)
    obs = rng.uniform(0, 20, size=500)
    pred = obs + rng.standard_normal(500) * 2.0
    tm = threshold_metrics(obs, pred, 10.0)
    assert frequency_bias(obs, pred, 10.0) == pytest.approx(tm.frequency_bias)


# ---------------------------------------------------------------------------
# Minimum-pairs gating
# ---------------------------------------------------------------------------
def test_short_arrays_return_nan_for_every_metric():
    tiny = np.array([1.0, 2.0])
    pred = np.array([1.1, 2.0])
    assert np.isnan(rmse(tiny, pred, min_pairs=DEFAULT_MIN_PAIRS))
    assert np.isnan(pearson_r(tiny, pred, min_pairs=DEFAULT_MIN_PAIRS))
    assert np.isnan(quantile_bias(tiny, pred, 0.5, min_pairs=DEFAULT_MIN_PAIRS))
    tm = threshold_metrics(tiny, pred, 5.0, min_pairs=DEFAULT_MIN_PAIRS)
    assert np.isnan(tm.hit_rate)
