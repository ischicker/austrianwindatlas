"""Unit tests for :mod:`austrianwindatlas.validation.aggregate`."""
from __future__ import annotations

import numpy as np
import pytest

from austrianwindatlas.validation.aggregate import (
    AggregatedMetrics,
    StationMetrics,
    aggregate_across_stations,
    station_metric_field_names,
    station_metrics,
)


# ---------------------------------------------------------------------------
# station_metrics
# ---------------------------------------------------------------------------
def test_station_metrics_populates_all_expected_fields():
    rng = np.random.default_rng(0)
    obs = rng.uniform(0, 20, size=1_000)
    pred = obs + rng.standard_normal(1_000) * 0.5
    res = station_metrics(obs, pred, station_id=42)
    assert isinstance(res, StationMetrics)
    assert res.station_id == 42
    assert res.n_pairs == 1_000
    for name in ("rmse", "mae", "bias", "pearson_r", "spearman_r", "r_squared",
                 "ks_statistic", "skill_vs_climatology", "skill_vs_persistence"):
        assert np.isfinite(getattr(res, name)), name
    # Quantile biases on the default quantiles.
    for q in (50, 90, 99):
        assert np.isfinite(getattr(res, f"quantile_bias_q{q}"))
    # Threshold metrics at each of 5, 10, 15.
    for t in (5, 10, 15):
        for family in ("hit_rate", "false_alarm_rate",
                       "critical_success_index", "frequency_bias"):
            # Values may be NaN if the threshold produced no events, but
            # the attribute must at least exist.
            assert hasattr(res, f"{family}_{t}"), f"missing {family}_{t}"


def test_station_metrics_returns_nans_when_too_few_pairs():
    obs = np.array([1.0, 2.0])
    pred = np.array([1.0, 2.0])
    res = station_metrics(obs, pred, min_pairs=10)
    assert res.n_pairs == 2
    assert np.isnan(res.rmse)
    assert np.isnan(res.pearson_r)


def test_station_metrics_perfect_prediction():
    obs = np.linspace(0, 10, 500)
    res = station_metrics(obs, obs)
    assert res.rmse == pytest.approx(0.0, abs=1e-10)
    assert res.pearson_r == pytest.approx(1.0)
    assert res.r_squared == pytest.approx(1.0)


def test_station_metrics_custom_thresholds_populate_dynamic_fields():
    """Pass a non-default threshold and verify the rounded-key field lands."""
    obs = np.linspace(0, 20, 500)
    pred = obs.copy()
    res = station_metrics(obs, pred, thresholds=(7.0,), quantiles=(0.5,))
    # Threshold 7 → integer suffix 7.
    assert res.hit_rate_7 == pytest.approx(1.0)
    assert res.critical_success_index_7 == pytest.approx(1.0)


def test_station_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="share shape"):
        station_metrics(np.zeros(10), np.zeros(5))


def test_station_metrics_asdict_roundtrips_key_fields():
    obs = np.linspace(0, 10, 500)
    res = station_metrics(obs, obs, station_id="S01")
    data = res.asdict()
    assert data["station_id"] == "S01"
    assert data["rmse"] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# aggregate_across_stations
# ---------------------------------------------------------------------------
def _build_station_list(n_stations=5, n_times=500, seed=0):
    rng = np.random.default_rng(seed)
    stations = []
    pooled_obs_parts, pooled_pred_parts = [], []
    for s in range(n_stations):
        obs = rng.uniform(0, 20, size=n_times)
        pred = obs + rng.standard_normal(n_times) * (0.3 + 0.1 * s)
        stations.append(station_metrics(obs, pred, station_id=s))
        pooled_obs_parts.append(obs)
        pooled_pred_parts.append(pred)
    return stations, np.concatenate(pooled_obs_parts), np.concatenate(pooled_pred_parts)


def test_aggregate_handles_empty_input():
    agg = aggregate_across_stations([], method_name="empty")
    assert isinstance(agg, AggregatedMetrics)
    assert agg.n_stations == 0
    assert agg.n_pairs == 0
    assert np.isnan(agg.mean_rmse)


def test_aggregate_fills_mean_median_std_rmse():
    stations, _, _ = _build_station_list()
    agg = aggregate_across_stations(stations, method_name="test")
    assert agg.n_stations == len(stations)
    assert agg.n_pairs == sum(s.n_pairs for s in stations)
    # mean / median / std are all finite.
    assert np.isfinite(agg.mean_rmse)
    assert np.isfinite(agg.median_rmse)
    assert np.isfinite(agg.std_rmse)
    # std across the per-station rmses should be > 0 by construction.
    assert agg.std_rmse > 0.0


def test_aggregate_pooled_only_when_arrays_given():
    stations, obs_pool, pred_pool = _build_station_list()
    agg_no = aggregate_across_stations(stations, method_name="no_pool")
    agg_yes = aggregate_across_stations(
        stations, method_name="with_pool",
        pooled_obs=obs_pool, pooled_pred=pred_pool,
    )
    assert np.isnan(agg_no.pooled_rmse)
    assert np.isfinite(agg_yes.pooled_rmse)
    assert np.isfinite(agg_yes.pooled_pearson_r)


def test_aggregate_mean_rmse_matches_direct_mean():
    stations, _, _ = _build_station_list()
    expected = float(np.mean([s.rmse for s in stations]))
    agg = aggregate_across_stations(stations, method_name="x")
    assert agg.mean_rmse == pytest.approx(expected, rel=1e-10)


def test_aggregate_threshold_metrics_land_in_output():
    stations, _, _ = _build_station_list()
    agg = aggregate_across_stations(stations, method_name="x")
    # At least one threshold key per family should be non-NaN given our synthetic data.
    assert np.isfinite(agg.mean_critical_success_index_5) or \
           np.isfinite(agg.mean_critical_success_index_10)


def test_aggregate_ignores_nan_rmse_in_mean():
    """A station with all-NaN RMSE must not NaN the whole average."""
    good = _build_station_list(n_stations=3)[0]
    nan_station = StationMetrics(station_id="broken", n_pairs=0)      # all NaN fields
    agg = aggregate_across_stations([*good, nan_station], method_name="mixed")
    assert np.isfinite(agg.mean_rmse)


def test_station_metric_field_names_covers_core_fields():
    names = station_metric_field_names()
    # Spot-check that the set includes the key metric columns.
    for required in ("rmse", "mae", "bias", "pearson_r", "spearman_r", "r_squared",
                     "ks_statistic", "skill_vs_climatology", "hit_rate_5",
                     "critical_success_index_15"):
        assert required in names
