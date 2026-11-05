"""Unit tests for :mod:`austrianwindatlas.validation.energy`."""
from __future__ import annotations

from math import gamma

import numpy as np
import pandas as pd
import pytest

from austrianwindatlas.validation.energy import (
    DEFAULT_AIR_DENSITY,
    RampEvent,
    WeibullParameters,
    capacity_factor,
    detect_ramp_events,
    fit_weibull,
    wind_power_density,
)


# ---------------------------------------------------------------------------
# fit_weibull
# ---------------------------------------------------------------------------
def test_fit_weibull_recovers_known_parameters():
    """Sample from a Weibull(k=2, A=8), fit, compare."""
    rng = np.random.default_rng(0)
    k_true, A_true = 2.0, 8.0
    ws = A_true * rng.weibull(k_true, size=10_000)
    params = fit_weibull(ws)
    assert isinstance(params, WeibullParameters)
    # Moment-match is approximate; aim for within 5 %.
    assert params.k == pytest.approx(k_true, rel=0.1)
    assert pytest.approx(A_true, rel=0.05) == params.A


def test_fit_weibull_returns_nan_on_short_input():
    out = fit_weibull(np.array([1.0, 2.0, 3.0]))
    assert np.isnan(out.k)
    assert np.isnan(out.A)


def test_fit_weibull_ignores_nonpositive_and_nan():
    rng = np.random.default_rng(0)
    ws = rng.weibull(2.0, size=1_000) * 5.0
    # Inject NaN and non-positive values.
    ws[::7] = np.nan
    ws[::13] = -1.0
    ws[::17] = 0.0
    out = fit_weibull(ws)
    assert np.isfinite(out.k) and np.isfinite(out.A)


def test_fit_weibull_k_respects_clip_bounds():
    # Nearly-constant wind → std/mean tiny → k shoots up; clip to 4.
    ws = np.full(1_000, 5.0) + np.random.default_rng(0).normal(0, 0.01, 1_000)
    assert fit_weibull(ws).k == pytest.approx(4.0, abs=1e-6)


# ---------------------------------------------------------------------------
# wind_power_density
# ---------------------------------------------------------------------------
def test_wind_power_density_matches_formula():
    ws = np.array([5.0, 8.0, 11.0, 14.0, 6.0, 7.0, 9.0, 10.0, 12.0, 13.0])
    expected = 0.5 * DEFAULT_AIR_DENSITY * float(np.mean(ws ** 3))
    assert wind_power_density(ws) == pytest.approx(expected, rel=1e-10)


def test_wind_power_density_custom_rho():
    ws = np.ones(100) * 10.0
    # At rho=1 and ws=10 everywhere → 0.5 * 10³ = 500.
    assert wind_power_density(ws, rho=1.0) == pytest.approx(500.0, rel=1e-10)


def test_wind_power_density_nan_on_short_input():
    assert np.isnan(wind_power_density(np.array([5.0])))


# ---------------------------------------------------------------------------
# capacity_factor
# ---------------------------------------------------------------------------
def test_capacity_factor_at_rated_is_one():
    ws = np.full(500, 15.0)       # above rated=12, below cut_out=25
    assert capacity_factor(ws) == pytest.approx(1.0)


def test_capacity_factor_below_cut_in_is_zero():
    ws = np.full(500, 1.0)
    assert capacity_factor(ws) == 0.0


def test_capacity_factor_above_cut_out_is_zero():
    ws = np.full(500, 30.0)
    assert capacity_factor(ws) == 0.0


def test_capacity_factor_half_rated_matches_cubic_curve():
    """At ws = (cut_in + rated) / 2 = 7.5, power = 0.5³ = 0.125."""
    ws = np.full(500, 7.5)
    expected = ((7.5 - 3.0) / (12.0 - 3.0)) ** 3
    assert capacity_factor(ws) == pytest.approx(expected, rel=1e-10)


def test_capacity_factor_rejects_invalid_power_curve():
    with pytest.raises(ValueError, match="invalid power curve"):
        capacity_factor(np.ones(200), cut_in=12.0, rated=3.0, cut_out=25.0)


def test_capacity_factor_handles_nan():
    rng = np.random.default_rng(0)
    ws = rng.uniform(0, 25, size=1_000)
    ws[::5] = np.nan
    out = capacity_factor(ws)
    assert 0.0 <= out <= 1.0


# ---------------------------------------------------------------------------
# detect_ramp_events
# ---------------------------------------------------------------------------
def test_detect_ramp_events_finds_known_ramps():
    times = pd.date_range("2020-01-01", periods=20, freq="h")
    ws = np.ones(20) * 3.0
    ws[5:8] = [3.0, 6.0, 10.0]           # ramp up around index 5-7
    ws[12:15] = [10.0, 4.0, 2.0]         # ramp down around index 12-14

    events = detect_ramp_events(ws, times, threshold=5.0, window_hours=2)
    assert any(ev.kind == "ramp_up" for ev in events)
    assert any(ev.kind == "ramp_down" for ev in events)


def test_detect_ramp_events_respects_threshold():
    times = pd.date_range("2020-01-01", periods=10, freq="h")
    ws = np.linspace(0, 10, 10)           # smooth climb
    # delta over any 3-hour window is ~3.33 — below a threshold of 5.
    events = detect_ramp_events(ws, times, threshold=5.0, window_hours=3)
    assert events == []


def test_detect_ramp_events_short_input_returns_empty():
    times = pd.date_range("2020-01-01", periods=3, freq="h")
    ws = np.array([1.0, 2.0, 10.0])
    assert detect_ramp_events(ws, times, window_hours=3) == []


def test_detect_ramp_events_skips_nan_windows():
    times = pd.date_range("2020-01-01", periods=5, freq="h")
    ws = np.array([1.0, np.nan, 2.0, 3.0, 10.0])
    events = detect_ramp_events(ws, times, threshold=5.0, window_hours=4)
    # Only window (0 → 4) is end-finite but has start = 1 (finite), delta = 9 → ramp_up.
    assert len(events) == 1
    assert events[0].kind == "ramp_up"


def test_detect_ramp_events_fields_populated_correctly():
    times = np.arange(10)
    ws = np.array([0, 0, 0, 10, 10, 10, 10, 10, 10, 10], dtype=float)
    events = detect_ramp_events(ws, times, threshold=5.0, window_hours=3)
    # First ramp: window (0 → 3), delta = 10, kind = ramp_up.
    ev = events[0]
    assert isinstance(ev, RampEvent)
    assert ev.start_index == 0
    assert ev.end_index == 3
    assert ev.start_wind == 0.0
    assert ev.end_wind == 10.0
    assert ev.delta == 10.0
    assert ev.kind == "ramp_up"


def test_detect_ramp_events_rejects_bad_window():
    with pytest.raises(ValueError, match="window_hours"):
        detect_ramp_events(np.zeros(10), np.arange(10), window_hours=0)


def test_detect_ramp_events_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="share length"):
        detect_ramp_events(np.zeros(10), np.arange(5), window_hours=3)


# ---------------------------------------------------------------------------
# gamma function sanity (sanity test of the imported math.gamma)
# ---------------------------------------------------------------------------
def test_weibull_scale_formula_matches_gamma():
    """For k=2, A = mean / gamma(1 + 1/k) = mean / gamma(1.5) ≈ mean / 0.8862."""
    rng = np.random.default_rng(0)
    ws = 10.0 * rng.weibull(2.0, size=20_000)
    out = fit_weibull(ws)
    expected_A = float(np.mean(ws[ws > 0])) / gamma(1.5)
    assert pytest.approx(expected_A, rel=0.05) == out.A
