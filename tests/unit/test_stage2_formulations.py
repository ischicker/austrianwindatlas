"""Unit tests for :mod:`austrianwindatlas.stage2.formulations`."""
from __future__ import annotations

import numpy as np
import pytest

from austrianwindatlas.stage2.formulations import (
    DEFAULT_CLASSIC_ALPHA,
    DEFAULT_CLASSIC_Z0,
    DEFAULT_REFERENCE_HEIGHT,
    classic_log_law,
    classic_power_law,
    enhanced_log_law,
    enhanced_power_law,
    hybrid_power_log,
    roughness_adaptive,
    stability_dependent,
)


# ---------------------------------------------------------------------------
# Classic power law
# ---------------------------------------------------------------------------
def test_classic_power_law_identity_at_reference_height():
    wind = np.ones((2, 3, 4)) * 7.5
    out = classic_power_law(wind, target_height=DEFAULT_REFERENCE_HEIGHT)
    np.testing.assert_allclose(out, wind, rtol=0, atol=1e-5)


def test_classic_power_law_matches_formula():
    wind = np.ones((1, 1, 1)) * 5.0
    out = classic_power_law(wind, target_height=100.0, alpha=0.143)
    expected = 5.0 * (100.0 / 10.0) ** 0.143
    assert out[0, 0, 0] == pytest.approx(expected, rel=1e-5)


def test_classic_power_law_clips_at_max_wind():
    wind = np.ones((1, 1, 1)) * 1000.0
    out = classic_power_law(wind, target_height=100.0, max_wind=50.0)
    assert out.max() == 50.0


def test_classic_power_law_returns_float32():
    out = classic_power_law(np.array([[1.0]]), target_height=50.0)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Classic log law
# ---------------------------------------------------------------------------
def test_classic_log_law_identity_at_reference_height():
    wind = np.ones((2, 3)) * 4.0
    out = classic_log_law(wind, target_height=DEFAULT_REFERENCE_HEIGHT, z0=0.1)
    np.testing.assert_allclose(out, wind, rtol=0, atol=1e-5)


def test_classic_log_law_matches_formula():
    out = classic_log_law(np.ones((1,)) * 5.0, target_height=100.0, z0=0.1)
    expected = 5.0 * np.log(100.0 / 0.1) / np.log(10.0 / 0.1)
    assert out[0] == pytest.approx(expected, rel=1e-5)


def test_classic_log_law_grows_with_height():
    out_50 = classic_log_law(np.array([5.0]), 50.0, z0=DEFAULT_CLASSIC_Z0)
    out_100 = classic_log_law(np.array([5.0]), 100.0, z0=DEFAULT_CLASSIC_Z0)
    assert out_100 > out_50


# ---------------------------------------------------------------------------
# Enhanced power law
# ---------------------------------------------------------------------------
def test_enhanced_power_law_with_scalar_alpha_matches_classic():
    rng = np.random.default_rng(0)
    wind = rng.uniform(1, 10, size=(4, 3, 5))
    alpha = np.full((3, 5), DEFAULT_CLASSIC_ALPHA)
    enh = enhanced_power_law(wind, 100.0, alpha)
    cls = classic_power_law(wind, 100.0, alpha=DEFAULT_CLASSIC_ALPHA)
    np.testing.assert_allclose(enh, cls, rtol=0, atol=1e-5)


def test_enhanced_power_law_higher_alpha_gives_stronger_amplification():
    wind = np.ones((1, 2, 2)) * 5.0
    low = enhanced_power_law(wind, 150.0, np.full((2, 2), 0.1))
    high = enhanced_power_law(wind, 150.0, np.full((2, 2), 0.3))
    assert high.min() > low.max()


# ---------------------------------------------------------------------------
# Enhanced log law
# ---------------------------------------------------------------------------
def test_enhanced_log_law_matches_classic_when_fields_uniform():
    rng = np.random.default_rng(1)
    wind = rng.uniform(1, 10, size=(2, 3, 4))
    z0 = np.full((3, 4), 0.1)
    h_ref = np.full((3, 4), 10.0)
    enh = enhanced_log_law(wind, 100.0, z0, h_ref)
    cls = classic_log_law(wind, 100.0, z0=0.1)
    np.testing.assert_allclose(enh, cls, rtol=0, atol=1e-4)


def test_enhanced_log_law_clips_coefficient():
    """Test that extreme z0 / h_ref inputs don't produce huge coefficients."""
    wind = np.ones((1, 2, 2)) * 5.0
    # Near-singular combo: z0 ≈ h_ref → log(h_ref/z0) ≈ 0 → coeff explodes.
    z0 = np.full((2, 2), 9.0)
    h_ref = np.full((2, 2), 9.5)
    out = enhanced_log_law(wind, 100.0, z0, h_ref, min_coeff=0.5, max_coeff=3.0)
    # Clipped by max_coeff * wind.
    assert out.max() <= 3.0 * 5.0 + 1e-3


def test_enhanced_log_law_handles_target_below_2z0_floor():
    """target_height < 2*z0 at some cells → floored to 2*z0."""
    wind = np.ones((1, 1, 1)) * 5.0
    z0 = np.full((1, 1), 20.0)
    h_ref = np.full((1, 1), 50.0)
    # target=10 m < 2*z0=40 m → h_target = 40.
    out = enhanced_log_law(wind, 10.0, z0, h_ref)
    expected_coeff = np.log(40.0 / 20.0) / np.log(50.0 / 20.0)
    assert out[0, 0, 0] == pytest.approx(5.0 * expected_coeff, rel=1e-5)


# ---------------------------------------------------------------------------
# Stability-dependent
# ---------------------------------------------------------------------------
def test_stability_dependent_weak_winds_see_larger_alpha():
    weak = np.ones((1, 1, 1)) * 2.0
    strong = np.ones((1, 1, 1)) * 15.0
    out_weak = stability_dependent(weak, 100.0)
    out_strong = stability_dependent(strong, 100.0)
    # Ratios: out / wind = (h/h_ref)^alpha_stab.  Larger ratio → larger alpha_stab.
    ratio_weak = out_weak[0, 0, 0] / 2.0
    ratio_strong = out_strong[0, 0, 0] / 15.0
    assert ratio_weak > ratio_strong


def test_stability_dependent_alpha_stays_in_band():
    rng = np.random.default_rng(0)
    wind = rng.uniform(0, 30, size=(5, 4, 4))
    out = stability_dependent(wind, 100.0, alpha_min=0.1, alpha_max=0.25)
    ratio = 100.0 / 10.0
    # Per-cell effective alpha = log(out/wind) / log(10) — must stay in band.
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha_eff = np.log(np.maximum(out / np.maximum(wind, 1e-9), 1e-9)) / np.log(ratio)
    alpha_valid = alpha_eff[np.isfinite(alpha_eff) & (wind > 0.1)]
    assert alpha_valid.min() >= 0.1 - 1e-5
    assert alpha_valid.max() <= 0.25 + 1e-5


# ---------------------------------------------------------------------------
# Roughness-adaptive
# ---------------------------------------------------------------------------
def test_roughness_adaptive_picks_per_cell_method():
    ny, nx = 1, 3
    wind = np.ones((1, ny, nx)) * 10.0
    enhanced_log = np.ones((1, ny, nx)) * 11.0           # smooth regime output
    enhanced_power = np.ones((1, ny, nx)) * 13.0         # moderate regime output
    smooth = np.array([[True, False, False]])
    moderate = np.array([[False, True, False]])
    rough = np.array([[False, False, True]])
    out = roughness_adaptive(
        wind, 100.0,
        enhanced_log=enhanced_log, enhanced_power=enhanced_power,
        smooth_mask=smooth, moderate_mask=moderate, rough_mask=rough,
        rough_alpha=0.25,
    )
    # Cell 0 → enhanced_log = 11.
    assert out[0, 0, 0] == pytest.approx(11.0, rel=1e-5)
    # Cell 1 → enhanced_power = 13.
    assert out[0, 0, 1] == pytest.approx(13.0, rel=1e-5)
    # Cell 2 → wind * (100/10)^0.25 = 10 * 1.778...
    assert out[0, 0, 2] == pytest.approx(10.0 * (10.0 ** 0.25), rel=1e-5)


# ---------------------------------------------------------------------------
# Hybrid
# ---------------------------------------------------------------------------
def test_hybrid_power_log_is_convex_blend():
    wind = np.ones((1, 1, 1)) * 10.0
    el = np.ones((1, 1, 1)) * 20.0
    ep = np.ones((1, 1, 1)) * 10.0
    z0 = np.full((1, 1), 0.1)
    out = hybrid_power_log(wind, enhanced_log=el, enhanced_power=ep, z0=z0)
    # Output must sit between the two inputs.
    assert 10.0 - 1e-5 <= out[0, 0, 0] <= 20.0 + 1e-5


def test_hybrid_weights_shift_with_wind_speed():
    """Stronger winds should move the output towards enhanced_log."""
    el = np.ones((1, 1, 1)) * 20.0
    ep = np.ones((1, 1, 1)) * 10.0
    z0 = np.full((1, 1), 0.1)
    low = hybrid_power_log(np.ones((1, 1, 1)) * 0.0,
                           enhanced_log=el, enhanced_power=ep, z0=z0)
    high = hybrid_power_log(np.ones((1, 1, 1)) * 30.0,
                            enhanced_log=el, enhanced_power=ep, z0=z0)
    # Higher wind → more weight on `el` (20) → higher output.
    assert high[0, 0, 0] > low[0, 0, 0]


def test_hybrid_respects_weight_bounds():
    rng = np.random.default_rng(0)
    wind = rng.uniform(0, 30, size=(3, 2, 2))
    el = rng.uniform(5, 25, size=(3, 2, 2))
    ep = rng.uniform(5, 25, size=(3, 2, 2))
    z0 = np.full((2, 2), 0.01)
    out = hybrid_power_log(wind, enhanced_log=el, enhanced_power=ep, z0=z0,
                           weight_min=0.1, weight_max=0.9)
    lo = np.minimum(el, ep)
    hi = np.maximum(el, ep)
    # With 0.1 ≤ w ≤ 0.9, the output must stay inside [min, max] of the two.
    assert np.all(out >= lo - 1e-4)
    assert np.all(out <= hi + 1e-4)
