"""Unit tests for :mod:`austrianwindatlas.stage2.extrapolator`."""
from __future__ import annotations

import numpy as np
import pytest

from austrianwindatlas.stage2.extrapolator import (
    METHOD_NAMES,
    ExtrapolationConfig,
    MultiHeightExtrapolator,
)


def _tiny_roughness(ny=4, nx=5):
    rng = np.random.default_rng(0)
    # Mix of smooth / moderate / rough cells.
    return rng.choice([0.01, 0.1, 0.5, 1.0], size=(ny, nx))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_from_roughness_builds_consistent_extrapolator():
    raw = _tiny_roughness()
    extr = MultiHeightExtrapolator.from_roughness(raw, target_heights=(100.0,))
    assert extr.roughness.shape == raw.shape
    assert extr.config.target_heights == (100.0,)


def test_config_default_heights():
    cfg = ExtrapolationConfig()
    assert len(cfg.target_heights) >= 5
    # Paper's expected floor is 80 m (lowest commercial hub height).
    assert cfg.target_heights[0] >= 80.0


def test_from_roughness_accepts_explicit_config():
    raw = _tiny_roughness()
    cfg = ExtrapolationConfig(target_heights=(120.0, 200.0), compute_ensemble=False)
    extr = MultiHeightExtrapolator.from_roughness(raw, config=cfg)
    assert extr.config.target_heights == (120.0, 200.0)
    assert extr.config.compute_ensemble is False


# ---------------------------------------------------------------------------
# Shape handling
# ---------------------------------------------------------------------------
def test_extrapolate_rejects_2d_input():
    extr = MultiHeightExtrapolator.from_roughness(_tiny_roughness())
    with pytest.raises(ValueError, match="3-D"):
        extr.extrapolate(np.ones((4, 5)))


def test_extrapolate_rejects_shape_mismatch():
    extr = MultiHeightExtrapolator.from_roughness(_tiny_roughness(ny=4, nx=5))
    with pytest.raises(ValueError, match="spatial shape"):
        extr.extrapolate(np.ones((3, 5, 4)))


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------
def test_extrapolate_returns_all_methods_for_each_height():
    raw = _tiny_roughness()
    heights = (100.0, 150.0)
    extr = MultiHeightExtrapolator.from_roughness(raw, target_heights=heights)
    wind = np.ones((3, *raw.shape)) * 7.0
    out = extr.extrapolate(wind)

    assert set(out.keys()) == set(heights)
    expected_methods = set(METHOD_NAMES)
    for h in heights:
        assert set(out[h].keys()) == expected_methods
        for arr in out[h].values():
            assert arr.shape == wind.shape
            assert arr.dtype == np.float32


def test_extrapolate_respects_disabled_methods():
    raw = _tiny_roughness()
    cfg = ExtrapolationConfig(
        target_heights=(100.0,),
        use_classic_power_law=False,
        use_stability_dependent=False,
        use_hybrid=False,
        compute_ensemble=False,
    )
    extr = MultiHeightExtrapolator.from_roughness(raw, config=cfg)
    wind = np.ones((2, *raw.shape)) * 5.0
    out = extr.extrapolate(wind)[100.0]
    assert "power_law" not in out
    assert "stability" not in out
    assert "hybrid" not in out
    assert "ensemble" not in out
    # Enabled methods still land.
    assert "log_law" in out
    assert "enhanced_log" in out


def test_extrapolate_values_match_formulations():
    """End-to-end: the batcher's power_law output must equal the
    standalone :func:`classic_power_law`.
    """
    from austrianwindatlas.stage2.formulations import classic_power_law

    raw = _tiny_roughness()
    extr = MultiHeightExtrapolator.from_roughness(
        raw, target_heights=(100.0,),
        config=ExtrapolationConfig(
            target_heights=(100.0,), compute_ensemble=False,
            use_classic_power_law=True, use_classic_log_law=False,
            use_enhanced_power_law=False, use_enhanced_log_law=False,
            use_stability_dependent=False, use_roughness_adaptive=False,
            use_hybrid=False,
        ),
    )
    wind = np.ones((2, *raw.shape)) * 6.0
    out = extr.extrapolate(wind)[100.0]
    expected = classic_power_law(
        np.where(np.isfinite(wind), wind, np.nan), 100.0,
    )
    np.testing.assert_allclose(out["power_law"], expected, rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------
def test_qc_nans_propagate_through_classic_methods():
    raw = _tiny_roughness()
    extr = MultiHeightExtrapolator.from_roughness(raw, target_heights=(100.0,))
    wind = np.ones((1, *raw.shape)) * 5.0
    wind[0, 0, 0] = np.nan
    out = extr.extrapolate(wind)[100.0]
    assert np.isnan(out["power_law"][0, 0, 0])
    # Non-NaN cells still produce finite output.
    assert np.isfinite(out["power_law"][0, 1, 1])


def test_qc_infs_become_nan():
    raw = _tiny_roughness()
    extr = MultiHeightExtrapolator.from_roughness(raw, target_heights=(100.0,))
    wind = np.ones((1, *raw.shape)) * 5.0
    wind[0, 1, 2] = np.inf
    out = extr.extrapolate(wind)[100.0]
    assert np.isnan(out["power_law"][0, 1, 2])


def test_qc_over_max_becomes_nan():
    raw = _tiny_roughness()
    cfg = ExtrapolationConfig(target_heights=(100.0,), max_wind_speed=20.0)
    extr = MultiHeightExtrapolator.from_roughness(raw, config=cfg)
    wind = np.ones((1, *raw.shape)) * 5.0
    wind[0, 0, 0] = 100.0           # above max_wind_speed → NaN'd
    out = extr.extrapolate(wind)[100.0]
    assert np.isnan(out["power_law"][0, 0, 0])


def test_qc_tiny_positive_clamped_to_floor():
    raw = _tiny_roughness()
    cfg = ExtrapolationConfig(target_heights=(100.0,), min_wind_speed=0.05)
    extr = MultiHeightExtrapolator.from_roughness(raw, config=cfg)
    wind = np.ones((1, *raw.shape)) * 5.0
    wind[0, 0, 0] = 0.01            # below min_wind_speed → clamped
    out = extr.extrapolate(wind)[100.0]
    # Classic power law: 0.05 * (10)^0.143 ≈ 0.07 — strictly above 0.05.
    assert out["power_law"][0, 0, 0] > 0.05


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
def test_ensemble_is_mean_of_individual_methods():
    raw = _tiny_roughness()
    extr = MultiHeightExtrapolator.from_roughness(raw, target_heights=(100.0,))
    wind = np.ones((1, *raw.shape)) * 5.0
    out = extr.extrapolate(wind)[100.0]
    # Build the expected ensemble from the methods that actually ran.
    method_arrays = [v for k, v in out.items() if k != "ensemble"]
    expected = np.nanmean(np.stack(method_arrays, axis=0), axis=0)
    np.testing.assert_allclose(out["ensemble"], expected.astype(np.float32),
                               rtol=0, atol=1e-5)


def test_ensemble_skipped_when_only_one_method():
    raw = _tiny_roughness()
    cfg = ExtrapolationConfig(
        target_heights=(100.0,),
        use_classic_power_law=True,
        use_classic_log_law=False,
        use_enhanced_power_law=False,
        use_enhanced_log_law=False,
        use_stability_dependent=False,
        use_roughness_adaptive=False,
        use_hybrid=False,
        compute_ensemble=True,
    )
    extr = MultiHeightExtrapolator.from_roughness(raw, config=cfg)
    wind = np.ones((1, *raw.shape)) * 5.0
    out = extr.extrapolate(wind)[100.0]
    assert "ensemble" not in out
