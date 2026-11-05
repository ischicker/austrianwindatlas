"""Unit tests for :mod:`austrianwindatlas.stage2.roughness`."""
from __future__ import annotations

import numpy as np
import pytest

from austrianwindatlas.stage2.roughness import (
    DEFAULT_Z0_FILL,
    DEFAULT_Z0_MAX,
    DEFAULT_Z0_MIN,
    RoughnessFields,
    clipped_roughness,
    derive_roughness_fields,
    log_law_reference_height,
    roughness_alpha,
    roughness_regime_masks,
)


# ---------------------------------------------------------------------------
# clipped_roughness
# ---------------------------------------------------------------------------
def test_clipped_roughness_output_dtype_and_shape():
    raw = np.array([[0.5, 1.0], [0.1, 0.01]])
    out = clipped_roughness(raw)
    assert out.shape == (2, 2)
    assert out.dtype == np.float32


def test_clipped_roughness_clamps_extremes():
    raw = np.array([[1e-6, 10.0], [100.0, 0.0001]])
    out = clipped_roughness(raw)
    assert out.min() == pytest.approx(DEFAULT_Z0_MIN, rel=1e-5)
    assert out.max() == pytest.approx(DEFAULT_Z0_MAX, rel=1e-5)


def test_clipped_roughness_replaces_nan_and_inf_with_fill():
    raw = np.array([[np.nan, np.inf], [-np.inf, 0.0]])
    out = clipped_roughness(raw)
    # All four get the fill value (then clipped — fill is inside the range).
    assert np.allclose(out, DEFAULT_Z0_FILL)


def test_clipped_roughness_respects_custom_fill():
    raw = np.array([[np.nan, 0.5]])
    out = clipped_roughness(raw, fill=0.8)
    assert out[0, 0] == pytest.approx(0.8, rel=1e-5)


# ---------------------------------------------------------------------------
# roughness_alpha
# ---------------------------------------------------------------------------
def test_roughness_alpha_matches_formula_at_reference_point():
    # z0 = 0.01 → ratio = 1 → log10(1) = 0 → alpha = 0.1.
    out = roughness_alpha(np.array([[0.01]]))
    assert out[0, 0] == pytest.approx(0.1, rel=1e-5)


def test_roughness_alpha_grows_with_z0_in_range():
    z0 = np.array([0.001, 0.01, 0.1, 1.0, 2.0])
    out = roughness_alpha(z0)
    # Strictly increasing where the formula isn't saturated by clip(z0/0.01, 0.1, 100).
    assert np.all(np.diff(out) >= 0)


def test_roughness_alpha_clips_at_bounds():
    out = roughness_alpha(np.array([0.001, 10.0]), alpha_min=0.07, alpha_max=0.25)
    assert out.min() >= 0.07 - 1e-6
    assert out.max() <= 0.25 + 1e-6


# ---------------------------------------------------------------------------
# log_law_reference_height
# ---------------------------------------------------------------------------
def test_log_law_reference_height_min_is_ten():
    # For z0 <= 5, max(10, 2*z0) == 10; for z0 > 5, it's 2*z0.
    out = log_law_reference_height(np.array([0.1, 1.0, 5.0, 8.0]))
    np.testing.assert_allclose(out, [10.0, 10.0, 10.0, 16.0], rtol=1e-6)


def test_log_law_reference_height_scales_with_z0_when_large():
    out = log_law_reference_height(np.array([10.0]), factor=2.0)
    assert out[0] == pytest.approx(20.0, rel=1e-6)


# ---------------------------------------------------------------------------
# roughness_regime_masks
# ---------------------------------------------------------------------------
def test_regime_masks_partition_the_grid():
    z0 = np.array([0.01, 0.1, 0.5, 1.5])
    smooth, moderate, rough = roughness_regime_masks(z0)
    # Each pixel in exactly one mask.
    np.testing.assert_array_equal(smooth.astype(int) + moderate.astype(int) + rough.astype(int),
                                   np.ones_like(z0, dtype=int))
    np.testing.assert_array_equal(smooth, [True, False, False, False])
    np.testing.assert_array_equal(moderate, [False, True, False, False])
    np.testing.assert_array_equal(rough, [False, False, True, True])


def test_regime_masks_respect_custom_thresholds():
    z0 = np.array([0.03, 0.1, 0.5])
    smooth, moderate, rough = roughness_regime_masks(
        z0, smooth_thresh=0.02, rough_thresh=0.3,
    )
    np.testing.assert_array_equal(smooth, [False, False, False])
    np.testing.assert_array_equal(moderate, [True, True, False])
    np.testing.assert_array_equal(rough, [False, False, True])


# ---------------------------------------------------------------------------
# derive_roughness_fields
# ---------------------------------------------------------------------------
def test_derive_roughness_fields_returns_consistent_bundle():
    raw = np.array([[0.005, 0.03], [0.1, 1.5]])
    fields = derive_roughness_fields(raw)
    assert isinstance(fields, RoughnessFields)
    assert fields.z0.shape == raw.shape
    assert fields.alpha.shape == raw.shape
    assert fields.h_ref.shape == raw.shape
    assert fields.smooth.shape == raw.shape
    # Masks still partition.
    assert np.all(fields.smooth | fields.moderate | fields.rough)
    assert not np.any(fields.smooth & fields.moderate)
    # And z0 is in the clip range.
    assert fields.z0.min() >= DEFAULT_Z0_MIN
    assert fields.z0.max() <= DEFAULT_Z0_MAX


def test_derive_roughness_fields_nan_safe():
    raw = np.full((3, 3), np.nan)
    fields = derive_roughness_fields(raw)
    # All pixels use the fill value → all moderate (0.1 falls in [0.05, 0.5)).
    assert np.all(fields.moderate)
    assert np.all(np.isfinite(fields.alpha))
