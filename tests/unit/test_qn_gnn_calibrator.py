"""Unit tests for :mod:`austrianwindatlas.stage1.qn_gnn.calibrator`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from austrianwindatlas.stage1.qn_gnn.calibrator import (
    AsymmetricCalibrator,
    ElevationBiasCorrector,
    SpatialWindCalibrator,
)


def _station_frame(n=20, seed=0):
    rng = np.random.default_rng(seed)
    alts = np.linspace(200.0, 2_000.0, n)
    # Mean wind grows with altitude (physically plausible).
    mean_wind = 2.0 + alts / 500.0
    wind = np.stack([
        rng.normal(mean_wind, 0.5 + alts / 5_000.0) for _ in range(500)
    ], axis=0)
    meta = pd.DataFrame({"alt": alts})
    return meta, wind


# ---------------------------------------------------------------------------
# SpatialWindCalibrator
# ---------------------------------------------------------------------------
def test_spatial_calibrator_grid_transform_shape_and_positive():
    meta, wind = _station_frame()
    cal = SpatialWindCalibrator(meta, wind)
    z = np.zeros((3, 4, 5))                    # zero z → should map to mu(elev)
    elev = np.linspace(500.0, 1_500.0, 20).reshape(4, 5)
    out = cal.transform_grid(z, elev)
    assert out.shape == (3, 4, 5)
    assert out.dtype == np.float32
    assert (out > 0).all()


def test_spatial_calibrator_zero_z_recovers_climatology():
    meta, wind = _station_frame()
    cal = SpatialWindCalibrator(meta, wind)
    # z = 0 → wind = mu(elev); test at a station's own altitude.
    z = np.zeros((1, 1, len(meta)))
    elev = meta["alt"].to_numpy().reshape(1, len(meta))
    out = cal.transform_grid(z, elev)
    mu_expected = np.nanmean(wind, axis=0)
    np.testing.assert_allclose(out[0, 0], mu_expected, rtol=0.02, atol=0.1)


def test_spatial_calibrator_clips_absurd_z():
    meta, wind = _station_frame()
    cal = SpatialWindCalibrator(meta, wind)
    z = np.ones((1, 2, 3)) * 100.0             # wildly out of distribution
    elev = np.ones((2, 3)) * 500.0
    out = cal.transform_grid(z, elev, clip_sigma=5.0)
    # Must stay below the 1.5 × global_max cap.
    assert out.max() < cal.global_max * 1.5 + 1e-4


def test_spatial_calibrator_station_transform():
    meta, wind = _station_frame()
    cal = SpatialWindCalibrator(meta, wind)
    z = np.zeros((2, len(meta)))
    out = cal.transform_stations(z, meta["alt"].to_numpy())
    mu_expected = np.nanmean(wind, axis=0)
    np.testing.assert_allclose(out[0], mu_expected, rtol=0.02, atol=0.1)


def test_spatial_calibrator_rejects_elevation_shape_mismatch():
    meta, wind = _station_frame()
    cal = SpatialWindCalibrator(meta, wind)
    with pytest.raises(ValueError, match="elevation shape"):
        cal.transform_grid(np.zeros((1, 2, 3)), np.zeros((4, 4)))


def test_spatial_calibrator_station_transform_rejects_length_mismatch():
    meta, wind = _station_frame()
    cal = SpatialWindCalibrator(meta, wind)
    with pytest.raises(ValueError, match="station_alts"):
        cal.transform_stations(np.zeros((2, 10)), np.zeros(5))


def test_spatial_calibrator_handles_all_nan_input():
    meta = pd.DataFrame({"alt": [500.0, 1_000.0]})
    wind = np.full((20, 2), np.nan)
    cal = SpatialWindCalibrator(meta, wind)
    # Should not crash; falls back to the physical-plausibility defaults.
    out = cal.transform_grid(np.zeros((1, 1, 2)), np.array([[500.0, 1_000.0]]))
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# AsymmetricCalibrator
# ---------------------------------------------------------------------------
def test_asymmetric_boosts_positive_z():
    meta, wind = _station_frame()
    linear = SpatialWindCalibrator(meta, wind)
    asym = AsymmetricCalibrator(meta, wind, stretch_power=1.5)
    # Compare at z = 2 (positive) and z = -2 (negative) at a fixed elev.
    z_pos = np.full((1, 1, 1), 2.0)
    z_neg = np.full((1, 1, 1), -2.0)
    elev = np.array([[1_000.0]])
    lin_pos = float(linear.transform_grid(z_pos, elev)[0, 0, 0])
    asym_pos = float(asym.transform_grid(z_pos, elev)[0, 0, 0])
    lin_neg = float(linear.transform_grid(z_neg, elev)[0, 0, 0])
    asym_neg = float(asym.transform_grid(z_neg, elev)[0, 0, 0])
    # Positive tail inflates; negative tail stays unchanged.
    assert asym_pos > lin_pos
    assert abs(asym_neg - lin_neg) < 0.01


def test_asymmetric_global_max_matches_linear():
    meta, wind = _station_frame()
    asym = AsymmetricCalibrator(meta, wind)
    assert asym.global_max == pytest.approx(np.percentile(wind, 99.9), abs=1e-6)


def test_asymmetric_rejects_wrong_shapes():
    meta, wind = _station_frame()
    asym = AsymmetricCalibrator(meta, wind)
    with pytest.raises(ValueError, match="3-D"):
        asym.transform_grid(np.zeros((2, 3)), np.zeros((3,)))
    with pytest.raises(ValueError, match="elevation shape"):
        asym.transform_grid(np.zeros((1, 2, 3)), np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# ElevationBiasCorrector
# ---------------------------------------------------------------------------
def test_elevation_bias_corrector_fit_bins_correctly():
    corrector = ElevationBiasCorrector()
    alts = np.array([150, 200, 250,          # bin 0 (0–300)
                     400, 500, 550,          # bin 1 (300–600)
                     800, 850,               # bin 2 (600–1000)
                     1200, 1300, 1400,       # bin 3 (1000–1500)
                     1800, 2000, 2500])      # bin 4 (1500–4500)
    # Bias per station: bin 0 → +1, bin 1 → +2, bin 2 → nan (filled nonetheless),
    # bin 3 → -1, bin 4 → -2.
    bias = np.array([1., 1., 1.,
                     2., 2., 2.,
                     np.nan, 0.,
                     -1., -1., -1.,
                     -2., -2., -2.])
    corrector.fit(bias, alts)
    # Minus sign — correction cancels bias.
    np.testing.assert_allclose(corrector.bin_corrections, [-1.0, -2.0, 0.0, 1.0, 2.0],
                               rtol=0, atol=1e-6)


def test_elevation_bias_corrector_too_few_stations_leaves_zero():
    corrector = ElevationBiasCorrector(min_valid_stations=5)
    alts = np.array([200, 200, 400, 1500])
    bias = np.array([1, 1, 2, -1])           # below min_valid_stations threshold
    corrector.fit(bias, alts)
    assert np.all(corrector.bin_corrections == 0.0)


def test_elevation_bias_corrector_rejects_shape_mismatch():
    corrector = ElevationBiasCorrector()
    with pytest.raises(ValueError, match="share shape"):
        corrector.fit(np.zeros(5), np.zeros(6))


def test_elevation_bias_corrector_apply_before_fit_raises():
    corrector = ElevationBiasCorrector()
    with pytest.raises(RuntimeError, match="fit the corrector"):
        corrector.apply(np.zeros((2, 2)), np.zeros((2, 2)))


def test_elevation_bias_corrector_apply_2d_wind():
    corrector = ElevationBiasCorrector()
    # 3 stations per bin so every bin passes min_valid_stations=3.
    alts_fit = np.array([150, 200, 250,          # bin 0
                         400, 500, 550,          # bin 1
                         700, 800, 900,          # bin 2
                         1200, 1300, 1400,       # bin 3
                         1800, 2000, 2500])      # bin 4
    bias_fit = np.ones_like(alts_fit, dtype=float)                       # +1 everywhere
    corrector.fit(bias_fit, alts_fit)
    # All corrections should be approximately -1.
    np.testing.assert_allclose(corrector.bin_corrections,
                               -1.0 * np.ones_like(corrector.bin_corrections),
                               atol=1e-6)
    out = corrector.apply(np.full((3, 3), 5.0), np.full((3, 3), 500.0))
    # 5.0 wind - 1.0 correction = 4.0 everywhere.
    np.testing.assert_allclose(out, 4.0, atol=1e-5)


def test_elevation_bias_corrector_apply_3d_wind():
    corrector = ElevationBiasCorrector()
    alts_fit = np.array([200] * 3 + [500] * 3 + [800] * 3 + [1200] * 3 + [2000] * 3)
    bias_fit = np.ones_like(alts_fit, dtype=float) * 0.5
    corrector.fit(bias_fit, alts_fit)
    out = corrector.apply(np.full((4, 3, 3), 7.0), np.full((3, 3), 500.0))
    assert out.shape == (4, 3, 3)
    np.testing.assert_allclose(out, 7.0 - 0.5, atol=1e-5)


def test_elevation_bias_corrector_clips_at_zero():
    corrector = ElevationBiasCorrector()
    alts_fit = np.array([200] * 5)
    bias_fit = np.full(5, 100.0)             # huge "bias" → correction of -100
    corrector.fit(bias_fit, alts_fit)
    out = corrector.apply(np.ones((2, 2)), np.full((2, 2), 200.0))
    # 1 - 100 clipped at 0.
    assert out.min() == 0.0
