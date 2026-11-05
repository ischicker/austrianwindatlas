"""Unit tests for :mod:`austrianwindatlas.stage1.qn_gnn.zscore`."""
from __future__ import annotations

import numpy as np
import pytest

from austrianwindatlas.stage1.qn_gnn.zscore import StationWiseZScoreTransformer


def test_fit_transform_mean_zero_std_one():
    rng = np.random.default_rng(0)
    wind = rng.normal(5.0, 2.0, size=(1000, 5))
    t = StationWiseZScoreTransformer().fit(wind)
    z = t.transform(wind)
    # Column-wise mean ~ 0, std ~ 1.
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(z.std(axis=0), 1.0, rtol=1e-10, atol=1e-10)


def test_inverse_transform_round_trip():
    rng = np.random.default_rng(1)
    wind = rng.normal(5.0, 2.0, size=(500, 4))
    t = StationWiseZScoreTransformer().fit(wind)
    z = t.transform(wind)
    back = t.inverse_transform(z)
    np.testing.assert_allclose(back, wind, rtol=1e-8, atol=1e-8)


def test_inverse_transform_clips_outliers():
    wind = np.array([[0.0, 0.0], [10.0, 10.0]])
    t = StationWiseZScoreTransformer().fit(wind)
    z = np.array([[100.0, -100.0]])
    back = t.inverse_transform(z, clip_sigma=7.0)
    # With mu=5, sigma=5 per column, clip to |z|<=7 → max |w - mu| = 35.
    assert back.max() <= 5.0 + 7.0 * 5.0 + 1e-6
    assert back.min() >= 5.0 - 7.0 * 5.0 - 1e-6


def test_min_std_floor_prevents_division_by_zero():
    wind = np.ones((50, 3))  # zero variance everywhere
    t = StationWiseZScoreTransformer(min_std=0.01).fit(wind)
    assert (t.stds_ >= 0.01).all()
    z = t.transform(wind)  # should not raise, should not produce NaN
    assert np.isfinite(z).all()


def test_transform_before_fit_raises():
    t = StationWiseZScoreTransformer()
    with pytest.raises(RuntimeError, match="not been fit"):
        t.transform(np.zeros((3, 2)))


def test_transform_shape_mismatch_raises():
    t = StationWiseZScoreTransformer().fit(np.zeros((10, 5)))
    with pytest.raises(ValueError, match="does not match"):
        t.transform(np.zeros((10, 3)))


def test_fit_handles_nan_columns():
    rng = np.random.default_rng(2)
    wind = rng.normal(5.0, 2.0, size=(100, 4))
    wind[:, 2] = np.nan  # an entirely-NaN station
    t = StationWiseZScoreTransformer().fit(wind)
    assert np.isfinite(t.means_).all()
    assert np.isfinite(t.stds_).all()


def test_fit_rejects_1d():
    with pytest.raises(ValueError, match="2-D"):
        StationWiseZScoreTransformer().fit(np.zeros(10))


def test_fit_stores_station_ids():
    t = StationWiseZScoreTransformer().fit(np.zeros((5, 3)), station_ids=[10, 20, 30])
    np.testing.assert_array_equal(t.station_ids, [10, 20, 30])
