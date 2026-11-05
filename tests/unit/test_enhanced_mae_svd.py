"""Unit tests for :mod:`austrianwindatlas.stage1.enhanced_mae.svd`."""
from __future__ import annotations

import numpy as np
import pytest

from austrianwindatlas.stage1.enhanced_mae.svd import (
    TemporalSVD,
    compute_temporal_svd,
    fill_nans_with_station_mean,
    verify_reconstruction,
)


# ---------------------------------------------------------------------------
# fill_nans_with_station_mean
# ---------------------------------------------------------------------------
def test_fill_nans_preserves_finite_values():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = fill_nans_with_station_mean(arr)
    np.testing.assert_array_equal(out, arr)


def test_fill_nans_fills_columnwise():
    arr = np.array([[1.0, np.nan], [3.0, np.nan], [5.0, 4.0]])
    out = fill_nans_with_station_mean(arr)
    assert not np.isnan(out).any()
    # Column 1: only finite value is 4.0 → NaN should be replaced with 4.0
    np.testing.assert_array_equal(out[:, 1], [4.0, 4.0, 4.0])


def test_fill_nans_all_nan_column_uses_global_mean():
    arr = np.array([[1.0, np.nan], [3.0, np.nan], [5.0, np.nan]])
    out = fill_nans_with_station_mean(arr)
    # Global mean of finite values = 3.0.
    np.testing.assert_array_equal(out[:, 1], [3.0, 3.0, 3.0])


def test_fill_nans_raises_when_all_nan():
    arr = np.full((3, 2), np.nan)
    with pytest.raises(ValueError, match="entirely NaN"):
        fill_nans_with_station_mean(arr)


def test_fill_nans_rejects_1d():
    with pytest.raises(ValueError, match="2-D"):
        fill_nans_with_station_mean(np.zeros(5))


# ---------------------------------------------------------------------------
# compute_temporal_svd
# ---------------------------------------------------------------------------
def _make_synthetic_rank_k(n_times: int = 200, n_stations: int = 20, k: int = 3,
                           noise: float = 0.0, seed: int = 42):
    """Generate wind data with a known temporal-SVD low-rank structure.

    V is forced to have zero column means so temporal centering (which
    subtracts the across-stations mean at each timestep) does not
    redistribute energy into/out of the sigmas — i.e. the singular
    values of the centred matrix equal ``sigma_true``.
    """
    rng = np.random.default_rng(seed)
    # Random orthonormal column spaces for U and V via QR; that way
    # the returned sigmas really are the singular values we fed in.
    U_raw = rng.standard_normal((n_times, k))
    U_true, _ = np.linalg.qr(U_raw)                     # (n_times, k) orthonormal cols
    V_raw = rng.standard_normal((n_stations, k))
    V_centered = V_raw - V_raw.mean(axis=0, keepdims=True)  # zero station mean
    V_true, _ = np.linalg.qr(V_centered)                # (n_stations, k) orthonormal cols
    sigma_true = np.sort(rng.uniform(2.0, 10.0, size=k))[::-1]
    temporal_mean = 5.0 + rng.standard_normal(n_times)
    wind = temporal_mean[:, None] + U_true @ np.diag(sigma_true) @ V_true.T
    if noise:
        wind = wind + rng.standard_normal(wind.shape) * noise
    return wind, U_true, sigma_true, V_true, temporal_mean


def test_compute_temporal_svd_recovers_low_rank_truth():
    wind, _, sigma_true, _, _ = _make_synthetic_rank_k(noise=0.0, k=3)
    svd = compute_temporal_svd(wind, n_eofs=10)
    assert isinstance(svd, TemporalSVD)
    assert svd.n_eofs == 10
    assert svd.n_times == 200
    assert svd.n_stations == 20
    # The first three singular values must match the true sigmas
    # (V had zero station-means so centering is a no-op on the signal);
    # the remaining ones should be essentially zero.
    np.testing.assert_allclose(svd.sigma[:3], sigma_true, rtol=1e-6, atol=1e-6)
    assert svd.sigma[3] < 1e-8


def test_compute_temporal_svd_variance_explained_monotone_and_bounded():
    wind, *_ = _make_synthetic_rank_k(noise=0.01)
    svd = compute_temporal_svd(wind, n_eofs=10)
    ve = svd.variance_explained
    # Cumulative → monotone non-decreasing.
    assert np.all(np.diff(ve) >= -1e-12)
    # Bounded by 1.
    assert ve[-1] <= 1.0 + 1e-12


def test_compute_temporal_svd_centering_is_temporal():
    wind, *_ = _make_synthetic_rank_k(noise=0.0)
    svd = compute_temporal_svd(wind, n_eofs=5)
    # Temporal centering: mean across stations at each timestep → that
    # is what the reconstruction formula adds back.
    # Reconstruction at every timestep:
    recon = svd.temporal_mean[:, None] + (svd.U * svd.sigma) @ svd.V.T
    np.testing.assert_allclose(recon, wind, rtol=1e-6, atol=1e-6)


def test_compute_temporal_svd_tolerates_nan():
    wind, *_ = _make_synthetic_rank_k(noise=0.0)
    # Drop ~10 % of entries.
    rng = np.random.default_rng(99)
    mask = rng.random(wind.shape) < 0.1
    wind_nan = wind.copy()
    wind_nan[mask] = np.nan
    svd = compute_temporal_svd(wind_nan, n_eofs=5)
    assert np.isfinite(svd.U).all()
    assert np.isfinite(svd.V).all()
    assert np.isfinite(svd.sigma).all()
    assert np.isfinite(svd.temporal_mean).all()


def test_compute_temporal_svd_caps_eofs_at_available_rank():
    wind, *_ = _make_synthetic_rank_k(n_times=10, n_stations=15, k=2, noise=0.0)
    svd = compute_temporal_svd(wind, n_eofs=50)
    assert svd.n_eofs == min(10, 15)


def test_compute_temporal_svd_rejects_zero_eofs():
    with pytest.raises(ValueError, match="n_eofs"):
        compute_temporal_svd(np.ones((5, 5)), n_eofs=0)


# ---------------------------------------------------------------------------
# verify_reconstruction
# ---------------------------------------------------------------------------
def test_verify_reconstruction_zero_rmse_on_exact_svd():
    wind, *_ = _make_synthetic_rank_k(k=3, noise=0.0)
    svd = compute_temporal_svd(wind, n_eofs=3)
    stats = verify_reconstruction(svd, wind, n_sample=100, rng=np.random.default_rng(0))
    assert stats["rmse"] < 1e-6
    assert stats["r"] == pytest.approx(1.0, abs=1e-6)


def test_verify_reconstruction_degrades_with_truncation():
    wind, *_ = _make_synthetic_rank_k(k=5, noise=0.0)
    full = verify_reconstruction(
        compute_temporal_svd(wind, n_eofs=5), wind,
        n_sample=50, rng=np.random.default_rng(0),
    )
    truncated = verify_reconstruction(
        compute_temporal_svd(wind, n_eofs=2), wind,
        n_sample=50, rng=np.random.default_rng(0),
    )
    assert full["rmse"] < truncated["rmse"]
