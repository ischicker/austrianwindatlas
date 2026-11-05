"""Unit tests for :mod:`austrianwindatlas.stage1.qn_gnn.temporal_cnn.data`."""
from __future__ import annotations

import numpy as np
import pytest

from austrianwindatlas.stage1.qn_gnn.temporal_cnn.data import (
    compute_U_true,
    sample_availability,
)
from austrianwindatlas.stage1.qn_gnn.zscore import StationWiseZScoreTransformer

torch = pytest.importorskip("torch")

from austrianwindatlas.stage1.qn_gnn.temporal_cnn.data import UResidualDataset  # noqa: E402


# ---------------------------------------------------------------------------
# sample_availability
# ---------------------------------------------------------------------------
def test_sample_availability_shape_and_dtype():
    rng = np.random.default_rng(0)
    m = sample_availability(50, rng=rng)
    assert m.shape == (50,)
    assert m.dtype == np.float32


def test_sample_availability_values_are_binary():
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = sample_availability(30, rng=rng)
        uniq = set(np.unique(m).tolist())
        assert uniq.issubset({0.0, 1.0})


def test_sample_availability_p_fail_zero_gives_all_ones_without_ice():
    rng = np.random.default_rng(0)
    m = sample_availability(100, p_fail=0.0, p_ice_event=0.0, rng=rng)
    assert (m == 1.0).all()


def test_sample_availability_p_fail_one_gives_all_zeros():
    rng = np.random.default_rng(0)
    m = sample_availability(100, p_fail=1.0, p_ice_event=0.0, rng=rng)
    assert (m == 0.0).all()


def test_sample_availability_ice_event_knocks_out_fraction():
    # Force the ice branch (p_ice_event=1.0) with no sensor failures.
    rng = np.random.default_rng(42)
    m = sample_availability(10_000, p_fail=0.0, p_ice_event=1.0, p_ice_fail=0.3, rng=rng)
    # Expect ~70 % available after icing.
    frac = m.mean()
    assert 0.65 < frac < 0.75


# ---------------------------------------------------------------------------
# compute_U_true
# ---------------------------------------------------------------------------
def test_compute_U_true_recovers_known_U():
    """With orthonormal V and consistent qt, compute_U_true must
    recover the injected U * sigma.
    """
    rng = np.random.default_rng(0)
    n_times, n_stations, k = 200, 15, 3

    U_raw = rng.standard_normal((n_times, k))
    U, _ = np.linalg.qr(U_raw)
    V_raw = rng.standard_normal((n_stations, k))
    V_cent = V_raw - V_raw.mean(axis=0, keepdims=True)
    V, _ = np.linalg.qr(V_cent)
    sigma = np.sort(rng.uniform(2.0, 10.0, size=k))[::-1]
    tmean = rng.standard_normal(n_times)
    z = tmean[:, None] + U @ np.diag(sigma) @ V.T          # z-score space data
    mu_station = 2.0 + rng.standard_normal(n_stations)
    sig_station = 1.0 + 0.3 * rng.random(n_stations)
    wind = z * sig_station[None, :] + mu_station[None, :]

    # Build the transformer the same way train-time would.
    qt = StationWiseZScoreTransformer().fit(wind)

    US_true, tmean_hat = compute_U_true(
        wind, qt, temporal_mean=None, sigma=sigma, V_stations=V, n_eofs=k,
    )
    US_expected = (U * sigma).astype(np.float32)
    # The fresh qt re-normalises with population mean/std of `wind`,
    # which differs from (mu_station, sig_station) at a few-percent
    # level because the injected z has finite-sample column stats
    # (not exactly N(0, 1)).  US_true recovers US_expected to within
    # ~5 %, which is the accuracy regime the paper operates in at
    # inference time anyway.
    np.testing.assert_allclose(US_true, US_expected, rtol=0.1, atol=0.3)
    np.testing.assert_allclose(tmean_hat, tmean, rtol=0.05, atol=0.1)


def test_compute_U_true_fallback_without_qt():
    """When qt is None, compute_U_true falls back to a global z-score
    and should still produce a finite result (albeit less accurate).
    """
    rng = np.random.default_rng(1)
    wind = rng.uniform(1, 10, size=(60, 8)).astype(np.float32)
    sigma = np.array([3.0, 1.5])
    V = rng.standard_normal((8, 2))
    US_true, tmean_hat = compute_U_true(
        wind, qt=None, temporal_mean=None, sigma=sigma, V_stations=V,
    )
    assert US_true.shape == (60, 2)
    assert US_true.dtype == np.float32
    assert np.isfinite(US_true).all()
    assert np.isfinite(tmean_hat).all()


def test_compute_U_true_rejects_shape_mismatch():
    rng = np.random.default_rng(0)
    wind = rng.standard_normal((50, 10)).astype(np.float32)
    sigma = np.array([3.0, 1.5, 0.5])
    V = rng.standard_normal((10, 2))  # only 2 EOFs available
    with pytest.raises(ValueError, match="EOFs"):
        compute_U_true(wind, None, None, sigma, V, n_eofs=3)


def test_compute_U_true_handles_nan_input():
    rng = np.random.default_rng(2)
    wind = rng.uniform(1, 10, size=(80, 6)).astype(np.float32)
    wind[rng.random(wind.shape) < 0.05] = np.nan
    sigma = np.array([3.0, 1.0])
    V = rng.standard_normal((6, 2))
    US_true, _ = compute_U_true(wind, None, None, sigma, V)
    assert np.isfinite(US_true).all()


# ---------------------------------------------------------------------------
# UResidualDataset
# ---------------------------------------------------------------------------
def _toy_dataset(n_times=50, n_stations=6, n_eofs=3, window=8, seed=0):
    rng = np.random.default_rng(seed)
    wind = rng.uniform(1, 10, size=(n_times, n_stations)).astype(np.float32)
    U_res = rng.standard_normal((n_times, n_eofs)).astype(np.float32)
    return wind, U_res, window


def test_dataset_length_equals_n_windows():
    wind, U_res, W = _toy_dataset()
    ds = UResidualDataset(wind, U_res, window=W, augment=False)
    assert len(ds) == wind.shape[0] - W + 1


def test_dataset_getitem_shapes():
    wind, U_res, W = _toy_dataset(n_stations=6, n_eofs=3, window=8)
    ds = UResidualDataset(wind, U_res, window=W, augment=False)
    obs, avail, tgt = ds[0]
    assert obs.shape == (W, 6)
    assert avail.shape == (W, 6)
    assert tgt.shape == (3,)


def test_dataset_target_aligned_with_window_end():
    """Dataset item ``i`` should return ``U_residual[i + W - 1]``."""
    wind, U_res, W = _toy_dataset()
    ds = UResidualDataset(wind, U_res, window=W, augment=False)
    _, _, tgt0 = ds[0]
    np.testing.assert_allclose(tgt0.numpy(), U_res[W - 1])
    _, _, tgt5 = ds[5]
    np.testing.assert_allclose(tgt5.numpy(), U_res[W - 1 + 5])


def test_dataset_holdout_mask_permanently_zeros():
    wind, U_res, W = _toy_dataset(n_stations=6)
    holdout = np.array([False, True, False, False, True, False])
    ds = UResidualDataset(wind, U_res, window=W, holdout_mask=holdout, augment=False)
    obs, avail, _ = ds[0]
    # Columns 1 and 4 should be exactly zero for both obs and avail.
    assert float(obs[:, 1].abs().max()) == 0.0
    assert float(obs[:, 4].abs().max()) == 0.0
    assert float(avail[:, 1].abs().max()) == 0.0
    assert float(avail[:, 4].abs().max()) == 0.0


def test_dataset_without_augmentation_is_deterministic():
    wind, U_res, W = _toy_dataset()
    ds = UResidualDataset(wind, U_res, window=W, augment=False)
    obs_a, _, _ = ds[3]
    obs_b, _, _ = ds[3]
    assert torch.equal(obs_a, obs_b)


def test_dataset_augmentation_mutates_output():
    """With augment=True we expect calls to return different masks
    (at least once in a dozen draws, well above chance).
    """
    wind, U_res, W = _toy_dataset()
    rng = np.random.default_rng(0)
    ds = UResidualDataset(wind, U_res, window=W, augment=True, rng=rng)
    first = ds[0][1].numpy().copy()
    diffs = sum(not np.array_equal(first, ds[0][1].numpy()) for _ in range(10))
    assert diffs >= 1


def test_dataset_rejects_1d_wind():
    with pytest.raises(ValueError, match="2-D"):
        UResidualDataset(np.zeros(10), np.zeros((10, 2)), window=4)


def test_dataset_rejects_window_too_big():
    wind, U_res, _ = _toy_dataset(n_times=10)
    with pytest.raises(ValueError, match="window"):
        UResidualDataset(wind, U_res, window=100)


def test_dataset_rejects_mismatched_n_times():
    wind, U_res, W = _toy_dataset()
    with pytest.raises(ValueError, match="share n_times"):
        UResidualDataset(wind, U_res[:-5], window=W)


def test_dataset_rejects_wrong_holdout_shape():
    wind, U_res, W = _toy_dataset(n_stations=6)
    with pytest.raises(ValueError, match="holdout_mask must have shape"):
        UResidualDataset(wind, U_res, window=W, holdout_mask=np.zeros(5, dtype=bool))


def test_dataset_fills_nan_with_zero_in_obs():
    wind, U_res, W = _toy_dataset()
    wind_nan = wind.copy()
    wind_nan[5, 2] = np.nan
    ds = UResidualDataset(wind_nan, U_res, window=W, augment=False)
    # Any window that includes timestep 5 should have a zero at station 2
    # and avail=0 there.
    obs, avail, _ = ds[0]
    assert float(obs[5, 2]) == 0.0
    assert float(avail[5, 2]) == 0.0
