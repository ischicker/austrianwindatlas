"""Unit tests for :mod:`austrianwindatlas.stage1.qn_gnn.svd`."""
from __future__ import annotations

import numpy as np
import pytest

from austrianwindatlas.stage1.qn_gnn.svd import ZScoreSVD, compute_zscore_svd


def _synth_zscore_rank_k(seed: int = 0, n_times: int = 200, n_stations: int = 20, k: int = 3):
    """Build wind data whose z-score + temporal centering gives a known SVD.

    V has orthonormal zero-station-mean columns so temporal centering
    does not redistribute energy; U has orthonormal columns; the
    singular values are injected at known magnitudes.  The per-station
    mu and sigma are then applied as the inverse of the z-score so
    the final matrix really reduces to the known factorisation after
    the transformer re-normalises.
    """
    rng = np.random.default_rng(seed)
    U_raw = rng.standard_normal((n_times, k))
    U, _ = np.linalg.qr(U_raw)
    V_raw = rng.standard_normal((n_stations, k))
    V_cent = V_raw - V_raw.mean(axis=0, keepdims=True)
    V, _ = np.linalg.qr(V_cent)
    sigma = np.sort(rng.uniform(2.0, 10.0, size=k))[::-1]
    temporal_mean = rng.standard_normal(n_times)

    z = temporal_mean[:, None] + U @ np.diag(sigma) @ V.T       # (n_times, n_stations)

    mu_station = 2.0 + rng.standard_normal(n_stations)
    sigma_station = 1.0 + 0.5 * rng.random(n_stations)
    wind = z * sigma_station[None, :] + mu_station[None, :]
    return wind, U, sigma, V, temporal_mean, mu_station, sigma_station


def test_compute_zscore_svd_captures_most_variance_in_top_k():
    """Rank-k structure must be preserved: >= 99 % of variance lives
    in the top k+1 EOFs.

    We don't assert exact sigma values because the fresh z-score
    transformer inside ``compute_zscore_svd`` re-normalises each
    station column with population std/mean, which differs from the
    per-station scale we injected at construction time.  The
    structure survives though — 99 %+ of variance ends up in roughly
    the first k modes.
    """
    wind, *_ = _synth_zscore_rank_k(k=3)
    svd = compute_zscore_svd(wind, n_eofs=10)
    assert isinstance(svd, ZScoreSVD)
    assert svd.n_eofs == 10
    # Top-4 EOFs capture essentially all variance (one extra mode
    # appears because the z-score re-normalisation is not a strict
    # rank-preserving operation; it adds one dof).
    assert svd.variance_explained[3] > 0.99
    # Strictly decreasing.
    assert svd.sigma[0] > svd.sigma[1] > svd.sigma[2] > 0


def test_compute_zscore_svd_variance_monotone_and_bounded():
    wind, *_ = _synth_zscore_rank_k()
    svd = compute_zscore_svd(wind, n_eofs=5)
    ve = svd.variance_explained
    assert np.all(np.diff(ve) >= -1e-12)
    assert ve[-1] <= 1.0 + 1e-12


def test_compute_zscore_svd_fills_nan_with_zero_after_normalisation():
    wind, *_ = _synth_zscore_rank_k()
    rng = np.random.default_rng(5)
    wind_nan = wind.copy()
    wind_nan[rng.random(wind.shape) < 0.05] = np.nan
    svd = compute_zscore_svd(wind_nan, n_eofs=5)
    # Should not raise; all outputs finite.
    for arr in (svd.U, svd.sigma, svd.V, svd.temporal_mean):
        assert np.isfinite(arr).all()


def test_qt_is_fit_and_round_trips():
    wind, *_ = _synth_zscore_rank_k()
    svd = compute_zscore_svd(wind, n_eofs=5)
    # Transformer was fit; inverse round-trip works.
    z = svd.qt.transform(wind)
    back = svd.qt.inverse_transform(z, clip_sigma=1e6)
    np.testing.assert_allclose(back, wind, rtol=1e-8, atol=1e-8)


def test_compute_zscore_svd_caps_eofs_at_rank():
    rng = np.random.default_rng(0)
    wind = rng.normal(5, 2, size=(8, 12))
    svd = compute_zscore_svd(wind, n_eofs=100)
    assert svd.n_eofs == min(wind.shape)


def test_compute_zscore_svd_rejects_zero_eofs():
    with pytest.raises(ValueError, match="n_eofs"):
        compute_zscore_svd(np.ones((5, 5)), n_eofs=0)
