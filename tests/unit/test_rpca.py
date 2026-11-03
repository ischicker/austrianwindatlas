"""Unit tests for `austrianwindatlas.core.rpca`.

The tests follow the recovery-guarantee of Candès et al. (2011):
if we construct ``M = L_true + S_true`` with ``L_true`` low rank
and ``S_true`` element-wise sparse, Principal Component Pursuit
recovers them (nearly) exactly.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from austrianwindatlas.core.rpca import (
    HAS_CUPY,
    RobustPCA,
    RPCAIteration,
    default_lambda,
    default_mu,
)

RNG_SEED = 20261103


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_low_rank_plus_sparse(
    n_rows: int = 80,
    n_cols: int = 100,
    rank: int = 2,
    sparsity: float = 0.03,
    sparse_scale: float = 2.0,
    seed: int = RNG_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (M, L_true, S_true) with M = L_true + S_true.

    The default parameters sit inside the Candès et al. (2011) exact-
    recovery regime: rank 2 of an 80 x 100 matrix is well below the
    bound ``rank <= 0.6 * min(n,m) / log(max(n,m))^2 ~ 2.8``, and the
    sparse-signal magnitudes stay on the order of the low-rank
    magnitudes so the inflating-mu ADMM converges to the true
    L/S split rather than to an alternative decomposition in which
    the sparse signal has leaked into L (which happens for
    sparse_scale >> ||L_true||_F / sqrt(nnz(S_true))).
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n_rows, rank))
    V = rng.standard_normal((rank, n_cols))
    L_true = (U @ V) / np.sqrt(rank)
    mask = rng.random((n_rows, n_cols)) < sparsity
    signs = rng.choice([-1.0, 1.0], size=(n_rows, n_cols))
    magnitudes = sparse_scale * (0.5 + 0.5 * rng.random((n_rows, n_cols)))
    S_true = np.where(mask, signs * magnitudes, 0.0)
    M = L_true + S_true
    return M, L_true, S_true


def _rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, "fro") / max(1e-12, np.linalg.norm(b, "fro")))


# ---------------------------------------------------------------------------
# Default-parameter helpers
# ---------------------------------------------------------------------------
class TestDefaults:
    def test_default_lambda_matches_candes(self):
        assert default_lambda(40, 60) == pytest.approx(1.0 / np.sqrt(60))
        assert default_lambda(100, 10) == pytest.approx(1.0 / np.sqrt(100))

    def test_default_mu_positive(self):
        mu = default_mu(matrix_abs_sum=1234.5, n_rows=40, n_cols=60)
        assert mu > 0

    def test_default_mu_rejects_nonpositive_abs_sum(self):
        with pytest.raises(ValueError):
            default_mu(matrix_abs_sum=0.0, n_rows=40, n_cols=60)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------
class TestConstructor:
    def test_defaults_do_not_raise(self):
        RobustPCA()

    def test_max_iter_must_be_positive(self):
        with pytest.raises(ValueError):
            RobustPCA(max_iter=0)

    def test_tol_must_be_strictly_positive(self):
        with pytest.raises(ValueError):
            RobustPCA(tol=0.0)
        with pytest.raises(ValueError):
            RobustPCA(tol=-1e-9)

    def test_rho_must_exceed_one(self):
        with pytest.raises(ValueError):
            RobustPCA(rho=1.0)
        with pytest.raises(ValueError):
            RobustPCA(rho=0.5)

    def test_gpu_requested_without_cupy_logs_warning(self, caplog):
        if HAS_CUPY:
            pytest.skip("CuPy is installed; cannot assert the fallback warning")
        with caplog.at_level(logging.WARNING, logger="austrianwindatlas.core.rpca"):
            m = RobustPCA(use_gpu=True)
        assert m.use_gpu is False
        assert any("CuPy is not available" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# fit(M) — recovery, convergence, residuals
# ---------------------------------------------------------------------------
class TestFitRecovery:
    def test_recovers_low_rank_and_sparse_components(self):
        M, L_true, S_true = _make_low_rank_plus_sparse()
        model = RobustPCA(max_iter=200).fit(M)
        assert model.converged_, "ADMM should converge inside the recovery regime"
        # (1) ADMM constraint: M = L + S holds to machine precision.
        assert _rel_error(model.L_ + model.S_, M) < 1e-4
        # (2) Component recovery.  In the Candès et al. exact-recovery
        # regime with sparse_scale ~ ||L||/sqrt(nnz(S)) the recovery is
        # near machine precision; 1e-6 is a comfortable practical bound.
        assert _rel_error(model.L_, L_true) < 1e-6
        assert _rel_error(model.S_, S_true) < 1e-6
        # (3) Rank of recovered L matches the planted rank.
        assert np.linalg.matrix_rank(model.L_, tol=1e-6) == 2
        # (4) Sparse support agreement is total.
        true_mask = S_true != 0
        recovered_mask = np.abs(model.S_) > 1e-6
        assert np.mean(recovered_mask[true_mask]) == 1.0

    def test_degenerate_zero_matrix(self):
        M = np.zeros((10, 15))
        model = RobustPCA(max_iter=50).fit(M)
        assert model.converged_
        assert model.n_iter_ == 0
        np.testing.assert_array_equal(model.L_, np.zeros_like(M))
        np.testing.assert_array_equal(model.S_, np.zeros_like(M))

    def test_history_rows_shape_and_types(self):
        M, _, _ = _make_low_rank_plus_sparse(n_rows=20, n_cols=20, rank=2)
        model = RobustPCA(max_iter=80).fit(M)
        assert len(model.history_) == model.n_iter_
        row = model.history_[0]
        assert isinstance(row, RPCAIteration)
        assert row.iter == 1
        assert row.rank >= 0
        assert 0.0 <= row.sparsity <= 1.0
        assert row.mu > 0

    def test_max_iter_boundary_still_returns(self):
        M, _, _ = _make_low_rank_plus_sparse(n_rows=20, n_cols=25, rank=2)
        model = RobustPCA(max_iter=2, tol=1e-20).fit(M)
        # tol is unreachable in 2 iterations: should return non-converged.
        assert model.converged_ is False
        assert model.n_iter_ == 2
        assert model.L_.shape == M.shape and model.S_.shape == M.shape

    def test_custom_lambda_and_mu_round_trip(self):
        M, _, _ = _make_low_rank_plus_sparse(n_rows=25, n_cols=35, rank=2)
        model = RobustPCA(lambda_param=0.1, mu=0.5, max_iter=400).fit(M)
        assert _rel_error(model.L_ + model.S_, M) < 1e-3


class TestFitInputValidation:
    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            RobustPCA().fit(np.zeros(5))

    def test_rejects_tiny_matrix(self):
        with pytest.raises(ValueError):
            RobustPCA().fit(np.zeros((1, 10)))
        with pytest.raises(ValueError):
            RobustPCA().fit(np.zeros((10, 1)))


# ---------------------------------------------------------------------------
# Sparse-timestep post-processing
# ---------------------------------------------------------------------------
class TestSparseTimesteps:
    def test_requires_fit(self):
        with pytest.raises(RuntimeError):
            RobustPCA().get_sparse_timesteps()

    def test_flags_timesteps_with_enough_active_stations(self):
        rng = np.random.default_rng(RNG_SEED + 1)
        n_times, n_stations = 50, 30
        M = 0.1 * rng.standard_normal((n_times, n_stations))
        # Inject a strong, broad sparse signal into timesteps 7 and 23.
        for t in (7, 23):
            affected = rng.choice(n_stations, size=10, replace=False)
            M[t, affected] += 15.0
        model = RobustPCA(max_iter=400).fit(M)
        flagged = model.get_sparse_timesteps(threshold_percentile=90, min_stations=5)
        assert flagged.shape == (n_times,)
        assert flagged[7] and flagged[23]

    def test_empty_sparse_returns_all_false(self):
        # Pure low-rank input; S should be ~0 and no timestep should be flagged.
        rng = np.random.default_rng(RNG_SEED + 2)
        U = rng.standard_normal((40, 3))
        V = rng.standard_normal((3, 50))
        M = (U @ V) / np.sqrt(3)
        model = RobustPCA(max_iter=200).fit(M)
        flagged = model.get_sparse_timesteps(threshold_percentile=99, min_stations=1)
        assert flagged.shape == (40,)
        assert not flagged.any()

    def test_invalid_threshold_rejected(self):
        M, _, _ = _make_low_rank_plus_sparse(n_rows=10, n_cols=15, rank=2)
        model = RobustPCA(max_iter=50).fit(M)
        with pytest.raises(ValueError):
            model.get_sparse_timesteps(threshold_percentile=0)
        with pytest.raises(ValueError):
            model.get_sparse_timesteps(threshold_percentile=110)
        with pytest.raises(ValueError):
            model.get_sparse_timesteps(min_stations=0)


# ---------------------------------------------------------------------------
# GPU path (skipped when CuPy is absent, which is the usual CI case)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
class TestGPUBackend:  # pragma: no cover
    def test_gpu_result_matches_cpu(self):
        M, _, _ = _make_low_rank_plus_sparse(n_rows=30, n_cols=40, rank=2)
        cpu = RobustPCA(max_iter=300, use_gpu=False).fit(M)
        gpu = RobustPCA(max_iter=300, use_gpu=True).fit(M)
        assert _rel_error(cpu.L_, gpu.L_) < 1e-4
        assert _rel_error(cpu.S_, gpu.S_) < 1e-4
