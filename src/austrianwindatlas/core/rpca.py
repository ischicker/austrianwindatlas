"""Robust Principal Component Analysis via Principal Component Pursuit (PCP).

Decomposes an observation matrix :math:`M` into

.. math::

    M \\;=\\; L \\;+\\; S

where ``L`` is low rank (common patterns) and ``S`` is element-wise
sparse (rare localised extremes), by solving

.. math::

    \\min_{L,S}\\; \\|L\\|_{*} + \\lambda \\|S\\|_{1}
    \\quad\\text{subject to}\\quad L + S = M .

The solver is the Alternating Direction Method of Multipliers (ADMM)
formulation of Candès et al. (2011); the implementation follows
Lin, Chen and Ma (2010) for the proximal updates and the inflating-
penalty schedule.

The algorithm is backend-agnostic: it runs on NumPy by default and
on CuPy if ``use_gpu=True`` and CuPy is importable.

References
----------
Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011).
    Robust Principal Component Analysis?
    *Journal of the ACM*, 58(3), 11.

Lin, Z., Chen, M., & Ma, Y. (2010).
    The Augmented Lagrange Multiplier Method for Exact Recovery of
    Corrupted Low-Rank Matrices.
    *arXiv:1009.5055*.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

try:
    import cupy as _cp

    HAS_CUPY: bool = True
except ImportError:  # pragma: no cover — covered by CI matrix
    _cp = None  # type: ignore[assignment]
    HAS_CUPY = False

__all__ = [
    "HAS_CUPY",
    "RobustPCA",
    "RPCAIteration",
    "default_lambda",
    "default_mu",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default-parameter helpers (so callers and tests can reuse them)
# ---------------------------------------------------------------------------
def default_lambda(n_rows: int, n_cols: int) -> float:
    r"""Return the theoretically optimal :math:`\\lambda = 1 / \\sqrt{\\max(n,m)}`.

    From Candès et al. (2011): for an :math:`n \\times m` matrix,
    PCP with this choice recovers the exact decomposition with
    high probability under incoherence assumptions on :math:`L_{\\text{true}}`
    and uniform sparsity of :math:`S_{\\text{true}}`.
    """
    return 1.0 / np.sqrt(max(n_rows, n_cols))


def default_mu(matrix_abs_sum: float, n_rows: int, n_cols: int) -> float:
    r"""Return the ADMM initial-penalty heuristic from Lin et al. (2010).

    .. math::

        \\mu_0 \\;=\\; \\dfrac{n\\,m}{4 \\sum_{i,j} |M_{ij}|}

    balances the primal and dual residuals on the first iteration.
    """
    if matrix_abs_sum <= 0:
        raise ValueError("matrix_abs_sum must be strictly positive")
    return n_rows * n_cols / (4.0 * matrix_abs_sum)


# ---------------------------------------------------------------------------
# Per-iteration diagnostic record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RPCAIteration:
    """One row of the ADMM convergence history."""

    iter: int
    """1-based iteration index."""

    primal_residual: float
    r""":math:`\\|M - L - S\\|_{F} / \\|M\\|_{F}`."""

    L_change: float
    r""":math:`\\|L_{k} - L_{k-1}\\|_{F} / \\max(1, \\|L_{k-1}\\|_{F})`."""

    S_change: float
    r"""Same relative-Frobenius change for :math:`S`."""

    rank: int
    """Rank of :math:`L` after this iteration."""

    sparsity: float
    """Fraction of entries of :math:`S` whose magnitude exceeds ``1e-10``."""

    mu: float
    """ADMM penalty parameter used at this iteration."""


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class RobustPCA:
    """Principal Component Pursuit solver (ADMM).

    Parameters
    ----------
    lambda_param : float, optional
        Regularisation weight on the sparse component.  When ``None``
        (the default) :func:`default_lambda` is used, i.e. the
        theoretically optimal :math:`1/\\sqrt{\\max(n,m)}`.
    max_iter : int, default 500
        Maximum ADMM iterations.
    tol : float, default 1e-6
        Convergence tolerance.  The solver stops when the primal
        residual *and* the relative change of :math:`L` *and* of
        :math:`S` all fall below ``tol``.
    mu : float, optional
        Initial ADMM penalty parameter :math:`\\mu_0`.  When ``None``
        the Lin et al. (2010) heuristic :func:`default_mu` is used.
    mu_max : float, default 1e10
        Cap on :math:`\\mu` so the dual update does not explode.
    rho : float, default 1.5
        Inflation factor: :math:`\\mu \\leftarrow \\min(\\rho\\,\\mu, \\mu_{\\max})`
        at the end of each iteration.
    use_gpu : bool, default False
        If ``True`` and CuPy is importable, run on CuPy; otherwise
        fall back to NumPy with a warning on the first fit.
    verbose : bool, default False
        If ``True``, emit per-20-iteration progress at
        :pyattr:`logging.INFO`.

    Attributes
    ----------
    L_ : np.ndarray or None
        Low-rank component after :meth:`fit`.
    S_ : np.ndarray or None
        Sparse component after :meth:`fit`.
    converged_ : bool
        ``True`` if the convergence criterion was met before
        ``max_iter``.
    n_iter_ : int
        Number of ADMM iterations actually run.
    history_ : list[RPCAIteration]
        Per-iteration diagnostics; useful for plotting convergence.
    """

    def __init__(
        self,
        *,
        lambda_param: float | None = None,
        max_iter: int = 500,
        tol: float = 1e-6,
        mu: float | None = None,
        mu_max: float = 1e10,
        rho: float = 1.5,
        use_gpu: bool = False,
        verbose: bool = False,
    ) -> None:
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if tol <= 0:
            raise ValueError("tol must be strictly positive")
        if rho <= 1.0:
            raise ValueError("rho must be > 1 (penalty must be non-decreasing)")

        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.tol = tol
        self.mu = mu
        self.mu_max = mu_max
        self.rho = rho
        self.use_gpu = bool(use_gpu and HAS_CUPY)
        self.verbose = verbose

        if use_gpu and not HAS_CUPY:
            logger.warning("use_gpu=True but CuPy is not available; falling back to NumPy")

        # Fitted state
        self.L_: NDArray[np.floating] | None = None
        self.S_: NDArray[np.floating] | None = None
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.history_: list[RPCAIteration] = field(default_factory=list)  # type: ignore[assignment]
        self.history_ = []

    # ---------- backend dispatch -----------------------------------------
    def _xp(self) -> Any:
        return _cp if self.use_gpu else np

    def _to_device(self, arr: NDArray[np.floating]) -> Any:
        return _cp.asarray(arr) if self.use_gpu else np.asarray(arr)

    def _to_host(self, arr: Any) -> NDArray[np.floating]:
        if self.use_gpu and hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    # ---------- proximal operators ---------------------------------------
    def _shrink(self, X: Any, tau: float) -> Any:
        r"""Soft-thresholding (L1 proximal): :math:`\\mathrm{sgn}(X)\\max(|X|-\\tau,0)`."""
        xp = self._xp()
        return xp.sign(X) * xp.maximum(xp.abs(X) - tau, 0)

    def _svd_threshold(self, X: Any, tau: float) -> tuple[Any, int]:
        r"""Singular-value thresholding (nuclear-norm proximal): :math:`U\\,\\mathrm{shrink}(\\Sigma,\\tau)\\,V^{T}`."""
        xp = self._xp()
        U, s, Vt = xp.linalg.svd(X, full_matrices=False)
        s_thresh = xp.maximum(s - tau, 0)
        rank = int(xp.sum(s_thresh > 0))
        if rank == 0:
            return xp.zeros_like(X), 0
        return (U[:, :rank] * s_thresh[:rank]) @ Vt[:rank, :], rank

    # ---------- public API -----------------------------------------------
    def fit(self, M: NDArray[np.floating]) -> RobustPCA:
        r"""Decompose ``M`` into :math:`L + S`.

        Parameters
        ----------
        M : (n, m) ndarray
            Input matrix.  Centre (subtract the row or column mean) if
            that is appropriate for your problem; this solver treats
            ``M`` as given.

        Returns
        -------
        self
            Fitted model, with :pyattr:`L_`, :pyattr:`S_`,
            :pyattr:`converged_`, :pyattr:`n_iter_` and
            :pyattr:`history_` populated.
        """
        if M.ndim != 2:
            raise ValueError(f"M must be 2-D; got shape {M.shape!r}")
        n_rows, n_cols = M.shape
        if n_rows < 2 or n_cols < 2:
            raise ValueError(f"M must have at least 2 rows and 2 columns; got {M.shape!r}")

        start = time.perf_counter()
        xp = self._xp()
        M_dev = self._to_device(M.astype(np.float64, copy=False))

        # Resolve default parameters.
        lam = self.lambda_param
        if lam is None:
            lam = default_lambda(n_rows, n_cols)

        norm_M = float(xp.linalg.norm(M_dev, "fro"))
        if norm_M == 0.0:
            # Degenerate but legal input — the decomposition is just L=S=0.
            self.L_ = np.zeros_like(M)
            self.S_ = np.zeros_like(M)
            self.converged_ = True
            self.n_iter_ = 0
            self.history_ = []
            return self

        mu = self.mu
        if mu is None:
            mu = default_mu(float(xp.sum(xp.abs(M_dev))), n_rows, n_cols)

        if self.verbose:
            logger.info(
                "RobustPCA.fit: %d x %d, lambda=%.4g, mu0=%.4g, backend=%s",
                n_rows, n_cols, lam, mu, "CuPy" if self.use_gpu else "NumPy",
            )

        L = xp.zeros_like(M_dev)
        S = xp.zeros_like(M_dev)
        Y = xp.zeros_like(M_dev)  # dual (Lagrange multiplier)

        self.history_ = []
        self.converged_ = False
        rank, sparsity = 0, 0.0

        for k in range(self.max_iter):
            L_prev, S_prev = L, S

            # Primal updates
            L, rank = self._svd_threshold(M_dev - S + Y / mu, 1.0 / mu)
            S = self._shrink(M_dev - L + Y / mu, lam / mu)

            # Dual update
            residual = M_dev - L - S
            Y = Y + mu * residual

            # Diagnostics
            primal = float(xp.linalg.norm(residual, "fro")) / norm_M
            dL = float(xp.linalg.norm(L - L_prev, "fro")) / max(
                1.0, float(xp.linalg.norm(L_prev, "fro"))
            )
            dS = float(xp.linalg.norm(S - S_prev, "fro")) / max(
                1.0, float(xp.linalg.norm(S_prev, "fro"))
            )
            n_sparse = int(xp.sum(xp.abs(S) > 1e-10))
            sparsity = n_sparse / (n_rows * n_cols)

            self.history_.append(
                RPCAIteration(
                    iter=k + 1,
                    primal_residual=primal,
                    L_change=dL,
                    S_change=dS,
                    rank=rank,
                    sparsity=sparsity,
                    mu=mu,
                )
            )

            if self.verbose and (k + 1) % 20 == 0:
                logger.info(
                    "  iter %4d: residual=%.3e, rank(L)=%d, sparsity(S)=%.4f",
                    k + 1, primal, rank, sparsity,
                )

            if primal < self.tol and dL < self.tol and dS < self.tol:
                self.converged_ = True
                break

            mu = min(mu * self.rho, self.mu_max)

        self.n_iter_ = k + 1
        self.L_ = self._to_host(L)
        self.S_ = self._to_host(S)

        if self.verbose:
            recon = float(np.linalg.norm(M - self.L_ - self.S_, "fro")) / norm_M
            logger.info(
                "RobustPCA.fit done in %.2fs: n_iter=%d, converged=%s, "
                "rank(L)=%d, sparsity(S)=%.4f, recon_err=%.2e",
                time.perf_counter() - start, self.n_iter_, self.converged_,
                rank, sparsity, recon,
            )

        return self

    def get_sparse_timesteps(
        self,
        *,
        threshold_percentile: float = 95.0,
        min_stations: int = 3,
    ) -> NDArray[np.bool_]:
        r"""Flag timesteps whose sparse component is active enough to warrant
        separate (non-EOF) interpolation.

        A row (= timestep, since ``M`` has shape ``(n_times, n_stations)``) is
        flagged when at least ``min_stations`` of its entries exceed the
        ``threshold_percentile`` of :math:`|S|` over the fitted model.

        Parameters
        ----------
        threshold_percentile : float, default 95.0
            Percentile of :math:`|S_{ij}|` over strictly positive entries
            used as the per-cell threshold.
        min_stations : int, default 3
            Minimum number of stations above threshold in a timestep.

        Returns
        -------
        (n_times,) bool array
        """
        if self.S_ is None:
            raise RuntimeError("RobustPCA has not been fitted; call .fit(M) first")
        if not (0 < threshold_percentile <= 100):
            raise ValueError("threshold_percentile must be in (0, 100]")
        if min_stations < 1:
            raise ValueError("min_stations must be >= 1")

        S_abs = np.abs(self.S_)
        nonzero = S_abs[S_abs > 0]
        if nonzero.size == 0:
            return np.zeros(self.S_.shape[0], dtype=bool)
        threshold = float(np.percentile(nonzero, threshold_percentile))
        per_time = np.sum(S_abs > threshold, axis=1)
        return per_time >= min_stations
