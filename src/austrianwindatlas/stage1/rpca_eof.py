"""Classical interpolators for RPCA / EOF spatial coefficients.

Given station EOF coefficients :math:`V \\in \\mathbb{R}^{n_{\\mathrm{stations}} \\times K}`
from the rPCA step (:class:`austrianwindatlas.core.rpca.RobustPCA`),
this module interpolates each of the :math:`K` columns onto a regular
grid. The resulting :math:`V_{\\mathrm{grid}}` is one of the inputs to
the reconstruction :math:`u(t,x) = \\overline{u}(x) + U(t)\\,\\Sigma\\,V_{\\mathrm{grid}}(x)^{T}`
performed by :mod:`austrianwindatlas.stage1.reconstruct` (next PR).

Four interpolators are provided; each has the same ``fit`` /
``predict`` signature pattern:

:class:`IDW`
    Inverse-distance weighting with k-nearest neighbours and a
    configurable power. No covariates.

:class:`MLR`
    Ordinary least squares on ``(x, y, covariates)`` with optional
    IDW-interpolated residuals. The "classical linear" baseline of
    the paper.

:class:`RegressionIDW`
    Ridge regression on covariates (coordinates omitted) with
    IDW-interpolated residuals. This is the family the paper labels
    "KED" — the label is historical; the implementation is
    regression kriging with an IDW surrogate for the semivariogram,
    not full ordinary / universal kriging. See the note under
    :class:`RegressionIDW` for details.

:func:`interpolate_eof_stack`
    Apply any of the three to each column of an
    :math:`(n_{\\mathrm{stations}} \\times K)` EOF-coefficient matrix,
    returning the :math:`(K \\times n_{\\mathrm{rows}} \\times n_{\\mathrm{cols}})`
    grid stack used by the reconstruction step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge

__all__ = [
    "IDW",
    "MLR",
    "RegressionIDW",
    "EOFInterpolator",
    "interpolate_eof_stack",
]


# ---------------------------------------------------------------------------
# Common Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class EOFInterpolator(Protocol):
    """Shared interface for the three EOF-coefficient interpolators.

    Each interpolator accepts station coordinates (and, for covariate-
    aware methods, per-station features) at fit time and predicts
    gridded values at predict time.
    """

    def fit(
        self,
        station_xy: NDArray[np.floating],
        values: NDArray[np.floating],
        station_features: NDArray[np.floating] | None = None,
    ) -> EOFInterpolator:
        ...

    def predict(
        self,
        grid_xy: NDArray[np.floating],
        grid_features: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_xy(name: str, xy: NDArray[np.floating]) -> NDArray[np.floating]:
    arr = np.asarray(xy, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be a (N, 2) array; got shape {arr.shape!r}")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one point")
    return arr


def _validate_values(xy_n: int, values: NDArray[np.floating]) -> NDArray[np.floating]:
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or v.shape[0] != xy_n:
        raise ValueError(
            f"values must be a 1-D array of length {xy_n} (matching station_xy); "
            f"got shape {v.shape!r}"
        )
    return v


def _validate_features(
    name: str,
    expected_rows: int,
    features: NDArray[np.floating] | None,
) -> NDArray[np.floating] | None:
    if features is None:
        return None
    arr = np.asarray(features, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != expected_rows:
        raise ValueError(
            f"{name} must be a 2-D array with {expected_rows} rows; "
            f"got shape {arr.shape!r}"
        )
    return arr


def _idw_predict(
    tree: cKDTree,
    values: NDArray[np.floating],
    grid_xy: NDArray[np.floating],
    power: float,
    k: int,
) -> NDArray[np.floating]:
    """Shared k-NN IDW evaluation."""
    k_effective = min(k, len(values))
    distances, indices = tree.query(grid_xy, k=k_effective)
    if k_effective == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    distances = np.maximum(distances, 1e-10)
    weights = 1.0 / (distances**power)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.sum(weights * values[indices], axis=1)


# ---------------------------------------------------------------------------
# IDW
# ---------------------------------------------------------------------------
@dataclass
class IDW:
    """k-NN inverse-distance weighting interpolator.

    Parameters
    ----------
    power : float, default 2.0
        Exponent on the inverse-distance weight.  2.0 is the Shepard
        default used by every IDW in the paper's benchmark.
    k_neighbors : int, default 20
        Number of nearest stations consulted for each grid point.
        Clipped silently to the number of available stations when
        smaller.
    """

    power: float = 2.0
    k_neighbors: int = 20

    def __post_init__(self) -> None:
        if self.power <= 0:
            raise ValueError("power must be strictly positive")
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        self._tree: cKDTree | None = None
        self._values: NDArray[np.floating] | None = None

    def fit(
        self,
        station_xy: NDArray[np.floating],
        values: NDArray[np.floating],
        station_features: NDArray[np.floating] | None = None,  # noqa: ARG002 — API parity
    ) -> IDW:
        xy = _validate_xy("station_xy", station_xy)
        self._values = _validate_values(xy.shape[0], values)
        self._tree = cKDTree(xy)
        return self

    def predict(
        self,
        grid_xy: NDArray[np.floating],
        grid_features: NDArray[np.floating] | None = None,  # noqa: ARG002 — API parity
    ) -> NDArray[np.floating]:
        if self._tree is None or self._values is None:
            raise RuntimeError("IDW has not been fitted; call .fit(...) first")
        g = _validate_xy("grid_xy", grid_xy)
        return _idw_predict(self._tree, self._values, g, self.power, self.k_neighbors)


# ---------------------------------------------------------------------------
# MLR
# ---------------------------------------------------------------------------
@dataclass
class MLR:
    """Ordinary least squares on coordinates + covariates, with optional
    IDW residual interpolation.

    The design matrix stacks the station coordinates and the supplied
    per-station features, standardises them (mean 0, std 1 per column)
    and fits an unregularised linear model against the target with an
    intercept. When ``use_residual_idw=True`` (the default), the
    fitted trend is corrected at prediction time by an IDW
    interpolation of the per-station residuals; this is how the paper's
    MLR configuration behaves.

    Parameters
    ----------
    power : float, default 2.0
        IDW exponent for the residual correction.
    k_neighbors : int, default 20
        Neighbour count for the residual IDW.
    use_residual_idw : bool, default True
        If ``False`` the predictor returns the bare linear trend.
    """

    power: float = 2.0
    k_neighbors: int = 20
    use_residual_idw: bool = True

    def __post_init__(self) -> None:
        if self.power <= 0:
            raise ValueError("power must be strictly positive")
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        self._mean: NDArray[np.floating] | None = None
        self._std: NDArray[np.floating] | None = None
        self._y_mean: float = 0.0
        self._beta: NDArray[np.floating] | None = None
        self._residuals: NDArray[np.floating] | None = None
        self._tree: cKDTree | None = None

    @staticmethod
    def _design_matrix(
        xy: NDArray[np.floating], features: NDArray[np.floating] | None
    ) -> NDArray[np.floating]:
        parts = [xy]
        if features is not None:
            parts.append(features)
        return np.nan_to_num(np.column_stack(parts), nan=0.0, posinf=0.0, neginf=0.0)

    def fit(
        self,
        station_xy: NDArray[np.floating],
        values: NDArray[np.floating],
        station_features: NDArray[np.floating] | None = None,
    ) -> MLR:
        xy = _validate_xy("station_xy", station_xy)
        y = _validate_values(xy.shape[0], values)
        feats = _validate_features("station_features", xy.shape[0], station_features)

        X = self._design_matrix(xy, feats)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        # Avoid divide-by-zero on a constant column.
        self._std = np.where(self._std == 0, 1.0, self._std)
        Xn = (X - self._mean) / self._std
        Xd = np.column_stack([np.ones(len(y)), Xn])

        self._y_mean = float(y.mean())
        yc = y - self._y_mean
        self._beta, *_ = np.linalg.lstsq(Xd, yc, rcond=None)

        if self.use_residual_idw:
            trend_at_stations = Xd @ self._beta + self._y_mean
            self._residuals = y - trend_at_stations
            self._tree = cKDTree(xy)
        return self

    def predict(
        self,
        grid_xy: NDArray[np.floating],
        grid_features: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        if self._beta is None:
            raise RuntimeError("MLR has not been fitted; call .fit(...) first")
        g = _validate_xy("grid_xy", grid_xy)
        gf = _validate_features("grid_features", g.shape[0], grid_features)

        Xg = self._design_matrix(g, gf)
        if self._mean is None or self._std is None:
            raise RuntimeError("MLR internal state is corrupt (mean/std missing)")
        if Xg.shape[1] != self._mean.shape[0]:
            raise ValueError(
                "grid feature count does not match the number of features used at fit "
                f"(expected {self._mean.shape[0] - 2}, got {Xg.shape[1] - 2})"
            )
        Xn = (Xg - self._mean) / self._std
        Xd = np.column_stack([np.ones(len(g)), Xn])
        trend = Xd @ self._beta + self._y_mean

        if self.use_residual_idw and self._residuals is not None and self._tree is not None:
            residual = _idw_predict(
                self._tree, self._residuals, g, self.power, self.k_neighbors
            )
            return trend + residual
        return trend


# ---------------------------------------------------------------------------
# Ridge + IDW residuals (paper's "KED")
# ---------------------------------------------------------------------------
@dataclass
class RegressionIDW:
    """Ridge regression on covariates + IDW-interpolated residuals.

    Also known as *regression kriging* in the geostatistics
    literature, with the IDW residual interpolation standing in for a
    semivariogram-based residual kriging. This is what the paper's
    multi-criteria benchmark labels **KED** — the label is historical
    (inherited from an earlier internal pipeline that did use
    ordinary kriging on the residuals) and is kept here for
    consistency with the published benchmark; the implementation is
    IDW-based.

    Parameters
    ----------
    ridge_alpha : float, default 1.0
        L2 regularisation strength for the Ridge trend.
    power : float, default 2.0
        IDW exponent for the residual correction.
    k_neighbors : int, default 20
        Neighbour count for the residual IDW.
    """

    ridge_alpha: float = 1.0
    power: float = 2.0
    k_neighbors: int = 20

    def __post_init__(self) -> None:
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be >= 0")
        if self.power <= 0:
            raise ValueError("power must be strictly positive")
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        self._feature_mean: NDArray[np.floating] | None = None
        self._feature_std: NDArray[np.floating] | None = None
        self._ridge: Ridge | None = None
        self._residuals: NDArray[np.floating] | None = None
        self._tree: cKDTree | None = None

    def fit(
        self,
        station_xy: NDArray[np.floating],
        values: NDArray[np.floating],
        station_features: NDArray[np.floating] | None = None,
    ) -> RegressionIDW:
        xy = _validate_xy("station_xy", station_xy)
        y = _validate_values(xy.shape[0], values)
        if station_features is None:
            raise ValueError(
                "RegressionIDW requires per-station features; pass station_features"
            )
        feats = _validate_features("station_features", xy.shape[0], station_features)
        assert feats is not None  # for type checkers; _validate_features returned non-None

        self._feature_mean = np.nanmean(feats, axis=0)
        self._feature_std = np.nanstd(feats, axis=0)
        self._feature_std = np.where(self._feature_std == 0, 1.0, self._feature_std)
        Xn = np.nan_to_num(
            (feats - self._feature_mean) / self._feature_std,
            nan=0.0, posinf=0.0, neginf=0.0,
        )

        self._ridge = Ridge(alpha=self.ridge_alpha)
        self._ridge.fit(Xn, y)

        trend_at_stations = self._ridge.predict(Xn)
        self._residuals = y - trend_at_stations
        self._tree = cKDTree(xy)
        return self

    def predict(
        self,
        grid_xy: NDArray[np.floating],
        grid_features: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        if self._ridge is None or self._tree is None or self._residuals is None:
            raise RuntimeError("RegressionIDW has not been fitted; call .fit(...) first")
        if grid_features is None:
            raise ValueError(
                "RegressionIDW requires per-grid features; pass grid_features"
            )
        g = _validate_xy("grid_xy", grid_xy)
        gf = _validate_features("grid_features", g.shape[0], grid_features)
        assert gf is not None

        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("RegressionIDW internal state is corrupt (mean/std missing)")
        if gf.shape[1] != self._feature_mean.shape[0]:
            raise ValueError(
                "grid feature count does not match the number of features used at fit "
                f"(expected {self._feature_mean.shape[0]}, got {gf.shape[1]})"
            )
        Xn = np.nan_to_num(
            (gf - self._feature_mean) / self._feature_std,
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        trend = self._ridge.predict(Xn)
        residual = _idw_predict(
            self._tree, self._residuals, g, self.power, self.k_neighbors
        )
        return trend + residual


# ---------------------------------------------------------------------------
# Convenience: whole EOF stack
# ---------------------------------------------------------------------------
def interpolate_eof_stack(
    interpolator: EOFInterpolator,
    station_xy: NDArray[np.floating],
    eof_coefficients: NDArray[np.floating],
    grid_xy: NDArray[np.floating],
    grid_shape: tuple[int, int],
    *,
    station_features: NDArray[np.floating] | None = None,
    grid_features: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Interpolate every column of ``eof_coefficients`` to the grid.

    The same interpolator instance is re-fit to each column (the fit
    is cheap compared with the predict step), so this helper runs
    exactly as fast as a hand-rolled per-column loop while keeping
    the caller's code to one line.

    Parameters
    ----------
    interpolator : EOFInterpolator
        An instance of :class:`IDW`, :class:`MLR` or
        :class:`RegressionIDW`.
    station_xy : (n_stations, 2) array
        Projected station coordinates (same CRS as ``grid_xy``).
    eof_coefficients : (n_stations, K) array
        The :math:`V` matrix from the rPCA step; each column is one
        EOF mode's spatial loading at the stations.
    grid_xy : (n_cells, 2) array
        Projected grid coordinates, flattened row-major.
    grid_shape : (n_rows, n_cols)
        Shape to reshape each interpolated column into.
    station_features, grid_features : array-like, optional
        Per-station and per-grid covariate matrices for
        covariate-aware interpolators. Must be omitted for :class:`IDW`.

    Returns
    -------
    (K, n_rows, n_cols) float32 array
    """
    V = np.asarray(eof_coefficients, dtype=float)
    if V.ndim != 2:
        raise ValueError(f"eof_coefficients must be 2-D; got shape {V.shape!r}")
    if V.shape[0] != len(station_xy):
        raise ValueError(
            f"eof_coefficients has {V.shape[0]} rows but station_xy has "
            f"{len(station_xy)} — must match"
        )
    n_rows, n_cols = grid_shape
    if n_rows * n_cols != len(grid_xy):
        raise ValueError(
            f"grid_shape {grid_shape!r} does not match grid_xy length {len(grid_xy)}"
        )

    n_eofs = V.shape[1]
    out = np.zeros((n_eofs, n_rows, n_cols), dtype=np.float32)
    for k in range(n_eofs):
        interpolator.fit(station_xy, V[:, k], station_features=station_features)
        pred = interpolator.predict(grid_xy, grid_features=grid_features)
        out[k] = pred.reshape(n_rows, n_cols).astype(np.float32, copy=False)
    return out
