"""Unit tests for `austrianwindatlas.stage1.rpca_eof`.

Each interpolator is tested against a known analytic field on a
small synthetic station-and-grid setup. The tests cover:

* API contract (fit/predict signatures, Protocol membership).
* Constructor validation.
* Shape and dtype of predictions.
* Predict-at-station recovery (exactness for IDW; regression-trend
  + residual recovery for MLR and RegressionIDW).
* Correct recovery of a pure linear field by MLR.
* Correct recovery of a pure-covariate field by RegressionIDW.
* Degenerate / edge cases (single-feature column, constant field,
  k > n_stations, NaN features).
* ``interpolate_eof_stack`` applies the interpolator to each EOF
  column and shapes the output correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

from austrianwindatlas.stage1.rpca_eof import (
    IDW,
    MLR,
    EOFInterpolator,
    RegressionIDW,
    interpolate_eof_stack,
)

RNG_SEED = 20261104


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
def _grid_xy(n_rows: int = 20, n_cols: int = 25) -> tuple[np.ndarray, tuple[int, int]]:
    """Return a flattened (n_cells, 2) coordinate array and the shape."""
    xs = np.linspace(0.0, 100.0, n_cols)
    ys = np.linspace(0.0, 80.0, n_rows)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()]), (n_rows, n_cols)


def _station_xy(n: int = 30, seed: int = RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [rng.uniform(0.0, 100.0, n), rng.uniform(0.0, 80.0, n)]
    )


# ---------------------------------------------------------------------------
# Protocol membership
# ---------------------------------------------------------------------------
class TestProtocol:
    def test_all_three_match_protocol_at_runtime(self):
        for inst in (IDW(), MLR(), RegressionIDW()):
            assert isinstance(inst, EOFInterpolator)


# ---------------------------------------------------------------------------
# IDW
# ---------------------------------------------------------------------------
class TestIDW:
    def test_constructor_rejects_bad_params(self):
        with pytest.raises(ValueError):
            IDW(power=0)
        with pytest.raises(ValueError):
            IDW(power=-1)
        with pytest.raises(ValueError):
            IDW(k_neighbors=0)

    def test_predict_shape_matches_grid(self):
        xy_s = _station_xy()
        xy_g, _ = _grid_xy()
        values = np.cos(xy_s[:, 0] / 20.0) + np.sin(xy_s[:, 1] / 15.0)
        pred = IDW().fit(xy_s, values).predict(xy_g)
        assert pred.shape == (xy_g.shape[0],)
        assert np.all(np.isfinite(pred))

    def test_predict_at_station_recovers_value_in_the_limit(self):
        """IDW with distance → 0 should recover the station value."""
        xy_s = _station_xy()
        values = np.cos(xy_s[:, 0] / 20.0) + np.sin(xy_s[:, 1] / 15.0)
        # Query exactly at a station: should return that station's value.
        pred = IDW().fit(xy_s, values).predict(xy_s[:5])
        np.testing.assert_allclose(pred, values[:5], rtol=1e-7, atol=1e-7)

    def test_k_greater_than_n_stations_is_clipped(self):
        xy_s = _station_xy(n=4)
        values = np.array([1.0, 2.0, 3.0, 4.0])
        xy_g, _ = _grid_xy(n_rows=3, n_cols=3)
        # k=20 default, only 4 stations → silently uses all 4.
        pred = IDW(k_neighbors=20).fit(xy_s, values).predict(xy_g)
        assert pred.shape == (9,)
        assert np.all(np.isfinite(pred))
        # All predictions must fall inside the convex hull of station values.
        assert pred.min() >= values.min() - 1e-8
        assert pred.max() <= values.max() + 1e-8

    def test_constant_field_stays_constant(self):
        xy_s = _station_xy()
        values = np.full(len(xy_s), 7.5)
        xy_g, _ = _grid_xy()
        pred = IDW().fit(xy_s, values).predict(xy_g)
        np.testing.assert_allclose(pred, 7.5)

    def test_predict_before_fit_raises(self):
        xy_g, _ = _grid_xy()
        with pytest.raises(RuntimeError):
            IDW().predict(xy_g)

    def test_fit_rejects_bad_xy_shape(self):
        with pytest.raises(ValueError):
            IDW().fit(np.zeros((10, 3)), np.zeros(10))
        with pytest.raises(ValueError):
            IDW().fit(np.zeros((10,)), np.zeros(10))

    def test_fit_rejects_value_length_mismatch(self):
        xy_s = _station_xy(n=5)
        with pytest.raises(ValueError):
            IDW().fit(xy_s, np.zeros(4))


# ---------------------------------------------------------------------------
# MLR
# ---------------------------------------------------------------------------
class TestMLR:
    def test_constructor_rejects_bad_params(self):
        with pytest.raises(ValueError):
            MLR(power=0)
        with pytest.raises(ValueError):
            MLR(k_neighbors=0)

    def test_recovers_pure_linear_field_in_coords_only(self):
        # f(x, y) = 2.5*x - 1.3*y + 7
        xy_s = _station_xy()
        values = 2.5 * xy_s[:, 0] - 1.3 * xy_s[:, 1] + 7.0
        xy_g, shape = _grid_xy()

        pred = MLR(use_residual_idw=False).fit(xy_s, values).predict(xy_g)
        truth = 2.5 * xy_g[:, 0] - 1.3 * xy_g[:, 1] + 7.0
        np.testing.assert_allclose(pred, truth, rtol=1e-8, atol=1e-6)
        assert pred.reshape(shape).shape == shape

    def test_recovers_linear_field_plus_covariate(self):
        xy_s = _station_xy()
        elev_s = 100.0 + xy_s[:, 0] * 2.0 + xy_s[:, 1] * 0.5
        values = -0.02 * elev_s + 0.3 * xy_s[:, 0]
        xy_g, _ = _grid_xy()
        elev_g = 100.0 + xy_g[:, 0] * 2.0 + xy_g[:, 1] * 0.5

        pred = (
            MLR(use_residual_idw=False)
            .fit(xy_s, values, station_features=elev_s[:, None])
            .predict(xy_g, grid_features=elev_g[:, None])
        )
        truth = -0.02 * elev_g + 0.3 * xy_g[:, 0]
        np.testing.assert_allclose(pred, truth, rtol=1e-8, atol=1e-6)

    def test_residual_idw_closes_the_residual_at_stations(self):
        xy_s = _station_xy()
        # A non-linear component the OLS trend cannot capture:
        values = 0.5 * xy_s[:, 0] + np.sin(xy_s[:, 0] / 10.0) * 3.0
        mlr = MLR(use_residual_idw=True).fit(xy_s, values)
        # Querying at station positions should recover the station values
        # (trend + IDW residual with distance → 0).
        pred = mlr.predict(xy_s)
        np.testing.assert_allclose(pred, values, rtol=1e-6, atol=1e-6)

    def test_grid_feature_count_mismatch_raises(self):
        xy_s = _station_xy()
        xy_g, _ = _grid_xy()
        feats_s = np.zeros((xy_s.shape[0], 3))
        feats_g = np.zeros((xy_g.shape[0], 2))
        model = MLR().fit(xy_s, np.ones(len(xy_s)), station_features=feats_s)
        with pytest.raises(ValueError):
            model.predict(xy_g, grid_features=feats_g)

    def test_tolerates_nan_in_features(self):
        xy_s = _station_xy(n=10)
        feats_s = np.ones((10, 2))
        feats_s[0, 0] = np.nan
        values = np.arange(10, dtype=float)
        # Must not raise; NaN is zero-filled per module contract.
        MLR().fit(xy_s, values, station_features=feats_s)


# ---------------------------------------------------------------------------
# RegressionIDW (the paper's "KED")
# ---------------------------------------------------------------------------
class TestRegressionIDW:
    def test_constructor_rejects_bad_params(self):
        with pytest.raises(ValueError):
            RegressionIDW(ridge_alpha=-0.1)
        with pytest.raises(ValueError):
            RegressionIDW(power=0)
        with pytest.raises(ValueError):
            RegressionIDW(k_neighbors=0)

    def test_requires_features_at_fit_and_predict(self):
        xy_s = _station_xy()
        xy_g, _ = _grid_xy()
        with pytest.raises(ValueError):
            RegressionIDW().fit(xy_s, np.ones(len(xy_s)))
        model = RegressionIDW().fit(
            xy_s, np.ones(len(xy_s)), station_features=np.zeros((len(xy_s), 1))
        )
        with pytest.raises(ValueError):
            model.predict(xy_g)

    def test_recovers_covariate_field_with_small_ridge(self):
        xy_s = _station_xy()
        elev_s = 100.0 + xy_s[:, 0] * 2.0 - xy_s[:, 1]
        values = 0.05 * elev_s - 1.0
        xy_g, _ = _grid_xy()
        elev_g = 100.0 + xy_g[:, 0] * 2.0 - xy_g[:, 1]

        pred = (
            RegressionIDW(ridge_alpha=1e-6)
            .fit(xy_s, values, station_features=elev_s[:, None])
            .predict(xy_g, grid_features=elev_g[:, None])
        )
        truth = 0.05 * elev_g - 1.0
        # Ridge with tiny alpha is OLS-equivalent in behaviour; the
        # IDW-residual correction is tiny (close to zero everywhere
        # since the model fits the data exactly) so recovery should
        # be excellent.
        np.testing.assert_allclose(pred, truth, rtol=1e-3, atol=1e-3)

    def test_station_feature_count_mismatch_raises(self):
        xy_s = _station_xy()
        xy_g, _ = _grid_xy()
        model = RegressionIDW().fit(
            xy_s, np.ones(len(xy_s)), station_features=np.zeros((len(xy_s), 3))
        )
        with pytest.raises(ValueError):
            model.predict(xy_g, grid_features=np.zeros((xy_g.shape[0], 2)))

    def test_constant_feature_column_does_not_break(self):
        xy_s = _station_xy(n=8)
        feats = np.ones((8, 2))  # Both columns constant → std=0
        feats[:, 1] = np.arange(8)
        values = np.arange(8, dtype=float)
        model = RegressionIDW(ridge_alpha=0.1).fit(
            xy_s, values, station_features=feats
        )
        # Predict at stations — residual IDW should close the residual
        # (ridge trend is nearly zero because the useful feature has
        # non-zero std).
        pred = model.predict(xy_s, grid_features=feats)
        np.testing.assert_allclose(pred, values, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# interpolate_eof_stack
# ---------------------------------------------------------------------------
class TestEOFStack:
    def test_shape_and_dtype(self):
        xy_s = _station_xy()
        xy_g, shape = _grid_xy()
        K = 4
        V = np.random.default_rng(1).standard_normal((xy_s.shape[0], K))
        out = interpolate_eof_stack(IDW(), xy_s, V, xy_g, shape)
        assert out.shape == (K, *shape)
        assert out.dtype == np.float32

    def test_rejects_bad_v_shape(self):
        xy_s = _station_xy(n=5)
        xy_g, shape = _grid_xy(n_rows=3, n_cols=4)
        with pytest.raises(ValueError):
            interpolate_eof_stack(IDW(), xy_s, np.zeros(5), xy_g, shape)  # 1-D
        with pytest.raises(ValueError):
            interpolate_eof_stack(
                IDW(), xy_s, np.zeros((6, 3)), xy_g, shape
            )  # wrong n_stations

    def test_rejects_grid_shape_mismatch(self):
        xy_s = _station_xy(n=5)
        xy_g, _ = _grid_xy(n_rows=3, n_cols=4)  # 12 cells
        V = np.zeros((5, 2))
        with pytest.raises(ValueError):
            interpolate_eof_stack(IDW(), xy_s, V, xy_g, (3, 5))  # 15 cells

    def test_per_column_results_match_direct_call(self):
        xy_s = _station_xy()
        xy_g, shape = _grid_xy()
        rng = np.random.default_rng(2)
        V = rng.standard_normal((xy_s.shape[0], 3))

        stacked = interpolate_eof_stack(IDW(), xy_s, V, xy_g, shape)
        for k in range(3):
            direct = IDW().fit(xy_s, V[:, k]).predict(xy_g).reshape(shape)
            np.testing.assert_allclose(stacked[k], direct.astype(np.float32), rtol=1e-6)

    def test_works_with_mlr_and_features(self):
        xy_s = _station_xy()
        xy_g, shape = _grid_xy()
        feats_s = np.column_stack([xy_s[:, 0], xy_s[:, 1] ** 2])
        feats_g = np.column_stack([xy_g[:, 0], xy_g[:, 1] ** 2])
        rng = np.random.default_rng(3)
        V = rng.standard_normal((xy_s.shape[0], 2))
        out = interpolate_eof_stack(
            MLR(),
            xy_s,
            V,
            xy_g,
            shape,
            station_features=feats_s,
            grid_features=feats_g,
        )
        assert out.shape == (2, *shape)
        assert np.all(np.isfinite(out))
