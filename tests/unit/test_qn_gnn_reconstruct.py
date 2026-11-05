"""Unit tests for :mod:`austrianwindatlas.stage1.qn_gnn.reconstruct`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from austrianwindatlas.io.windfield import GridSpec
from austrianwindatlas.stage1.qn_gnn.calibrator import SpatialWindCalibrator
from austrianwindatlas.stage1.qn_gnn.reconstruct import reconstruct_month

torch = pytest.importorskip("torch")

from austrianwindatlas.stage1.qn_gnn.reconstruct import compute_cnn_U  # noqa: E402
from austrianwindatlas.stage1.qn_gnn.temporal_cnn.model import build_temporal_cnn  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _small_grid() -> GridSpec:
    return GridSpec(xmin=0.0, xmax=4.0, ymin=0.0, ymax=2.0,
                    ncols=5, nrows=3, crs="EPSG:31287")


def _simple_calibrator(n_stations=10):
    import pandas as pd

    rng = np.random.default_rng(0)
    alts = np.linspace(200, 2_000, n_stations)
    wind = 2.0 + alts / 500.0 + rng.standard_normal((300, n_stations)) * 0.5
    meta = pd.DataFrame({"alt": alts})
    return SpatialWindCalibrator(meta, wind)


# ---------------------------------------------------------------------------
# compute_cnn_U
# ---------------------------------------------------------------------------
def test_compute_cnn_U_output_shape_and_dtype():
    model = build_temporal_cnn(n_stations=6, n_eofs=3, window=8, proj_dim=16)
    wind = np.random.default_rng(0).uniform(1, 10, size=(32, 6)).astype(np.float32)
    sigma = np.array([5.0, 3.0, 1.0], dtype=np.float32)
    out = compute_cnn_U(model, wind, sigma, window=8, batch_size=16)
    assert out.shape == (32, 3)
    assert out.dtype == np.float32


def test_compute_cnn_U_zero_at_init_head():
    """Zero-init head → ΔU ≈ 0 at init."""
    model = build_temporal_cnn(n_stations=5, n_eofs=2, window=8, proj_dim=16)
    wind = np.random.default_rng(0).uniform(1, 10, size=(20, 5)).astype(np.float32)
    sigma = np.ones(2, dtype=np.float32) * 3.0
    out = compute_cnn_U(model, wind, sigma, window=8, batch_size=16)
    assert np.allclose(out, 0.0)


def test_compute_cnn_U_handles_nan_input():
    model = build_temporal_cnn(n_stations=5, n_eofs=2, window=8, proj_dim=16)
    rng = np.random.default_rng(0)
    wind = rng.uniform(1, 10, size=(20, 5)).astype(np.float32)
    wind[rng.random(wind.shape) < 0.1] = np.nan
    sigma = np.array([3.0, 1.0], dtype=np.float32)
    out = compute_cnn_U(model, wind, sigma, window=8)
    assert np.isfinite(out).all()


def test_compute_cnn_U_scales_by_sigma():
    """Doubling sigma should halve the returned ΔU."""
    model = build_temporal_cnn(n_stations=5, n_eofs=2, window=8, proj_dim=16)
    # Make head slightly non-zero so the test can see the scaling.
    with torch.no_grad():
        model.head.weight.add_(torch.randn_like(model.head.weight) * 0.01)
        model.head.bias.add_(torch.randn_like(model.head.bias) * 0.01)

    wind = np.random.default_rng(0).uniform(1, 10, size=(16, 5)).astype(np.float32)
    s1 = np.array([2.0, 2.0], dtype=np.float32)
    s2 = s1 * 2.0
    u1 = compute_cnn_U(model, wind, s1, window=8)
    u2 = compute_cnn_U(model, wind, s2, window=8)
    # ΔU = raw_output / sigma → doubling sigma halves ΔU.
    np.testing.assert_allclose(u2, u1 * 0.5, atol=1e-6)


def test_compute_cnn_U_rejects_wrong_dims():
    model = build_temporal_cnn(n_stations=5, n_eofs=2, window=8, proj_dim=16)
    with pytest.raises(ValueError, match="2-D"):
        compute_cnn_U(model, np.zeros(10), np.array([1.0]), window=8)
    with pytest.raises(ValueError, match="1-D"):
        compute_cnn_U(model, np.zeros((10, 5)), np.zeros((2, 2)), window=8)


# ---------------------------------------------------------------------------
# reconstruct_month
# ---------------------------------------------------------------------------
def _assemble_reconstruct_inputs():
    grid = _small_grid()
    k = 2
    ny, nx = grid.nrows, grid.ncols
    rng = np.random.default_rng(0)
    V_grid = rng.standard_normal((k, ny, nx)) * 0.3
    sigma = np.array([3.0, 1.5])
    ts = pd.date_range("2020-01-01", periods=48, freq="h")
    U = rng.standard_normal((48, k)) * 0.5
    tmean = rng.standard_normal(48) * 0.2
    elev = rng.uniform(500, 1_500, size=(ny, nx))
    calibrator = _simple_calibrator()
    return grid, V_grid, sigma, U, tmean, ts, elev, calibrator


def test_reconstruct_month_shape_and_physical_bounds():
    grid, V_grid, sigma, U, tmean, ts, elev, cal = _assemble_reconstruct_inputs()
    ds = reconstruct_month(
        V_grid, sigma, U, tmean, ts,
        elevation_grid=elev, calibrator=cal,
        year=2020, month=1, grid=grid,
    )
    assert ds is not None
    assert ds["wind_speed"].dims == ("time", "y", "x")
    assert ds.sizes == {"time": 48, "y": grid.nrows, "x": grid.ncols}
    assert float(ds["wind_speed"].min()) >= 0.0
    assert float(ds["wind_speed"].max()) <= 50.0
    assert ds.attrs["background_field"] == "none"
    assert ds.attrs["cnn_mode"] == "none"
    assert ds.attrs["centering"] == "zscore_temporal"


def test_reconstruct_month_returns_none_for_empty_month():
    grid, V_grid, sigma, U, tmean, ts, elev, cal = _assemble_reconstruct_inputs()
    out = reconstruct_month(
        V_grid, sigma, U, tmean, ts,
        elevation_grid=elev, calibrator=cal,
        year=2020, month=3, grid=grid,            # ts doesn't cover March
    )
    assert out is None


def test_reconstruct_month_additive_mode_adds_delta_U():
    grid, V_grid, sigma, U, tmean, ts, elev, cal = _assemble_reconstruct_inputs()
    delta_U = np.ones_like(U) * 0.1
    ds_no = reconstruct_month(
        V_grid, sigma, U, tmean, ts,
        elevation_grid=elev, calibrator=cal,
        year=2020, month=1, grid=grid,
    )
    ds_add = reconstruct_month(
        V_grid, sigma, U, tmean, ts,
        elevation_grid=elev, calibrator=cal,
        year=2020, month=1, grid=grid,
        delta_U=delta_U, cnn_mode="additive",
    )
    assert ds_add.attrs["cnn_mode"] == "additive"
    # Different reconstructions → fields differ.
    assert not np.allclose(ds_no["wind_speed"].values, ds_add["wind_speed"].values)


def test_reconstruct_month_replace_mode_discards_U_gnn():
    """With cnn_mode='replace', using U_gnn=random and delta_U=0 should
    be identical to using U_gnn=0 and delta_U=0 — i.e. only delta_U
    enters the formula.
    """
    grid, V_grid, sigma, U, tmean, ts, elev, cal = _assemble_reconstruct_inputs()
    zeros = np.zeros_like(U)
    a = reconstruct_month(
        V_grid, sigma, U, tmean, ts,
        elevation_grid=elev, calibrator=cal,
        year=2020, month=1, grid=grid,
        delta_U=zeros, cnn_mode="replace",
    )
    b = reconstruct_month(
        V_grid, sigma, zeros, tmean, ts,
        elevation_grid=elev, calibrator=cal,
        year=2020, month=1, grid=grid,
        delta_U=zeros, cnn_mode="replace",
    )
    np.testing.assert_allclose(a["wind_speed"].values, b["wind_speed"].values)


def test_reconstruct_month_background_blend():
    grid, V_grid, sigma, U, tmean, ts, elev, cal = _assemble_reconstruct_inputs()
    bg = np.ones((48, grid.nrows, grid.ncols)) * 10.0
    ds = reconstruct_month(
        V_grid, sigma, U, tmean, ts,
        elevation_grid=elev, calibrator=cal,
        year=2020, month=1, grid=grid,
        background=bg, background_weight=1.0, background_source="era5",
    )
    # Weight=1 → pure background, clipped at [0, max_wind].
    np.testing.assert_allclose(ds["wind_speed"].values, 10.0, atol=1e-5)
    assert ds.attrs["background_field"] == "era5"


def test_reconstruct_month_rejects_grid_shape_mismatch():
    grid, V_grid, sigma, U, tmean, ts, elev, cal = _assemble_reconstruct_inputs()
    bad_V = V_grid[:, :, :-1]                  # wrong nx
    with pytest.raises(ValueError, match="does not match grid"):
        reconstruct_month(
            bad_V, sigma, U, tmean, ts,
            elevation_grid=elev, calibrator=cal,
            year=2020, month=1, grid=grid,
        )


def test_reconstruct_month_rejects_elevation_shape_mismatch():
    grid, V_grid, sigma, U, tmean, ts, _, cal = _assemble_reconstruct_inputs()
    bad_elev = np.zeros((grid.nrows + 1, grid.ncols))
    with pytest.raises(ValueError, match="elevation_grid"):
        reconstruct_month(
            V_grid, sigma, U, tmean, ts,
            elevation_grid=bad_elev, calibrator=cal,
            year=2020, month=1, grid=grid,
        )


def test_reconstruct_month_rejects_unknown_cnn_mode():
    grid, V_grid, sigma, U, tmean, ts, elev, cal = _assemble_reconstruct_inputs()
    with pytest.raises(ValueError, match="unknown CNN mode"):
        reconstruct_month(
            V_grid, sigma, U, tmean, ts,
            elevation_grid=elev, calibrator=cal,
            year=2020, month=1, grid=grid,
            delta_U=np.zeros_like(U), cnn_mode="bogus",  # type: ignore[arg-type]
        )
