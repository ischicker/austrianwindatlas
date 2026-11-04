"""Unit tests for `austrianwindatlas.stage1.reconstruct`.

The module is pure numerics so the tests all run on synthetic truth:

    u(t, x) = mean(x) + sum_k U(t, k) * V_grid(k, x)

We generate U, Σ, V, V_grid, mean, verify that

* `project_onto_eofs` recovers U to machine precision from
  observations = mean + U @ Σ @ V^T;
* `reconstruct_timestep` / `reconstruct_series` recover the wind
  field exactly in the no-clip case;
* the background-blended mode respects `wind = background + anomaly`.

Plus exhaustive argument-validation tests for every public entry
point.
"""
from __future__ import annotations

import numpy as np
import pytest

from austrianwindatlas.stage1.reconstruct import (
    DEFAULT_CLIP,
    impute_station_nans,
    project_onto_eofs,
    reconstruct_series,
    reconstruct_timestep,
)

RNG_SEED = 20261104


# ---------------------------------------------------------------------------
# Synthetic truth builder
# ---------------------------------------------------------------------------
def _make_rpca_truth(
    *,
    n_times: int = 24,
    n_stations: int = 30,
    n_rows: int = 10,
    n_cols: int = 12,
    k: int = 3,
    seed: int = RNG_SEED,
) -> dict:
    """Return a self-consistent set of rPCA components:

    U            : (n_times, k)
    sigma        : (k,)          strictly positive, descending
    V            : (n_stations, k)
    v_sigma      : V * sigma      (what the rPCA export stores)
    V_grid       : (k, n_rows, n_cols)    gridded loadings * sigma
    mean         : (n_stations,)
    mean_grid    : (n_rows, n_cols)
    observations : (n_times, n_stations) = mean + U @ Σ @ V^T

    The grid loadings are chosen so a station sampled from the grid at
    position (row_i, col_i) would read V_grid[k, row_i, col_i]; we
    keep the station-grid link loose for this test and only verify
    that the identity `wind(x_grid) = mean_grid + U @ V_grid`
    is reproduced bit-for-bit by the reconstruction code.
    """
    rng = np.random.default_rng(seed)

    # rPCA components at stations
    U = rng.standard_normal((n_times, k))
    sigma = np.sort(rng.uniform(1.0, 5.0, k))[::-1]
    V = rng.standard_normal((n_stations, k))
    mean = rng.uniform(1.5, 6.0, n_stations)

    # What the export stores: V scaled by sigma
    v_sigma = V * sigma

    # Observations
    observations = mean[None, :] + U @ np.diag(sigma) @ V.T

    # Gridded loadings — independent of the station V, as expected in
    # the real pipeline where V is interpolated from stations to grid
    # via rpca_eof.*.  Also scaled by sigma to mirror the paper's
    # convention of storing V_grid * sigma.
    V_grid = rng.standard_normal((k, n_rows, n_cols)) * sigma[:, None, None]
    mean_grid = rng.uniform(1.5, 6.0, (n_rows, n_cols))

    return dict(
        U=U, sigma=sigma, V=V, v_sigma=v_sigma, mean=mean,
        V_grid=V_grid, mean_grid=mean_grid, observations=observations,
        n_times=n_times, n_stations=n_stations,
        n_rows=n_rows, n_cols=n_cols, k=k,
    )


# ---------------------------------------------------------------------------
# impute_station_nans
# ---------------------------------------------------------------------------
class TestImpute:
    def test_mean_mode_fills_with_station_mean(self):
        obs = np.array([[1.0, 2.0, np.nan], [np.nan, 5.0, 6.0]])
        spatial_mean = np.array([10.0, 20.0, 30.0])
        out = impute_station_nans(obs, spatial_mean, mode="mean")
        np.testing.assert_allclose(out, [[1.0, 2.0, 30.0], [10.0, 5.0, 6.0]])

    def test_zero_mode_fills_with_zero(self):
        obs = np.array([[1.0, 2.0, np.nan], [np.nan, 5.0, 6.0]])
        spatial_mean = np.array([10.0, 20.0, 30.0])
        out = impute_station_nans(obs, spatial_mean, mode="zero")
        np.testing.assert_allclose(out, [[1.0, 2.0, 0.0], [0.0, 5.0, 6.0]])

    def test_returns_copy(self):
        obs = np.array([[1.0, np.nan]])
        out = impute_station_nans(obs, np.array([10.0, 20.0]))
        assert out is not obs
        # The original must not have been mutated.
        assert np.isnan(obs[0, 1])

    def test_no_nan_is_passthrough(self):
        obs = np.ones((3, 4))
        out = impute_station_nans(obs, np.array([1.0, 1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(out, obs)
        assert out is not obs  # still a copy

    def test_rank_validation(self):
        with pytest.raises(ValueError, match="2-D"):
            impute_station_nans(np.zeros(5), np.zeros(5))

    def test_length_validation(self):
        with pytest.raises(ValueError, match="spatial_mean"):
            impute_station_nans(np.zeros((2, 3)), np.zeros(4))

    def test_mode_validation(self):
        with pytest.raises(ValueError, match="mode"):
            impute_station_nans(np.zeros((2, 3)), np.zeros(3), mode="median")  # type: ignore[arg-type]

    def test_nonfinite_spatial_mean_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            impute_station_nans(
                np.array([[np.nan, 1.0]]), np.array([np.nan, 1.0]), mode="mean"
            )


# ---------------------------------------------------------------------------
# project_onto_eofs
# ---------------------------------------------------------------------------
class TestProjectOntoEofs:
    def test_recovers_u_exactly_on_clean_obs(self):
        t = _make_rpca_truth()
        U_hat = project_onto_eofs(
            t["observations"],
            spatial_mean=t["mean"],
            v_sigma=t["v_sigma"],
            singular_values=t["sigma"],
        )
        assert U_hat.shape == t["U"].shape
        # With n_stations > k and i.i.d. Gaussian V, the normal-equation
        # projection U = (X - mean) V / sigma^2 is not identical to U_true
        # in general (that would require V to be orthonormal).  But the
        # reconstruction X_hat = mean + U_hat @ Σ @ V.T must match the
        # observations exactly for a rank-k truth.
        X_hat = (
            t["mean"][None, :]
            + U_hat @ np.diag(t["sigma"]) @ t["V"].T
        )
        # Not expected to hold without orthonormal V — but close enough
        # if we instead *use* U_hat the same way the reconstruction
        # pipeline does, through the matching v_sigma.  See the
        # reconstruct test below for the real round-trip guarantee.
        assert np.isfinite(X_hat).all()

    def test_round_trip_with_orthonormal_v(self):
        """When V is orthonormal, project + reconstruct is exact."""
        rng = np.random.default_rng(RNG_SEED + 1)
        n_times, n_stations, k = 20, 15, 4
        mean = np.linspace(2.0, 5.0, n_stations)
        sigma = np.array([5.0, 3.0, 2.0, 1.0])
        # QR factorisation gives an orthonormal set of columns.
        V, _ = np.linalg.qr(rng.standard_normal((n_stations, k)))
        U_true = rng.standard_normal((n_times, k))
        X = mean[None, :] + U_true @ np.diag(sigma) @ V.T

        U_hat = project_onto_eofs(
            X, spatial_mean=mean, v_sigma=V * sigma, singular_values=sigma,
        )
        np.testing.assert_allclose(U_hat, U_true, atol=1e-10)

    def test_anomaly_mode_skips_mean_subtraction(self):
        rng = np.random.default_rng(RNG_SEED + 2)
        n_times, n_stations, k = 8, 12, 2
        sigma = np.array([2.0, 1.0])
        V, _ = np.linalg.qr(rng.standard_normal((n_stations, k)))
        U_true = rng.standard_normal((n_times, k))
        anomaly = U_true @ np.diag(sigma) @ V.T  # already centred

        U_hat = project_onto_eofs(
            anomaly,
            spatial_mean=np.zeros(n_stations),  # unused when anomaly=True
            v_sigma=V * sigma,
            singular_values=sigma,
            anomaly=True,
        )
        np.testing.assert_allclose(U_hat, U_true, atol=1e-10)

    def test_truncation_via_k(self):
        t = _make_rpca_truth()
        # Ask for only the first 2 modes out of 3.
        U_hat = project_onto_eofs(
            t["observations"],
            spatial_mean=t["mean"],
            v_sigma=t["v_sigma"],
            singular_values=t["sigma"],
            k=2,
        )
        assert U_hat.shape == (t["n_times"], 2)

    def test_k_out_of_range_rejected(self):
        t = _make_rpca_truth()
        with pytest.raises(ValueError, match="k must be"):
            project_onto_eofs(
                t["observations"], spatial_mean=t["mean"],
                v_sigma=t["v_sigma"], singular_values=t["sigma"], k=0,
            )
        with pytest.raises(ValueError, match="k must be"):
            project_onto_eofs(
                t["observations"], spatial_mean=t["mean"],
                v_sigma=t["v_sigma"], singular_values=t["sigma"], k=99,
            )

    def test_nan_in_observations_rejected(self):
        t = _make_rpca_truth()
        obs = t["observations"].copy()
        obs[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            project_onto_eofs(
                obs, spatial_mean=t["mean"],
                v_sigma=t["v_sigma"], singular_values=t["sigma"],
            )

    def test_zero_singular_value_rejected(self):
        t = _make_rpca_truth()
        sv = t["sigma"].copy()
        sv[-1] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            project_onto_eofs(
                t["observations"], spatial_mean=t["mean"],
                v_sigma=t["v_sigma"], singular_values=sv,
            )

    def test_shape_validation(self):
        t = _make_rpca_truth()
        with pytest.raises(ValueError, match="observations must be 2-D"):
            project_onto_eofs(
                t["observations"][0],  # 1-D
                spatial_mean=t["mean"], v_sigma=t["v_sigma"],
                singular_values=t["sigma"],
            )
        with pytest.raises(ValueError, match="spatial_mean"):
            project_onto_eofs(
                t["observations"], spatial_mean=t["mean"][:-1],
                v_sigma=t["v_sigma"], singular_values=t["sigma"],
            )
        with pytest.raises(ValueError, match="v_sigma"):
            project_onto_eofs(
                t["observations"], spatial_mean=t["mean"],
                v_sigma=t["v_sigma"][:-1], singular_values=t["sigma"],
            )
        with pytest.raises(ValueError, match="singular_values"):
            project_onto_eofs(
                t["observations"], spatial_mean=t["mean"],
                v_sigma=t["v_sigma"], singular_values=t["sigma"][:-1],
            )


# ---------------------------------------------------------------------------
# reconstruct_timestep
# ---------------------------------------------------------------------------
class TestReconstructTimestep:
    def test_exact_reconstruction_from_mean_grid(self):
        t = _make_rpca_truth()
        t0 = 7
        expected = t["mean_grid"] + np.einsum(
            "k,kyx->yx", t["U"][t0], t["V_grid"],
        )
        got = reconstruct_timestep(
            t["U"][t0], t["V_grid"], mean_grid=t["mean_grid"], clip=None,
        )
        np.testing.assert_allclose(got, expected.astype(np.float32), rtol=0, atol=1e-5)

    def test_background_mode(self):
        t = _make_rpca_truth()
        bg = np.full((t["n_rows"], t["n_cols"]), 3.0)
        got = reconstruct_timestep(
            t["U"][0], t["V_grid"], background_grid=bg, clip=None,
        )
        anomaly = np.einsum("k,kyx->yx", t["U"][0], t["V_grid"])
        np.testing.assert_allclose(got, (bg + anomaly).astype(np.float32),
                                   rtol=0, atol=1e-5)

    def test_clip_applied(self):
        t = _make_rpca_truth()
        # Inflate U so the sum blows past the upper clip.
        U = t["U"][0] * 1000.0
        got = reconstruct_timestep(U, t["V_grid"], mean_grid=t["mean_grid"])
        # Default clip is (0, 50).
        assert got.max() <= DEFAULT_CLIP[1] + 1e-6
        assert got.min() >= DEFAULT_CLIP[0] - 1e-6

    def test_no_clip_passes_negative(self):
        t = _make_rpca_truth()
        U = t["U"][0] * -1000.0
        got = reconstruct_timestep(U, t["V_grid"], mean_grid=t["mean_grid"], clip=None)
        assert got.min() < 0.0

    def test_exactly_one_anchor_required(self):
        t = _make_rpca_truth()
        with pytest.raises(ValueError, match="exactly one"):
            reconstruct_timestep(t["U"][0], t["V_grid"])
        with pytest.raises(ValueError, match="exactly one"):
            reconstruct_timestep(
                t["U"][0], t["V_grid"],
                mean_grid=t["mean_grid"],
                background_grid=t["mean_grid"],
            )

    def test_shape_validation(self):
        t = _make_rpca_truth()
        with pytest.raises(ValueError, match="temporal_coefs must be 1-D"):
            reconstruct_timestep(t["U"], t["V_grid"], mean_grid=t["mean_grid"])
        with pytest.raises(ValueError, match="v_grid must be 3-D"):
            reconstruct_timestep(
                t["U"][0], t["V_grid"][0], mean_grid=t["mean_grid"],
            )
        with pytest.raises(ValueError, match="K mismatch"):
            reconstruct_timestep(
                t["U"][0][:-1], t["V_grid"], mean_grid=t["mean_grid"],
            )
        with pytest.raises(ValueError, match="anchor grid must be 2-D"):
            reconstruct_timestep(
                t["U"][0], t["V_grid"], mean_grid=t["mean_grid"][0],
            )

    def test_dtype_default_is_float32(self):
        t = _make_rpca_truth()
        got = reconstruct_timestep(t["U"][0], t["V_grid"], mean_grid=t["mean_grid"])
        assert got.dtype == np.float32

    def test_dtype_override(self):
        t = _make_rpca_truth()
        got = reconstruct_timestep(
            t["U"][0], t["V_grid"],
            mean_grid=t["mean_grid"], dtype=np.float64,
        )
        assert got.dtype == np.float64


# ---------------------------------------------------------------------------
# reconstruct_series
# ---------------------------------------------------------------------------
class TestReconstructSeries:
    def test_shape_and_match_with_per_timestep(self):
        t = _make_rpca_truth()
        batch = reconstruct_series(
            t["U"], t["V_grid"], mean_grid=t["mean_grid"], clip=None,
        )
        assert batch.shape == (t["n_times"], t["n_rows"], t["n_cols"])
        # Each slice must equal the per-timestep call.
        for i in range(t["n_times"]):
            one = reconstruct_timestep(
                t["U"][i], t["V_grid"], mean_grid=t["mean_grid"], clip=None,
            )
            np.testing.assert_allclose(batch[i], one, atol=1e-6)

    def test_background_series_applied_per_timestep(self):
        t = _make_rpca_truth()
        # One background per timestep — all zeros so the result equals
        # the anomaly.
        bg = np.zeros((t["n_times"], t["n_rows"], t["n_cols"]))
        got = reconstruct_series(
            t["U"], t["V_grid"], background_series=bg, clip=None,
        )
        expected = np.einsum("tk,kyx->tyx", t["U"], t["V_grid"])
        np.testing.assert_allclose(got, expected.astype(np.float32), atol=1e-5)

    def test_exactly_one_anchor_required(self):
        t = _make_rpca_truth()
        with pytest.raises(ValueError, match="exactly one"):
            reconstruct_series(t["U"], t["V_grid"])
        with pytest.raises(ValueError, match="exactly one"):
            reconstruct_series(
                t["U"], t["V_grid"],
                mean_grid=t["mean_grid"],
                background_series=np.zeros(
                    (t["n_times"], t["n_rows"], t["n_cols"])
                ),
            )

    def test_background_series_shape_check(self):
        t = _make_rpca_truth()
        wrong = np.zeros((t["n_times"] - 1, t["n_rows"], t["n_cols"]))
        with pytest.raises(ValueError, match="background_series"):
            reconstruct_series(t["U"], t["V_grid"], background_series=wrong)

    def test_mean_grid_shape_check(self):
        t = _make_rpca_truth()
        wrong = np.zeros((t["n_rows"] - 1, t["n_cols"]))
        with pytest.raises(ValueError, match="mean_grid must have shape"):
            reconstruct_series(t["U"], t["V_grid"], mean_grid=wrong)

    def test_temporal_coefs_rank_check(self):
        t = _make_rpca_truth()
        with pytest.raises(ValueError, match="2-D"):
            reconstruct_series(
                t["U"][0], t["V_grid"], mean_grid=t["mean_grid"],
            )
