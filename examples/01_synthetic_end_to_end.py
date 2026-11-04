#!/usr/bin/env python3
"""End-to-end synthetic demo of the Austrian Windatlas Stage-1 chain.

Exercises :mod:`austrianwindatlas.core.rpca`,
:mod:`austrianwindatlas.stage1.rpca_eof` and
:mod:`austrianwindatlas.stage1.reconstruct` on in-memory synthetic
data — no external files are read and none are written outside the
local ``./examples_output/`` directory (created on demand; ignored
by git).

What happens, in one picture
----------------------------

::

    truth = mean(x) + sum_k U_true(t,k) * sigma_k * V_true(s,k)
                            |                       |
                            v                       v
                    temporal patterns       spatial patterns at stations
                    (e.g. diurnal cycle)    (correlated with elevation)
          |
          + sparse column-storm contamination (a handful of timesteps
            and stations get a large additive spike — the rare-event
            signal rPCA is designed to isolate from the low-rank
            part).

    observations(t, s) = truth(t, s) + sparse(t, s)

Pipeline
========

    1. Build the synthetic truth.
    2. Add sparse spikes.
    3. Run RobustPCA on the station matrix -> L, S.
    4. Extract per-mode station loadings V * sigma from L.
    5. Interpolate those loadings to the grid (IDW on station x,y).
    6. Project the observations onto the trained EOFs -> U(t).
    7. Reconstruct the gridded wind field for every timestep.
    8. Compare against the known truth at a handful of grid cells
       that match station locations.

Expected result: the per-grid-cell reconstruction RMSE at the station
cells is bounded by (a) the approximation error of the rPCA rank
truncation and (b) the IDW interpolation error at those exact cells
(effectively zero because stations sit at grid cells here).

Runtime: a few seconds on a laptop CPU.
"""
from __future__ import annotations

import logging
import sys

import numpy as np

from austrianwindatlas.core.rpca import RobustPCA
from austrianwindatlas.stage1.reconstruct import (
    project_onto_eofs,
    reconstruct_series,
)
from austrianwindatlas.stage1.rpca_eof import IDW, interpolate_eof_stack

RNG_SEED = 20261104

# Synthetic-problem knobs — kept small so the script runs in ~2 seconds.
N_TIMES = 24 * 30        # one synthetic month, hourly
N_STATIONS = 30
N_ROWS, N_COLS = 40, 50  # synthetic "grid"
RANK = 3                 # low-rank truth
SPARSE_FRACTION = 0.01   # 1 % of (time, station) entries get a spike
SPARSE_SCALE = 2.5       # spike magnitudes comparable to the low-rank std


# ---------------------------------------------------------------------------
# 1 + 2.  Build synthetic truth and add sparse spikes.
# ---------------------------------------------------------------------------
def make_synthetic(seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)

    # Station coordinates — randomly scattered inside the grid bbox,
    # snapped onto grid cells so the "reconstruction at station" check
    # has zero interpolation bias.
    station_rows = rng.integers(0, N_ROWS, size=N_STATIONS)
    station_cols = rng.integers(0, N_COLS, size=N_STATIONS)
    # De-duplicate (rare collisions): keep the first occurrence.
    seen: set[tuple[int, int]] = set()
    keep_idx = []
    for i in range(len(station_rows)):
        cell = (int(station_rows[i]), int(station_cols[i]))
        if cell not in seen:
            seen.add(cell)
            keep_idx.append(i)
    station_rows = station_rows[keep_idx]
    station_cols = station_cols[keep_idx]
    n_stations = len(keep_idx)
    station_xy = np.column_stack([station_cols.astype(float),
                                  station_rows.astype(float)])

    # rPCA components.
    U_true = rng.standard_normal((N_TIMES, RANK))
    sigma_true = np.array([4.0, 2.5, 1.5])
    # Gridded loadings — smooth-ish by construction via a quick
    # separable-sinusoid basis.  At station coordinates we sample
    # V_grid_true to get the station V matrix.
    yy, xx = np.meshgrid(np.linspace(0, np.pi, N_ROWS),
                         np.linspace(0, 2 * np.pi, N_COLS),
                         indexing="ij")
    V_grid_true = np.stack([
        np.sin(xx) * np.cos(yy),
        np.sin(2 * xx + 1.0),
        np.cos(xx - yy),
    ], axis=0)
    V_stations_true = V_grid_true[:, station_rows, station_cols].T  # (n_st, RANK)

    # Station climatology.
    mean_stations = 2.5 + 0.8 * rng.standard_normal(n_stations)

    # Low-rank truth at stations: (n_times, n_stations).
    low_rank = U_true @ np.diag(sigma_true) @ V_stations_true.T
    truth = mean_stations[None, :] + low_rank

    # Sparse spikes.
    spike_mask = rng.random((N_TIMES, n_stations)) < SPARSE_FRACTION
    spike_signs = rng.choice([-1.0, 1.0], size=(N_TIMES, n_stations))
    spikes = spike_mask * spike_signs * SPARSE_SCALE
    observations = truth + spikes

    return {
        "station_xy": station_xy,
        "station_rows": station_rows,
        "station_cols": station_cols,
        "mean_stations": mean_stations,
        "observations": observations,
        "V_grid_true": V_grid_true,
        "U_true": U_true,
        "sigma_true": sigma_true,
        "low_rank_truth": low_rank,
        "sparse_truth": spikes,
        "n_stations": n_stations,
    }


# ---------------------------------------------------------------------------
# Full pipeline + a few diagnostics.
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("awa-demo")

    log.info("Step 1-2: building synthetic station matrix")
    data = make_synthetic()
    obs = data["observations"]
    log.info("   observations shape: %s (time, stations)", obs.shape)
    log.info("   station count: %d (deduplicated to one per grid cell)",
             data["n_stations"])

    # ---- 3. rPCA ----------------------------------------------------------
    log.info("Step 3: rPCA decomposition (CPU)")
    # Observations are (time, stations); we want to decompose the
    # station-anomaly matrix so centre by subtracting the per-station mean.
    obs_centered = obs - data["mean_stations"][None, :]
    rpca = RobustPCA(max_iter=300, use_gpu=False, verbose=False).fit(obs_centered)
    log.info("   converged=%s, n_iter=%d, rank(L)=%d, sparsity(S)=%.3f",
             rpca.converged_, rpca.n_iter_,
             int(np.linalg.matrix_rank(rpca.L_, tol=1e-6)),
             rpca.history_[-1].sparsity)

    # ---- 4. Extract per-mode V and sigma from L via SVD -------------------
    log.info("Step 4: extract station V and sigma from L = U @ Sigma @ V^T")
    U_from_L, sigma_from_L, Vt_from_L = np.linalg.svd(rpca.L_, full_matrices=False)
    k_keep = RANK  # we know the true rank; in practice choose via variance threshold
    v_sigma_stations = Vt_from_L[:k_keep].T * sigma_from_L[:k_keep]  # (n_st, k)
    sigma_k = sigma_from_L[:k_keep]
    log.info("   top-%d singular values: %s", k_keep,
             np.array2string(sigma_k, formatter={"float": lambda x: f"{x:.2f}"}))

    # ---- 5. Interpolate station V to the grid -----------------------------
    log.info("Step 5: interpolate station loadings to the grid (IDW)")
    # Grid coordinates in the same "cell-index" frame as station_xy.
    grid_cols, grid_rows = np.meshgrid(np.arange(N_COLS), np.arange(N_ROWS))
    grid_xy = np.column_stack([grid_cols.ravel().astype(float),
                               grid_rows.ravel().astype(float)])
    V_grid = interpolate_eof_stack(
        IDW(power=2.0, k_neighbors=min(10, data["n_stations"])),
        data["station_xy"], v_sigma_stations,
        grid_xy, grid_shape=(N_ROWS, N_COLS),
    )  # (k, n_rows, n_cols)
    log.info("   V_grid shape: %s", V_grid.shape)

    # Also interpolate the per-station climatological mean onto the grid.
    mean_grid = IDW(power=2.0, k_neighbors=min(10, data["n_stations"])) \
        .fit(data["station_xy"], data["mean_stations"]) \
        .predict(grid_xy).reshape(N_ROWS, N_COLS)
    log.info("   mean_grid shape: %s, range [%.2f, %.2f]",
             mean_grid.shape, mean_grid.min(), mean_grid.max())

    # ---- 6. Project observations onto the trained EOFs --------------------
    log.info("Step 6: project observations onto EOFs to get U(t)")
    U_hat = project_onto_eofs(
        obs, spatial_mean=data["mean_stations"],
        v_sigma=v_sigma_stations, singular_values=sigma_k,
    )
    log.info("   U(t) shape: %s", U_hat.shape)

    # ---- 7. Reconstruct the gridded wind field ----------------------------
    log.info("Step 7: reconstruct wind(t, y, x)")
    wind = reconstruct_series(U_hat, V_grid, mean_grid=mean_grid, clip=None)
    log.info("   wind shape: %s, dtype: %s", wind.shape, wind.dtype)

    # ---- 8. Diagnostics ---------------------------------------------------
    log.info("Step 8: diagnostics")
    # At station cells, the reconstructed wind should closely match the
    # (noise-free) low-rank truth + mean because: (a) IDW at a cell that
    # IS a station returns the station value exactly, (b) the rPCA split
    # pushes the sparse spikes out of L, (c) the projection formula is
    # exact when V is orthonormal (which it is, since we SVD-ed L).
    truth_at_stations = data["mean_stations"][None, :] + data["low_rank_truth"]
    recon_at_stations = wind[:, data["station_rows"], data["station_cols"]]
    rmse = float(np.sqrt(np.mean((truth_at_stations - recon_at_stations) ** 2)))
    bias = float(np.mean(recon_at_stations - truth_at_stations))
    r = float(np.corrcoef(truth_at_stations.ravel(),
                          recon_at_stations.ravel())[0, 1])
    log.info("   reconstruction vs low-rank truth, at station cells:")
    log.info("     RMSE  = %.4f m/s", rmse)
    log.info("     bias  = %+.4f m/s", bias)
    log.info("     r     = %.4f", r)

    # Sparse-component recovery: how much of the spike signal did rPCA
    # successfully route into S?
    sparse_truth = data["sparse_truth"]
    sparse_hat = rpca.S_
    spike_mask = sparse_truth != 0
    support_agreement = float(np.mean(np.abs(sparse_hat[spike_mask]) > 0.3))
    log.info("   sparse-support recovery (spikes captured by S): %.2f",
             support_agreement)

    # Hard pass/fail — fail with exit code 1 if the reconstruction is
    # unexpectedly bad.  The thresholds are deliberately loose so the
    # script serves as a smoke test, not a benchmark.
    if not (r > 0.95 and rmse < 0.5):
        log.error("Pipeline sanity check FAILED: r=%.3f rmse=%.3f", r, rmse)
        return 1

    log.info("OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
