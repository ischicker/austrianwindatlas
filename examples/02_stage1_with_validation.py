#!/usr/bin/env python3
"""End-to-end Stage-1 pipeline WITH validation scoring.

Extends :mod:`examples.01_synthetic_end_to_end` by:

1. Holding out a handful of stations from the rPCA fit.
2. Running the full decomposition → EOF-interpolation → reconstruction
   pipeline on the remaining training stations.
3. Scoring the reconstruction at the *held-out* stations using the
   :mod:`austrianwindatlas.validation` harness — exactly the
   verification flow the paper uses on 54 held-out TAWES stations.

Pipeline
========

::

    build synthetic truth  ─────────────────┐
             │                              │
             ▼                              │
    split (train / held-out)                │
             │                              │
             ▼                              │
    Robust PCA on train → L, S              │
             │                              │
             ▼                              │
    SVD(L) → station V, sigma               │
             │                              │
             ▼                              │
    IDW-interpolate V to grid               │
             │                              │
             ▼                              │
    project held-out obs onto V             │
             │                              │
             ▼                              │
    reconstruct wind field at grid cells    │
             │                              │
             ▼                              │
    score reconstruction at held-out cells  │
             │                              │
             ▼                              │
    aggregate: per-station + pooled         │
                                            │
                        obs_true at held-out stations
                                            ◀
                                   compared against

Expected result: aggregated r > 0.8 and RMSE < 4 m/s on the 5
held-out stations.  The held-out metrics are naturally weaker than
the training-cell numbers in example 01 because (a) the held-out
stations have NO training data in the IDW fit and (b) their sparse
spikes can't be learned from the rPCA basis.

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
from austrianwindatlas.validation import (
    aggregate_across_stations,
    station_metrics,
)

RNG_SEED = 20261105

# Synthetic knobs — kept small so the demo completes in ~3 seconds.
N_TIMES = 24 * 30            # one synthetic month, hourly
N_STATIONS = 40
N_HELDOUT = 5
N_ROWS, N_COLS = 40, 50      # synthetic grid
RANK = 3
SPARSE_FRACTION = 0.01
SPARSE_SCALE = 2.5


# ---------------------------------------------------------------------------
# 1. Build synthetic truth + sparse spikes
# ---------------------------------------------------------------------------
def make_synthetic(seed: int = RNG_SEED):
    """Reuse the same construction as example 01, but return more structure
    so we can split the station set later."""
    rng = np.random.default_rng(seed)

    # Station coordinates — distinct grid cells only.
    rows_raw = rng.integers(0, N_ROWS, size=N_STATIONS)
    cols_raw = rng.integers(0, N_COLS, size=N_STATIONS)
    seen: set[tuple[int, int]] = set()
    keep = []
    for i in range(len(rows_raw)):
        cell = (int(rows_raw[i]), int(cols_raw[i]))
        if cell not in seen:
            seen.add(cell)
            keep.append(i)
    station_rows = rows_raw[keep]
    station_cols = cols_raw[keep]
    n_stations = len(keep)
    station_xy = np.column_stack([station_cols.astype(float),
                                  station_rows.astype(float)])

    # Low-rank signal.
    U_true = rng.standard_normal((N_TIMES, RANK))
    sigma_true = np.array([4.0, 2.5, 1.5])
    yy, xx = np.meshgrid(np.linspace(0, np.pi, N_ROWS),
                         np.linspace(0, 2 * np.pi, N_COLS),
                         indexing="ij")
    V_grid_true = np.stack([
        np.sin(xx) * np.cos(yy),
        np.sin(2 * xx + 1.0),
        np.cos(xx - yy),
    ], axis=0)
    V_stations_true = V_grid_true[:, station_rows, station_cols].T

    mean_stations = 2.5 + 0.8 * rng.standard_normal(n_stations)
    low_rank = U_true @ np.diag(sigma_true) @ V_stations_true.T
    truth = mean_stations[None, :] + low_rank

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
        "low_rank_truth": low_rank,
        "n_stations": n_stations,
    }


# ---------------------------------------------------------------------------
# 2. Pipeline + 3. validation
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("awa-demo")

    log.info("Step 1: build synthetic station matrix with held-out stations")
    data = make_synthetic()
    n_sta = data["n_stations"]

    # Hold out the first N_HELDOUT stations (deterministic on the seed).
    holdout = np.zeros(n_sta, dtype=bool)
    holdout[:N_HELDOUT] = True
    train = ~holdout
    log.info("  training stations: %d, held out: %d", int(train.sum()), int(holdout.sum()))

    obs_all = data["observations"]
    mean_train = data["mean_stations"][train]

    # ---- 2. rPCA on training stations ------------------------------------
    log.info("Step 2: rPCA decomposition on training stations")
    obs_train_centered = obs_all[:, train] - mean_train[None, :]
    rpca = RobustPCA(max_iter=300, use_gpu=False, verbose=False).fit(obs_train_centered)
    log.info("  rPCA converged=%s, n_iter=%d", rpca.converged_, rpca.n_iter_)

    # ---- 3. SVD → station V and sigma ------------------------------------
    log.info("Step 3: extract V and sigma from L")
    _, sigma_from_L, Vt_from_L = np.linalg.svd(rpca.L_, full_matrices=False)
    k = RANK
    v_sigma_stations = Vt_from_L[:k].T * sigma_from_L[:k]
    sigma_k = sigma_from_L[:k]

    # ---- 4. Interpolate V to the grid ------------------------------------
    log.info("Step 4: interpolate station V and mean to the grid (IDW)")
    station_xy_train = data["station_xy"][train]
    grid_cols, grid_rows = np.meshgrid(np.arange(N_COLS), np.arange(N_ROWS))
    grid_xy = np.column_stack([grid_cols.ravel().astype(float),
                               grid_rows.ravel().astype(float)])
    V_grid = interpolate_eof_stack(
        IDW(power=2.0, k_neighbors=min(10, int(train.sum()))),
        station_xy_train, v_sigma_stations,
        grid_xy, grid_shape=(N_ROWS, N_COLS),
    )
    mean_grid = IDW(power=2.0, k_neighbors=min(10, int(train.sum()))) \
        .fit(station_xy_train, mean_train) \
        .predict(grid_xy).reshape(N_ROWS, N_COLS)

    # ---- 5. Project held-out obs onto the trained EOFs -------------------
    log.info("Step 5: project held-out observations onto trained EOFs")
    obs_heldout = obs_all[:, holdout]
    # project_onto_eofs needs a spatial_mean per station — use the held-out
    # stations' *own* climatology, since we don't want to leak train info
    # beyond what the DNN pipeline would at inference time.
    mean_heldout = np.nanmean(obs_heldout, axis=0)
    # Compute V_sigma at the HELD-OUT station locations by sampling V_grid.
    heldout_rows = data["station_rows"][holdout]
    heldout_cols = data["station_cols"][holdout]
    v_sigma_heldout = V_grid[:, heldout_rows, heldout_cols].T   # (n_held, k)
    U_hat = project_onto_eofs(
        obs_heldout, spatial_mean=mean_heldout,
        v_sigma=v_sigma_heldout, singular_values=sigma_k,
    )

    # ---- 6. Reconstruct wind field over the full grid --------------------
    log.info("Step 6: reconstruct wind(t, y, x)")
    wind = reconstruct_series(U_hat, V_grid, mean_grid=mean_grid, clip=None)

    # ---- 7. Score reconstruction at held-out cells -----------------------
    log.info("Step 7: score reconstruction at held-out stations")
    station_results = []
    pooled_obs_parts, pooled_pred_parts = [], []
    for i in range(len(heldout_rows)):
        row, col = int(heldout_rows[i]), int(heldout_cols[i])
        obs_i = obs_heldout[:, i]
        pred_i = wind[:, row, col]
        pooled_obs_parts.append(obs_i)
        pooled_pred_parts.append(pred_i)
        sm = station_metrics(obs_i, pred_i, station_id=int(i))
        station_results.append(sm)
        log.info(
            "  station %d: n=%d  RMSE=%.3f  bias=%+.3f  r=%.3f",
            i, sm.n_pairs, sm.rmse, sm.bias, sm.pearson_r,
        )

    agg = aggregate_across_stations(
        station_results, method_name="synthetic_demo",
        pooled_obs=np.concatenate(pooled_obs_parts),
        pooled_pred=np.concatenate(pooled_pred_parts),
    )
    log.info("Aggregated across %d held-out stations:", agg.n_stations)
    log.info("  mean  RMSE=%.3f  bias=%+.3f  r=%.3f",
             agg.mean_rmse, agg.mean_bias, agg.mean_pearson_r)
    log.info("  pooled RMSE=%.3f  bias=%+.3f  r=%.3f",
             agg.pooled_rmse, agg.pooled_bias, agg.pooled_pearson_r)

    # Hard pass/fail — fails CI if the reconstruction is unexpectedly bad.
    # Held-out stations have no training data in the IDW fit AND carry
    # sparse spikes the rPCA basis can't model, so bounds are looser
    # than the "reconstruction at a training station" case in example 01.
    if not (agg.mean_pearson_r > 0.8 and agg.mean_rmse < 4.0):
        log.error("Validation FAILED: mean_r=%.3f mean_rmse=%.3f",
                  agg.mean_pearson_r, agg.mean_rmse)
        return 1

    log.info("OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
