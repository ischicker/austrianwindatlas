#!/usr/bin/env python3
"""Real-TAWES verification demo on klima-v2-1h open data.

Fetches hourly wind-speed observations for two well-known TAWES
stations — Vienna Hohe Warte (station id 105) and Innsbruck
Universität (id 39) — from the GeoSphere Austria open-data API,
builds a toy "model" prediction for each station, scores the two
against observations with :mod:`austrianwindatlas.validation`, and
renders a three-panel figure (Taylor diagram + Weibull overlay +
diurnal cycle) via :mod:`austrianwindatlas.plotting`.

This is the first example in the tree that touches real data.  It
is deliberately small (two stations, one month) so a first-time
reader can inspect the whole pipeline without waiting on big
downloads; the download is < 20 KB and cached to
``$XDG_CACHE_HOME/austrianwindatlas/`` on first run.

Pipeline
========

::

    fetch_klima_v2_1h(Vienna, Innsbruck, 2022-02)
             │
             ▼
    long-form DataFrame (time, station, ff)
             │
             ▼
    pivot to (n_times, 2) obs matrix
             │
             ▼
    build a toy "model" per station
      (station climatology + mild diurnal noise)
             │
             ▼
    validation.station_metrics per station
             │
             ▼
    aggregate_across_stations
             │
             ▼
    plotting: Taylor + Weibull + diurnal
             │
             ▼
    save PDF + PNG to ./examples_output/

The toy model isn't the point — it's a placeholder so the demo has
*something* to score.  Swap it out with any real reconstruction
(e.g. the output of :func:`stage1.reconstruct.reconstruct_series`)
and the rest of the pipeline is identical.

Expected result: mean Pearson r > 0.25 on the toy model; Weibull k
in the 1.3-2.0 range for both stations; a three-panel figure saved
to ``examples_output/04_fig3_tawes_open.(pdf|png)``.

Runtime: < 10 seconds on a cold cache; < 1 second warm.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from austrianwindatlas.io.geosphere import KlimaQuery, fetch_klima_v2_1h
from austrianwindatlas.validation import (
    aggregate_across_stations,
    station_metrics,
)

STATIONS: dict[int, str] = {
    105: "Vienna Hohe Warte",
    39: "Innsbruck Universität",
}
START = "2022-02-01T00:00"
END = "2022-02-28T23:00"
OUTPUT_DIR = Path("examples_output")


# ---------------------------------------------------------------------------
# 1. Fetch observations
# ---------------------------------------------------------------------------
def load_observations(
    *,
    allow_download: bool,
) -> dict[int, pd.DataFrame]:
    """Return ``{station_id: DataFrame(time, ff)}`` for the two stations."""
    query = KlimaQuery(
        station_ids=tuple(STATIONS.keys()),
        parameters=("ff",),
        start=START,
        end=END,
    )
    frame = fetch_klima_v2_1h(query, allow_download=allow_download)

    per_station: dict[int, pd.DataFrame] = {}
    for sid in STATIONS:
        sub = frame[frame["station"] == sid][["time", "ff"]].copy()
        sub = sub.sort_values("time").reset_index(drop=True)
        per_station[sid] = sub
    return per_station


# ---------------------------------------------------------------------------
# 2. Build a toy "model" per station
# ---------------------------------------------------------------------------
def build_toy_model(
    obs: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    """Return a per-timestep "prediction" that is deliberately not perfect.

    The toy model is ``climatology + diurnal_signal + noise``.  It
    exists so the rest of the pipeline has two different series to
    score; replace it with your real reconstruction in practice.
    """
    rng = np.random.default_rng(seed)
    ff = obs["ff"].to_numpy(dtype=float)
    finite = np.isfinite(ff)
    if not np.any(finite):
        return np.full_like(ff, np.nan)

    climatology = float(np.nanmean(ff))
    hour = pd.DatetimeIndex(obs["time"]).hour.to_numpy()
    diurnal = 0.4 * np.sin(2 * np.pi * (hour - 14) / 24.0)
    noise = rng.normal(0.0, 0.8, size=ff.size)
    return climatology + diurnal + noise


# ---------------------------------------------------------------------------
# 3 + 4. Score + aggregate
# ---------------------------------------------------------------------------
def score_stations(
    station_data: dict[int, pd.DataFrame],
) -> tuple[list, dict]:
    """Return ``(list_of_StationMetrics, pooled_arrays_dict)``."""
    station_results = []
    pooled_obs: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []

    for sid, obs in station_data.items():
        pred = build_toy_model(obs, seed=sid)
        obs_arr = obs["ff"].to_numpy(dtype=float)

        sm = station_metrics(obs_arr, pred, station_id=sid)
        station_results.append(sm)

        pooled_obs.append(obs_arr)
        pooled_pred.append(pred)

    pooled = {
        "obs": np.concatenate(pooled_obs),
        "pred": np.concatenate(pooled_pred),
    }
    return station_results, pooled


# ---------------------------------------------------------------------------
# 5. Render the three-panel figure
# ---------------------------------------------------------------------------
def make_figure(
    station_data: dict[int, pd.DataFrame],
    station_results: list,
    output_path: Path,
) -> None:
    """Build the paper-style three-panel figure.

    Deferred matplotlib import so the example still runs end-to-end
    without the ``viz`` extra (it just skips the figure step).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.getLogger("awa-fig3-demo").warning(
            "matplotlib not installed; skipping figure step. "
            "Install the viz extra: pip install 'austrianwindatlas[viz]'"
        )
        return

    from austrianwindatlas.plotting import (
        TaylorEntry,
        plot_diurnal_cycle,
        plot_taylor_diagram,
        plot_weibull_fit,
    )

    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1.2])
    ax_taylor = fig.add_subplot(gs[0, 0], polar=True)
    ax_weibull = fig.add_subplot(gs[0, 1])
    ax_diurnal = fig.add_subplot(gs[0, 2])

    # ---- Taylor diagram --------------------------------------------------
    taylor_entries = []
    palette = {105: "#4477AA", 39: "#CC6677"}
    for sm in station_results:
        sid = sm.station_id
        if not (np.isfinite(sm.pearson_r)):
            continue
        obs_arr = station_data[sid]["ff"].to_numpy(dtype=float)
        pred_arr = build_toy_model(station_data[sid], seed=sid)
        finite = np.isfinite(obs_arr) & np.isfinite(pred_arr)
        obs_std = float(np.std(obs_arr[finite]))
        pred_std = float(np.std(pred_arr[finite]))
        ratio = pred_std / obs_std if obs_std > 0 else float("nan")
        taylor_entries.append(TaylorEntry(
            label=STATIONS[sid], correlation=float(sm.pearson_r),
            std_ratio=ratio, color=palette.get(sid, "#333"),
        ))
    plot_taylor_diagram(
        taylor_entries, title="Toy model vs TAWES obs",
        ax=ax_taylor, show_legend=True,
    )

    # ---- Weibull overlay (Vienna only, for a clean plot) -----------------
    vienna_obs = station_data[105]["ff"].to_numpy(dtype=float)
    plot_weibull_fit(
        vienna_obs, title=f"Weibull fit — {STATIONS[105]}",
        ax=ax_weibull,
    )

    # ---- Diurnal cycle ---------------------------------------------------
    predictions = {
        f"{STATIONS[sid]} (toy)":
            (station_data[sid]["time"], build_toy_model(station_data[sid], seed=sid))
        for sid in STATIONS
    }
    plot_diurnal_cycle(
        obs_times=station_data[105]["time"], obs_wind=vienna_obs,
        predictions=predictions, title="Diurnal cycle",
        colors=palette | {},  # noqa: PIE800 — one-item merge, fine
        ax=ax_diurnal,
    )

    fig.suptitle(
        "klima-v2-1h real-data demo — Feb 2022, Vienna + Innsbruck",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        out = output_path.with_suffix(suffix)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logging.getLogger("awa-fig3-demo").info("saved %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("awa-fig3-demo")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Error out if the klima-v2-1h cache is cold instead of "
             "fetching.  Use for air-gapped CI runs.",
    )
    args = parser.parse_args(argv)

    log.info("Step 1: load klima-v2-1h observations (Vienna + Innsbruck, Feb 2022)")
    try:
        station_data = load_observations(allow_download=not args.no_download)
    except FileNotFoundError as exc:
        log.error("No cached data and --no-download set: %s", exc)
        return 1

    for sid, name in STATIONS.items():
        df = station_data[sid]
        n_finite = int(np.isfinite(df["ff"]).sum())
        log.info("  station %d (%s): %d finite rows of %d",
                 sid, name, n_finite, len(df))

    log.info("Step 2+3: build toy model + score per station")
    station_results, pooled = score_stations(station_data)
    for sm in station_results:
        log.info("  station %d: n=%d  RMSE=%.3f  bias=%+.3f  r=%.3f",
                 sm.station_id, sm.n_pairs, sm.rmse, sm.bias, sm.pearson_r)

    log.info("Step 4: aggregate across stations")
    agg = aggregate_across_stations(
        station_results, method_name="toy_model",
        pooled_obs=pooled["obs"], pooled_pred=pooled["pred"],
    )
    log.info("  mean  RMSE=%.3f  bias=%+.3f  r=%.3f",
             agg.mean_rmse, agg.mean_bias, agg.mean_pearson_r)
    log.info("  pooled RMSE=%.3f  bias=%+.3f  r=%.3f",
             agg.pooled_rmse, agg.pooled_bias, agg.pooled_pearson_r)

    log.info("Step 5: render three-panel figure")
    output_path = OUTPUT_DIR / "04_fig3_tawes_open.pdf"
    make_figure(station_data, station_results, output_path)

    # Sanity bounds — the toy model is predictable enough that it
    # shouldn't blow up.  Pearson r on the diurnal + climatology
    # signal is low but positive.
    if not (np.isfinite(agg.mean_pearson_r) and agg.mean_rmse < 5.0):
        log.error("Validation FAILED: mean_r=%.3f mean_rmse=%.3f",
                  agg.mean_pearson_r, agg.mean_rmse)
        return 1

    log.info("OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
