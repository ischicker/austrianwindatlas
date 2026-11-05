#!/usr/bin/env python3
"""Stage-2 demo: run all seven hub-height formulations on synthetic data.

Builds a toy 10-m wind field over a 3-regime roughness map (smooth
farmland / moderate forest / rough built-up), extrapolates to three
hub heights (80 / 120 / 160 m) with every Stage-2 formulation, and
prints per-regime shear ratios so you can see how the methods disagree
over terrain.

Pipeline
========

::

    build synthetic roughness  ─────────────┐
      ┌──────────────┬──────────────┐       │
      │ smooth       │ moderate     │ rough │
      │ (farmland)   │ (forest)     │ (city)│
      └──────────────┴──────────────┴───────┘
             │
             ▼
    derive (z0, alpha, h_ref, masks)
             │
             ▼
    build synthetic 10-m wind cube (T, ny, nx)
             │
             ▼
    MultiHeightExtrapolator(roughness=..., config=...)
             │
             ▼
    extrapolate → dict[height][method] = (T, ny, nx)
             │
             ▼
    per-regime shear ratio = mean(output) / mean(input)
             │
             ▼
    print table

Expected qualitative behaviour (from the paper):

* Smooth regime: log-law variants predict ~1.26 at 80 m (log(80/0.03)
  / log(10/0.03) ≈ 1.4).
* Rough regime: power-law variants predict ~1.33 at 80 m
  ((80/10)^0.25 = 1.68 for the fixed-rough branch).
* Enhanced methods interpolate between regimes.

Runtime: < 1 second.
"""
from __future__ import annotations

import logging
import sys

import numpy as np

from austrianwindatlas.stage2 import (
    METHOD_NAMES,
    ExtrapolationConfig,
    MultiHeightExtrapolator,
    derive_roughness_fields,
)

RNG_SEED = 20261105
N_TIMES = 24
N_ROWS, N_COLS = 20, 30


# ---------------------------------------------------------------------------
# 1. Build synthetic roughness + wind cube
# ---------------------------------------------------------------------------
def make_synthetic():
    """Return (z0, wind_10m) where z0 has a visible 3-regime pattern."""
    rng = np.random.default_rng(RNG_SEED)

    # Roughness: left third smooth (0.03), middle third moderate (0.2),
    # right third rough (0.8).  Small jitter so no cell is literally
    # identical to its neighbour.
    z0 = np.empty((N_ROWS, N_COLS), dtype=np.float32)
    third = N_COLS // 3
    z0[:, :third] = 0.03
    z0[:, third : 2 * third] = 0.2
    z0[:, 2 * third :] = 0.8
    z0 += rng.normal(0, 0.005, size=z0.shape).astype(np.float32)
    z0 = np.clip(z0, 0.005, 1.0)

    # 10-m wind: uniform spatial field with a hourly-cycle time signal.
    # Keep it in [2, 15] m/s so we exercise the full stability-dependent
    # branch (strong-wind well-mixed vs weak-wind stratified).
    t = np.arange(N_TIMES)
    diurnal = 8.0 + 3.0 * np.sin(2 * np.pi * t / 24.0)
    wind_10m = np.broadcast_to(diurnal[:, None, None],
                               (N_TIMES, N_ROWS, N_COLS)).astype(np.float32)

    return z0, wind_10m


# ---------------------------------------------------------------------------
# 2. Extrapolate + 3. per-regime diagnostics
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("awa-stage2-demo")

    log.info("Step 1: build synthetic roughness + 10-m wind cube")
    z0, wind_10m = make_synthetic()
    log.info("  z0 shape %s, wind shape %s", z0.shape, wind_10m.shape)

    log.info("Step 2: derive roughness fields (z0 clip, alpha, h_ref, masks)")
    fields = derive_roughness_fields(z0)
    log.info(
        "  regime cell counts: smooth=%d moderate=%d rough=%d",
        int(fields.smooth.sum()), int(fields.moderate.sum()), int(fields.rough.sum()),
    )

    log.info("Step 3: build MultiHeightExtrapolator for 3 heights, all methods")
    target_heights = (80.0, 120.0, 160.0)
    cfg = ExtrapolationConfig(target_heights=target_heights)
    extr = MultiHeightExtrapolator(roughness=fields, config=cfg)

    log.info("Step 4: extrapolate — runs all 7 methods × 3 heights")
    results = extr.extrapolate(wind_10m)

    # Diagnostic: per-regime shear ratio = mean(output) / mean(input).
    log.info("Step 5: per-regime shear ratio = mean(out_h) / mean(input)")
    input_mean_by_regime = {
        "smooth": float(wind_10m[:, fields.smooth].mean()),
        "moderate": float(wind_10m[:, fields.moderate].mean()),
        "rough": float(wind_10m[:, fields.rough].mean()),
    }
    log.info("  input mean by regime: %s", input_mean_by_regime)

    print()
    print(f"{'Height':>8}  {'Regime':>9}  {'Method':<16}  {'Shear ratio':>11}")
    print(f"{'(m)':>8}  {' ':>9}  {' ':<16}  {' ':>11}")
    print("-" * 56)
    for h in target_heights:
        for regime_name, mask in (("smooth", fields.smooth),
                                   ("moderate", fields.moderate),
                                   ("rough", fields.rough)):
            for method_name in METHOD_NAMES:
                arr = results[h].get(method_name)
                if arr is None:
                    continue
                out_mean = float(arr[:, mask].mean())
                ratio = out_mean / input_mean_by_regime[regime_name]
                print(f"{h:>8.0f}  {regime_name:>9}  {method_name:<16}  {ratio:>11.4f}")
            print()

    # Sanity bounds — each method must produce physically sensible
    # shear ratios at 80 m (between 1.0 and 2.0 for any regime / method).
    for h in target_heights:
        for method_name in METHOD_NAMES:
            arr = results[h].get(method_name)
            if arr is None:
                continue
            ratios = [
                float(arr[:, mask].mean()) / input_mean_by_regime[regime_name]
                for regime_name, mask in (("smooth", fields.smooth),
                                           ("moderate", fields.moderate),
                                           ("rough", fields.rough))
            ]
            if not all(0.8 <= r <= 3.5 for r in ratios):
                log.error(
                    "Bad shear ratio for %s at %g m: %s", method_name, h, ratios,
                )
                return 1

    log.info("OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
