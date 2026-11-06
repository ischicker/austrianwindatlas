# Examples

Each file is a self-contained script.  Run them with the library's
runtime deps installed (`pip install -e .`).  They print diagnostics
to stdout; no file is written outside `./examples_output/` (which
the scripts create on demand and the top-level `.gitignore` keeps
out of version control).

| File | What it shows | External data? |
|---|---|---|
| `01_synthetic_end_to_end.py` | rPCA decomposition → EOF interpolation → observation projection → wind-field reconstruction, all in memory. Validates the paper's full Stage-1 maths chain in ~2 seconds. | No |
| `02_stage1_with_validation.py` | Same pipeline as 01 but with a held-out-station split and `validation` scoring (per-station + aggregated). Mirrors the paper's 54-held-out verification flow on a synthetic problem. | No |
| `03_stage2_extrapolation_demo.py` | Build a 3-regime roughness map + diurnal 10-m wind cube, run all seven Stage-2 formulations at 80 / 120 / 160 m, print per-regime shear-ratio table. | No |
| `04_reproduce_fig3_tawes_open.py` | Hourly TAWES observations for Vienna Hohe Warte + Innsbruck Universität (Feb 2022) fetched from the klima-v2-1h open-data API, scored against a toy model, rendered as a Fig.3-style three-panel figure (Taylor + Weibull + diurnal). | Yes (klima-v2-1h; cached on first run) |

Caching
-------

Example 04 caches the downloaded CSV under
`$XDG_CACHE_HOME/austrianwindatlas/` (or `~/.cache/austrianwindatlas/`
if that's unset).  Pass `--no-download` to fail fast on a cold cache
— useful in CI and air-gapped environments.

Coming soon
-----------

As the method ports progress (see `AWA_TRANSITION.md` in the paper
repo), more examples land here:

- `05_bamlss_quick_docker.py` — `stage1.gam_bamlss.run_gam` in
  `--quick` mode on synthetic data; requires R + mgcv (install via
  `environment.yml` or run inside the Docker image).
- `06_loyo_fold.py` — one leave-one-year-out fold end-to-end.
