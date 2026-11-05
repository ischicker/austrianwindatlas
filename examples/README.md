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

As the method ports progress (see `AWA_TRANSITION.md` in the paper
repo), more examples land here:

- `02_bamlss_quick_docker.py` — `stage1.gam_bamlss.run_gam` in
  `--quick` mode on synthetic data; requires R + mgcv (install via
  `environment.yml` or run inside the Docker image).
- `04_reproduce_fig3_tawes_open.py` — Fig. 3 of Schicker et al.
  (2026) on real TAWES open-data stations
  (<https://data.hub.geosphere.at/dataset/klima-v2-10min>).
- `05_loyo_fold.py` — one leave-one-year-out fold end-to-end.
