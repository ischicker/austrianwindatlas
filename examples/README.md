# Examples

Each file is a self-contained script.  Run them with the library's
runtime deps installed (`pip install -e .`).  They print diagnostics
to stdout; no file is written outside `./examples_output/` (which
the scripts create on demand and the top-level `.gitignore` keeps
out of version control).

| File | What it shows | External data? |
|---|---|---|
| `01_synthetic_end_to_end.py` | rPCA decomposition → EOF interpolation to grid → observation projection → wind-field reconstruction, all on in-memory synthetic data.  Validates the paper's full Stage-1 maths chain in ~2 seconds. | No |

As the method ports progress (see `AWA_TRANSITION.md` in the paper
repo), more examples land here:

- `02_bamlss_quick_docker.py` — `stage1.gam_bamlss.run_gam` in
  `--quick` mode on synthetic data; requires R + mgcv (install via
  `environment.yml` or run inside the Docker image).
- `03_reproduce_fig3_tawes_open.py` — Fig. 3 of Schicker et al.
  (2026) on real TAWES open-data stations
  (<https://data.hub.geosphere.at/dataset/klima-v2-10min>).
- `04_loyo_fold.py` — one leave-one-year-out fold end-to-end.
