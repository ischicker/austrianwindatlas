# Austrian Windatlas

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ischicker/austrianwindatlas/blob/main/notebooks/01_synthetic_end_to_end.ipynb)

Observation-constrained hourly wind climatology for Austria —
reference implementation and companion library to Schicker et al.
(2026, *Climate*, MS 4613178).

The library wraps the nine Stage-1 interpolation families (IDW,
RK-IDW, KED, MLR, GAM/BAMLSS, QN-GNN-CNN, Amato, Enhanced-MAE,
Enhanced-WMSE) in a common EOF/rPCA framework, the seven Stage-2
hub-height extrapolation formulations, and the validation harness
used in the paper (54 withheld TAWES stations, 251 SCADA turbines,
NEWA comparison, spectral + Weibull + spread-calibration
diagnostics).

> **Status:** skeleton. The method modules land one PR at a time;
> the day-1 commit contains only the repo layout, licence,
> environments, Docker image, path handling and CI. See
> `AWA_TRANSITION.md` in the paper repo for the per-file port
> plan.

## Install

```bash
# clone + create the conda env with Python + R + BAMLSS + mgcv
git clone https://github.com/ischicker/austrianwindatlas.git
cd austrianwindatlas
conda env create -f environment.yml
conda activate austrianwindatlas
pip install -e .
```

A ready-to-go Docker image (Python + R + BAMLSS + mgcv) is
available via `docker build -t austrianwindatlas .`

## Quick smoke test

```bash
python -c "import austrianwindatlas as awa; print(awa.__version__)"
```

## Try it in Colab (no install)

The [`notebooks/`](notebooks/) directory has a small end-to-end demo
that runs in ~3 seconds on a free-tier CPU and exercises the full
Stage-1 chain (rPCA → EOF interpolation → reconstruction) on fully
synthetic data.  Click the **Open in Colab** badge at the top of this
README to launch it directly — no local Python or data required.

The notebooks are derived from the scripts under `examples/`; see
`notebooks/build_notebooks.py` for the one-liner that rebuilds them
(edit the `.py`, rerun the builder, never hand-edit the `.ipynb`).

## Data

Only code is shipped. Example data is not bundled:

- Real TAWES station observations — download via
  <https://data.hub.geosphere.at/dataset/klima-v2-10min>
  (GeoSphere Austria open data, CC-BY-4.0).
- EU-DEM v1.1 and CORINE Land Cover 2018 — Copernicus Land
  Monitoring Service.
- NEWA — fetched via the WPS API by
  `austrianwindatlas.io.newa`.

Set `AWA_DATA_ROOT` to the directory where you keep the downloads;
every I/O helper in the library resolves paths against it.

```bash
export AWA_DATA_ROOT=/path/to/wind_data
```

## Reproducing the paper

See `examples/reproduce_paper/` (populated as the method modules
land). Each example is self-contained and operates on a small
synthetic station fixture by default; set `AWA_DATA_ROOT` to point
at a real download for full-scale runs.

## Licence

MIT (see `LICENSE`). Third-party components ship under their own
terms (see `NOTICE`). The Amato-family Stage-1 module is a stub
pending an upstream licence (see `docs/amato.md`).

## Citing

See `CITATION.cff` for both the software citation and the paper
citation.
