# Austrian Windatlas

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ischicker/austrianwindatlas/blob/main/notebooks/01_synthetic_end_to_end.ipynb)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PENDING.svg)](https://doi.org/10.5281/zenodo.PENDING)

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

> **Status: v0.1.0 feature-complete** — all nine Stage-1 methods
> ported (Amato is a stub pending upstream licence), all seven
> Stage-2 formulations with optional cupy GPU dispatch, full
> validation harness, four runnable examples, 500+ tests passing.
> See the issue tracker on GitHub for the v1.0 roadmap.

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

Both the software release and the companion paper should be cited.

**Software** — pin a specific version from Zenodo (the DOI badge at
the top of this README resolves to the latest release):

```bibtex
@software{austrianwindatlas,
  title   = {Austrian Windatlas},
  author  = {Schicker, Irene and Andre, Konrad and Lexer, Annemarie
             and Janisch, Stefan and Bisko, Nina},
  year    = {2026},
  doi     = {10.5281/zenodo.PENDING},
  url     = {https://github.com/ischicker/austrianwindatlas},
  version = {0.1.0}
}
```

**Paper** — Schicker et al. (2026, *Climate*, MS 4613178); see
[`CITATION.cff`](CITATION.cff) for the full BibTeX.

Tools that understand the Citation File Format (GitHub's "Cite this
repository" button, Zotero, …) will pull the author list and DOI
automatically from `CITATION.cff`.
