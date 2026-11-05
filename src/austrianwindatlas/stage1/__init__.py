"""Stage-1 near-surface interpolators and reconstruction.

Currently ported:

- :mod:`austrianwindatlas.stage1.rpca_eof` — IDW, MLR and
  RegressionIDW (the paper's "KED") for interpolating rPCA / EOF
  spatial coefficients onto a regular grid.
- :mod:`austrianwindatlas.stage1.gam_bamlss` — subprocess bridge to
  the BAMLSS / mgcv R scripts in ``R/`` (CSV in, NetCDF out).
- :mod:`austrianwindatlas.stage1.reconstruct` — pure numerical core
  that combines the spatial mean, interpolated EOF loadings and
  temporal coefficients into hourly wind fields, with optional
  ERA5 / CERRA background blending.
- :mod:`austrianwindatlas.stage1.enhanced_mae` — DL variant that
  predicts station-side EOF loadings from terrain covariates with
  configurable loss (MSE / MAE / asymmetric / focal / weighted-*),
  per the Enhanced-MAE / Enhanced-WMSE methods in Schicker et al.
  (2026).  Re-exported as a sub-package to keep its internal API
  (``LossFunction``, ``TrainConfig``, ``predict_grid`` etc.) off
  the top-level ``stage1`` namespace where it would collide with
  the pure-rPCA :mod:`.reconstruct` helpers.
- :mod:`austrianwindatlas.stage1.qn_gnn` — the QN-GNN method:
  per-station z-score normalisation, terrain-aware KNN graph with
  ridge-height penalty, GAT attention + MLP dual path, spatial
  validation holdout.  Ships training + grid prediction.  The
  Temporal CNN refinement and the production reconstruction glue
  (ERA5 / CERRA blending, calibrator ``z → m/s``) land in later PRs.

- :mod:`austrianwindatlas.stage1.amato` — **stub** pending upstream
  licence (<https://github.com/federhub/ST_DeepLearning> has no
  declared licence).  The sub-package is importable and introspectable
  (``amato.AMATO_STATUS``, ``amato.AMATO_UPSTREAM_URL``) but every
  callable raises :class:`~.amato.AmatoUnavailableError`.  See
  :doc:`docs/amato.md` and use :mod:`~.enhanced_mae` instead.
"""
from austrianwindatlas.stage1 import (
    amato,
    enhanced_mae,
    gam_bamlss,
    qn_gnn,
    reconstruct,
    rpca_eof,
)
from austrianwindatlas.stage1.gam_bamlss import (
    RExportInputs,
    RScriptResult,
    load_eof_grids,
    prepare_rpca_export,
    r_script_path,
    rscript_executable,
    run_bamlss,
    run_gam,
)
from austrianwindatlas.stage1.reconstruct import (
    DEFAULT_CLIP,
    impute_station_nans,
    project_onto_eofs,
    reconstruct_series,
    reconstruct_timestep,
)
from austrianwindatlas.stage1.rpca_eof import (
    IDW,
    MLR,
    EOFInterpolator,
    RegressionIDW,
    interpolate_eof_stack,
)

__all__ = [
    # rpca_eof
    "rpca_eof",
    "IDW",
    "MLR",
    "RegressionIDW",
    "EOFInterpolator",
    "interpolate_eof_stack",
    # gam_bamlss
    "gam_bamlss",
    "RExportInputs",
    "RScriptResult",
    "load_eof_grids",
    "prepare_rpca_export",
    "r_script_path",
    "rscript_executable",
    "run_bamlss",
    "run_gam",
    # reconstruct
    "reconstruct",
    "DEFAULT_CLIP",
    "impute_station_nans",
    "project_onto_eofs",
    "reconstruct_series",
    "reconstruct_timestep",
    # enhanced_mae (sub-package only; access via `stage1.enhanced_mae.*`)
    "enhanced_mae",
    # qn_gnn (sub-package only; access via `stage1.qn_gnn.*`)
    "qn_gnn",
    # amato (stub sub-package; see docs/amato.md)
    "amato",
]
