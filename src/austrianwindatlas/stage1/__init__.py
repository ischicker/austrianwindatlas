"""Stage-1 near-surface interpolators.

Currently ported:

- :mod:`austrianwindatlas.stage1.rpca_eof` — IDW, MLR and
  RegressionIDW (the paper's "KED") for interpolating rPCA / EOF
  spatial coefficients onto a regular grid.
- :mod:`austrianwindatlas.stage1.gam_bamlss` — subprocess bridge to
  the BAMLSS / mgcv R scripts in ``R/`` (CSV in, NetCDF out).

Pending (one per PR; see ``AWA_TRANSITION.md``):

- ``austrianwindatlas.stage1.reconstruct`` — combine
  :math:`\\overline{u}(x) + U(t)\\Sigma V_{\\mathrm{grid}}(x)^{T}` into
  hourly wind-speed NetCDFs with optional ERA5 / CERRA background
  blending.
- ``austrianwindatlas.stage1.{enhanced_mae,qn_gnn,temporal_cnn}`` —
  the three deep-learning families.
- ``austrianwindatlas.stage1.amato`` — stub pending upstream
  licence (see ``docs/amato.md``).
"""
from austrianwindatlas.stage1 import gam_bamlss, rpca_eof
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
]
