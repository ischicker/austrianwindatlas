"""Austrian Windatlas — observation-constrained wind climatology library.

The library is organised as five sub-packages:

- :mod:`austrianwindatlas.core` — rPCA/EOF decomposition, grid definitions,
  path and config handling.
- :mod:`austrianwindatlas.stage1` — the nine near-surface interpolation
  families compared in Schicker et al. (2026).
- :mod:`austrianwindatlas.stage2` — the seven hub-height extrapolation
  formulations plus the equal-weight ensemble + inter-formulation spread.
- :mod:`austrianwindatlas.validation` — the TAWES hold-out / SCADA /
  NEWA / spectral / Weibull / spread-calibration diagnostics.
- :mod:`austrianwindatlas.io` — data loaders for TAWES, SCADA, CORINE,
  NEWA and the native NetCDF/GeoTIFF formats.

Methods land one PR at a time; see ``AWA_TRANSITION.md`` in the paper
repository for the per-file port status.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__", "paths"]

from austrianwindatlas.core import paths  # noqa: F401 — re-export for convenience
