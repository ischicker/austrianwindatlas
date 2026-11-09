"""Stage-2 hub-height extrapolation.

Vertical extrapolation of 10-m gridded wind fields to turbine hub
heights (80–250 m AGL, per the paper's production run).  Seven
formulations ship — all pure NumPy, same input contract, so a driver
can run every method in one pass and pick the best per cell /
timestep / season.

Submodules:

* :mod:`.roughness` — CORINE-derived roughness fields (``z0``, ``alpha``,
  ``h_ref``, regime masks) used by the five terrain-aware methods.
* :mod:`.formulations` — the seven extrapolation functions as standalone
  callables: ``classic_power_law``, ``classic_log_law``,
  ``enhanced_power_law``, ``enhanced_log_law``, ``stability_dependent``,
  ``roughness_adaptive``, ``hybrid_power_log``.
* :mod:`.extrapolator` — :class:`MultiHeightExtrapolator` that batches
  every enabled method across a stack of target heights.
"""
from __future__ import annotations

from . import formulations, roughness
from .extrapolator import (
    HAS_CUPY,
    METHOD_NAMES,
    ExtrapolationConfig,
    MultiHeightExtrapolator,
)
from .formulations import (
    DEFAULT_CLASSIC_ALPHA,
    DEFAULT_CLASSIC_Z0,
    DEFAULT_MAX_WIND,
    DEFAULT_REFERENCE_HEIGHT,
    classic_log_law,
    classic_power_law,
    enhanced_log_law,
    enhanced_power_law,
    hybrid_power_log,
    roughness_adaptive,
    stability_dependent,
)
from .roughness import (
    DEFAULT_Z0_FILL,
    DEFAULT_Z0_MAX,
    DEFAULT_Z0_MIN,
    RoughnessFields,
    clipped_roughness,
    derive_roughness_fields,
    log_law_reference_height,
    roughness_alpha,
    roughness_regime_masks,
)

__all__ = [
    # roughness
    "roughness",
    "DEFAULT_Z0_MIN",
    "DEFAULT_Z0_MAX",
    "DEFAULT_Z0_FILL",
    "RoughnessFields",
    "clipped_roughness",
    "derive_roughness_fields",
    "log_law_reference_height",
    "roughness_alpha",
    "roughness_regime_masks",
    # formulations
    "formulations",
    "DEFAULT_CLASSIC_ALPHA",
    "DEFAULT_CLASSIC_Z0",
    "DEFAULT_REFERENCE_HEIGHT",
    "DEFAULT_MAX_WIND",
    "classic_power_law",
    "classic_log_law",
    "enhanced_power_law",
    "enhanced_log_law",
    "stability_dependent",
    "roughness_adaptive",
    "hybrid_power_log",
    # extrapolator
    "HAS_CUPY",
    "METHOD_NAMES",
    "ExtrapolationConfig",
    "MultiHeightExtrapolator",
]
