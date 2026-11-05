"""Roughness-derived fields for Stage-2 extrapolation.

Three gridded fields the Stage-2 formulations consume:

* :func:`clipped_roughness` — ``z0`` clipped to a physically-plausible
  range, replacing ``nodata`` / non-finite entries with a safe value.
* :func:`roughness_alpha` — the paper's spatially-variable ``alpha``
  for the enhanced power-law:

  .. math::

      \\alpha(x, y) = 0.1 + 0.15 \\log_{10}\\!\\left(\\mathrm{clip}(z_0 / 0.01, 0.1, 100)\\right)

  Result clipped to ``[0.05, 0.4]`` — a safeguard against runaway
  values on heavily modified CORINE pixels.
* :func:`log_law_reference_height` — the "floor" reference height for
  the enhanced log-law:

  .. math::

      h_\\mathrm{ref}(x, y) = \\max(10\\,\\mathrm{m}, 2\\,z_0(x, y))

  so the base ratio ``log(h_ref / z0)`` stays well-behaved even for
  extremely rough cells (dense forest with ``z0 > 1`` m).

Plus a convenience derivation of a *roughness regime* classifier
(``smooth`` / ``moderate`` / ``rough``), used by the roughness-adaptive
and hybrid methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = [
    "DEFAULT_Z0_MIN",
    "DEFAULT_Z0_MAX",
    "DEFAULT_Z0_FILL",
    "RoughnessFields",
    "clipped_roughness",
    "roughness_alpha",
    "log_law_reference_height",
    "roughness_regime_masks",
    "derive_roughness_fields",
]

#: Lower clip for ``z0`` (metres).  Lower than this and the log-law
#: denominator ``log(h/z0)`` blows up for any sensible height.
DEFAULT_Z0_MIN: float = 0.001

#: Upper clip for ``z0`` (metres).  Above this the log-law reference
#: height dominates the ratio and the formulations lose physical sense.
DEFAULT_Z0_MAX: float = 2.0

#: Value substituted for non-finite / nodata pixels in the input DEM /
#: CORINE roughness raster.  Represents open grassland, roughly.
DEFAULT_Z0_FILL: float = 0.1


def clipped_roughness(
    raw_z0: NDArray[np.floating],
    *,
    z0_min: float = DEFAULT_Z0_MIN,
    z0_max: float = DEFAULT_Z0_MAX,
    fill: float = DEFAULT_Z0_FILL,
) -> NDArray[np.float32]:
    """Return a NaN-free, clipped ``z0`` field in metres.

    Replaces ``NaN`` / ``Inf`` / negative values with ``fill`` *before*
    clipping, so the clipped output never silently converts a
    data-quality problem into a physically-plausible edge value.
    """
    arr = np.asarray(raw_z0, dtype=float)
    cleaned = np.where(np.isfinite(arr) & (arr > 0), arr, fill)
    return np.clip(cleaned, z0_min, z0_max).astype(np.float32)


def roughness_alpha(
    z0: NDArray[np.floating],
    *,
    alpha_min: float = 0.05,
    alpha_max: float = 0.4,
) -> NDArray[np.float32]:
    """Spatially-variable power-law exponent from roughness.

    See module docstring for the formula.  Pass the output of
    :func:`clipped_roughness` to avoid NaN-propagation through the log.
    """
    z0 = np.asarray(z0, dtype=float)
    ratio = np.clip(z0 / 0.01, 0.1, 100.0)
    alpha = 0.1 + 0.15 * np.log10(ratio)
    return np.clip(alpha, alpha_min, alpha_max).astype(np.float32)


def log_law_reference_height(
    z0: NDArray[np.floating],
    *,
    min_height: float = 10.0,
    factor: float = 2.0,
) -> NDArray[np.float32]:
    """Return ``h_ref = max(min_height, factor * z0)`` for the log-law.

    Guarantees the log-law reference is above the roughness length even
    for dense-forest pixels with ``z0`` up to the clip ceiling.
    """
    z0 = np.asarray(z0, dtype=float)
    return np.maximum(min_height, factor * z0).astype(np.float32)


@dataclass(frozen=True)
class RoughnessFields:
    """Bundled Stage-2 roughness derivatives.

    Attributes
    ----------
    z0
        Clipped roughness length (metres).
    alpha
        Spatially-variable power-law exponent.
    h_ref
        Log-law reference height (metres).
    smooth, moderate, rough
        Boolean ``(ny, nx)`` masks of the three roughness regimes used
        by the roughness-adaptive formulation.  Each pixel lands in
        exactly one mask: ``smooth`` (z0 < 0.05), ``moderate``
        (0.05 ≤ z0 < 0.5), ``rough`` (z0 ≥ 0.5).  Thresholds match the
        paper's ``FastGPUExtrapolator`` exactly.
    """

    z0: NDArray[np.float32]
    alpha: NDArray[np.float32]
    h_ref: NDArray[np.float32]
    smooth: NDArray[np.bool_]
    moderate: NDArray[np.bool_]
    rough: NDArray[np.bool_]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.z0.shape


def roughness_regime_masks(
    z0: NDArray[np.floating],
    *,
    smooth_thresh: float = 0.05,
    rough_thresh: float = 0.5,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Return ``(smooth, moderate, rough)`` boolean masks of ``z0``."""
    z0 = np.asarray(z0, dtype=float)
    smooth = z0 < smooth_thresh
    rough = z0 >= rough_thresh
    moderate = ~(smooth | rough)
    return smooth, moderate, rough


def derive_roughness_fields(
    raw_z0: NDArray[np.floating],
    *,
    z0_min: float = DEFAULT_Z0_MIN,
    z0_max: float = DEFAULT_Z0_MAX,
    fill: float = DEFAULT_Z0_FILL,
    alpha_min: float = 0.05,
    alpha_max: float = 0.4,
    smooth_thresh: float = 0.05,
    rough_thresh: float = 0.5,
) -> RoughnessFields:
    """One-stop constructor: raw roughness → fully-derived Stage-2 fields.

    Equivalent to running :func:`clipped_roughness`,
    :func:`roughness_alpha`, :func:`log_law_reference_height` and
    :func:`roughness_regime_masks` on the same input, then packing the
    results into a :class:`RoughnessFields` dataclass.
    """
    z0 = clipped_roughness(raw_z0, z0_min=z0_min, z0_max=z0_max, fill=fill)
    alpha = roughness_alpha(z0, alpha_min=alpha_min, alpha_max=alpha_max)
    h_ref = log_law_reference_height(z0)
    smooth, moderate, rough = roughness_regime_masks(
        z0, smooth_thresh=smooth_thresh, rough_thresh=rough_thresh,
    )
    return RoughnessFields(
        z0=z0, alpha=alpha, h_ref=h_ref,
        smooth=smooth, moderate=moderate, rough=rough,
    )
