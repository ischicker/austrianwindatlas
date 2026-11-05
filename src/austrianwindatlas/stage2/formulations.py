"""Seven hub-height extrapolation formulations.

Pure NumPy functions, no torch / cupy / xarray.  Each takes a 10-m
wind field ``wind_10m[t, y, x]`` plus whatever auxiliary fields the
formulation needs, and returns the extrapolated wind at the target
height.  All operate on the *same input* and the same shape, so a
driver can run every method per timestep without re-materialising
inputs.

The seven formulations, matching Schicker et al. (2026):

1. :func:`classic_power_law` — fixed-alpha power law.
2. :func:`classic_log_law` — fixed-z0 log law.
3. :func:`enhanced_power_law` — power law with spatially variable
   ``alpha(x, y)`` from roughness.
4. :func:`enhanced_log_law` — log law with spatially variable
   ``z0(x, y)`` and matching reference height.
5. :func:`stability_dependent` — power law with exponent modulated by
   the surface wind speed (a crude PBL-stability proxy).
6. :func:`roughness_adaptive` — region-switched: smooth → enhanced-log,
   moderate → enhanced-power, rough → fixed ``(h/10)^0.25``.
7. :func:`hybrid_power_log` — weighted blend of enhanced log and
   enhanced power, weight driven by wind speed and roughness.

Every function clamps the output at ``[0, max_wind]`` (default 100 m/s)
and returns ``float32``.  Non-finite inputs stay non-finite in the
output — the caller is responsible for quality control if needed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = [
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
]

#: Standard IEC-61400 one-seventh exponent.  Paper default.
DEFAULT_CLASSIC_ALPHA: float = 0.143

#: Open-country roughness length (metres).  Paper default for the
#: fixed-z0 log law.
DEFAULT_CLASSIC_Z0: float = 0.1

#: Reference height at which the input wind is measured (metres).
#: The paper uses 10 m (TAWES mast height, INCA grid).
DEFAULT_REFERENCE_HEIGHT: float = 10.0

#: Clip ceiling applied to every formulation's output (m/s).  Guards
#: against pathological inputs; also a sanity cap on Kelvin-Helmholtz
#: artefacts from the stability-dependent variant.
DEFAULT_MAX_WIND: float = 100.0


def _finalise(wind: NDArray[np.floating], *, max_wind: float) -> NDArray[np.float32]:
    """Clip at ``[0, max_wind]`` and cast to ``float32``."""
    return np.clip(wind, 0.0, max_wind).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# 1. Classic power law — fixed alpha, uniform across the grid
# ---------------------------------------------------------------------------
def classic_power_law(
    wind_10m: NDArray[np.floating],
    target_height: float,
    *,
    alpha: float = DEFAULT_CLASSIC_ALPHA,
    reference_height: float = DEFAULT_REFERENCE_HEIGHT,
    max_wind: float = DEFAULT_MAX_WIND,
) -> NDArray[np.float32]:
    """``wind(h) = wind_10m * (h / h_ref) ** alpha``.

    The oldest and simplest of the seven.  No spatial variation.
    """
    ratio = float(target_height) / float(reference_height)
    return _finalise(np.asarray(wind_10m, dtype=float) * (ratio ** float(alpha)),
                     max_wind=max_wind)


# ---------------------------------------------------------------------------
# 2. Classic log law — fixed z0, uniform across the grid
# ---------------------------------------------------------------------------
def classic_log_law(
    wind_10m: NDArray[np.floating],
    target_height: float,
    *,
    z0: float = DEFAULT_CLASSIC_Z0,
    reference_height: float = DEFAULT_REFERENCE_HEIGHT,
    max_wind: float = DEFAULT_MAX_WIND,
) -> NDArray[np.float32]:
    """``wind(h) = wind_10m * log(h / z0) / log(h_ref / z0)``.

    Fixed open-country ``z0=0.1`` m.  Works well at airports, poorly
    over Alpine terrain.
    """
    coeff = np.log(float(target_height) / float(z0)) / np.log(float(reference_height) / float(z0))
    return _finalise(np.asarray(wind_10m, dtype=float) * coeff, max_wind=max_wind)


# ---------------------------------------------------------------------------
# 3. Enhanced power law — spatially variable alpha
# ---------------------------------------------------------------------------
def enhanced_power_law(
    wind_10m: NDArray[np.floating],
    target_height: float,
    alpha: NDArray[np.floating],
    *,
    reference_height: float = DEFAULT_REFERENCE_HEIGHT,
    max_wind: float = DEFAULT_MAX_WIND,
) -> NDArray[np.float32]:
    """``wind(h) = wind_10m * (h / h_ref) ** alpha(x, y)``.

    ``alpha`` must broadcast against the spatial axes of ``wind_10m``
    (i.e. ``(ny, nx)`` for a ``(T, ny, nx)`` input, or ``(ny, nx)``
    for a 2-D single-timestep input).  See
    :func:`austrianwindatlas.stage2.roughness.roughness_alpha` for the
    paper's canonical ``alpha(x, y)``.
    """
    wind = np.asarray(wind_10m, dtype=float)
    alpha_arr = np.asarray(alpha, dtype=float)
    ratio = float(target_height) / float(reference_height)
    out = wind * np.power(ratio, alpha_arr)
    return _finalise(out, max_wind=max_wind)


# ---------------------------------------------------------------------------
# 4. Enhanced log law — spatially variable z0 and matching h_ref
# ---------------------------------------------------------------------------
def enhanced_log_law(
    wind_10m: NDArray[np.floating],
    target_height: float,
    z0: NDArray[np.floating],
    h_ref: NDArray[np.floating],
    *,
    min_coeff: float = 0.5,
    max_coeff: float = 3.0,
    max_wind: float = DEFAULT_MAX_WIND,
) -> NDArray[np.float32]:
    """``wind(h) = wind_10m * log(max(h, 2*z0) / z0) / log(h_ref / z0)``.

    The ``max(h, 2*z0)`` floor keeps the numerator above the
    roughness length at extremely rough cells.  Final log-ratio clipped
    to ``[min_coeff, max_coeff]`` — the paper uses ``[0.5, 3.0]`` so a
    single pathological pixel can't blow up a whole batch.
    """
    wind = np.asarray(wind_10m, dtype=float)
    z0_arr = np.asarray(z0, dtype=float)
    h_ref_arr = np.asarray(h_ref, dtype=float)

    h_target = np.maximum(float(target_height), 2.0 * z0_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        coeff = np.log(h_target / z0_arr) / np.log(h_ref_arr / z0_arr)
    coeff = np.where(np.isfinite(coeff), coeff, 1.0)
    coeff = np.clip(coeff, min_coeff, max_coeff)
    return _finalise(wind * coeff, max_wind=max_wind)


# ---------------------------------------------------------------------------
# 5. Stability-dependent power law
# ---------------------------------------------------------------------------
def stability_dependent(
    wind_10m: NDArray[np.floating],
    target_height: float,
    *,
    reference_height: float = DEFAULT_REFERENCE_HEIGHT,
    wind_scale: float = 15.0,
    alpha_base: float = 0.15,
    alpha_min: float = 0.08,
    alpha_max: float = 0.3,
    stab_clip: tuple[float, float] = (0.8, 1.3),
    max_wind: float = DEFAULT_MAX_WIND,
) -> NDArray[np.float32]:
    """Power law with exponent modulated by surface wind speed.

    Captures the paper's crude PBL-stability proxy: strong surface
    winds → well-mixed boundary layer → smaller ``alpha``; weak winds
    → stratified layer → larger ``alpha``.  Formula::

        stab        = 1.2 - clip(wind_10m / wind_scale, 0, 0.4)
        alpha_stab  = clip(alpha_base * clip(stab, 0.8, 1.3),
                           alpha_min, alpha_max)
        wind(h)     = wind_10m * (h / h_ref) ** alpha_stab
    """
    wind = np.asarray(wind_10m, dtype=float)
    ratio = float(target_height) / float(reference_height)

    stab = 1.2 - np.clip(wind / wind_scale, 0.0, 0.4)
    alpha_s = np.clip(alpha_base * np.clip(stab, stab_clip[0], stab_clip[1]),
                      alpha_min, alpha_max)
    return _finalise(wind * np.power(ratio, alpha_s), max_wind=max_wind)


# ---------------------------------------------------------------------------
# 6. Roughness-adaptive: switch method by regime
# ---------------------------------------------------------------------------
def roughness_adaptive(
    wind_10m: NDArray[np.floating],
    target_height: float,
    *,
    enhanced_log: NDArray[np.floating],
    enhanced_power: NDArray[np.floating],
    smooth_mask: NDArray[np.bool_],
    moderate_mask: NDArray[np.bool_],
    rough_mask: NDArray[np.bool_],
    rough_alpha: float = 0.25,
    reference_height: float = DEFAULT_REFERENCE_HEIGHT,
    max_wind: float = DEFAULT_MAX_WIND,
) -> NDArray[np.float32]:
    """Pick a per-regime formulation for each cell.

    * ``smooth_mask`` → use the already-computed ``enhanced_log``.
    * ``moderate_mask`` → use the already-computed ``enhanced_power``.
    * ``rough_mask`` → fixed-alpha power law with
      ``alpha = rough_alpha`` (default 0.25).

    All three masks must cover every pixel (their logical OR should be
    all-true); the three outputs of :func:`enhanced_power_law` and
    :func:`enhanced_log_law` must share the shape of ``wind_10m``.
    """
    wind = np.asarray(wind_10m, dtype=float)
    el = np.asarray(enhanced_log, dtype=float)
    ep = np.asarray(enhanced_power, dtype=float)

    ratio = float(target_height) / float(reference_height)
    rough_output = wind * (ratio ** float(rough_alpha))

    # Broadcast the (ny, nx) masks across the leading time axis when
    # wind is 3-D.  np.where does this for us.
    result = np.where(smooth_mask, el, np.where(moderate_mask, ep, rough_output))
    return _finalise(result, max_wind=max_wind)


# ---------------------------------------------------------------------------
# 7. Hybrid power / log
# ---------------------------------------------------------------------------
def hybrid_power_log(
    wind_10m: NDArray[np.floating],
    *,
    enhanced_log: NDArray[np.floating],
    enhanced_power: NDArray[np.floating],
    z0: NDArray[np.floating],
    wind_scale: float = 15.0,
    z0_scale: float = 0.01,
    weight_min: float = 0.1,
    weight_max: float = 0.9,
    max_wind: float = DEFAULT_MAX_WIND,
) -> NDArray[np.float32]:
    """Convex blend of enhanced-log and enhanced-power predictions.

    Weight formula::

        w = 0.5
          + 0.3 * clip(wind_10m / wind_scale, 0, 1)
          + 0.2 * clip(1 - log10(clip(z0 / z0_scale, 0.1, 100)), 0, 1)
        w = clip(w, weight_min, weight_max)

    then ``wind(h) = w * enhanced_log + (1 - w) * enhanced_power``.

    * Stronger winds → shift towards the log-law prediction (which is
      better anchored in the stable regime).
    * Smaller roughness → similarly shift towards log-law.

    Thresholds / constants match the paper's ``_extrapolate_cpu``
    hybrid branch exactly.
    """
    wind = np.asarray(wind_10m, dtype=float)
    el = np.asarray(enhanced_log, dtype=float)
    ep = np.asarray(enhanced_power, dtype=float)
    z0_arr = np.asarray(z0, dtype=float)

    w = (
        0.5
        + 0.3 * np.clip(wind / wind_scale, 0.0, 1.0)
        + 0.2 * np.clip(1.0 - np.log10(np.clip(z0_arr / z0_scale, 0.1, 100.0)), 0.0, 1.0)
    )
    w = np.clip(w, weight_min, weight_max)
    return _finalise(w * el + (1.0 - w) * ep, max_wind=max_wind)
