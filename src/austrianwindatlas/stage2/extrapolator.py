"""Run the seven hub-height formulations in one pass.

:class:`MultiHeightExtrapolator` batches every method across a stack
of target heights and a ``(T, ny, nx)`` wind cube.  Equivalent to the
paper's ``FastGPUExtrapolator._extrapolate_cpu`` but factored into
small composable pieces.

Usage::

    from austrianwindatlas.stage2 import MultiHeightExtrapolator

    extr = MultiHeightExtrapolator.from_roughness(
        corine_z0,                        # (ny, nx) raw roughness
        target_heights=(100.0, 150.0),
    )
    out = extr.extrapolate(wind_10m)      # dict[height -> dict[method -> arr]]

The result is a nested dict so callers can pick any one method at any
one height without materialising the others — important when running
many months and only the ensemble is needed on disk.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

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
from .roughness import RoughnessFields, derive_roughness_fields

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = [
    "METHOD_NAMES",
    "ExtrapolationConfig",
    "MultiHeightExtrapolator",
]

logger = logging.getLogger(__name__)

#: Canonical method-name order used by :class:`MultiHeightExtrapolator`.
#: Matches the paper's :class:`FastGPUExtrapolator` output keys.
METHOD_NAMES: tuple[str, ...] = (
    "power_law",
    "log_law",
    "enhanced_power",
    "enhanced_log",
    "stability",
    "adaptive",
    "hybrid",
    "ensemble",
)


@dataclass
class ExtrapolationConfig:
    """Which methods to run + their shared constants.

    Flip any ``use_*`` to False to skip that method entirely (useful
    for benchmark runs that only want the best three).  Defaults match
    the paper's production configuration.
    """

    target_heights: tuple[float, ...] = (80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 250.0)
    reference_height: float = DEFAULT_REFERENCE_HEIGHT

    use_classic_power_law: bool = True
    use_classic_log_law: bool = True
    use_enhanced_power_law: bool = True
    use_enhanced_log_law: bool = True
    use_stability_dependent: bool = True
    use_roughness_adaptive: bool = True
    use_hybrid: bool = True
    compute_ensemble: bool = True

    classic_alpha: float = DEFAULT_CLASSIC_ALPHA
    classic_z0: float = DEFAULT_CLASSIC_Z0
    rough_alpha: float = 0.25
    max_wind: float = DEFAULT_MAX_WIND

    # QC thresholds applied to the input before extrapolation.
    max_wind_speed: float = 50.0
    min_wind_speed: float = 0.05


@dataclass
class MultiHeightExtrapolator:
    """Batches every hub-height formulation across a ``(T, ny, nx)`` input.

    Build either from a raw roughness raster via
    :meth:`from_roughness` (one-shot constructor that derives all
    gridded fields internally) or by passing a pre-computed
    :class:`RoughnessFields` instance to the regular constructor.

    The ``(ny, nx)`` shape of the roughness must match the spatial
    shape of the wind cube passed to :meth:`extrapolate`; a clear
    error is raised otherwise.
    """

    roughness: RoughnessFields
    config: ExtrapolationConfig = field(default_factory=ExtrapolationConfig)

    @classmethod
    def from_roughness(
        cls,
        raw_z0: NDArray[np.floating],
        *,
        target_heights: tuple[float, ...] = (80.0, 100.0, 120.0, 140.0, 160.0, 180.0,
                                             200.0, 220.0, 250.0),
        config: ExtrapolationConfig | None = None,
    ) -> MultiHeightExtrapolator:
        """Shortcut constructor: raw CORINE roughness → ready-to-run extrapolator."""
        fields = derive_roughness_fields(raw_z0)
        cfg = config if config is not None else ExtrapolationConfig(target_heights=tuple(target_heights))
        if config is None:
            cfg.target_heights = tuple(target_heights)
        return cls(roughness=fields, config=cfg)

    def _quality_control(self, wind_10m: NDArray[np.floating]) -> NDArray[np.float32]:
        """Replace Inf / out-of-range with NaN; clamp tiny positives to the floor.

        Matches ``FastGPUExtrapolator._extrapolate_cpu``'s QC.
        """
        arr = np.asarray(wind_10m, dtype=np.float32).copy()
        arr[np.isinf(arr)] = np.nan
        arr[arr > self.config.max_wind_speed] = np.nan
        tiny = (arr > 0) & (arr < self.config.min_wind_speed)
        arr[tiny] = self.config.min_wind_speed
        return arr

    def _check_shape(self, wind_10m: NDArray[np.floating]) -> None:
        if wind_10m.ndim != 3:
            raise ValueError(
                f"wind_10m must be 3-D (T, ny, nx); got shape {wind_10m.shape!r}"
            )
        if wind_10m.shape[1:] != self.roughness.shape:
            raise ValueError(
                f"wind_10m spatial shape {wind_10m.shape[1:]!r} does not match "
                f"roughness shape {self.roughness.shape!r}"
            )

    def extrapolate(
        self,
        wind_10m: NDArray[np.floating],
    ) -> dict[float, dict[str, NDArray[np.float32]]]:
        """Compute every enabled method at every target height.

        Returns a nested dict ``result[height][method]`` with values
        of shape ``(T, ny, nx)``, dtype ``float32``.  ``method`` keys
        appear in the order listed by :data:`METHOD_NAMES` and only
        for methods actually enabled by the config.
        """
        self._check_shape(wind_10m)
        wind = self._quality_control(wind_10m)

        all_results: dict[float, dict[str, NDArray[np.float32]]] = {}
        cfg = self.config
        rf = self.roughness
        h_ref = cfg.reference_height

        for height in cfg.target_heights:
            results: dict[str, NDArray[np.float32]] = {}
            stack: list[NDArray[np.float32]] = []

            if cfg.use_classic_power_law:
                r = classic_power_law(
                    wind, height, alpha=cfg.classic_alpha,
                    reference_height=h_ref, max_wind=cfg.max_wind,
                )
                results["power_law"] = r
                stack.append(r)

            if cfg.use_classic_log_law:
                r = classic_log_law(
                    wind, height, z0=cfg.classic_z0,
                    reference_height=h_ref, max_wind=cfg.max_wind,
                )
                results["log_law"] = r
                stack.append(r)

            if cfg.use_enhanced_power_law:
                r = enhanced_power_law(
                    wind, height, rf.alpha,
                    reference_height=h_ref, max_wind=cfg.max_wind,
                )
                results["enhanced_power"] = r
                stack.append(r)

            if cfg.use_enhanced_log_law:
                r = enhanced_log_law(
                    wind, height, rf.z0, rf.h_ref,
                    max_wind=cfg.max_wind,
                )
                results["enhanced_log"] = r
                stack.append(r)

            if cfg.use_stability_dependent:
                r = stability_dependent(
                    wind, height, reference_height=h_ref,
                    max_wind=cfg.max_wind,
                )
                results["stability"] = r
                stack.append(r)

            have_both_enhanced = "enhanced_log" in results and "enhanced_power" in results
            if cfg.use_roughness_adaptive and have_both_enhanced:
                r = roughness_adaptive(
                    wind, height,
                    enhanced_log=results["enhanced_log"],
                    enhanced_power=results["enhanced_power"],
                    smooth_mask=rf.smooth,
                    moderate_mask=rf.moderate,
                    rough_mask=rf.rough,
                    rough_alpha=cfg.rough_alpha,
                    reference_height=h_ref, max_wind=cfg.max_wind,
                )
                results["adaptive"] = r
                stack.append(r)

            if cfg.use_hybrid and have_both_enhanced:
                r = hybrid_power_log(
                    wind,
                    enhanced_log=results["enhanced_log"],
                    enhanced_power=results["enhanced_power"],
                    z0=rf.z0,
                    max_wind=cfg.max_wind,
                )
                results["hybrid"] = r
                stack.append(r)

            if cfg.compute_ensemble and len(stack) >= 2:
                stacked = np.stack(stack, axis=0)
                results["ensemble"] = np.nanmean(stacked, axis=0).astype(np.float32)

            all_results[float(height)] = results

        return all_results
