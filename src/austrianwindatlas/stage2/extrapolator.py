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

import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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

try:
    import cupy as _cp

    HAS_CUPY: bool = True
except ImportError:  # pragma: no cover — environment-gated
    _cp = None  # type: ignore[assignment]
    HAS_CUPY = False

__all__ = [
    "HAS_CUPY",
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

    # GPU dispatch.  When :data:`True` and cupy is importable, the
    # per-method arithmetic runs on-device; the input is moved once
    # and every height is batched on the same device.  Silently
    # falls back on cupy ImportError or on CUDA runtime errors.
    use_gpu: bool = False


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

    def _quality_control(
        self, wind_10m: NDArray[np.floating], *, xp: Any = np,
    ) -> NDArray[np.float32]:
        """Replace Inf / out-of-range with NaN; clamp tiny positives to the floor.

        Matches ``FastGPUExtrapolator._extrapolate_cpu``'s QC.  Works on
        both numpy and cupy arrays — the only ``xp``-specific calls are
        ``xp.asarray`` and ``xp.isinf``; the rest is broadcast arithmetic.
        """
        arr = xp.asarray(wind_10m, dtype=np.float32).copy()
        arr[xp.isinf(arr)] = np.nan
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

    def _resolve_backend(self) -> tuple[Any, bool]:
        """Decide whether to run on GPU and return ``(xp, on_gpu)``.

        Honours :attr:`ExtrapolationConfig.use_gpu` but falls back on
        the CPU path if cupy is missing.  A warning is logged so the
        fallback is visible in production logs.
        """
        if not self.config.use_gpu:
            return np, False
        if not HAS_CUPY:
            logger.warning(
                "stage2.extrapolator: use_gpu=True but cupy is not installed; "
                "falling back to numpy.  Install the gpu extra to enable "
                "GPU dispatch."
            )
            return np, False
        return _cp, True

    def extrapolate(
        self,
        wind_10m: NDArray[np.floating],
    ) -> dict[float, dict[str, NDArray[np.float32]]]:
        """Compute every enabled method at every target height.

        Returns a nested dict ``result[height][method]`` with values
        of shape ``(T, ny, nx)``, dtype ``float32`` (always on host —
        even when :attr:`ExtrapolationConfig.use_gpu` is :data:`True`
        the results are copied back to host before return).
        ``method`` keys appear in the order listed by :data:`METHOD_NAMES`
        and only for methods actually enabled by the config.
        """
        self._check_shape(wind_10m)

        xp, on_gpu = self._resolve_backend()
        if on_gpu:
            try:
                return self._extrapolate_impl(wind_10m, xp=xp, on_gpu=True)
            except Exception as exc:  # noqa: BLE001 — any GPU error → CPU retry
                logger.warning(
                    "stage2.extrapolator: GPU path failed (%s); retrying on CPU.",
                    exc,
                )
                # Release device buffers before the CPU retry.
                with contextlib.suppress(Exception):  # pragma: no cover
                    _cp.get_default_memory_pool().free_all_blocks()
                return self._extrapolate_impl(wind_10m, xp=np, on_gpu=False)
        return self._extrapolate_impl(wind_10m, xp=xp, on_gpu=False)

    def _extrapolate_impl(
        self,
        wind_10m: NDArray[np.floating],
        *,
        xp: Any,
        on_gpu: bool,
    ) -> dict[float, dict[str, NDArray[np.float32]]]:
        """Shared batched loop over target heights.  CPU and GPU share
        this code path — only ``xp`` differs."""
        wind = self._quality_control(wind_10m, xp=xp)

        cfg = self.config
        rf = self.roughness
        h_ref = cfg.reference_height

        # Move roughness fields to the same device as ``wind``.
        rf_alpha = xp.asarray(rf.alpha)
        rf_z0 = xp.asarray(rf.z0)
        rf_h_ref = xp.asarray(rf.h_ref)
        rf_smooth = xp.asarray(rf.smooth)
        rf_moderate = xp.asarray(rf.moderate)
        rf_rough = xp.asarray(rf.rough)

        all_results: dict[float, dict[str, NDArray[np.float32]]] = {}

        for height in cfg.target_heights:
            results_on_device: dict[str, Any] = {}
            stack: list[Any] = []

            if cfg.use_classic_power_law:
                r = classic_power_law(
                    wind, height, alpha=cfg.classic_alpha,
                    reference_height=h_ref, max_wind=cfg.max_wind, xp=xp,
                )
                results_on_device["power_law"] = r
                stack.append(r)

            if cfg.use_classic_log_law:
                r = classic_log_law(
                    wind, height, z0=cfg.classic_z0,
                    reference_height=h_ref, max_wind=cfg.max_wind, xp=xp,
                )
                results_on_device["log_law"] = r
                stack.append(r)

            if cfg.use_enhanced_power_law:
                r = enhanced_power_law(
                    wind, height, rf_alpha,
                    reference_height=h_ref, max_wind=cfg.max_wind, xp=xp,
                )
                results_on_device["enhanced_power"] = r
                stack.append(r)

            if cfg.use_enhanced_log_law:
                r = enhanced_log_law(
                    wind, height, rf_z0, rf_h_ref,
                    max_wind=cfg.max_wind, xp=xp,
                )
                results_on_device["enhanced_log"] = r
                stack.append(r)

            if cfg.use_stability_dependent:
                r = stability_dependent(
                    wind, height, reference_height=h_ref,
                    max_wind=cfg.max_wind, xp=xp,
                )
                results_on_device["stability"] = r
                stack.append(r)

            have_both_enhanced = (
                "enhanced_log" in results_on_device
                and "enhanced_power" in results_on_device
            )
            if cfg.use_roughness_adaptive and have_both_enhanced:
                r = roughness_adaptive(
                    wind, height,
                    enhanced_log=results_on_device["enhanced_log"],
                    enhanced_power=results_on_device["enhanced_power"],
                    smooth_mask=rf_smooth,
                    moderate_mask=rf_moderate,
                    rough_mask=rf_rough,
                    rough_alpha=cfg.rough_alpha,
                    reference_height=h_ref, max_wind=cfg.max_wind, xp=xp,
                )
                results_on_device["adaptive"] = r
                stack.append(r)

            if cfg.use_hybrid and have_both_enhanced:
                r = hybrid_power_log(
                    wind,
                    enhanced_log=results_on_device["enhanced_log"],
                    enhanced_power=results_on_device["enhanced_power"],
                    z0=rf_z0,
                    max_wind=cfg.max_wind, xp=xp,
                )
                results_on_device["hybrid"] = r
                stack.append(r)

            if cfg.compute_ensemble and len(stack) >= 2:
                stacked = xp.stack(stack, axis=0)
                results_on_device["ensemble"] = (
                    xp.nanmean(stacked, axis=0).astype(np.float32)
                )

            # Move everything back to host — the public API promises
            # numpy arrays, not device arrays.
            all_results[float(height)] = {
                name: (arr.get() if on_gpu and hasattr(arr, "get") else np.asarray(arr))
                for name, arr in results_on_device.items()
            }

        return all_results
