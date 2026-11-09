"""Unit tests for the GPU-dispatch path in :mod:`austrianwindatlas.stage2.extrapolator`.

cupy isn't installed on the CI runners (it needs a CUDA device), so we
test the dispatch logic with a **fake array module**: an object that
exposes exactly the numpy functions the extrapolator calls, plus a
``.get()`` method on its arrays to mimic cupy's device→host copy.

The fake is backed by real numpy under the hood, so the arithmetic
result equals the CPU path bit-for-bit — which gives us a
regression-safe way to verify the GPU dispatch doesn't drift from
the CPU formulas.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from austrianwindatlas.stage2 import formulations
from austrianwindatlas.stage2.extrapolator import (
    HAS_CUPY,
    ExtrapolationConfig,
    MultiHeightExtrapolator,
)


# ---------------------------------------------------------------------------
# Fake cupy — numpy under the hood but exposes .get() like cupy ndarrays do
# ---------------------------------------------------------------------------
class _FakeArray(np.ndarray):
    """Numpy subclass that pretends to be on-device."""

    def get(self):
        """Mimic cupy's device→host copy."""
        return np.asarray(self)


def _wrap(value):
    """Wrap a numpy array in :class:`_FakeArray`.  Scalars pass through."""
    if isinstance(value, np.ndarray):
        return value.view(_FakeArray)
    return value


class _FakeCupyModule:
    """A duck-typed stand-in for :mod:`cupy`.

    Every method delegates to numpy and wraps the result so
    downstream code sees ``.get()`` and other cupy-ndarray surface.
    Only the methods the extrapolator actually touches are wrapped.
    """

    get_default_memory_pool_called = False

    def asarray(self, a, dtype=None):
        arr = np.asarray(a, dtype=dtype)
        return _wrap(arr.copy())        # copy = simulate host→device transfer

    def stack(self, arrays, axis=0):
        return _wrap(np.stack(arrays, axis=axis))

    def nanmean(self, a, axis=None):
        return _wrap(np.nanmean(np.asarray(a), axis=axis))

    # Functions the formulations call.
    def clip(self, a, lo, hi, *args, **kwargs):
        return _wrap(np.clip(np.asarray(a), lo, hi))

    def log(self, a):
        return _wrap(np.log(np.asarray(a)))

    def log10(self, a):
        return _wrap(np.log10(np.asarray(a)))

    def power(self, base, exp):
        return _wrap(np.power(base, np.asarray(exp)) if hasattr(exp, "__array__")
                     else np.power(np.asarray(base) if hasattr(base, "__array__") else base,
                                   exp))

    def where(self, cond, a, b):
        return _wrap(np.where(np.asarray(cond), np.asarray(a), np.asarray(b)))

    def isfinite(self, a):
        return _wrap(np.isfinite(np.asarray(a)))

    def isinf(self, a):
        return _wrap(np.isinf(np.asarray(a)))

    def maximum(self, a, b):
        return _wrap(np.maximum(np.asarray(a) if hasattr(a, "__array__") else a,
                                np.asarray(b) if hasattr(b, "__array__") else b))

    def zeros_like(self, a):
        return _wrap(np.zeros_like(a))


def _tiny_roughness(ny=4, nx=5):
    rng = np.random.default_rng(0)
    return rng.choice([0.01, 0.1, 0.5, 1.0], size=(ny, nx))


# ---------------------------------------------------------------------------
# Formulations respect the xp kwarg
# ---------------------------------------------------------------------------
def test_classic_power_law_with_fake_xp_matches_numpy():
    """Using xp=fake must produce the same result as xp=None (numpy)."""
    rng = np.random.default_rng(0)
    wind = rng.uniform(1, 10, size=(4, 3, 5))
    cpu = formulations.classic_power_law(wind, 100.0)
    gpu_like = formulations.classic_power_law(wind, 100.0, xp=_FakeCupyModule())
    np.testing.assert_allclose(cpu, np.asarray(gpu_like), rtol=0, atol=1e-6)


def test_all_formulations_respect_xp_kwarg_without_crashing():
    """Every public formulation accepts xp= and returns a sensible shape."""
    rng = np.random.default_rng(0)
    xp = _FakeCupyModule()
    wind = rng.uniform(1, 10, size=(2, 3, 5))
    alpha_grid = rng.uniform(0.1, 0.3, size=(3, 5))
    z0 = rng.uniform(0.01, 1.0, size=(3, 5))
    h_ref = np.maximum(10.0, 2.0 * z0)
    smooth = z0 < 0.05
    moderate = (z0 >= 0.05) & (z0 < 0.5)
    rough = z0 >= 0.5
    el = formulations.enhanced_log_law(wind, 100.0, z0, h_ref, xp=xp)
    ep = formulations.enhanced_power_law(wind, 100.0, alpha_grid, xp=xp)
    results = {
        "classic_power": formulations.classic_power_law(wind, 100.0, xp=xp),
        "classic_log": formulations.classic_log_law(wind, 100.0, xp=xp),
        "enhanced_power": ep,
        "enhanced_log": el,
        "stability": formulations.stability_dependent(wind, 100.0, xp=xp),
        "adaptive": formulations.roughness_adaptive(
            wind, 100.0,
            enhanced_log=el, enhanced_power=ep,
            smooth_mask=smooth, moderate_mask=moderate, rough_mask=rough,
            xp=xp,
        ),
        "hybrid": formulations.hybrid_power_log(
            wind, enhanced_log=el, enhanced_power=ep, z0=z0, xp=xp,
        ),
    }
    for name, arr in results.items():
        assert arr.shape == wind.shape, name
        assert arr.dtype == np.float32, name


# ---------------------------------------------------------------------------
# Config carries use_gpu and the dispatcher honours it
# ---------------------------------------------------------------------------
def test_extrapolation_config_has_use_gpu_default_false():
    cfg = ExtrapolationConfig()
    assert cfg.use_gpu is False


def test_use_gpu_false_runs_on_cpu(monkeypatch):
    """The default path doesn't touch cupy at all."""
    raw = _tiny_roughness()
    extr = MultiHeightExtrapolator.from_roughness(raw, target_heights=(100.0,))

    # Resolve backend directly — should return (np, False).
    xp, on_gpu = extr._resolve_backend()
    assert xp is np
    assert on_gpu is False


def test_use_gpu_without_cupy_falls_back_to_cpu(monkeypatch, caplog):
    """use_gpu=True + HAS_CUPY=False logs a warning and runs on CPU."""
    monkeypatch.setattr(
        "austrianwindatlas.stage2.extrapolator.HAS_CUPY", False,
    )
    raw = _tiny_roughness()
    cfg = ExtrapolationConfig(target_heights=(100.0,), use_gpu=True)
    extr = MultiHeightExtrapolator.from_roughness(raw, config=cfg)

    with caplog.at_level(logging.WARNING,
                         logger="austrianwindatlas.stage2.extrapolator"):
        xp, on_gpu = extr._resolve_backend()
    assert xp is np
    assert on_gpu is False
    assert any("cupy is not installed" in m for m in caplog.messages)


def test_use_gpu_with_fake_cupy_matches_cpu_output(monkeypatch):
    """End-to-end: inject the fake cupy module, run with use_gpu=True,
    and verify the result equals the CPU-only run bit-for-bit."""
    fake_cp = _FakeCupyModule()
    monkeypatch.setattr(
        "austrianwindatlas.stage2.extrapolator._cp", fake_cp,
    )
    monkeypatch.setattr(
        "austrianwindatlas.stage2.extrapolator.HAS_CUPY", True,
    )

    raw = _tiny_roughness()
    rng = np.random.default_rng(1)
    wind = rng.uniform(1, 10, size=(3, *raw.shape)).astype(np.float32)

    cpu_cfg = ExtrapolationConfig(target_heights=(100.0, 150.0), use_gpu=False)
    gpu_cfg = ExtrapolationConfig(target_heights=(100.0, 150.0), use_gpu=True)
    cpu_extr = MultiHeightExtrapolator.from_roughness(raw, config=cpu_cfg)
    gpu_extr = MultiHeightExtrapolator.from_roughness(raw, config=gpu_cfg)

    cpu_out = cpu_extr.extrapolate(wind)
    gpu_out = gpu_extr.extrapolate(wind)

    assert set(cpu_out) == set(gpu_out)
    for h in cpu_out:
        assert set(cpu_out[h]) == set(gpu_out[h])
        for method in cpu_out[h]:
            # The fake cupy is numpy underneath → identical arithmetic.
            np.testing.assert_allclose(
                cpu_out[h][method], gpu_out[h][method], rtol=0, atol=1e-6,
                err_msg=f"CPU vs fake-GPU mismatch at height={h}, method={method}",
            )
            # Result is on host (numpy).
            assert isinstance(gpu_out[h][method], np.ndarray)
            # And a plain ndarray — ``get()`` has been called.
            assert type(gpu_out[h][method]) is np.ndarray


def test_gpu_path_falls_back_on_runtime_error(monkeypatch, caplog):
    """If the GPU backend raises during extrapolate(), the dispatcher
    retries on CPU rather than propagating the exception."""
    class _BrokenCupy(_FakeCupyModule):
        def asarray(self, a, dtype=None):
            raise RuntimeError("simulated CUDA OOM")

        def get_default_memory_pool(self):          # called by the retry cleanup
            class _Pool:
                def free_all_blocks(self):
                    _BrokenCupy.get_default_memory_pool_called = True
            return _Pool()

    broken = _BrokenCupy()
    monkeypatch.setattr(
        "austrianwindatlas.stage2.extrapolator._cp", broken,
    )
    monkeypatch.setattr(
        "austrianwindatlas.stage2.extrapolator.HAS_CUPY", True,
    )

    raw = _tiny_roughness()
    cfg = ExtrapolationConfig(target_heights=(100.0,), use_gpu=True)
    extr = MultiHeightExtrapolator.from_roughness(raw, config=cfg)
    wind = np.ones((1, *raw.shape), dtype=np.float32) * 5.0

    with caplog.at_level(logging.WARNING,
                         logger="austrianwindatlas.stage2.extrapolator"):
        out = extr.extrapolate(wind)
    # GPU failure logged + CPU retry succeeded.
    assert any("GPU path failed" in m for m in caplog.messages)
    assert 100.0 in out
    assert "power_law" in out[100.0]
    # And the memory-pool cleanup ran.
    assert _BrokenCupy.get_default_memory_pool_called is True


def test_use_gpu_stored_on_config():
    cfg = ExtrapolationConfig(use_gpu=True)
    assert cfg.use_gpu is True


# ---------------------------------------------------------------------------
# Live GPU test — only if real cupy is actually installed on the runner
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_CUPY, reason="cupy not installed")
def test_live_cupy_path_matches_cpu():  # pragma: no cover
    """Live-GPU path: only runs on hosts with a working CUDA + cupy stack.

    Does the full ExtrapolationConfig(use_gpu=True) → extrapolate() run
    and compares the result to the CPU reference arithmetic.
    """
    raw = _tiny_roughness()
    rng = np.random.default_rng(42)
    wind = rng.uniform(1, 10, size=(2, *raw.shape)).astype(np.float32)

    cpu = MultiHeightExtrapolator.from_roughness(
        raw, config=ExtrapolationConfig(target_heights=(100.0,))
    ).extrapolate(wind)
    gpu = MultiHeightExtrapolator.from_roughness(
        raw, config=ExtrapolationConfig(target_heights=(100.0,), use_gpu=True)
    ).extrapolate(wind)

    for method in cpu[100.0]:
        np.testing.assert_allclose(
            cpu[100.0][method], gpu[100.0][method], rtol=0, atol=1e-5,
            err_msg=f"CPU vs cupy mismatch at method={method}",
        )
