"""Core utilities: paths, config, rPCA/EOF decomposition, grid definitions."""
from austrianwindatlas.core import paths, rpca  # noqa: F401
from austrianwindatlas.core.rpca import HAS_CUPY, RobustPCA, RPCAIteration

__all__ = ["paths", "rpca", "RobustPCA", "RPCAIteration", "HAS_CUPY"]
