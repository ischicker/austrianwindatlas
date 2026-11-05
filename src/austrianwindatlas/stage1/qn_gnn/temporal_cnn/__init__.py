"""Temporal CNN refinement for the QN-GNN pipeline.

Stage-2 of QN-GNN+TCNN: a lightweight 1-D CNN that reads a 48-hour
sliding window of station observations and predicts a per-timestep
``ΔU`` correction for the GNN's EOF coefficients.  The GNN handles
the spatial mapping (terrain → V); the CNN brings in the synoptic
state so the full reconstruction has non-trivial temporal skill at
high-wind events (:math:`hw_r` jumps from ~0.05 for pure GNN to
competitive levels).

Submodules:

* :mod:`.model` — :class:`TemporalCNN` nn.Module factory
* :mod:`.data`  — :func:`sample_availability`, :func:`compute_U_true`,
  :class:`UResidualDataset`
* :mod:`.train` — :class:`TrainConfig`, :func:`train_temporal_cnn`,
  :func:`save_cnn`, :func:`load_cnn`
"""
from __future__ import annotations

from .data import UResidualDataset, compute_U_true, sample_availability
from .model import TemporalCNN, build_temporal_cnn
from .train import (
    EpochRecord,
    TrainConfig,
    TrainResult,
    load_cnn,
    save_cnn,
    train_temporal_cnn,
)

__all__ = [
    # model
    "TemporalCNN",
    "build_temporal_cnn",
    # data
    "UResidualDataset",
    "compute_U_true",
    "sample_availability",
    # train
    "TrainConfig",
    "EpochRecord",
    "TrainResult",
    "train_temporal_cnn",
    "save_cnn",
    "load_cnn",
]
