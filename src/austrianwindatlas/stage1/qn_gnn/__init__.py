"""QN-GNN method: terrain-aware graph attention + MLP dual path.

Predicts per-station EOF loadings ``V`` from terrain covariates via a
Graph Attention Network with a terrain-barrier distance bias
(``graph.build_station_graph``), trained with a spatial validation
holdout so early-stopping signals *interpolation to unseen stations*
— the same task the paper's verification measures.

Reconstruction formula (same as the Enhanced-MAE path but in
z-score space before the final calibrator)::

    recon_z(t, s)   = temporal_mean(t) + sum_k U(t,k) * sigma_k * V(s,k)
    recon_phys(t,s) = recon_z(t, s) * sigma_s + mu_s           # linear inverse

Submodules:

* :mod:`.zscore` — :class:`StationWiseZScoreTransformer`
* :mod:`.graph`  — :func:`build_station_graph`
* :mod:`.svd`    — :class:`ZScoreSVD`, :func:`compute_zscore_svd`
* :mod:`.losses` — directional / variance / physical-recon / spatial-smoothness
* :mod:`.model`  — :class:`TerrainGNN`
* :mod:`.train`  — :class:`TrainConfig`, :func:`train_gnn`
* :mod:`.predict`— :func:`predict_grid`, :func:`spatial_bias_map`,
  :func:`variance_correct`
* :mod:`.temporal_cnn` — Stage-2 Temporal CNN that predicts a
  per-timestep ``ΔU`` correction (see sub-package docstring).
"""
from __future__ import annotations

from . import temporal_cnn
from .calibrator import (
    AsymmetricCalibrator,
    ElevationBiasCorrector,
    SpatialWindCalibrator,
)
from .graph import StationGraph, build_station_graph
from .losses import (
    DirectionalLossKind,
    directional_loss,
    high_wind_time_weights,
    physical_recon_loss,
    spatial_smoothness,
    variance_penalty,
)
from .model import TerrainGNN, build_terrain_gnn
from .predict import predict_grid, spatial_bias_map, variance_correct
from .reconstruct import CalibratorLike, CNNMode, compute_cnn_U, reconstruct_month
from .svd import ZScoreSVD, compute_zscore_svd
from .train import EpochRecord, TrainConfig, TrainResult, train_gnn
from .zscore import StationWiseZScoreTransformer

__all__ = [
    # zscore
    "StationWiseZScoreTransformer",
    # graph
    "StationGraph",
    "build_station_graph",
    # svd
    "ZScoreSVD",
    "compute_zscore_svd",
    # losses
    "DirectionalLossKind",
    "directional_loss",
    "variance_penalty",
    "physical_recon_loss",
    "spatial_smoothness",
    "high_wind_time_weights",
    # model
    "TerrainGNN",
    "build_terrain_gnn",
    # train
    "TrainConfig",
    "EpochRecord",
    "TrainResult",
    "train_gnn",
    # predict
    "predict_grid",
    "spatial_bias_map",
    "variance_correct",
    # calibrator
    "SpatialWindCalibrator",
    "AsymmetricCalibrator",
    "ElevationBiasCorrector",
    # reconstruct
    "CalibratorLike",
    "CNNMode",
    "compute_cnn_U",
    "reconstruct_month",
    # temporal_cnn sub-package (access via `stage1.qn_gnn.temporal_cnn.*`)
    "temporal_cnn",
]
