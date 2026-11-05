"""Enhanced-MAE / Enhanced-WMSE DL method.

Trains a terrain-aware DNN to predict the station-side EOF loadings
``V`` of a temporally-centred wind matrix, then reconstructs the
gridded wind field via :math:`\\mathrm{wind}(t, s) =
\\overline{\\mathrm{wind}}(t) + \\sum_k U(t,k)\\sigma_k V(s,k)`.

The two variants shipped with the paper are:

* **Enhanced-MAE** — ``loss_type="asymmetric"`` or ``"mae"``, no
  station weights.  Best RMSE in the paper's open-air verification
  (RMSE=1.40, r=0.685 on 2020 with CERRA background).
* **Enhanced-WMSE** — ``loss_type="weighted_mse"`` with
  ``weight_method="mean_wind"``.  Best bias in the paper
  (bias=-0.005 with ERA5 background).

Submodules:

* :mod:`.losses` — :class:`LossFunction`, :func:`compute_station_weights`
* :mod:`.svd` — temporal-centering SVD + NaN handling
* :mod:`.features` — terrain-feature extraction at stations and on-grid
* :mod:`.model` — PyTorch model + training loop
* :mod:`.predict` — grid prediction, projection, reconstruction
"""
from __future__ import annotations

from .features import (
    DEFAULT_COVARIATE_FILES,
    FeatureSpec,
    build_feature_spec_for_covariates,
    sample_features_at_stations,
    sample_features_on_grid,
)
from .losses import LossFunction, LossKind, WeightMethod, compute_station_weights
from .model import EpochRecord, SpatialPatternDNN, TrainConfig, TrainResult, train_dnn
from .predict import predict_grid, project_onto_eofs, reconstruct_month
from .svd import (
    TemporalSVD,
    compute_temporal_svd,
    fill_nans_with_station_mean,
    verify_reconstruction,
)

__all__ = [
    # losses
    "LossFunction",
    "LossKind",
    "WeightMethod",
    "compute_station_weights",
    # svd
    "TemporalSVD",
    "compute_temporal_svd",
    "fill_nans_with_station_mean",
    "verify_reconstruction",
    # features
    "DEFAULT_COVARIATE_FILES",
    "FeatureSpec",
    "build_feature_spec_for_covariates",
    "sample_features_at_stations",
    "sample_features_on_grid",
    # model
    "SpatialPatternDNN",
    "TrainConfig",
    "TrainResult",
    "EpochRecord",
    "train_dnn",
    # predict
    "predict_grid",
    "project_onto_eofs",
    "reconstruct_month",
]
