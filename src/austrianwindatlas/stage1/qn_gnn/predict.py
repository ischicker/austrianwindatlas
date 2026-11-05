"""Grid prediction + spatial bias correction for the QN-GNN method.

At inference time the GNN half of the model is retired (the training
subgraph doesn't extend to the grid), so :func:`predict_grid` relies
on :meth:`TerrainGNN.forward_mlp` for a station-wise terrain →
EOF-loading mapping.  Two downstream corrections are applied before
the result leaves this module:

1. **Per-EOF variance correction.**  The DNN tends to under-emit
   variance; we rescale each EOF column so its std matches what the
   training stations saw.
2. **Spatial bias map.**  A 3-D inverse-distance interpolator
   (``dx^2 + dy^2 + (2 * dz)^2``) of the training residuals.
   Subtracted at reconstruction time to remove the systematic
   per-cell offset the DNN left behind.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ...io.windfield import INCA_GRID, GridSpec
from ..enhanced_mae.features import FeatureSpec, sample_features_on_grid
from .svd import ZScoreSVD

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import torch.nn as nn
    from numpy.typing import NDArray
    from sklearn.preprocessing import StandardScaler

__all__ = [
    "predict_grid",
    "spatial_bias_map",
    "variance_correct",
]

logger = logging.getLogger(__name__)


def variance_correct(
    V_grid: NDArray[np.floating],
    V_stations: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Rescale each column of ``V_grid`` so its std matches ``V_stations``.

    ``V_grid`` is ``(n_points, n_eofs)``; the returned array has the
    same shape and dtype float32.  Columns with near-zero grid std are
    left unchanged.
    """
    V_grid = np.asarray(V_grid, dtype=float)
    V_stations = np.asarray(V_stations, dtype=float)
    if V_grid.ndim != 2 or V_stations.ndim != 2:
        raise ValueError(
            f"both V_grid and V_stations must be 2-D; got shapes "
            f"{V_grid.shape!r} and {V_stations.shape!r}"
        )
    if V_grid.shape[1] != V_stations.shape[1]:
        raise ValueError(
            f"EOF count mismatch: V_grid has {V_grid.shape[1]}, "
            f"V_stations has {V_stations.shape[1]}"
        )

    out = np.zeros_like(V_grid, dtype=np.float32)
    for k in range(V_grid.shape[1]):
        col = V_grid[:, k]
        gmean = col.mean()
        gstd = col.std()
        sstd = V_stations[:, k].std()
        if gstd > 1e-10 and sstd > 1e-10:
            out[:, k] = ((col - gmean) * (sstd / gstd) + gmean).astype(np.float32)
        else:
            out[:, k] = col.astype(np.float32)
    return out


def spatial_bias_map(
    training_residuals: NDArray[np.floating],
    meta_df: pd.DataFrame,
    *,
    grid: GridSpec = INCA_GRID,
    elevation_grid: NDArray[np.floating] | None = None,
    elev_weight: float = 2.0,
    chunk_size: int = 20_000,
) -> NDArray[np.float32]:
    """Inverse-distance interpolate per-station residuals onto the grid.

    Distance metric is ``dx^2 + dy^2 + (elev_weight * dz)^2`` — so a
    grid cell at 500 m elevation gets most of its weight from
    stations at similar elevations, not just stations that are
    horizontally close.  Matches the paper's bias-map construction.

    Parameters
    ----------
    training_residuals
        ``(n_stations,)`` residual in m/s (``mean_t(recon - obs)`` on
        the training stations).  NaN stations are ignored.
    meta_df
        Must have ``x``, ``y`` columns; ``alt`` is optional (defaults
        to zero when missing).
    grid
        Target grid.
    elevation_grid
        Optional ``(nrows, ncols)`` DEM to drive the elevation
        component of the distance metric.  Defaults to zeros.
    elev_weight
        Metres-per-metre ratio applied to the vertical delta before
        squaring.  Paper default: 2.0.
    chunk_size
        Cells per batch — keeps the pairwise-distance matrix memory
        bounded.
    """
    sx = meta_df["x"].to_numpy(dtype=float)
    sy = meta_df["y"].to_numpy(dtype=float)
    sa = (
        meta_df["alt"].to_numpy(dtype=float)
        if "alt" in meta_df.columns else np.zeros(len(meta_df))
    )
    sr = np.asarray(training_residuals, dtype=float)

    valid = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(sr)
    sx, sy, sa, sr = sx[valid], sy[valid], sa[valid], sr[valid]
    if sr.size == 0:
        return np.zeros(grid.shape, dtype=np.float32)

    xx, yy = np.meshgrid(grid.x_axis(), grid.y_axis())
    n_cells = xx.size
    gx = xx.ravel().astype(np.float64)
    gy = yy.ravel().astype(np.float64)
    if elevation_grid is None:
        ga = np.zeros(n_cells, dtype=np.float64)
    else:
        ga = np.asarray(elevation_grid, dtype=float).ravel()
        if ga.size != n_cells:
            raise ValueError(
                f"elevation_grid size {ga.size} does not match grid cells {n_cells}"
            )

    out = np.empty(n_cells, dtype=np.float32)
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        dx = gx[start:end, None] - sx[None, :]
        dy = gy[start:end, None] - sy[None, :]
        dz = (ga[start:end, None] - sa[None, :]) * elev_weight
        dist2 = np.maximum(dx * dx + dy * dy + dz * dz, 1e6)
        w = 1.0 / dist2
        w = w / w.sum(axis=1, keepdims=True)
        out[start:end] = (w * sr[None, :]).sum(axis=1).astype(np.float32)

    return out.reshape(grid.shape)


def predict_grid(
    model: nn.Module,
    X_scaler: StandardScaler,
    V_scaler: StandardScaler,
    feature_spec: FeatureSpec,
    svd: ZScoreSVD,
    *,
    device: str = "cpu",
    batch_size: int = 50_000,
    altitude_source: str | None = None,
) -> NDArray[np.float32]:
    """Run the TerrainGNN's MLP head on every cell of the target grid.

    Parameters
    ----------
    model
        Trained :class:`TerrainGNN` (needs ``.forward_mlp``).
    X_scaler, V_scaler
        Scalers fitted during training.
    feature_spec
        Describes which features to sample on the grid; MUST match
        the feature stack used at training time.
    svd
        The ZScoreSVD used at training time.  Supplies the per-EOF
        station variance for the correction step.
    device
        Torch device string.
    batch_size
        Cells per GPU/CPU batch — keeps peak memory bounded.
    altitude_source
        Optional override for the DEM GeoTIFF used by the
        ``alt_norm`` feature; see
        :func:`austrianwindatlas.stage1.enhanced_mae.features.sample_features_on_grid`.

    Returns
    -------
    ``V_grid``: ``(n_eofs, ny, nx)``, dtype float32, variance-corrected.
    """
    import torch

    X_grid = sample_features_on_grid(feature_spec, altitude_source=altitude_source)
    if X_grid.shape[1] != X_scaler.n_features_in_:
        raise ValueError(
            f"feature stack width {X_grid.shape[1]} does not match scaler "
            f"n_features_in_ {X_scaler.n_features_in_}; feature_spec must match training."
        )
    X_scaled = X_scaler.transform(X_grid)

    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X_scaled.shape[0], batch_size):
            end = min(start + batch_size, X_scaled.shape[0])
            batch = torch.as_tensor(X_scaled[start:end], dtype=torch.float32, device=device)
            chunks.append(model.forward_mlp(batch).cpu().numpy())
    V_flat = np.vstack(chunks)                                  # (n_cells, n_eofs)
    V_flat = V_scaler.inverse_transform(V_flat)
    V_flat = variance_correct(V_flat, svd.V)

    grid = feature_spec.grid
    n_eofs = V_flat.shape[1]
    V_grid = V_flat.reshape(grid.nrows, grid.ncols, n_eofs).transpose(2, 0, 1)
    return V_grid.astype(np.float32)
