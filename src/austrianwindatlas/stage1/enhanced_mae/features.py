"""Terrain-feature extraction for the Enhanced-MAE DNN.

The DNN predicts the station-side EOF loadings ``V`` from terrain
covariates sampled at station locations.  At inference time the same
features are needed on the full target grid.

The feature stack (ordered, for stable DNN inputs):

* ``x_norm``, ``y_norm`` — projected-coordinate position in ``[0, 1]``
  on the target grid.
* ``alt_norm`` — altitude in km (``raw_m / 3000``).
* six raster covariates: ``elev``, ``tpi``, ``slope``, ``sin``, ``cos``,
  ``rough`` (CORINE 2018 roughness).  Available ones are added in a
  fixed order; missing ones are skipped (and the inference grid also
  skips them, so the DNN input dim stays consistent).

The module exposes two sampling entry points — one for stations, one
for the full grid — and shares a single :class:`FeatureSpec` so any
mismatch between training and inference is a type error, not a silent
wrong-features bug.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ...io.windfield import INCA_GRID, GridSpec

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from numpy.typing import NDArray

__all__ = [
    "DEFAULT_COVARIATE_FILES",
    "ALT_SCALE_METERS",
    "FeatureSpec",
    "build_feature_spec_for_covariates",
    "sample_features_at_stations",
    "sample_features_on_grid",
]

logger = logging.getLogger(__name__)

#: Scaling factor for altitude: raw metres divided by this → ``alt_norm``.
#: 3 km is a reasonable upper bound for station altitudes in Austria,
#: so the result stays in ``[0, 1]`` for the whole TAWES network.
ALT_SCALE_METERS: float = 3000.0

#: Mapping from feature name to GeoTIFF filename on disk.  Matches the
#: paper's ``DEM_DERIVATIVES/COVARIATES/1KM/`` directory.
DEFAULT_COVARIATE_FILES: dict[str, str] = {
    "elev": "EUDEM_AUSTRIA_1km.tif",
    "tpi": "TPI_2000m_1km.tif",
    "slope": "SLOPE_1km.tif",
    "sin": "ASPECT_SIN_1km.tif",
    "cos": "ASPECT_COS_1km.tif",
    "rough": "ROUGHNESS2018_REVISED_1km.tif",
}

#: Feature names that are *always* present (position + altitude), in
#: order.  Raster-covariate names follow and are added only if the
#: corresponding file was found on disk.
_BASE_FEATURE_NAMES: tuple[str, ...] = ("x_norm", "y_norm", "alt_norm")


@dataclass(frozen=True)
class FeatureSpec:
    """Declares which features are used, in what order, and where to
    find the raster covariates on disk.

    Attributes
    ----------
    names
        Feature names in the order they appear in the feature matrix.
        Shape of ``X`` is always ``(n_points, len(names))``.
    grid
        Target projected grid (defaults to :data:`~austrianwindatlas.io.windfield.INCA_GRID`).
    covariate_dir
        Directory containing the raster files named in
        :data:`DEFAULT_COVARIATE_FILES`.  :data:`None` means no raster
        covariates are used — the feature matrix will only contain
        the base features (``x_norm``, ``y_norm``, ``alt_norm``).
    covariate_files
        Per-feature filename overrides.  Keys that don't appear in
        :attr:`names` are ignored.
    """

    names: tuple[str, ...] = _BASE_FEATURE_NAMES
    grid: GridSpec = INCA_GRID
    covariate_dir: Path | None = None
    covariate_files: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_COVARIATE_FILES))

    @property
    def n_features(self) -> int:
        return len(self.names)

    def covariate_names(self) -> tuple[str, ...]:
        """Feature names that are read from raster files (i.e. not
        ``x_norm`` / ``y_norm`` / ``alt_norm``).
        """
        return tuple(n for n in self.names if n in self.covariate_files)


def build_feature_spec_for_covariates(
    covariate_dir: str | Path | None,
    *,
    grid: GridSpec = INCA_GRID,
    include_altitude: bool = True,
    extra_files: dict[str, str] | None = None,
) -> FeatureSpec:
    """Build a :class:`FeatureSpec` by probing which covariate files exist.

    Only raster files that are actually present on disk are added to
    ``names``, so a training run on a reduced covariate set produces a
    DNN whose inference-time feature probe is guaranteed to match.
    """
    files = dict(DEFAULT_COVARIATE_FILES)
    if extra_files:
        files.update(extra_files)

    base: list[str] = ["x_norm", "y_norm"]
    if include_altitude:
        base.append("alt_norm")

    if covariate_dir is None:
        return FeatureSpec(names=tuple(base), grid=grid, covariate_dir=None, covariate_files=files)

    root = Path(covariate_dir)
    extras: list[str] = []
    for name, filename in files.items():
        if (root / filename).is_file():
            extras.append(name)
        else:
            logger.debug("covariate file %s missing; skipping feature %r", filename, name)
    return FeatureSpec(
        names=tuple(base + extras),
        grid=grid,
        covariate_dir=root,
        covariate_files=files,
    )


def _sample_raster_at_xy(
    raster_path: Path,
    xs: NDArray[np.floating],
    ys: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return raster values at the given projected-coordinate points.

    NaNs come out for points outside the raster bounds or for
    ``nodata`` pixels.
    """
    import rasterio  # lazy import — rasterio is a heavy dep

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        transform = src.transform
        nodata = src.nodata

    n = len(xs)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        x, y = xs[i], ys[i]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        col_f, row_f = ~transform * (x, y)
        col, row = int(col_f), int(row_f)
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            val = data[row, col]
            if nodata is not None and val == nodata:
                continue
            out[i] = val
    return out


def _normalise_xy(
    xs: NDArray[np.floating],
    ys: NDArray[np.floating],
    grid: GridSpec,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Map projected ``(x, y)`` to ``[0, 1]`` using the grid extents."""
    x_norm = (xs - grid.xmin) / (grid.xmax - grid.xmin)
    y_norm = (ys - grid.ymin) / (grid.ymax - grid.ymin)
    return x_norm, y_norm


def _fill_feature_nans(
    X: NDArray[np.floating],
    names: tuple[str, ...],
) -> NDArray[np.floating]:
    """Column-wise NaN fill with the column mean (0 if the column is
    entirely NaN).  Emits a warning per column so it's visible in logs.
    """
    out = X.copy()
    for j, name in enumerate(names):
        mask = np.isnan(out[:, j])
        if not mask.any():
            continue
        finite = out[~mask, j]
        fill = float(finite.mean()) if finite.size else 0.0
        logger.warning(
            "feature %r has %d NaN values; filling with column mean %.3f",
            name, int(mask.sum()), fill,
        )
        out[mask, j] = fill
    return out


def sample_features_at_stations(
    station_meta: pd.DataFrame,
    spec: FeatureSpec,
    *,
    x_col: str = "x",
    y_col: str = "y",
    alt_col: str = "alt",
) -> NDArray[np.float32]:
    """Sample the feature stack at station locations.

    Parameters
    ----------
    station_meta
        DataFrame with at least ``x_col``, ``y_col`` columns (and
        ``alt_col`` if :attr:`FeatureSpec.names` contains ``"alt_norm"``).
    spec
        Describes which features to assemble and which grid to normalise
        against.
    x_col, y_col, alt_col
        Column names in ``station_meta`` — swap these to accommodate
        data frames that use ``lon``/``lat`` or ``station_x``/``station_y``.

    Returns
    -------
    ndarray of shape ``(n_stations, n_features)``, dtype ``float32``.
    NaNs are filled column-wise before return.
    """
    xs = np.asarray(station_meta[x_col].values, dtype=float)
    ys = np.asarray(station_meta[y_col].values, dtype=float)

    # Fill NaN coordinates with the column mean so normalisation doesn't
    # propagate NaN into the position features.
    for arr in (xs, ys):
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            arr[nan_mask] = np.nanmean(arr) if (~nan_mask).any() else 0.0

    cols: list[np.ndarray] = []
    x_norm, y_norm = _normalise_xy(xs, ys, spec.grid)

    for name in spec.names:
        if name == "x_norm":
            cols.append(x_norm)
        elif name == "y_norm":
            cols.append(y_norm)
        elif name == "alt_norm":
            if alt_col not in station_meta.columns:
                raise KeyError(f"station_meta has no {alt_col!r} column but spec requests 'alt_norm'")
            alt = np.asarray(station_meta[alt_col].values, dtype=float)
            alt = np.where(np.isnan(alt), np.nanmean(alt) if np.isfinite(np.nanmean(alt)) else 0.0, alt)
            cols.append(alt / ALT_SCALE_METERS)
        elif name in spec.covariate_files:
            if spec.covariate_dir is None:
                raise ValueError(f"spec lists feature {name!r} but covariate_dir is None")
            raster = spec.covariate_dir / spec.covariate_files[name]
            cols.append(_sample_raster_at_xy(raster, xs, ys))
        else:
            raise ValueError(f"unknown feature name {name!r}")

    X = np.column_stack(cols).astype(float)
    X = _fill_feature_nans(X, spec.names)
    return X.astype(np.float32)


def sample_features_on_grid(
    spec: FeatureSpec,
    *,
    altitude_source: Path | str | None = None,
) -> NDArray[np.float32]:
    """Sample the feature stack on every cell of the target grid.

    Parameters
    ----------
    spec
        Feature spec.  Must have a non-``None`` :attr:`FeatureSpec.covariate_dir`
        if ``"alt_norm"`` is in :attr:`FeatureSpec.names` and
        ``altitude_source`` is not given.
    altitude_source
        Optional path to the DEM GeoTIFF used for ``alt_norm``.
        Defaults to ``spec.covariate_dir / spec.covariate_files["elev"]``.

    Returns
    -------
    ndarray of shape ``(ny * nx, n_features)``, dtype ``float32``.
    Column order matches :attr:`FeatureSpec.names`.  The caller is
    responsible for reshaping the DNN output back to ``(ny, nx)``.
    """
    import rasterio  # lazy

    grid = spec.grid
    x_1d = grid.x_axis()
    y_1d = grid.y_axis()
    xx, yy = np.meshgrid(x_1d, y_1d)
    n_points = xx.size

    x_norm, y_norm = _normalise_xy(xx.ravel(), yy.ravel(), grid)

    cols: list[np.ndarray] = []
    for name in spec.names:
        if name == "x_norm":
            cols.append(x_norm)
            continue
        if name == "y_norm":
            cols.append(y_norm)
            continue
        if name == "alt_norm":
            dem = Path(altitude_source) if altitude_source is not None else (
                (spec.covariate_dir / spec.covariate_files["elev"])
                if spec.covariate_dir is not None else None
            )
            if dem is None or not dem.is_file():
                raise FileNotFoundError(f"no DEM GeoTIFF for alt_norm (looked at {dem!r})")
            with rasterio.open(dem) as src:
                elev = src.read(1).astype(float)
                nodata = src.nodata
            # Paper flips vertically because the DEM is written
            # top-to-bottom while our y_axis goes bottom-to-top.
            elev = np.flipud(elev)
            if nodata is not None:
                elev = np.where(elev == nodata, np.nan, elev)
            if elev.shape != (grid.nrows, grid.ncols):
                raise ValueError(
                    f"DEM shape {elev.shape!r} does not match grid shape "
                    f"{(grid.nrows, grid.ncols)!r}"
                )
            cols.append((elev / ALT_SCALE_METERS).ravel())
            continue
        if name in spec.covariate_files:
            if spec.covariate_dir is None:
                raise ValueError(f"spec lists feature {name!r} but covariate_dir is None")
            path = spec.covariate_dir / spec.covariate_files[name]
            with rasterio.open(path) as src:
                data = src.read(1).astype(float)
                nodata = src.nodata
            data = np.flipud(data)
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            if data.shape != (grid.nrows, grid.ncols):
                raise ValueError(
                    f"covariate {name!r} shape {data.shape!r} does not match grid "
                    f"shape {(grid.nrows, grid.ncols)!r}"
                )
            cols.append(data.ravel())
            continue
        raise ValueError(f"unknown feature name {name!r}")

    X = np.column_stack(cols)
    assert X.shape == (n_points, spec.n_features)
    X = _fill_feature_nans(X, spec.names)
    return X.astype(np.float32)
