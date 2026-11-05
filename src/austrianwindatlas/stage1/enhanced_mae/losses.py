"""Loss functions and station-importance weights for Enhanced-MAE / Enhanced-WMSE.

Two deliverables:

* :func:`compute_station_weights` — pure-NumPy helper that turns a
  wind-data matrix into per-station weights.  Three schemes, all
  mapped to ``[0.5, 2.0]`` (for ``mean_wind`` / ``variance``) or
  ``{1.0, 2.0}`` (for ``quantile``).  Used by the weighted-MSE /
  weighted-MAE variants.
* :class:`LossFunction` — a configurable PyTorch loss over the
  predicted EOF coefficients.  Supports MSE, MAE, an asymmetric MAE
  (penalising under-estimation more), a focal-MAE, and the two
  weighted variants.  Imported lazily so the module loads without
  torch; :class:`LossFunction` itself requires torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from numpy.typing import NDArray

__all__ = [
    "LossKind",
    "WeightMethod",
    "LossFunction",
    "compute_station_weights",
]

#: Supported loss kinds (string literal type for public API docs).
LossKind = Literal["mse", "mae", "asymmetric", "focal", "weighted_mse", "weighted_mae"]

#: Supported station-weight methods.
WeightMethod = Literal["none", "mean_wind", "variance", "quantile"]


def compute_station_weights(
    wind_data: NDArray[np.floating],
    method: WeightMethod = "mean_wind",
) -> NDArray[np.float32]:
    """Return per-station weights in ``[0.5, 2.0]`` (or ``{1, 2}``) for
    weighted-loss training.

    Parameters
    ----------
    wind_data
        Array of shape ``(n_times, n_stations)``.  NaNs are ignored
        (``np.nanmean`` / ``np.nanstd``).
    method
        ``"mean_wind"``
            Linear rescale of each station's climatological mean into
            ``[0.5, 2.0]``.  Windier stations weigh more.
        ``"variance"``
            Same, but based on per-station standard deviation.
        ``"quantile"``
            Top quartile gets weight 2; everything else weight 1.
        ``"none"``
            All stations weight 1.

    Returns
    -------
    ndarray of shape ``(n_stations,)``, dtype ``float32``.

    Raises
    ------
    ValueError
        For an unknown ``method``.
    """
    arr = np.asarray(wind_data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"wind_data must be 2-D (time, station); got shape {arr.shape!r}")

    n_stations = arr.shape[1]

    if method == "mean_wind":
        stat = np.nanmean(arr, axis=0)
    elif method == "variance":
        stat = np.nanstd(arr, axis=0)
    elif method == "quantile":
        stat = np.nanmean(arr, axis=0)
        q75 = np.nanpercentile(stat, 75)
        return np.where(stat > q75, 2.0, 1.0).astype(np.float32)
    elif method == "none":
        return np.ones(n_stations, dtype=np.float32)
    else:
        raise ValueError(f"unknown weight method {method!r}")

    # Linear rescale to [0.5, 2.0]; +1e-8 guards the all-equal case.
    lo, hi = np.nanmin(stat), np.nanmax(stat)
    weights = 0.5 + 1.5 * (stat - lo) / (hi - lo + 1e-8)
    return weights.astype(np.float32)


@dataclass
class LossFunction:
    """Configurable loss over predicted EOF coefficients.

    Call the instance like a function::

        loss_fn = LossFunction("asymmetric", alpha=0.7)
        loss = loss_fn(pred, target)

    Supported ``loss_type``\\ s:

    ``"mse"``
        :math:`\\mathbb{E}\\|p-t\\|^2` — plain MSE.
    ``"mae"``
        :math:`\\mathbb{E}\\|p-t\\|_1` — plain MAE.
    ``"asymmetric"``
        Weighted absolute error with weight ``alpha`` on
        under-estimation (where ``pred < target``) and ``1 - alpha``
        on over-estimation.  ``alpha > 0.5`` penalises under-estimation
        more — the regime used in Enhanced-MAE.
    ``"focal"``
        :math:`(1 + \\mathrm{MAE})^\\gamma \\cdot \\mathrm{MAE}` —
        emphasises hard examples.
    ``"weighted_mse"`` / ``"weighted_mae"``
        Standard MSE / MAE multiplied by per-station weights supplied
        via ``station_weights`` (shape ``(n_stations,)``).  The call
        site must also pass the station indices for the current
        mini-batch via ``indices``.  Falls back to the unweighted
        version when either is missing.

    All variants return a scalar tensor suitable for ``loss.backward()``.
    """

    loss_type: LossKind = "mse"
    alpha: float = 0.7
    gamma: float = 2.0
    station_weights: torch.Tensor | None = None

    def __post_init__(self) -> None:
        kind = self.loss_type.lower()
        valid = {"mse", "mae", "asymmetric", "focal", "weighted_mse", "weighted_mae"}
        if kind not in valid:
            raise ValueError(f"unknown loss_type {self.loss_type!r}; expected one of {sorted(valid)}")
        self.loss_type = kind  # type: ignore[assignment]

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import torch
        import torch.nn.functional as F

        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        if self.loss_type == "mae":
            return F.l1_loss(pred, target)
        if self.loss_type == "asymmetric":
            diff = target - pred
            weights = torch.where(diff > 0, self.alpha, 1.0 - self.alpha)
            return (weights * diff.abs()).mean()
        if self.loss_type == "focal":
            mae = (pred - target).abs()
            focal = (1.0 + mae) ** self.gamma
            return (focal * mae).mean()
        if self.loss_type == "weighted_mse":
            if self.station_weights is None or indices is None:
                return F.mse_loss(pred, target)
            w = self.station_weights[indices].unsqueeze(1)
            return (w * (pred - target).pow(2)).mean()
        if self.loss_type == "weighted_mae":
            if self.station_weights is None or indices is None:
                return F.l1_loss(pred, target)
            w = self.station_weights[indices].unsqueeze(1)
            return (w * (pred - target).abs()).mean()
        raise AssertionError(f"unreachable: {self.loss_type!r}")  # pragma: no cover
