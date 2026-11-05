"""TerrainGNN — GAT attention + MLP dual-path model for QN-GNN.

Two heads sharing the same ``(x_s)`` → ``V_s`` task:

* **GNN path**: graph-attention message passing with terrain-barrier
  distance bias.  Captures long-range dependencies between stations
  that share a terrain regime.
* **MLP path**: pure station-wise terrain regression.  Serves as a
  reliable fallback at grid cells that have no neighbours in the
  training station graph (i.e. essentially every grid cell at
  inference).

The two are blended at training time via ``mlp_weight`` so the model
learns to agree between them; at grid-inference time only the MLP
head is evaluated (because the inference graph would be the union of
grid + stations which doesn't exist as a training structure).

Architecture is faithful to the paper's ``TerrainGNNv6`` — the "v6"
designation drops here because this IS the canonical version shipped
in Schicker et al. (2026).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch
    import torch.nn as nn

__all__ = ["TerrainGNN", "build_terrain_gnn"]


def build_terrain_gnn(
    n_features: int,
    n_eofs: int,
    *,
    hidden_dim: int = 128,
    n_gnn_layers: int = 6,
    n_mlp_layers: int = 4,
    dropout: float = 0.20,
) -> nn.Module:
    """Instantiate a TerrainGNN with the paper's defaults.

    Parameters
    ----------
    n_features
        Terrain-feature count fed per station.
    n_eofs
        Output dimension (predicted EOF loadings per station).
    hidden_dim
        Width of the GNN hidden representation.  MLP uses
        ``2 * hidden_dim`` internally.
    n_gnn_layers
        GAT layer count.  6 in the paper.
    n_mlp_layers
        MLP depth (counting input + hidden blocks).  4 in the paper.
    dropout
        Applied after each ReLU.  0.20 in the paper.

    Returns
    -------
    An ``nn.Module`` with two public forwards:

    * ``forward_gnn(x, edge_index, edge_attr) -> V`` — graph path.
    * ``forward_mlp(x) -> V`` — station-wise path (used at grid inference).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TerrainGNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_features = n_features
            self.n_eofs = n_eofs
            self.hidden_dim = hidden_dim

            # ----- GNN path -------------------------------------------------
            self.gnn_input = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.gnn_layers = nn.ModuleList()
            for _ in range(n_gnn_layers):
                self.gnn_layers.append(nn.ModuleDict({
                    "conv": nn.Linear(hidden_dim, hidden_dim),
                    "msg": nn.Linear(hidden_dim, hidden_dim),
                    "attn": nn.Linear(2 * hidden_dim, 1, bias=False),
                    "norm": nn.LayerNorm(hidden_dim),
                    "drop": nn.Dropout(dropout),
                }))
            self.gnn_output = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_eofs),
            )

            # ----- MLP path -------------------------------------------------
            mlp_layers: list[nn.Module] = [
                nn.Linear(n_features, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            for _ in range(n_mlp_layers - 1):
                mlp_layers += [
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            mlp_layers.append(nn.Linear(hidden_dim * 2, n_eofs))
            self.mlp = nn.Sequential(*mlp_layers)

        def forward_gnn(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
        ) -> torch.Tensor:
            """GAT message passing.

            ``edge_attr`` is the normalised terrain-barrier distance in
            ``[0, 1]``; subtracted from the raw attention score so
            close / low-barrier neighbours win the softmax when the
            dot-product signal is weak.
            """
            h = self.gnn_input(x)
            source, target = edge_index[0], edge_index[1]
            n_nodes = x.size(0)

            dist_bias = edge_attr.squeeze(-1) if edge_attr.dim() == 2 else edge_attr
            scale = math.sqrt(self.hidden_dim)

            for layer in self.gnn_layers:
                h_self = layer["conv"](h)                      # (N, d)

                h_src = h[source]                              # (E, d)
                h_tgt = h[target]                              # (E, d)
                attn_raw = layer["attn"](
                    torch.cat([h_src, h_tgt], dim=-1)
                ).squeeze(-1)                                   # (E,)
                attn_raw = attn_raw / scale - dist_bias

                # Per-destination softmax with scatter.
                attn_max = torch.full((n_nodes,), float("-inf"), device=x.device)
                attn_max.scatter_reduce_(
                    0, target, attn_raw, reduce="amax", include_self=True,
                )
                attn_exp = torch.exp(
                    (attn_raw - attn_max[target]).clamp(min=-30)
                )                                               # (E,)
                attn_sum = torch.zeros(n_nodes, device=x.device)
                attn_sum.index_add_(0, target, attn_exp)
                attn_w = attn_exp / (attn_sum[target] + 1e-10)  # (E,)

                messages = layer["msg"](h_src) * attn_w.unsqueeze(-1)
                aggr = torch.zeros_like(h_self)
                aggr.index_add_(0, target, messages)

                h = h + layer["drop"](F.relu(layer["norm"](h_self + aggr)))

            return self.gnn_output(h)

        def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
            return self.mlp(x)

    return TerrainGNN()


class TerrainGNN:  # pragma: no cover — thin constructor facade
    """Public constructor for :func:`build_terrain_gnn`.

    ``TerrainGNN(...)`` is identical to ``build_terrain_gnn(...)``;
    keeps the module-summary type name readable.
    """

    def __new__(
        cls,
        n_features: int,
        n_eofs: int,
        *,
        hidden_dim: int = 128,
        n_gnn_layers: int = 6,
        n_mlp_layers: int = 4,
        dropout: float = 0.20,
    ):  # type: ignore[override]
        return build_terrain_gnn(
            n_features, n_eofs,
            hidden_dim=hidden_dim,
            n_gnn_layers=n_gnn_layers,
            n_mlp_layers=n_mlp_layers,
            dropout=dropout,
        )
