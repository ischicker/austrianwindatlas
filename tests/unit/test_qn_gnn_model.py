"""Unit tests for :mod:`austrianwindatlas.stage1.qn_gnn.model`."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from austrianwindatlas.stage1.qn_gnn.model import TerrainGNN, build_terrain_gnn  # noqa: E402


def test_builder_returns_nn_module():
    import torch.nn as nn
    model = build_terrain_gnn(
        n_features=4, n_eofs=3, hidden_dim=16,
        n_gnn_layers=2, n_mlp_layers=2, dropout=0.1,
    )
    assert isinstance(model, nn.Module)
    # Two forwards present.
    assert hasattr(model, "forward_gnn")
    assert hasattr(model, "forward_mlp")


def test_forward_mlp_shape():
    model = build_terrain_gnn(n_features=5, n_eofs=7, hidden_dim=16,
                              n_gnn_layers=1, n_mlp_layers=2)
    model.eval()
    x = torch.randn(10, 5)
    y = model.forward_mlp(x)
    assert y.shape == (10, 7)


def test_forward_gnn_shape():
    model = build_terrain_gnn(n_features=5, n_eofs=7, hidden_dim=16,
                              n_gnn_layers=1, n_mlp_layers=2)
    model.eval()
    x = torch.randn(4, 5)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.tensor([0.1, 0.2, 0.3, 0.4])
    y = model.forward_gnn(x, edge_index, edge_attr)
    assert y.shape == (4, 7)


def test_forward_gnn_accepts_2d_edge_attr():
    model = build_terrain_gnn(n_features=5, n_eofs=3, hidden_dim=16,
                              n_gnn_layers=1, n_mlp_layers=2)
    model.eval()
    x = torch.randn(3, 5)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.tensor([[0.1], [0.2]])  # shape (E, 1)
    y = model.forward_gnn(x, edge_index, edge_attr)
    assert y.shape == (3, 3)


def test_terraingnn_class_facade():
    model = TerrainGNN(n_features=4, n_eofs=2, hidden_dim=8,
                       n_gnn_layers=1, n_mlp_layers=1)
    model.eval()
    assert hasattr(model, "forward_mlp")


def test_gnn_layer_count_influences_parameter_count():
    small = build_terrain_gnn(n_features=4, n_eofs=2, hidden_dim=16, n_gnn_layers=1)
    large = build_terrain_gnn(n_features=4, n_eofs=2, hidden_dim=16, n_gnn_layers=6)
    small_n = sum(p.numel() for p in small.parameters())
    large_n = sum(p.numel() for p in large.parameters())
    assert large_n > small_n
