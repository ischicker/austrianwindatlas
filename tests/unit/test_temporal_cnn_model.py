"""Unit tests for :mod:`austrianwindatlas.stage1.qn_gnn.temporal_cnn.model`."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from austrianwindatlas.stage1.qn_gnn.temporal_cnn.model import (  # noqa: E402
    TemporalCNN,
    build_temporal_cnn,
)


def test_build_returns_nn_module():
    import torch.nn as nn
    m = build_temporal_cnn(n_stations=20, n_eofs=5, window=16, proj_dim=16)
    assert isinstance(m, nn.Module)


def test_forward_output_shape():
    m = build_temporal_cnn(n_stations=10, n_eofs=4, window=8, proj_dim=8)
    m.train()  # BatchNorm needs > 1 element even in eval → train is fine here too
    obs = torch.randn(3, 8, 10)
    avail = torch.ones(3, 8, 10)
    y = m(obs, avail)
    assert y.shape == (3, 4)


def test_head_starts_at_zero():
    """At epoch 0 the model must produce ΔU ≈ 0 (pipeline degrades to
    pure-GNN reconstruction when the CNN hasn't learned anything yet).
    """
    m = build_temporal_cnn(n_stations=12, n_eofs=6, window=8, proj_dim=16)
    m.eval()
    # Any obs / avail — the zero-initialised head gives zero output.
    obs = torch.randn(4, 8, 12)
    avail = torch.ones(4, 8, 12)
    with torch.no_grad():
        y = m(obs, avail)
    assert torch.allclose(y, torch.zeros_like(y))


def test_rejects_wrong_station_count():
    m = build_temporal_cnn(n_stations=15, n_eofs=4, window=8, proj_dim=16)
    obs = torch.randn(2, 8, 10)         # wrong n_stations
    avail = torch.ones_like(obs)
    with pytest.raises(ValueError, match="stations"):
        m(obs, avail)


def test_rejects_wrong_window():
    m = build_temporal_cnn(n_stations=10, n_eofs=4, window=16, proj_dim=16)
    obs = torch.randn(2, 8, 10)         # wrong window
    avail = torch.ones_like(obs)
    with pytest.raises(ValueError, match="window length"):
        m(obs, avail)


def test_rejects_shape_mismatch_between_obs_and_avail():
    m = build_temporal_cnn(n_stations=10, n_eofs=4, window=8, proj_dim=16)
    obs = torch.randn(2, 8, 10)
    avail = torch.ones(2, 8, 11)        # one too many stations
    with pytest.raises(ValueError, match="avail shape"):
        m(obs, avail)


def test_facade_class_constructs_same_module():
    import torch.nn as nn
    m = TemporalCNN(n_stations=8, n_eofs=3, window=8, proj_dim=16)
    assert isinstance(m, nn.Module)
