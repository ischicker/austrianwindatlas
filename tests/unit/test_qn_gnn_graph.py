"""Unit tests for :mod:`austrianwindatlas.stage1.qn_gnn.graph`."""
from __future__ import annotations

import numpy as np
import pytest

from austrianwindatlas.stage1.qn_gnn.graph import StationGraph, build_station_graph


def test_empty_input_returns_empty_graph():
    g = build_station_graph(np.zeros((0, 2)))
    assert isinstance(g, StationGraph)
    assert g.n_nodes == 0
    assert g.n_edges == 0
    assert g.avg_degree == 0.0


def test_rejects_wrong_coord_shape():
    with pytest.raises(ValueError, match=r"coords must be \(n, 2\)"):
        build_station_graph(np.zeros((3, 3)))


def test_edge_count_dense_small_graph():
    # 4 colinear stations; with k_nearest=8 (larger than n), every node
    # connects to every other.
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    g = build_station_graph(coords, max_distance=1.0, k_nearest=10)
    # Directed graph: n*(n-1) edges.
    assert g.n_edges == 12
    assert g.n_nodes == 4


def test_edge_attr_is_euclidean_without_elevations():
    coords = np.array([[0.0, 0.0], [3.0, 4.0]])  # dist = 5
    g = build_station_graph(coords, max_distance=1.0, k_nearest=2)
    # Two directed edges with distance 5.
    assert g.n_edges == 2
    np.testing.assert_allclose(g.edge_attr, [5.0, 5.0], rtol=0, atol=1e-6)


def test_ridge_penalty_increases_effective_distance():
    # Three stations in a line; the middle one is a "mountain" between
    # the two low outer stations.  Edge from 0 → 2 should carry a
    # ridge penalty; edges 0 → 1 and 1 → 2 should not.
    coords = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
    elevations = np.array([200.0, 1_500.0, 200.0])
    g = build_station_graph(coords, max_distance=1.0, k_nearest=3,
                            elevations=elevations, elev_barrier_weight=1.0)
    attr_dict = {(int(g.edge_index[0, i]), int(g.edge_index[1, i])): float(g.edge_attr[i])
                 for i in range(g.n_edges)}
    # No ridge: endpoint_max >= waypoint_max.
    assert attr_dict[(0, 1)] == pytest.approx(50.0, abs=1e-6)
    assert attr_dict[(1, 2)] == pytest.approx(50.0, abs=1e-6)
    # Ridge across (0, 2): snap-to-nearest waypoints land on station 1
    # (elevation 1500), ridge = 1500 - 200 = 1300.
    # eff = 100 + 1.0 * 1300 = 1400.
    assert attr_dict[(0, 2)] == pytest.approx(1400.0, abs=1e-6)


def test_elevation_shape_mismatch_raises():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="elevations must have shape"):
        build_station_graph(coords, elevations=np.zeros(3))


def test_k_nearest_includes_knn_even_beyond_max_distance():
    # With max_distance tiny, only k_nearest neighbours should still
    # make it into the graph.
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 1_000_000, size=(10, 2))
    g = build_station_graph(coords, max_distance=1.0, k_nearest=3)
    # Each node gets at least its 3 nearest neighbours.
    degrees = np.bincount(g.edge_index[0], minlength=10)
    assert (degrees >= 3).all()


def test_graph_avg_degree_property():
    coords = np.array([[0.0, 0.0], [10.0, 0.0]])
    g = build_station_graph(coords, max_distance=100.0, k_nearest=1)
    # 2 nodes, 2 directed edges → avg degree 1.0.
    assert g.avg_degree == 1.0
