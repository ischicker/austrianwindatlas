"""Terrain-aware station graph for the QN-GNN.

Each station is a graph node.  Edges carry an *effective* distance:

.. math::

    d_\\mathrm{eff}(i, j) = \\|p_i - p_j\\|_2
                           + \\alpha \\cdot \\max(0, h_\\mathrm{max}(i, j) - \\max(h_i, h_j))

where :math:`h_\\mathrm{max}(i, j)` is the maximum elevation sampled
at a handful of waypoints along the straight line between :math:`i`
and :math:`j`.  The second term is a *ridge penalty* — two stations
separated by a mountain ridge get a larger effective distance even if
their Euclidean separation is small, so the GAT layer discounts
messages passing across the ridge.

The implementation is faithful to the paper's ``build_station_graph``
(``dl_qn_gnn.py``) but cleaned up: a single pure function, no
logging side effects inside the core numerical loop, no reliance on
mutable module state.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = ["StationGraph", "build_station_graph"]

logger = logging.getLogger(__name__)

#: Waypoint fractions along each edge at which the ridge-max elevation is
#: sampled.  Matches the paper exactly (five interior samples).
_WAYPOINTS: tuple[float, ...] = (0.25, 0.4, 0.5, 0.6, 0.75)


@dataclass(frozen=True)
class StationGraph:
    """Edges of the station graph in COO form.

    Attributes
    ----------
    edge_index
        ``(2, n_edges)`` array of ``(source, target)`` node indices.
    edge_attr
        ``(n_edges,)`` float array of effective distances.  Normalise
        (e.g. divide by max) at the call site if the trainer expects
        values in ``[0, 1]``.
    n_nodes
        Number of nodes (``len(coords)``).
    """

    edge_index: NDArray[np.int64]
    edge_attr: NDArray[np.float32]
    n_nodes: int

    @property
    def n_edges(self) -> int:
        return int(self.edge_attr.size)

    @property
    def avg_degree(self) -> float:
        if self.n_nodes == 0:
            return 0.0
        return self.n_edges / self.n_nodes


def build_station_graph(
    coords: NDArray[np.floating],
    *,
    max_distance: float = 100_000.0,
    k_nearest: int = 8,
    elevations: NDArray[np.floating] | None = None,
    elev_barrier_weight: float = 0.002,
) -> StationGraph:
    """Build a terrain-aware station graph.

    Parameters
    ----------
    coords
        ``(n_stations, 2)`` projected coordinates ``(x, y)`` in metres.
    max_distance
        Metres; any neighbour closer than this gets an edge in addition
        to the ``k_nearest`` closest.  Paper default: 100 km.
    k_nearest
        Every node also gets edges to its ``k_nearest`` closest
        neighbours.  Paper default: 8.
    elevations
        Optional ``(n_stations,)`` elevation array.  When given,
        effective distance includes the ridge-height penalty with
        weight ``elev_barrier_weight``.  Without it, effective
        distance is plain Euclidean.
    elev_barrier_weight
        Metres-per-metre penalty applied to ``max(0, ridge - endpoint_max)``.
        Paper default: 0.002.

    Returns
    -------
    StationGraph
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2); got shape {coords.shape!r}")
    n = coords.shape[0]
    if n == 0:
        return StationGraph(
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_attr=np.zeros(0, dtype=np.float32),
            n_nodes=0,
        )

    if elevations is not None:
        elevations = np.asarray(elevations, dtype=float)
        if elevations.shape != (n,):
            raise ValueError(
                f"elevations must have shape ({n},); got {elevations.shape!r}"
            )

    # Full pair-distance matrix — O(n^2) memory, fine for the ~280
    # TAWES stations the paper uses.
    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.linalg.norm(diffs, axis=2)

    edges: list[tuple[int, int]] = []
    eff_dists: list[float] = []
    for i in range(n):
        dists_i = distances[i]
        order = np.argsort(dists_i)
        knn = order[1 : k_nearest + 1]                         # skip self (dist 0)
        within = np.where((dists_i > 0) & (dists_i < max_distance))[0]
        neighbours = np.unique(np.concatenate([knn, within]))

        for j in neighbours:
            if i == j:
                continue
            eucl = dists_i[j]
            if elevations is not None and eucl > 0:
                endpoint_max = max(elevations[i], elevations[j])
                max_along = endpoint_max
                for t in _WAYPOINTS:
                    mid = coords[i] + t * (coords[j] - coords[i])
                    # Snap to the nearest station — the paper uses this
                    # proxy for a DEM lookup because it already has the
                    # station elevations in hand.  Not exact but monotone.
                    idx_nearest = int(np.argmin(np.linalg.norm(coords - mid, axis=1)))
                    max_along = max(max_along, float(elevations[idx_nearest]))
                ridge_height = max(0.0, max_along - endpoint_max)
                eff = float(eucl) + elev_barrier_weight * ridge_height
            else:
                eff = float(eucl)
            edges.append((i, int(j)))
            eff_dists.append(eff)

    if not edges:
        return StationGraph(
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_attr=np.zeros(0, dtype=np.float32),
            n_nodes=n,
        )

    edge_index = np.asarray(edges, dtype=np.int64).T              # (2, n_edges)
    edge_attr = np.asarray(eff_dists, dtype=np.float32)
    return StationGraph(edge_index=edge_index, edge_attr=edge_attr, n_nodes=n)
