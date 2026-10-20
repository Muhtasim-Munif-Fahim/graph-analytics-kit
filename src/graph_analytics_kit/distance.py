"""All-pairs shortest paths, negative weights, and graph distance measures.

* :func:`bellman_ford` -- single-source shortest paths that allow negative
  edge weights (directed graphs), with negative-cycle detection that reports
  an actual cycle.
* :func:`floyd_warshall` -- dense all-pairs shortest-path matrix (``O(n^3)``,
  vectorised one pivot at a time) with successor pointers for path recovery
  via :func:`floyd_warshall_path`.
* Distance measures built on the all-pairs matrix: :func:`eccentricity`,
  :func:`diameter`, :func:`radius`, :func:`center`, :func:`periphery`,
  :func:`average_shortest_path_length`, :func:`wiener_index` and
  :func:`global_efficiency` (Latora & Marchiori, PRL 2001), plus
  :func:`distance_summary` for reports.

Every function takes ``weighted``: ``False`` (default) counts hops, ``True``
uses edge weights. Eccentricity-based measures need a (strongly) connected
graph and raise ``ValueError`` otherwise; global efficiency is well defined
on disconnected graphs because unreachable pairs contribute ``1/inf = 0``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .graph import Graph

__all__ = [
    "NegativeCycleError",
    "bellman_ford",
    "floyd_warshall",
    "floyd_warshall_path",
    "eccentricity",
    "diameter",
    "radius",
    "center",
    "periphery",
    "average_shortest_path_length",
    "wiener_index",
    "global_efficiency",
    "distance_summary",
]


class NegativeCycleError(ValueError):
    """Raised when a negative-weight cycle is reachable; ``cycle`` lists its nodes."""

    def __init__(self, cycle: List[int]):
        self.cycle = list(cycle)
        super().__init__(f"graph contains a negative-weight cycle: {' -> '.join(map(str, cycle))}")


def _weight(g: Graph, u: int, v: int, weighted: bool) -> float:
    return float(g.get_weight(u, v)) if weighted else 1.0


def bellman_ford(
    g: Graph, source: int, *, weighted: bool = True
) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """Shortest-path distances from ``source`` allowing negative edge weights.

    Returns ``(distances, predecessors)`` in the same format as
    :func:`graph_analytics_kit.dijkstra_shortest_path`. On an undirected graph
    any negative edge is itself a negative cycle (``u -> v -> u``). Raises
    :class:`NegativeCycleError` if a negative cycle is reachable from
    ``source``.
    """

    if source not in g.nodes:
        raise KeyError(f"{source} is not a node in the graph")
    dist: Dict[int, float] = {source: 0.0}
    prev: Dict[int, Optional[int]] = {source: None}
    edges = []
    for u in g.nodes:
        for v in g.neighbors(u):
            edges.append((u, v, _weight(g, u, v, weighted)))
    n = g.number_of_nodes()
    updated = None
    for _ in range(n):
        updated = None
        for u, v, w in edges:
            if u in dist and (v not in dist or dist[u] + w < dist[v]):
                dist[v] = dist[u] + w
                prev[v] = u
                updated = v
        if updated is None:
            return dist, prev
    # still relaxing after n rounds: walk back n steps to land on the cycle
    node = updated
    for _ in range(n):
        node = prev[node]
    cycle = [node]
    cur = prev[node]
    while cur != node:
        cycle.append(cur)
        cur = prev[cur]
    cycle.append(node)
    cycle.reverse()
    raise NegativeCycleError(cycle)


def floyd_warshall(g: Graph, *, weighted: bool = False) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """All-pairs shortest-path lengths.

    Returns ``(nodes, D, nxt)``: ``D[i, j]`` is the distance from ``nodes[i]``
    to ``nodes[j]`` (``inf`` if unreachable) and ``nxt[i, j]`` is the index of
    the next node on a shortest path (``-1`` if none). Raises
    :class:`NegativeCycleError` if any ``D[i, i] < 0``.
    """

    nodes = g.nodes
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    D = np.full((n, n), np.inf)
    nxt = np.full((n, n), -1, dtype=int)
    np.fill_diagonal(D, 0.0)
    for i in range(n):
        nxt[i, i] = i
    for u in nodes:
        i = index[u]
        for v in g.neighbors(u):
            j = index[v]
            w = _weight(g, u, v, weighted)
            if i == j:
                if w < 0:
                    raise NegativeCycleError([u, u])
                continue
            if w < D[i, j]:
                D[i, j] = w
                nxt[i, j] = j
    for k in range(n):
        via = D[:, k : k + 1] + D[k : k + 1, :]
        better = via < D
        if better.any():
            D = np.where(better, via, D)
            nxt = np.where(better, nxt[:, k : k + 1], nxt)
    negative = np.flatnonzero(np.diag(D) < 0)
    if negative.size:
        start = int(negative[0])
        cycle = [start]
        cur = int(nxt[start, start])
        seen = {start}
        while cur not in seen and cur != -1:
            cycle.append(cur)
            seen.add(cur)
            cur = int(nxt[cur, start])
        cycle.append(start)
        raise NegativeCycleError([nodes[i] for i in cycle])
    return nodes, D, nxt


def floyd_warshall_path(nodes: List[int], nxt: np.ndarray, source: int, target: int) -> List[int]:
    """Rebuild a shortest path from the ``nxt`` matrix of :func:`floyd_warshall`."""

    index = {node: i for i, node in enumerate(nodes)}
    if source not in index or target not in index:
        raise KeyError("source and target must be nodes of the graph")
    i, j = index[source], index[target]
    if nxt[i, j] == -1:
        raise KeyError(f"{target} is not reachable from {source}")
    path = [source]
    steps = 0
    while i != j:
        i = int(nxt[i, j])
        path.append(nodes[i])
        steps += 1
        if steps > len(nodes):
            raise RuntimeError("successor matrix is inconsistent")
    return path


def _connected_matrix(g: Graph, weighted: bool) -> Tuple[List[int], np.ndarray]:
    if g.number_of_nodes() == 0:
        raise ValueError("graph has no nodes")
    nodes, D, _ = floyd_warshall(g, weighted=weighted)
    if not np.all(np.isfinite(D)):
        kind = "strongly connected" if g.directed else "connected"
        raise ValueError(f"graph is not {kind}; eccentricity-based measures are infinite")
    return nodes, D


def eccentricity(g: Graph, *, weighted: bool = False) -> Dict[int, float]:
    """``ecc(v) = max_u d(v, u)`` for every node (out-distances on digraphs)."""

    nodes, D = _connected_matrix(g, weighted)
    return {node: float(D[i].max()) for i, node in enumerate(nodes)}


def diameter(g: Graph, *, weighted: bool = False) -> float:
    """Largest eccentricity."""

    return max(eccentricity(g, weighted=weighted).values())


def radius(g: Graph, *, weighted: bool = False) -> float:
    """Smallest eccentricity."""

    return min(eccentricity(g, weighted=weighted).values())


def center(g: Graph, *, weighted: bool = False, tol: float = 1e-12) -> List[int]:
    """Nodes whose eccentricity equals the radius."""

    ecc = eccentricity(g, weighted=weighted)
    r = min(ecc.values())
    return [node for node, e in ecc.items() if e <= r + tol]


def periphery(g: Graph, *, weighted: bool = False, tol: float = 1e-12) -> List[int]:
    """Nodes whose eccentricity equals the diameter."""

    ecc = eccentricity(g, weighted=weighted)
    d = max(ecc.values())
    return [node for node, e in ecc.items() if e >= d - tol]


def average_shortest_path_length(g: Graph, *, weighted: bool = False) -> float:
    """Mean of ``d(u, v)`` over ordered pairs ``u != v`` of a connected graph."""

    nodes, D = _connected_matrix(g, weighted)
    n = len(nodes)
    if n < 2:
        return 0.0
    return float((D.sum() - np.trace(D)) / (n * (n - 1)))


def wiener_index(g: Graph, *, weighted: bool = False) -> float:
    """Sum of distances over unordered pairs (ordered pairs for digraphs)."""

    _, D = _connected_matrix(g, weighted)
    total = float(D.sum())
    return total if g.directed else total / 2.0


def global_efficiency(g: Graph, *, weighted: bool = False) -> float:
    """``E = mean_{u != v} 1 / d(u, v)``; unreachable pairs contribute 0.

    Equals 1 for a complete graph and 0 for a graph without edges.
    """

    n = g.number_of_nodes()
    if n < 2:
        return 0.0
    _, D, _ = floyd_warshall(g, weighted=weighted)
    off = ~np.eye(n, dtype=bool)
    with np.errstate(divide="ignore"):
        inv = np.where(np.isfinite(D) & (D > 0), 1.0 / D, 0.0)
    return float(inv[off].sum() / (n * (n - 1)))


def distance_summary(g: Graph, *, weighted: bool = False) -> Dict[str, object]:
    """Diameter, radius, center, periphery, average path length, Wiener index, efficiency."""

    ecc = eccentricity(g, weighted=weighted)
    r = min(ecc.values())
    d = max(ecc.values())
    return {
        "weighted": weighted,
        "diameter": d,
        "radius": r,
        "center": [v for v, e in ecc.items() if e <= r + 1e-12],
        "periphery": [v for v, e in ecc.items() if e >= d - 1e-12],
        "average_shortest_path_length": average_shortest_path_length(g, weighted=weighted),
        "wiener_index": wiener_index(g, weighted=weighted),
        "global_efficiency": global_efficiency(g, weighted=weighted),
        "eccentricity": ecc,
    }
