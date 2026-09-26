"""Centrality measures for graph nodes."""
from __future__ import annotations

from collections import deque
from typing import Dict

import numpy as np

from .eigenvector import eigenvector_centrality
from .graph import Graph
from .hits import hits
from .katz import katz_centrality
from .kcore import core_number
from .pagerank import pagerank


def degree_centrality(g: Graph) -> Dict[int, int]:
    """Return the degree (number of neighbors) for each node."""
    return {node: g.degree(node) for node in g.nodes}


def closeness_centrality(g: Graph) -> Dict[int, float]:
    """Return the closeness centrality for each node.

    Closeness is defined as ``(n - 1) / sum(distances)`` where the sum is over
    shortest-path distances to all other reachable nodes. Nodes that are
    isolated (no path to any other node) receive a score of 0.0.
    """
    n = g.number_of_nodes()
    if n <= 1:
        return {node: 0.0 for node in g.nodes}
    result: Dict[int, float] = {}
    for source in g.nodes:
        dist = _bfs_distances(g, source)
        reachable = sum(d for d in dist.values() if d > 0)
        if reachable == 0:
            result[source] = 0.0
        else:
            result[source] = (n - 1) / reachable
    return result



def harmonic_centrality(g: Graph) -> Dict[int, float]:
    """Return the harmonic centrality for each node.

    Harmonic centrality is the sum of reciprocal shortest-path distances
    ``sum_{v != u} 1 / d(u, v)`` over nodes reachable from ``u``. Isolated
    nodes (and nodes that reach nobody) receive a score of ``0.0``. Unlike
    closeness, harmonic centrality stays well-defined on disconnected
    graphs because unreachable pairs contribute nothing rather than
    forcing the whole sum to zero.
    """
    n = g.number_of_nodes()
    if n <= 1:
        return {node: 0.0 for node in g.nodes}
    result: Dict[int, float] = {}
    for source in g.nodes:
        dist = _bfs_distances(g, source)
        total = 0.0
        for d in dist.values():
            if d > 0:
                total += 1.0 / d
        result[source] = total
    return result


def betweenness_centrality(g: Graph, normalized: bool = True) -> Dict[int, float]:
    """Return the betweenness centrality for each node using Brandes' algorithm.

    Betweenness counts the fraction of shortest paths between all pairs of
    nodes that pass through a given node. When ``normalized`` is ``True`` the
    scores are scaled to the range ``[0, 1]``.
    """
    nodes = g.nodes
    n = len(nodes)
    if n < 3:
        return {node: 0.0 for node in nodes}

    betweenness: Dict[int, float] = {node: 0.0 for node in nodes}

    for source in nodes:
        S: list[int] = []
        P: Dict[int, list[int]] = {node: [] for node in nodes}
        sigma: Dict[int, int] = {node: 0 for node in nodes}
        sigma[source] = 1
        dist: Dict[int, int] = {node: -1 for node in nodes}
        dist[source] = 0
        delta: Dict[int, float] = {node: 0.0 for node in nodes}

        queue: deque[int] = deque([source])
        while queue:
            v = queue.popleft()
            S.append(v)
            for w in g.neighbors(v):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != source:
                betweenness[w] += delta[w]

    if not normalized:
        scale = 0.5 if not g.directed else 1.0
    elif n <= 2:
        scale = 1.0
    else:
        scale = 1.0 / ((n - 1) * (n - 2))

    return {node: betweenness[node] * scale for node in nodes}


def _bfs_distances(g: Graph, source: int) -> Dict[int, int]:
    """Single-source BFS returning distance from *source* to every reachable node.

    Unreachable nodes simply do not appear in the returned dict.
    """
    dist: Dict[int, int] = {source: 0}
    queue: deque[int] = deque([source])
    while queue:
        v = queue.popleft()
        for w in g.neighbors(v):
            if w not in dist:
                dist[w] = dist[v] + 1
                queue.append(w)
    return dist


def degree_centrality_array(g: Graph) -> np.ndarray:
    """Return degree centrality as a NumPy array ordered by ``g.nodes``."""
    return np.array([g.degree(node) for node in g.nodes], dtype=float)


__all__ = [
    "degree_centrality",
    "degree_centrality_array",
    "closeness_centrality",
    "harmonic_centrality",
    "betweenness_centrality",
    "pagerank",
    "eigenvector_centrality",
    "katz_centrality",
    "hits",
    "core_number",
]
