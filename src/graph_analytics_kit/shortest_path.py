"""Shortest-path algorithms for weighted graphs."""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

from .graph import Graph


def dijkstra_shortest_path(
    g: Graph,
    source: int,
    target: Optional[int] = None,
) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """Compute shortest-path distances from *source* using Dijkstra's algorithm.

    Works on both directed and undirected graphs whose edge weights are
    non-negative (the class default weight is 1.0). When *target* is given,
    the search terminates as soon as the shortest path to *target* is found.

    Returns a tuple ``(distances, predecessors)`` where ``distances`` maps each
    reachable node to its shortest-path distance from *source* and
    ``predecessors`` maps each node to its predecessor on a shortest path
    (or ``None`` for the source itself). Unreachable nodes are absent from
    ``distances``.
    """
    dist: Dict[int, float] = {source: 0.0}
    prev: Dict[int, Optional[int]] = {source: None}
    visited: set[int] = set()
    queue: List[Tuple[float, int]] = [(0.0, source)]

    while queue:
        d, u = heapq.heappop(queue)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for v in g.neighbors(u):
            if v in visited:
                continue
            weight = g._adj[u][v]
            alt = d + weight
            if v not in dist or alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(queue, (alt, v))

    return dist, prev


def reconstruct_path(
    predecessors: Dict[int, Optional[int]],
    source: int,
    target: int,
) -> List[int]:
    """Reconstruct a shortest path from predecessor mapping.

    Given the ``predecessors`` dict returned by :func:`dijkstra_shortest_path`,
    rebuild the path from *source* to *target* as a list of node labels.
    Raises ``KeyError`` if *target* is unreachable from *source*.
    """
    if target not in predecessors:
        raise KeyError(f"{target} is not reachable from {source}")
    path: List[int] = []
    node: Optional[int] = target
    while node is not None:
        path.append(node)
        node = predecessors[node]
    path.reverse()
    if path[0] != source:
        raise KeyError(f"{target} is not reachable from {source}")
    return path


__all__ = ["dijkstra_shortest_path", "reconstruct_path"]
