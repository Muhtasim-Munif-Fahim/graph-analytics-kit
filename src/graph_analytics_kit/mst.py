"""Minimum-spanning-tree algorithms for weighted graphs."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .graph import Graph, WeightedEdge


def _find(parent: Dict[int, int], node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _union(parent: Dict[int, int], rank: Dict[int, int], a: int, b: int) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


def kruskal_mst(g: Graph) -> List[WeightedEdge]:
    """Return the minimum spanning tree of *g* using Kruskal's algorithm.

    Edges are processed in ascending weight order. A union-find structure
    rejects any edge that would create a cycle. The returned list contains
    ``n - 1`` edges for a connected graph (or fewer if the graph is
    disconnected, in which case a minimum spanning forest is returned).
    """
    nodes = g.nodes
    parent: Dict[int, int] = {node: node for node in nodes}
    rank: Dict[int, int] = {node: 0 for node in nodes}
    edges = sorted(g.edges, key=lambda e: e[2])
    mst: List[WeightedEdge] = []
    for u, v, w in edges:
        if _find(parent, u) != _find(parent, v):
            _union(parent, rank, u, v)
            mst.append((u, v, w))
    return mst


def mst_weight(g: Graph) -> float:
    """Return the total weight of the minimum spanning tree."""
    return float(sum(w for _, _, w in kruskal_mst(g)))


__all__ = ["kruskal_mst", "mst_weight"]
