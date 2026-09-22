"""k-core decomposition: core number of each node.

Closeness centrality already ships in :mod:`graph_analytics_kit.centrality`
(Wasserman–Faust ``(n - 1) / sum of distances``). This module is the
replacement structural measure: the Batagelj–Zaversnik core decomposition.
Brandes betweenness is left untouched.
"""
from __future__ import annotations

from typing import Dict, List

from .graph import Graph


def core_number(g: Graph) -> Dict[int, int]:
    """Return the core number of every node in *g*.

    The core number of a node is the largest integer ``k`` such that the
    node belongs to the ``k``-core: the unique maximal subgraph in which
    every node has degree at least ``k``. Equivalently, nodes are peeled
    in order of increasing degree; a node's core number is its degree at
    the moment it is removed.

    The implementation is the linear-time bin-sort algorithm of Batagelj
    and Zaversnik (2003). Edge weights are ignored: each simple edge
    contributes one to the degree. Self-loops are ignored. On a directed
    graph the decomposition is taken on the underlying simple undirected
    graph (an arc in either direction is one undirected edge).

    Isolated nodes have core number ``0``. An empty graph returns an
    empty dict. A complete graph on ``n`` nodes (``n >= 1``) assigns
    every node the core number ``n - 1``. A nonempty path or star assigns
    every node the core number ``1``.

    Parameters
    ----------
    g:
        The input graph.

    Returns
    -------
    dict[int, int]
        Node label -> core number.
    """
    nodes = g.nodes
    n = len(nodes)
    if n == 0:
        return {}

    adj = _simple_adjacency(g, nodes)
    degree = {node: len(adj[node]) for node in nodes}
    ordered, bin_start, node_pos = _bin_sort(nodes, degree)
    core = dict(degree)
    nbrs = {node: list(adj[node]) for node in nodes}

    for v in ordered:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos = node_pos[u]
                bin_pos = bin_start[core[u]]
                occupant = ordered[bin_pos]
                node_pos[u] = bin_pos
                node_pos[occupant] = pos
                ordered[bin_pos], ordered[pos] = ordered[pos], ordered[bin_pos]
                bin_start[core[u]] += 1
                core[u] -= 1
    return core


def _simple_adjacency(g: Graph, nodes: List[int]) -> Dict[int, List[int]]:
    """Undirected simple adjacency, ignoring self-loops and duplicate arcs."""
    adj: Dict[int, List[int]] = {node: [] for node in nodes}
    seen = {node: set() for node in nodes}
    for u, v, _weight in g.edges:
        if u == v or u not in seen or v not in seen or u in seen[v]:
            continue
        seen[u].add(v)
        seen[v].add(u)
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _bin_sort(
    nodes: List[int],
    degree: Dict[int, int],
) -> tuple[List[int], List[int], Dict[int, int]]:
    """Order *nodes* by nondecreasing degree and record bin boundaries."""
    n = len(nodes)
    max_degree = max(degree.values()) if nodes else 0
    counts = [0] * (max_degree + 1)
    for value in degree.values():
        counts[value] += 1
    start = 0
    for d in range(max_degree + 1):
        count = counts[d]
        counts[d] = start
        start += count
    bin_start = list(counts)
    ordered = [0] * n
    node_pos: Dict[int, int] = {}
    for node in nodes:
        pos = counts[degree[node]]
        ordered[pos] = node
        node_pos[node] = pos
        counts[degree[node]] += 1
    return ordered, bin_start, node_pos


__all__ = ["core_number"]
