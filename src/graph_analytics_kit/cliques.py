"""Maximal cliques (Bron–Kerbosch with Tomita pivoting) and triangle counts.

A *clique* is a set of nodes that are all pairwise adjacent; it is
*maximal* when no further node can be added. Cliques are the densest
possible substructures, so they complement the k-core decomposition and
community detection: every member of a ``k``-clique has core number at
least ``k - 1``.

:func:`find_cliques` enumerates every maximal clique with the
Bron–Kerbosch algorithm using the Tomita–Tanaka–Takahashi (2006) pivot
rule (pivot on the vertex of ``P ∪ X`` with the most neighbours in ``P``),
which is worst-case optimal at ``O(3^(n/3))``. The outer level iterates
vertices in a degeneracy (smallest-last) order, as in Eppstein, Löffler &
Strash (2010), which keeps the candidate sets small on sparse graphs.

Edge weights and self-loops are ignored. Directed graphs are treated as
their underlying simple undirected graph (an arc in either direction is
one undirected edge), matching :func:`core_number`.
"""
from __future__ import annotations

from typing import Dict, List, Set

from .graph import Graph

__all__ = [
    "find_cliques",
    "max_clique",
    "clique_number",
    "node_clique_number",
    "number_of_maximal_cliques",
    "triangles",
    "total_triangles",
]


def _simple_undirected(g: Graph) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {node: set() for node in g.nodes}
    for node in g.nodes:
        for nbr in g.neighbors(node):
            if nbr != node:
                adj[node].add(nbr)
                adj[nbr].add(node)
    return adj


def _degeneracy_order(adj: Dict[int, Set[int]]) -> List[int]:
    """Smallest-last vertex order (ties broken by node label)."""
    degree = {v: len(nbrs) for v, nbrs in adj.items()}
    remaining = set(adj)
    order: List[int] = []
    while remaining:
        v = min(remaining, key=lambda node: (degree[node], node))
        order.append(v)
        remaining.remove(v)
        for u in adj[v]:
            if u in remaining:
                degree[u] -= 1
    return order


def _bron_kerbosch(adj, r: List[int], p: Set[int], x: Set[int], out: List[List[int]]) -> None:
    # An explicit stack (instead of recursion) avoids Python's recursion limit
    # on large cliques; every frame carries its own snapshot of (R, P, X).
    stack = [(r, p, x)]
    while stack:
        r, p, x = stack.pop()
        if not p:
            if not x:
                out.append(sorted(r))
            continue
        pivot = max(p | x, key=lambda u: (len(adj[u] & p), -u))
        for v in sorted(p - adj[pivot], reverse=True):
            stack.append((r + [v], p & adj[v], x & adj[v]))
            p = p - {v}
            x = x | {v}


def find_cliques(g: Graph, min_size: int = 1) -> List[List[int]]:
    """Return every maximal clique with at least ``min_size`` nodes.

    Each clique is a sorted list of node labels; the list is sorted by
    decreasing size, then lexicographically, so the first entry is a
    maximum clique. Isolated nodes are maximal cliques of size 1.
    """
    if not isinstance(min_size, int) or isinstance(min_size, bool) or min_size < 1:
        raise ValueError("min_size must be a positive integer")
    adj = _simple_undirected(g)
    out: List[List[int]] = []
    order = _degeneracy_order(adj)
    position = {v: i for i, v in enumerate(order)}
    for v in order:
        later = {u for u in adj[v] if position[u] > position[v]}
        earlier = {u for u in adj[v] if position[u] < position[v]}
        _bron_kerbosch(adj, [v], later, earlier, out)
    cliques = [c for c in out if len(c) >= min_size]
    cliques.sort(key=lambda c: (-len(c), c))
    return cliques


def number_of_maximal_cliques(g: Graph) -> int:
    """Number of maximal cliques (isolated nodes count as size-1 cliques)."""
    return len(find_cliques(g))


def max_clique(g: Graph) -> List[int]:
    """A maximum clique (largest size; lexicographically smallest on ties)."""
    cliques = find_cliques(g)
    return cliques[0] if cliques else []


def clique_number(g: Graph) -> int:
    """Size of the largest clique (``omega(G)``); ``0`` for an empty graph."""
    return len(max_clique(g))


def node_clique_number(g: Graph) -> Dict[int, int]:
    """Size of the largest maximal clique containing each node."""
    best = {node: 0 for node in g.nodes}
    for clique in find_cliques(g):
        size = len(clique)
        for node in clique:
            if size > best[node]:
                best[node] = size
    return best


def triangles(g: Graph) -> Dict[int, int]:
    """Number of triangles through each node (undirected, weights ignored).

    Uses forward edge iteration over a degree ordering, so each triangle is
    found exactly once in ``O(m^{3/2})`` time.
    """
    adj = _simple_undirected(g)
    rank = {v: (len(adj[v]), v) for v in adj}
    forward = {v: {u for u in adj[v] if rank[u] > rank[v]} for v in adj}
    counts = {v: 0 for v in adj}
    for v in adj:
        for u in forward[v]:
            for w in forward[v] & forward[u]:
                counts[v] += 1
                counts[u] += 1
                counts[w] += 1
    return counts


def total_triangles(g: Graph) -> int:
    """Total number of distinct triangles in the graph."""
    return sum(triangles(g).values()) // 3
