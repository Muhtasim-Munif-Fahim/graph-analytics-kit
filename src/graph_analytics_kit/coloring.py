"""Vertex colouring: greedy strategies, DSATUR, and exact chromatic number.

A proper vertex colouring assigns every node an integer colour such that no
two adjacent nodes share a colour. Finding the chromatic number is NP-hard,
so practical tools use *greedy sequential* colouring: nodes are visited in a
strategy-specific order and each receives the smallest colour not used by
an already-coloured neighbour. The order matters a lot:

``largest_first``
    Welsh–Powell: decreasing degree (stable, ties keep insertion order —
    identical to ``networkx.greedy_color(strategy="largest_first")``).
``smallest_last``
    Matula–Beck degeneracy order: repeatedly remove a minimum-degree node and
    colour in reverse removal order. Uses at most ``degeneracy + 1`` colours.
``dsatur``
    Brélaz (1979): always colour the node with the most distinct neighbour
    colours (saturation), breaking ties by degree in the uncoloured subgraph.
    Optimal on bipartite graphs, cycles and wheels.
``connected_sequential_bfs`` / ``connected_sequential_dfs``
    BFS/DFS order per connected component (each node after the first has a
    coloured neighbour when visited).
``random_sequential``
    Uniformly random order (seeded).

:func:`chromatic_number` computes the exact chromatic number with a DSATUR
branch-and-bound (Brélaz order, clique lower bound) and is intended for
small and medium graphs. Edge weights and directions are ignored (directed
graphs are coloured on their underlying undirected graph); self-loops make a
proper colouring impossible and raise ``ValueError``.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Sequence, Set

from .graph import Graph

__all__ = [
    "COLORING_STRATEGIES",
    "greedy_color",
    "chromatic_number",
    "is_proper_coloring",
    "color_classes",
    "coloring_summary",
]

COLORING_STRATEGIES = (
    "largest_first",
    "smallest_last",
    "dsatur",
    "connected_sequential_bfs",
    "connected_sequential_dfs",
    "random_sequential",
)


def _undirected_adjacency(g: Graph) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {u: set() for u in g.nodes}
    for u in g.nodes:
        for v in g.neighbors(u):
            if u == v:
                raise ValueError(
                    f"node {u} has a self-loop; no proper vertex colouring exists"
                )
            adj[u].add(v)
            adj[v].add(u)
    return adj


def _smallest_free(colors: Dict[int, int], nbrs: Set[int]) -> int:
    used = {colors[v] for v in nbrs if v in colors}
    c = 0
    while c in used:
        c += 1
    return c


def _order_largest_first(adj: Dict[int, Set[int]]) -> List[int]:
    return sorted(adj, key=lambda u: len(adj[u]), reverse=True)


def _order_smallest_last(adj: Dict[int, Set[int]]) -> List[int]:
    degree = {u: len(nbrs) for u, nbrs in adj.items()}
    max_deg = max(degree.values(), default=0)
    buckets: List[Dict[int, None]] = [dict() for _ in range(max_deg + 1)]
    for u in adj:  # insertion order for deterministic ties
        buckets[degree[u]][u] = None
    removed: Set[int] = set()
    removal: List[int] = []
    low = 0
    for _ in range(len(adj)):
        low = max(low - 1, 0)
        while not buckets[low]:
            low += 1
        u = next(iter(buckets[low]))
        del buckets[low][u]
        removed.add(u)
        removal.append(u)
        for v in adj[u]:
            if v in removed:
                continue
            d = degree[v]
            del buckets[d][v]
            degree[v] = d - 1
            buckets[d - 1][v] = None
    return removal[::-1]


def _order_connected(adj: Dict[int, Set[int]], *, bfs: bool) -> List[int]:
    position = {u: i for i, u in enumerate(adj)}
    seen: Set[int] = set()
    order: List[int] = []
    for root in adj:
        if root in seen:
            continue
        seen.add(root)
        if bfs:
            queue = deque([root])
            while queue:
                u = queue.popleft()
                order.append(u)
                for v in sorted(adj[u], key=position.__getitem__):
                    if v not in seen:
                        seen.add(v)
                        queue.append(v)
        else:
            stack = [root]
            seen.discard(root)
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                order.append(u)
                for v in sorted(adj[u], key=lambda x: -position[x]):
                    if v not in seen:
                        stack.append(v)
    return order


def _dsatur(adj: Dict[int, Set[int]]) -> Dict[int, int]:
    colors: Dict[int, int] = {}
    neighbour_colors: Dict[int, Set[int]] = {u: set() for u in adj}
    uncolored_degree = {u: len(adj[u]) for u in adj}
    position = {u: i for i, u in enumerate(adj)}
    remaining = set(adj)
    while remaining:
        u = max(
            remaining,
            key=lambda x: (len(neighbour_colors[x]), uncolored_degree[x], -position[x]),
        )
        c = 0
        while c in neighbour_colors[u]:
            c += 1
        colors[u] = c
        remaining.discard(u)
        for v in adj[u]:
            if v in remaining:
                neighbour_colors[v].add(c)
                uncolored_degree[v] -= 1
    return colors


def greedy_color(
    g: Graph,
    strategy: str = "largest_first",
    *,
    seed: Optional[int] = None,
) -> Dict[int, int]:
    """Return a proper colouring ``{node: colour}`` with colours ``0..k-1``.

    ``strategy`` is one of :data:`COLORING_STRATEGIES` (see module docs).
    ``seed`` only affects ``random_sequential``.
    """
    if strategy not in COLORING_STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; choose from {', '.join(COLORING_STRATEGIES)}"
        )
    adj = _undirected_adjacency(g)
    if not adj:
        return {}
    if strategy == "dsatur":
        return _dsatur(adj)
    if strategy == "largest_first":
        order = _order_largest_first(adj)
    elif strategy == "smallest_last":
        order = _order_smallest_last(adj)
    elif strategy == "random_sequential":
        order = list(adj)
        random.Random(seed).shuffle(order)
    else:
        order = _order_connected(adj, bfs=strategy == "connected_sequential_bfs")
    colors: Dict[int, int] = {}
    for u in order:
        colors[u] = _smallest_free(colors, adj[u])
    return colors


def is_proper_coloring(g: Graph, coloring: Dict[int, int]) -> bool:
    """True iff every node is coloured and no edge joins two equal colours."""
    for u in g.nodes:
        if u not in coloring:
            return False
        for v in g.neighbors(u):
            if v in coloring and coloring[u] == coloring[v]:
                return False
    return True


def color_classes(coloring: Dict[int, int]) -> List[List[int]]:
    """Group nodes by colour: ``classes[c]`` lists the nodes with colour ``c``."""
    k = max(coloring.values(), default=-1) + 1
    classes: List[List[int]] = [[] for _ in range(k)]
    for node, c in coloring.items():
        classes[c].append(node)
    return [sorted(cls) for cls in classes]


def _greedy_clique(adj: Dict[int, Set[int]]) -> List[int]:
    position = {u: i for i, u in enumerate(adj)}
    best: List[int] = []
    for start in sorted(adj, key=lambda u: -len(adj[u])):
        clique = [start]
        cand = set(adj[start])
        while cand:
            v = max(cand, key=lambda x: (len(adj[x] & cand), -position[x]))
            clique.append(v)
            cand &= adj[v]
        if len(clique) > len(best):
            best = clique
    return best


def chromatic_number(g: Graph, *, max_nodes: int = 200, return_coloring: bool = False):
    """Exact chromatic number via DSATUR branch-and-bound.

    The search starts from the DSATUR upper bound and a greedy clique lower
    bound, colouring nodes in dynamic saturation order and pruning branches
    that cannot beat the incumbent. Worst-case exponential: ``max_nodes``
    guards against accidentally huge inputs. With ``return_coloring=True``
    returns ``(chi, coloring)``.
    """
    adj = _undirected_adjacency(g)
    n = len(adj)
    if n > max_nodes:
        raise ValueError(f"graph has {n} nodes > max_nodes={max_nodes}")
    if n == 0:
        return (0, {}) if return_coloring else 0
    best_coloring = _dsatur(adj)
    best_k = max(best_coloring.values()) + 1
    clique = _greedy_clique(adj)
    lower = len(clique)
    if best_k > lower:
        colors: Dict[int, int] = {}
        # Pre-colour the clique (symmetry breaking).
        for i, u in enumerate(clique):
            colors[u] = i
        position = {u: i for i, u in enumerate(adj)}

        def search(k_used: int) -> None:
            nonlocal best_k, best_coloring
            if best_k == lower:
                return
            if len(colors) == n:
                if k_used < best_k:
                    best_k = k_used
                    best_coloring = dict(colors)
                return
            # Pick uncoloured node with max saturation, then max degree.
            best_u, best_key = None, None
            for u in adj:
                if u in colors:
                    continue
                sat = len({colors[v] for v in adj[u] if v in colors})
                key = (sat, len(adj[u]), -position[u])
                if best_key is None or key > best_key:
                    best_u, best_key = u, key
            u = best_u
            used = {colors[v] for v in adj[u] if v in colors}
            for c in range(min(k_used + 1, best_k - 1)):
                if c in used:
                    continue
                colors[u] = c
                search(max(k_used, c + 1))
                del colors[u]
                if best_k == lower:
                    return

        search(lower)
    if return_coloring:
        return best_k, best_coloring
    return best_k


def coloring_summary(g: Graph, *, seed: Optional[int] = 0) -> Dict[str, int]:
    """Number of colours used by every greedy strategy (lower is better)."""
    out: Dict[str, int] = {}
    for strategy in COLORING_STRATEGIES:
        col = greedy_color(g, strategy, seed=seed)
        out[strategy] = max(col.values(), default=-1) + 1
    return out
