"""Matchings in general graphs and the assignment problem.

``maximum_matching`` finds a maximum-cardinality matching in **any**
undirected graph using Edmonds' blossom algorithm. Odd cycles (blossoms)
met during the alternating-tree search are contracted to their base, so
augmenting paths that pass through them are still found. It runs in
``O(V^3)`` (``V`` BFS searches, each with ``O(V^2)`` blossom work), after a
greedy warm start. In a non-bipartite graph Hopcroft-Karp can miss
augmenting paths, and this algorithm does not.

``linear_sum_assignment`` solves the (rectangular) assignment problem
``min sum_i C[i, sigma(i)]`` with the Hungarian method in its
shortest-augmenting-path form with row/column potentials (Kuhn-Munkres,
Jonker-Volgenant), ``O(n^2 m)``. ``maximize=True`` and ``inf`` entries
(forbidden pairs) are supported. ``maximum_weight_bipartite_matching``
builds on it for a weighted bipartite graph, optionally restricted to
maximum-cardinality matchings.

``minimum_edge_cover`` uses Gallai's identity ``rho(G) = |V| - nu(G)``:
take a maximum matching and cover each unmatched vertex with any incident
edge.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .graph import Graph

__all__ = [
    "maximum_matching",
    "is_matching",
    "is_perfect_matching",
    "matching_edges",
    "linear_sum_assignment",
    "maximum_weight_bipartite_matching",
    "minimum_edge_cover",
]

Edge = Tuple[int, int]


def _undirected_adjacency(g: Graph) -> Tuple[List[int], List[List[int]]]:
    nodes = g.nodes
    index = {u: i for i, u in enumerate(nodes)}
    adj: List[Set[int]] = [set() for _ in nodes]
    for u, v, _ in g.edges:
        if u == v:
            continue
        a, b = index[u], index[v]
        adj[a].add(b)
        adj[b].add(a)
    return nodes, [sorted(s) for s in adj]


def maximum_matching(g: Graph) -> Dict[int, int]:
    """Maximum-cardinality matching of a general graph (Edmonds' blossom algorithm).

    Edge direction, weights and self-loops are ignored. Returns a symmetric
    dict ``{u: v, v: u}``, the same format as :func:`maximum_bipartite_matching`.
    ``len(result) // 2`` is the matching number ``nu(G)``.
    """
    nodes, adj = _undirected_adjacency(g)
    n = len(nodes)
    match = [-1] * n

    # greedy warm start
    for v in range(n):
        if match[v] == -1:
            for w in adj[v]:
                if match[w] == -1:
                    match[v], match[w] = w, v
                    break

    def find_augmenting_path(root: int) -> Tuple[int, List[int]]:
        used = [False] * n
        parent = [-1] * n
        base = list(range(n))
        used[root] = True
        queue = deque([root])

        def lca(a: int, b: int) -> int:
            seen = [False] * n
            while True:
                a = base[a]
                seen[a] = True
                if match[a] == -1:
                    break
                a = parent[match[a]]
            while True:
                b = base[b]
                if seen[b]:
                    return b
                b = parent[match[b]]

        def mark_path(v: int, b: int, child: int, blossom: List[bool]) -> None:
            while base[v] != b:
                blossom[base[v]] = blossom[base[match[v]]] = True
                parent[v] = child
                child = match[v]
                v = parent[match[v]]

        while queue:
            v = queue.popleft()
            for to in adj[v]:
                if base[v] == base[to] or match[v] == to:
                    continue
                if to == root or (match[to] != -1 and parent[match[to]] != -1):
                    cur = lca(v, to)
                    blossom = [False] * n
                    mark_path(v, cur, to, blossom)
                    mark_path(to, cur, v, blossom)
                    for i in range(n):
                        if blossom[base[i]]:
                            base[i] = cur
                            if not used[i]:
                                used[i] = True
                                queue.append(i)
                elif parent[to] == -1:
                    parent[to] = v
                    if match[to] == -1:
                        return to, parent
                    used[match[to]] = True
                    queue.append(match[to])
        return -1, parent

    for root in range(n):
        if match[root] != -1:
            continue
        end, parent = find_augmenting_path(root)
        while end != -1:  # flip the alternating path
            pv = parent[end]
            nxt = match[pv]
            match[end], match[pv] = pv, end
            end = nxt

    return {nodes[i]: nodes[j] for i, j in enumerate(match) if j != -1}


def matching_edges(matching: Dict[int, int]) -> List[Edge]:
    """Sorted list of ``(u, v)`` pairs with ``u < v`` from a symmetric matching dict."""
    return sorted({(min(u, v), max(u, v)) for u, v in matching.items()})


def is_matching(g: Graph, matching) -> bool:
    """True if ``matching`` (symmetric dict or edge list) is a set of disjoint graph edges."""
    pairs = matching_edges(matching) if isinstance(matching, dict) else [tuple(e) for e in matching]
    if isinstance(matching, dict) and any(matching.get(v) != u for u, v in matching.items()):
        return False
    seen: Set[int] = set()
    for u, v in pairs:
        if u == v or u in seen or v in seen:
            return False
        if not (g.has_edge(u, v) or g.has_edge(v, u)):
            return False
        seen.update((u, v))
    return True


def is_perfect_matching(g: Graph, matching) -> bool:
    """True if ``matching`` is a matching that covers every node."""
    if not is_matching(g, matching):
        return False
    pairs = matching_edges(matching) if isinstance(matching, dict) else list(matching)
    return 2 * len(pairs) == g.number_of_nodes()


def linear_sum_assignment(cost, maximize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the assignment problem with the Hungarian algorithm.

    Parameters
    ----------
    cost : array-like of shape (r, c)
        Cost matrix. Entries may be ``inf`` (``-inf`` when maximizing) to
        forbid a pair. Rectangular matrices assign ``min(r, c)`` pairs.
    maximize : bool
        Maximize the total instead of minimizing it.

    Returns
    -------
    row_ind, col_ind : ndarray
        Assigned pairs ``(row_ind[k], col_ind[k])`` sorted by row, with the
        same convention as ``scipy.optimize.linear_sum_assignment``.

    Raises
    ------
    ValueError
        If no complete assignment avoids the forbidden entries.
    """
    C = np.array(cost, dtype=float)
    if C.ndim != 2:
        raise ValueError("cost must be a 2-D matrix")
    if np.isnan(C).any():
        raise ValueError("cost contains NaN")
    if maximize:
        C = -C
    if np.isneginf(C).any():
        raise ValueError("cost contains -inf (inf when maximizing)")
    r, c = C.shape
    if r == 0 or c == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    transposed = r > c
    if transposed:
        C = C.T
        r, c = c, r
    forbidden = np.isinf(C)
    finite = C[~forbidden]
    if finite.size == 0:
        raise ValueError("cost matrix is infeasible")
    big = (np.abs(finite).max() + 1.0) * (r + 1)
    W = np.where(forbidden, big, C)

    INF = float("inf")
    u = np.zeros(r + 1)
    v = np.zeros(c + 1)
    p = np.zeros(c + 1, dtype=int)  # p[j] = row assigned to column j (1-based), 0 = free
    way = np.zeros(c + 1, dtype=int)
    for i in range(1, r + 1):
        p[0] = i
        j0 = 0
        minv = np.full(c + 1, INF)
        used = np.zeros(c + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = W[i0 - 1] - u[i0] - v[1:]
            cols = np.flatnonzero(free) + 1
            better = cur[cols - 1] < minv[cols]
            upd = cols[better]
            minv[upd] = cur[upd - 1]
            way[upd] = j0
            j1 = cols[np.argmin(minv[cols])]
            delta = minv[j1]
            used_idx = np.flatnonzero(used)
            u[p[used_idx]] += delta
            v[used_idx] -= delta
            minv[cols] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:  # augment along the alternating path
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows, cols = [], []
    for j in range(1, c + 1):
        if p[j]:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    rows_a = np.array(rows, dtype=int)
    cols_a = np.array(cols, dtype=int)
    if forbidden[rows_a, cols_a].any():
        raise ValueError("cost matrix is infeasible")
    if transposed:
        rows_a, cols_a = cols_a, rows_a
    order = np.argsort(rows_a, kind="stable")
    return rows_a[order], cols_a[order]


def maximum_weight_bipartite_matching(
    g: Graph,
    left: Optional[Iterable[int]] = None,
    maximum_cardinality: bool = False,
) -> Dict[int, int]:
    """Maximum-weight matching of a weighted bipartite graph (Hungarian method).

    ``left`` lists one side; when omitted the graph is 2-coloured. Edges
    with non-positive weight are never worth matching unless
    ``maximum_cardinality=True``, which returns the heaviest matching among
    those of maximum size. Returns a symmetric dict ``{u: v, v: u}``.
    """
    from .flow import _two_colour

    left_set = _two_colour(g) if left is None else set(left)
    unknown = left_set - set(g.nodes)
    if unknown:
        raise ValueError(f"left contains nodes not in the graph: {sorted(unknown)}")
    L = [u for u in g.nodes if u in left_set]
    R = [u for u in g.nodes if u not in left_set]
    if not L or not R:
        return {}
    li = {u: i for i, u in enumerate(L)}
    ri = {u: i for i, u in enumerate(R)}
    weight = np.full((len(L), len(R)), np.nan)
    for u, v, w in g.edges:
        if u == v:
            continue
        if u in left_set and v in left_set:
            raise ValueError("left side has an internal edge; graph is not bipartite on this split")
        if u not in left_set and v not in left_set:
            raise ValueError("right side has an internal edge; graph is not bipartite on this split")
        a, b = (u, v) if u in left_set else (v, u)
        weight[li[a], ri[b]] = np.nanmax([weight[li[a], ri[b]], w])
    is_edge = ~np.isnan(weight)
    if maximum_cardinality:
        M = np.abs(weight[is_edge]).sum() + 1.0 if is_edge.any() else 1.0
        profit = np.where(is_edge, np.nan_to_num(weight) + M, 0.0)
    else:
        profit = np.where(is_edge & (np.nan_to_num(weight) > 0), np.nan_to_num(weight), 0.0)
    rows, cols = linear_sum_assignment(profit, maximize=True)
    out: Dict[int, int] = {}
    for i, j in zip(rows, cols):
        if not is_edge[i, j]:
            continue
        if not maximum_cardinality and weight[i, j] <= 0:
            continue
        out[L[i]] = R[j]
        out[R[j]] = L[i]
    return out


def minimum_edge_cover(g: Graph) -> List[Edge]:
    """A minimum edge cover (every node touches a chosen edge), via ``rho = |V| - nu``.

    Raises ``ValueError`` if some node is isolated (no cover exists).
    """
    nodes, adj = _undirected_adjacency(g)
    for i, nb in enumerate(adj):
        if not nb:
            raise ValueError(f"node {nodes[i]} is isolated; no edge cover exists")
    m = maximum_matching(g)
    cover = set(matching_edges(m))
    for i, u in enumerate(nodes):
        if u not in m:
            v = nodes[adj[i][0]]
            cover.add((min(u, v), max(u, v)))
    return sorted(cover)
