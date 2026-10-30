"""Travelling salesman tours: exact DP, constructive heuristics, local search, bounds.

The functions accept either a :class:`~graph_analytics_kit.graph.Graph` or a
square distance matrix.

* A graph that is not complete can use its **metric closure**
  (``metric_closure=True``). Each pair is then at shortest-path distance
  (Floyd-Warshall), and :func:`solve_tsp` expands the tour back to real edges
  with ``expand_path=True``.
* :func:`held_karp` -- the exact Bellman-Held-Karp dynamic program,
  ``O(2^n n^2)``. It handles symmetric and asymmetric instances up to about
  16 cities.
* :func:`nearest_neighbor_tour` -- the greedy constructive tour. It can be
  run from every start city.
* :func:`double_tree_tour` -- a preorder walk of the minimum spanning tree.
  On metric instances its cost is at most twice the optimum.
* :func:`two_opt` -- 2-opt local search (segment reversal) until no
  improving move remains. Asymmetric instances use full tour re-evaluation.
* :func:`or_opt` -- moves segments of 1-3 consecutive cities to a better
  position. It complements 2-opt.
* :func:`one_tree_bound` -- the Held-Karp 1-tree lower bound (an MST on
  ``n-1`` cities plus the two cheapest edges to the remaining one, maximised
  over that city). It certifies heuristic tours on symmetric instances.
* :func:`solve_tsp` -- the solver users normally call. It is exact for small
  ``n`` and otherwise runs a multi-start heuristic plus local search. The
  result reports the optimality gap against the 1-tree bound.

Only ``numpy`` is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .graph import Graph

__all__ = [
    "TSPResult",
    "tsp_distance_matrix",
    "tour_cost",
    "held_karp",
    "nearest_neighbor_tour",
    "double_tree_tour",
    "two_opt",
    "or_opt",
    "one_tree_bound",
    "solve_tsp",
]

Instance = Union[Graph, np.ndarray, Sequence[Sequence[float]]]


@dataclass
class TSPResult:
    """A Hamiltonian cycle and its cost.

    ``tour`` lists each city once, using node labels for graphs and indices
    for matrices. The cycle returns to ``tour[0]``. ``path`` is the closed
    walk over real graph edges when a metric closure was used.
    """

    tour: List[int]
    cost: float
    method: str
    optimal: bool
    lower_bound: Optional[float]
    path: Optional[List[int]] = None

    @property
    def gap(self) -> Optional[float]:
        """Relative gap ``(cost - lower_bound) / lower_bound``."""

        if self.lower_bound is None or self.lower_bound <= 0:
            return None
        return (self.cost - self.lower_bound) / self.lower_bound


def tsp_distance_matrix(g: Graph, *, metric_closure: bool = False) -> Tuple[List[int], np.ndarray]:
    """``(nodes, D)`` for a weighted graph.

    Missing edges are ``inf``. With ``metric_closure`` they become all-pairs
    shortest-path distances, so the triangle inequality holds and every
    disconnected pair stays ``inf``.
    """

    if metric_closure:
        from .distance import floyd_warshall

        nodes, D, _ = floyd_warshall(g, weighted=True)
        return nodes, D
    nodes = g.nodes
    idx = {u: i for i, u in enumerate(nodes)}
    D = np.full((len(nodes), len(nodes)), np.inf)
    np.fill_diagonal(D, 0.0)
    for u in nodes:
        for v in g.neighbors(u):
            if u != v:
                D[idx[u], idx[v]] = g.get_weight(u, v)
    return nodes, D


def _as_matrix(instance: Instance, metric_closure: bool = False) -> Tuple[Optional[List[int]], np.ndarray]:
    if isinstance(instance, Graph):
        nodes, D = tsp_distance_matrix(instance, metric_closure=metric_closure)
    else:
        nodes, D = None, np.asarray(instance, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distance matrix must be square")
    if D.shape[0] < 2:
        raise ValueError("need at least 2 cities")
    if np.isnan(D).any():
        raise ValueError("distance matrix contains NaN")
    return nodes, D


def tour_cost(D, tour: Sequence[int]) -> float:
    """Cost of the closed tour ``tour[0] -> ... -> tour[-1] -> tour[0]``."""

    D = np.asarray(D, dtype=float)
    t = np.asarray(tour, dtype=int)
    return float(D[t, np.roll(t, -1)].sum())


def _check_tour(n: int, tour: Sequence[int]) -> List[int]:
    t = [int(x) for x in tour]
    if sorted(t) != list(range(n)):
        raise ValueError("tour must visit every city exactly once")
    return t


def held_karp(D, *, max_n: int = 16) -> Tuple[float, List[int]]:
    """Exact optimum via Held-Karp DP. Returns ``(cost, tour)``, starting at 0.

    ``dp[S, j]`` is the cheapest path that starts at city 0, visits exactly
    the set ``S`` of other cities and ends at ``j``. Each layer is vectorised
    over the predecessor. Raises ``ValueError`` when ``n > max_n``, because
    memory is ``2^(n-1) (n-1)``.
    """

    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if n > max_n:
        raise ValueError(f"held_karp is exponential; n={n} exceeds max_n={max_n}")
    if n == 2:
        return float(D[0, 1] + D[1, 0]), [0, 1]
    m = n - 1
    full = 1 << m
    dp = np.full((full, m), np.inf)
    parent = np.full((full, m), -1, dtype=np.int64)
    W = D[1:, 1:]  # W[k, j] = cost k -> j among cities 1..n-1
    for j in range(m):
        dp[1 << j, j] = D[0, j + 1]
    masks_by_size: List[List[int]] = [[] for _ in range(m + 1)]
    for mask in range(1, full):
        masks_by_size[bin(mask).count("1")].append(mask)
    for size in range(2, m + 1):
        for mask in masks_by_size[size]:
            for j in range(m):
                bit = 1 << j
                if not mask & bit:
                    continue
                prev = mask ^ bit
                cand = dp[prev] + W[:, j]
                k = int(np.argmin(cand))
                dp[mask, j] = cand[k]
                parent[mask, j] = k
    closing = dp[full - 1] + D[1:, 0]
    j = int(np.argmin(closing))
    cost = float(closing[j])
    if not np.isfinite(cost):
        raise ValueError("no Hamiltonian cycle exists (infinite distances)")
    order = []
    mask = full - 1
    while j != -1:
        order.append(j + 1)
        pj = int(parent[mask, j])
        mask ^= 1 << j
        j = pj
    return cost, [0] + order[::-1]


def nearest_neighbor_tour(D, start: int = 0) -> List[int]:
    """Greedy tour: repeatedly move to the closest unvisited city."""

    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if not 0 <= start < n:
        raise ValueError("start out of range")
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        row = np.where(visited, np.inf, D[tour[-1]])
        nxt = int(np.argmin(row))
        if visited[nxt]:  # all remaining at inf distance; pick any unvisited
            nxt = int(np.flatnonzero(~visited)[0])
        tour.append(nxt)
        visited[nxt] = True
    return tour


def _prim(D) -> Tuple[np.ndarray, float]:
    n = D.shape[0]
    S = np.minimum(D, D.T)
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    par = np.full(n, -1)
    best[0] = 0.0
    total = 0.0
    for _ in range(n):
        u = int(np.argmin(np.where(in_tree, np.inf, best)))
        in_tree[u] = True
        total += best[u]
        upd = (~in_tree) & (S[u] < best)
        best[upd] = S[u][upd]
        par[upd] = u
    return par, float(total)


def double_tree_tour(D, start: int = 0) -> List[int]:
    """Preorder walk of the MST rooted at ``start``. Its cost is at most 2 OPT
    when ``D`` is metric."""

    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    order = np.r_[start, np.delete(np.arange(n), start)]
    par, _ = _prim(D[np.ix_(order, order)])
    children = [[] for _ in range(n)]
    for v in range(1, n):
        children[int(par[v])].append(v)
    tour, stack = [], [0]
    while stack:
        u = stack.pop()
        tour.append(int(order[u]))
        stack.extend(sorted(children[u], key=lambda c: -D[order[u], order[c]]))
    return tour


def _symmetric(D) -> bool:
    return bool(np.allclose(D, D.T, equal_nan=False))


def two_opt(D, tour: Sequence[int], *, max_passes: int = 1000) -> List[int]:
    """2-opt local search: reverse segments while doing so lowers the cost."""

    D = np.asarray(D, dtype=float)
    t = _check_tour(D.shape[0], tour)
    n = len(t)
    if n < 4:
        return t
    sym = _symmetric(D)
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n if i > 0 else n - 1):
                a, b = t[i], t[i + 1]
                c, d = t[j], t[(j + 1) % n]
                if sym:
                    delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                else:
                    cand = t[: i + 1] + t[i + 1 : j + 1][::-1] + t[j + 1 :]
                    delta = tour_cost(D, cand) - tour_cost(D, t)
                if delta < -1e-12:
                    t[i + 1 : j + 1] = t[i + 1 : j + 1][::-1]
                    improved = True
        if not improved:
            break
    return t


def or_opt(D, tour: Sequence[int], *, max_segment: int = 3, max_passes: int = 1000) -> List[int]:
    """Or-opt: relocate segments of 1..``max_segment`` consecutive cities,
    in either orientation, while this lowers the cost."""

    D = np.asarray(D, dtype=float)
    t = _check_tour(D.shape[0], tour)
    n = len(t)
    if n < 4:
        return t
    cur = tour_cost(D, t)
    for _ in range(max_passes):
        improved = False
        for L in range(1, min(max_segment, n - 2) + 1):
            for i in range(n):
                seg_idx = [(i + s) % n for s in range(L)]
                seg = [t[k] for k in seg_idx]
                rest = [c for k, c in enumerate(t) if k not in seg_idx]
                for pos in range(len(rest)):
                    for s in (seg, seg[::-1]):
                        cand = rest[: pos + 1] + s + rest[pos + 1 :]
                        c = tour_cost(D, cand)
                        if c < cur - 1e-12:
                            t, cur, improved = cand, c, True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return t


def one_tree_bound(D) -> float:
    """Held-Karp 1-tree lower bound on the optimal tour of a symmetric instance.

    For each special city ``s``, the bound is ``MST(V - s)`` plus the two
    cheapest edges from ``s``. The largest such bound is returned. Every tour
    is a 1-tree, so this never exceeds the optimum.
    """

    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if n < 3:
        return float(D[0, 1] + D[1, 0]) if n == 2 else 0.0
    S = np.minimum(D, D.T)
    best = -np.inf
    for s in range(n):
        keep = np.delete(np.arange(n), s)
        _, w = _prim(S[np.ix_(keep, keep)])
        two = np.sort(S[s, keep])[:2].sum()
        best = max(best, w + two)
    return float(best)


def _expand(nodes, g, tour_labels):
    from .shortest_path import dijkstra_shortest_path, reconstruct_path

    walk = [tour_labels[0]]
    for a, b in zip(tour_labels, tour_labels[1:] + tour_labels[:1]):
        _, prev = dijkstra_shortest_path(g, a, b)
        walk.extend(reconstruct_path(prev, a, b)[1:])
    return walk


def solve_tsp(
    instance: Instance,
    *,
    method: str = "auto",
    metric_closure: bool = False,
    expand_path: bool = False,
    exact_max_n: int = 12,
    n_starts: Optional[int] = None,
) -> TSPResult:
    """Find a short (or optimal) Hamiltonian cycle.

    ``method`` can be one of:

    * ``"auto"`` -- exact Held-Karp when ``n <= exact_max_n``, otherwise
      ``"heuristic"``.
    * ``"exact"`` -- Held-Karp.
    * ``"heuristic"`` -- nearest neighbour from ``n_starts`` start cities
      (default: all of them) plus the double-tree tour, each refined with
      2-opt and Or-opt. The best tour is kept.
    * ``"nearest-neighbor"`` / ``"double-tree"`` -- that construction alone.

    Infinite distances (missing edges) are allowed, and the result is then
    ``inf`` if no finite tour is found. Use ``metric_closure=True`` on sparse
    graphs. ``expand_path=True`` also returns the walk over real edges.
    """

    nodes, D = _as_matrix(instance, metric_closure)
    n = D.shape[0]
    if method not in {"auto", "exact", "heuristic", "nearest-neighbor", "double-tree"}:
        raise ValueError(f"unknown method {method!r}")
    sym = _symmetric(D)
    if method == "auto":
        method = "exact" if n <= exact_max_n else "heuristic"
    optimal = False
    if method == "exact":
        cost, tour = held_karp(D, max_n=max(16, exact_max_n))
        optimal = True
    elif method == "nearest-neighbor":
        tour = nearest_neighbor_tour(D)
    elif method == "double-tree":
        tour = double_tree_tour(D)
    else:
        starts = range(n) if n_starts is None else np.linspace(0, n - 1, min(n_starts, n)).astype(int)
        candidates = [nearest_neighbor_tour(D, int(s)) for s in starts] + [double_tree_tour(D)]
        best_t, best_c = None, np.inf
        for cand in candidates:
            cand = or_opt(D, two_opt(D, cand))
            c = tour_cost(D, cand)
            if best_t is None or c < best_c:
                best_t, best_c = cand, c
        tour = best_t
    cost = tour_cost(D, tour)
    lb = cost if optimal else (one_tree_bound(D) if sym and np.isfinite(D).all() else None)
    labels = tour if nodes is None else [nodes[i] for i in tour]
    path = None
    if expand_path:
        if not isinstance(instance, Graph):
            raise ValueError("expand_path needs a Graph instance")
        path = _expand(nodes, instance, labels)
    return TSPResult(tour=labels, cost=cost, method=method, optimal=optimal, lower_bound=lb, path=path)
