"""PageRank centrality for directed and undirected graphs."""
from __future__ import annotations

from typing import Dict

from .graph import Graph


def pagerank(
    g: Graph,
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-06,
) -> Dict[int, float]:
    """Compute the PageRank score for every node in *g*.

    PageRank models a random surfer who follows outgoing links with
    probability ``alpha`` and jumps to a uniformly random node with
    probability ``1 - alpha``. For undirected graphs every edge is treated
    as bidirectional. Nodes with out-degree zero (dangling nodes) leak
    their remaining rank uniformly to every node.

    The power-iteration method converges to a stationary distribution; the
    algorithm stops when the L1 norm of the residual between successive
    iterations falls below ``tol`` or after ``max_iter`` iterations.

    Parameters
    ----------
    g:
        The input graph.
    alpha:
        Damping factor (probability of following a link). Must be in (0, 1).
    max_iter:
        Maximum number of power-iteration steps.
    tol:
        Convergence tolerance on the L1 norm of the score delta.

    Returns
    -------
    dict[int, float]
        Node label -> PageRank score. Scores sum to 1.0 (within rounding).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    nodes = g.nodes
    n = len(nodes)
    if n == 0:
        return {}

    neighbors = {node: g.neighbors(node) for node in nodes}
    rank = {node: 1.0 / n for node in nodes}
    teleport = (1.0 - alpha) / n

    for _ in range(max_iter):
        new_rank: Dict[int, float] = {node: teleport for node in nodes}
        dangling_mass = 0.0
        for node in nodes:
            nbrs = neighbors[node]
            if not nbrs:
                dangling_mass += alpha * rank[node]
                continue
            share = alpha * rank[node] / len(nbrs)
            for v in nbrs:
                new_rank[v] += share
        if dangling_mass:
            extra = dangling_mass / n
            for node in nodes:
                new_rank[node] += extra

        delta = sum(abs(new_rank[node] - rank[node]) for node in nodes)
        rank = new_rank
        if delta < tol:
            break

    total = sum(rank.values())
    if total > 0:
        rank = {node: score / total for node, score in rank.items()}
    return rank


__all__ = ["pagerank"]
