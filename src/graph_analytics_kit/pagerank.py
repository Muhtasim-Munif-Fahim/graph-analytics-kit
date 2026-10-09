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




def personalized_pagerank(
    g: Graph,
    personalization: Dict[int, float] | None = None,
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-06,
) -> Dict[int, float]:
    """Compute Personalized PageRank (random walk with restart) on *g*.

    Unlike uniform PageRank, the teleport (restart) distribution is not
    uniform over nodes. When the surfer jumps, it lands according to the
    non-negative *personalization* weights (renormalized to sum to 1).
    Setting a single seed node to weight 1 recovers classic random-walk-
    with-restart scores used for local neighborhood ranking and
    recommendation (Haveliwala, 2002; Tong, Faloutsos & Pan, 2006).

    For undirected graphs every edge is treated as bidirectional. Dangling
    nodes redistribute their remaining mass according to the same
    personalization distribution. When *personalization* is ``None`` or
    empty, the teleport is uniform and the result matches :func:`pagerank`.

    Parameters
    ----------
    g:
        The input graph.
    personalization:
        Optional mapping of node label -> non-negative restart weight.
        Nodes missing from the map (or from *g*) receive weight 0. At
        least one node present in *g* must have a positive weight when
        the map is non-empty.
    alpha:
        Probability of following an outgoing link. Must be in (0, 1).
    max_iter:
        Maximum number of power-iteration steps.
    tol:
        Convergence tolerance on the L1 norm of the score delta.

    Returns
    -------
    dict[int, float]
        Node label -> Personalized PageRank score. Scores sum to 1.0
        (within rounding).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    nodes = g.nodes
    n = len(nodes)
    if n == 0:
        return {}

    if personalization is None or len(personalization) == 0:
        teleport_dist = {node: 1.0 / n for node in nodes}
    else:
        raw = {node: float(personalization.get(node, 0.0)) for node in nodes}
        if any(w < 0.0 for w in raw.values()):
            raise ValueError("personalization weights must be non-negative")
        total_w = sum(raw.values())
        if total_w <= 0.0:
            raise ValueError(
                "personalization must assign a positive weight to at least "
                "one node in the graph"
            )
        teleport_dist = {node: raw[node] / total_w for node in nodes}

    neighbors = {node: g.neighbors(node) for node in nodes}
    rank = {node: teleport_dist[node] for node in nodes}

    for _ in range(max_iter):
        new_rank: Dict[int, float] = {
            node: (1.0 - alpha) * teleport_dist[node] for node in nodes
        }
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
            for node in nodes:
                new_rank[node] += dangling_mass * teleport_dist[node]

        delta = sum(abs(new_rank[node] - rank[node]) for node in nodes)
        rank = new_rank
        if delta < tol:
            break

    total = sum(rank.values())
    if total > 0:
        rank = {node: score / total for node, score in rank.items()}
    return rank


__all__ = ["pagerank", "personalized_pagerank"]
