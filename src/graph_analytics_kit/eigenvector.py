"""Eigenvector centrality via power iteration."""
from __future__ import annotations

import math
from typing import Dict

from .graph import Graph


def eigenvector_centrality(
    g: Graph,
    *,
    max_iter: int = 100,
    tol: float = 1e-06,
) -> Dict[int, float]:
    """Compute eigenvector centrality for every node in *g*.

    A node's score is proportional to the weighted scores of nodes that
    point to it (neighbors, on an undirected graph). This is the principal
    eigenvector of the adjacency matrix: ``A^T x = λ x``. Power iteration
    applies ``A^T + I`` so the leading eigenvalue is unique in magnitude
    even on bipartite graphs; ``A`` and ``A + I`` share the same
    eigenvectors whenever ``A`` is diagonalizable, and the extra identity
    term vanishes after L2 normalization of the principal direction.

    Isolated nodes receive a score approaching 0.0 when any edge exists.
    An empty graph returns an empty dict. A graph with no edges returns a
    uniform unit vector. Scores are non-negative and L2-normalized.

    Parameters
    ----------
    g:
        The input graph. Directed edges contribute along the in-edge
        (prestige) direction.
    max_iter:
        Maximum number of power-iteration steps.
    tol:
        Convergence tolerance on the L1 norm of the score delta.

    Returns
    -------
    dict[int, float]
        Node label -> eigenvector centrality. The vector has L2 norm 1
        (within rounding) when the graph is non-empty.
    """
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    nodes = g.nodes
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: 1.0}

    outgoing = {
        node: [(nbr, g.get_weight(node, nbr)) for nbr in g.neighbors(node)]
        for node in nodes
    }

    inv_norm = 1.0 / math.sqrt(n)
    score: Dict[int, float] = {node: inv_norm for node in nodes}

    for _ in range(max_iter):
        # (A^T + I) x: identity shift keeps the iterate in the positive
        # orthant on bipartite graphs without changing the principal
        # direction of a non-negative adjacency matrix.
        new_score: Dict[int, float] = {node: score[node] for node in nodes}
        for node in nodes:
            value = score[node]
            for nbr, weight in outgoing[node]:
                new_score[nbr] += value * weight

        norm = math.sqrt(sum(value * value for value in new_score.values()))
        if norm == 0.0:
            return {node: inv_norm for node in nodes}

        new_score = {node: value / norm for node, value in new_score.items()}
        delta = sum(abs(new_score[node] - score[node]) for node in nodes)
        score = new_score
        if delta < tol:
            break

    if sum(score.values()) < 0.0:
        score = {node: -value for node, value in score.items()}
    return score


__all__ = ["eigenvector_centrality"]
