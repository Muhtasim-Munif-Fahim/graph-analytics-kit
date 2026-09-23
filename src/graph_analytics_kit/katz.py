"""Katz centrality via attenuated walk iteration."""
from __future__ import annotations

import math
from typing import Dict

from .graph import Graph


def katz_centrality(
    g: Graph,
    *,
    alpha: float = 0.1,
    beta: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-06,
    normalized: bool = True,
) -> Dict[int, float]:
    """Compute Katz centrality for every node in *g*.

    Katz centrality scores a node by the walks that end at it, with longer
    walks discounted by the attenuation factor ``alpha``:

    ``x = alpha * A^T x + beta * 1``

    ``A`` is the adjacency matrix (``A[u, v]`` is the weight of the edge
    from ``u`` to ``v``). Directed edges therefore add prestige to the
    target. On an undirected graph every edge is bidirectional. ``beta``
    is the score given to a walk of length zero (the constant term).

    The series converges only when ``alpha`` is strictly smaller than the
    reciprocal of the largest eigenvalue of ``A``. Iteration stops when the
    L1 change between successive vectors falls below ``tol``. If that does
    not happen within ``max_iter`` steps, ``ValueError`` is raised.

    Parameters
    ----------
    g:
        The input graph.
    alpha:
        Attenuation factor applied to each additional step of a walk.
        Must be a positive finite number, and small enough that the
        iteration converges.
    beta:
        Constant added at every node on each update. Must be finite.
        The default ``1.0`` counts every node once before any walks.
    max_iter:
        Maximum number of iterations.
    tol:
        Convergence tolerance on the L1 norm of the score delta.
    normalized:
        When ``True`` (default), scale the solution to L2 norm 1 while
        keeping its sign. The zero solution is returned unchanged.

    Returns
    -------
    dict[int, float]
        Node label -> Katz centrality. An empty graph returns an empty
        dict.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a positive finite number")
    if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise ValueError("alpha must be a positive finite number")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise ValueError("beta must be a finite number")
    if not math.isfinite(float(beta)):
        raise ValueError("beta must be a finite number")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(normalized, bool):
        raise ValueError("normalized must be a bool")

    alpha = float(alpha)
    beta = float(beta)
    nodes = g.nodes
    n = len(nodes)
    if n == 0:
        return {}

    outgoing = {
        node: [(nbr, g.get_weight(node, nbr)) for nbr in g.neighbors(node)]
        for node in nodes
    }
    score: Dict[int, float] = {node: beta for node in nodes}

    converged = False
    for _ in range(max_iter):
        new_score: Dict[int, float] = {node: beta for node in nodes}
        for node in nodes:
            value = alpha * score[node]
            for nbr, weight in outgoing[node]:
                new_score[nbr] += value * weight
        if any(not math.isfinite(value) for value in new_score.values()):
            break
        delta = sum(abs(new_score[node] - score[node]) for node in nodes)
        score = new_score
        if delta < tol:
            converged = True
            break

    if not converged:
        raise ValueError(
            "Katz iteration failed to converge; decrease alpha "
            "(it must be smaller than 1/lambda_max) or increase max_iter"
        )

    if not normalized:
        return score

    norm = math.sqrt(sum(value * value for value in score.values()))
    if norm == 0.0:
        return score
    return {node: value / norm for node, value in score.items()}


__all__ = ["katz_centrality"]
