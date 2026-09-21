"""HITS (hubs and authorities) centrality via Kleinberg iteration."""
from __future__ import annotations

import math
from typing import Dict, Tuple

from .graph import Graph

HubAuthScores = Tuple[Dict[int, float], Dict[int, float]]


def hits(
    g: Graph,
    *,
    max_iter: int = 100,
    tol: float = 1e-06,
) -> HubAuthScores:
    """Compute Kleinberg HITS hub and authority scores for every node in *g*.

    A node is a good **authority** if it is pointed to by good hubs, and a
    good **hub** if it points to good authorities. Each iteration updates

    * ``a = A^T h`` (authority from incoming hub weight)
    * ``h = A a`` (hub from outgoing authority weight)

    then L2-normalizes both vectors. Edge weights are used when present.
    Directed edges contribute only in the stated direction; on an undirected
    graph every edge is treated as bidirectional.

    Isolated nodes receive a score of 0.0 when any edge exists. An empty
    graph returns two empty dicts. A graph with no edges returns a uniform
    unit vector for both hubs and authorities.

    Parameters
    ----------
    g:
        The input graph.
    max_iter:
        Maximum number of HITS iterations.
    tol:
        Convergence tolerance on the summed L1 residual of both vectors.

    Returns
    -------
    tuple[dict[int, float], dict[int, float]]
        ``(hubs, authorities)`` mapping node labels to L2-normalized scores
        (within rounding) when the graph is non-empty.
    """
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    nodes = g.nodes
    n = len(nodes)
    if n == 0:
        return {}, {}
    if n == 1:
        return {nodes[0]: 1.0}, {nodes[0]: 1.0}

    outgoing = {
        node: [(nbr, g.get_weight(node, nbr)) for nbr in g.neighbors(node)]
        for node in nodes
    }

    inv_norm = 1.0 / math.sqrt(n)
    hubs: Dict[int, float] = {node: inv_norm for node in nodes}
    authorities: Dict[int, float] = {node: inv_norm for node in nodes}

    for _ in range(max_iter):
        new_auth: Dict[int, float] = {node: 0.0 for node in nodes}
        for node in nodes:
            value = hubs[node]
            for nbr, weight in outgoing[node]:
                new_auth[nbr] += value * weight

        new_hubs: Dict[int, float] = {node: 0.0 for node in nodes}
        for node in nodes:
            for nbr, weight in outgoing[node]:
                new_hubs[node] += new_auth[nbr] * weight

        new_auth = _l2_normalize(new_auth, inv_norm)
        new_hubs = _l2_normalize(new_hubs, inv_norm)

        delta = sum(abs(new_auth[node] - authorities[node]) for node in nodes)
        delta += sum(abs(new_hubs[node] - hubs[node]) for node in nodes)
        hubs, authorities = new_hubs, new_auth
        if delta < tol:
            break

    return hubs, authorities


def _l2_normalize(scores: Dict[int, float], inv_norm: float) -> Dict[int, float]:
    norm = math.sqrt(sum(value * value for value in scores.values()))
    if norm == 0.0:
        return {node: inv_norm for node in scores}
    return {node: value / norm for node, value in scores.items()}


__all__ = ["hits"]
