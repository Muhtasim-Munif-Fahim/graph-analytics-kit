"""SimRank structural similarity (Jeh & Widom, KDD 2002).

"Two objects are similar if they are referenced by similar objects." SimRank
is the fixed point of

    s(a, a) = 1
    s(a, b) = C / (|I(a)| |I(b)|) * sum_{i in I(a)} sum_{j in I(b)} s(i, j)

where ``I(x)`` are the in-neighbours of ``x`` (all neighbours on an undirected
graph) and ``C`` in ``(0, 1)`` is the decay factor. A node with no
in-neighbours is similar only to itself. In matrix form, with ``W`` the
column-normalised adjacency matrix, each sweep computes
``S <- C W^T S W`` and then resets the diagonal to one. The iterates increase
monotonically to the fixed point and the error after ``k`` sweeps is at most
``C^(k+1)``.

Unlike neighbourhood-overlap indices (common neighbours, Jaccard, Adamic-Adar),
SimRank is non-zero for pairs at distance greater than two and for pairs
with no common neighbour. That makes it a useful link predictor and
role-similarity measure, e.g. on bipartite graphs.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .graph import Graph

Pair = Tuple[int, int]


def _validate(decay: float, max_iter: int, tol: float) -> None:
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must lie strictly inside (0, 1)")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be at least 1")
    if not tol > 0.0:
        raise ValueError("tol must be positive")


def simrank_matrix(
    g: Graph,
    *,
    decay: float = 0.8,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[List[int], np.ndarray]:
    """Return ``(node_order, S)`` with the full SimRank similarity matrix.

    ``node_order`` follows :meth:`Graph.node_labels`, and ``S[i, j]`` is the
    similarity of ``node_order[i]`` and ``node_order[j]``. Edge weights are
    ignored, as in the original definition. Directed graphs use in-neighbours.
    Iteration stops once the largest entry change drops below ``tol`` or
    after ``max_iter`` sweeps. The result is symmetric with a unit diagonal and
    entries in ``[0, 1]``.
    """
    _validate(decay, max_iter, tol)
    nodes = g.node_labels()
    n = len(nodes)
    if n == 0:
        return nodes, np.zeros((0, 0))
    A = (g.adjacency_matrix() != 0).astype(float)
    in_degree = A.sum(axis=0)
    W = np.divide(A, in_degree, out=np.zeros_like(A), where=in_degree > 0)
    S = np.eye(n)
    for _ in range(int(max_iter)):
        new = decay * (W.T @ S @ W)
        np.fill_diagonal(new, 1.0)
        delta = float(np.max(np.abs(new - S)))
        S = new
        if delta < tol:
            break
    S = 0.5 * (S + S.T)  # remove round-off asymmetry
    return nodes, S


def simrank_similarity(
    g: Graph,
    source: Optional[int] = None,
    target: Optional[int] = None,
    *,
    decay: float = 0.8,
    max_iter: int = 100,
    tol: float = 1e-6,
):
    """SimRank similarities in dictionary form.

    * ``source`` and ``target`` both given: the float ``s(source, target)``.
    * only ``source``: ``{node: s(source, node)}``.
    * neither: ``{u: {v: s(u, v)}}`` for every pair.

    Unknown nodes raise ``KeyError``.
    """
    nodes, S = simrank_matrix(g, decay=decay, max_iter=max_iter, tol=tol)
    index = {node: i for i, node in enumerate(nodes)}
    for label, node in (("source", source), ("target", target)):
        if node is not None and node not in index:
            raise KeyError(f"{label} node {node} is not in the graph")
    if target is not None and source is None:
        raise ValueError("pass source together with target")
    if source is not None and target is not None:
        return float(S[index[source], index[target]])
    if source is not None:
        row = S[index[source]]
        return {node: float(row[i]) for i, node in enumerate(nodes)}
    return {u: {v: float(S[i, j]) for j, v in enumerate(nodes)} for i, u in enumerate(nodes)}


def simrank_scores(
    g: Graph,
    pairs: Iterable[Pair],
    *,
    decay: float = 0.8,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> List[float]:
    """SimRank link-prediction scores for many pairs (matrix computed once).

    Pairs that involve a node missing from the graph score ``0.0``, matching
    the other link-prediction helpers.
    """
    pair_list: List[Pair] = [(int(a), int(b)) for a, b in pairs]
    if not pair_list:
        return []
    nodes, S = simrank_matrix(g, decay=decay, max_iter=max_iter, tol=tol)
    index = {node: i for i, node in enumerate(nodes)}
    out: List[float] = []
    for u, v in pair_list:
        if u not in index or v not in index:
            out.append(0.0)
        else:
            out.append(float(S[index[u], index[v]]))
    return out


def most_similar(
    g: Graph,
    node: int,
    k: int = 5,
    *,
    decay: float = 0.8,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> List[Tuple[int, float]]:
    """The ``k`` nodes most SimRank-similar to ``node`` (excluding itself).

    Ties break by node label, so the ranking is deterministic.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    row: Dict[int, float] = simrank_similarity(
        g, node, decay=decay, max_iter=max_iter, tol=tol
    )
    ranked = sorted(
        ((other, score) for other, score in row.items() if other != node),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[:k]
