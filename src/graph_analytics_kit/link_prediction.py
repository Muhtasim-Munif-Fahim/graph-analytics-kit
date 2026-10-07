"""Link-prediction scores for undirected graphs."""
from __future__ import annotations

import math

import numpy as np
from typing import Iterable, List, Tuple

from .graph import Graph

__all__ = ["adamic_adar", "adamic_adar_scores", "jaccard", "jaccard_scores", "resource_allocation", "resource_allocation_scores", "preferential_attachment", "preferential_attachment_scores", "common_neighbors", "common_neighbors_scores", "katz_index", "katz_index_scores", "hub_promoted_index", "hub_promoted_index_scores"]

Pair = Tuple[int, int]


def _neighbor_sets(g: Graph, u: int, v: int) -> Tuple[set, set]:
    """Return neighbor sets of *u* and *v* excluding the endpoints themselves."""
    nu = set(g.neighbors(u))
    nv = set(g.neighbors(v))
    # Self-loops / endpoints are not part of the neighborhood for link prediction.
    nu.discard(u)
    nu.discard(v)
    nv.discard(u)
    nv.discard(v)
    return nu, nv


def _common_neighbors(g: Graph, u: int, v: int) -> List[int]:
    """Return common neighbors of *u* and *v* (excluding u and v)."""
    nu, nv = _neighbor_sets(g, u, v)
    return sorted(nu & nv)


def adamic_adar(g: Graph, u: int, v: int) -> float:
    """Return the Adamic–Adar index between nodes *u* and *v*.

    The score sums ``1 / log(deg(w))`` over common neighbors ``w`` of ``u``
    and ``v`` (Adamic & Adar, 2003). Neighbors with degree ``<= 1`` are
    skipped because ``log(1) = 0``. Higher scores indicate a stronger
    predicted link. The score is symmetric: ``adamic_adar(g, u, v) ==
    adamic_adar(g, v, u)``. When *u* or *v* is missing from *g*, or when
    the pair shares no eligible common neighbors, the score is ``0.0``.

    Parameters
    ----------
    g:
        Input graph. Directed graphs are treated via each node's
        out-neighborhood (same convention as :func:`degree_assortativity`).
    u, v:
        Node labels.

    Returns
    -------
    float
        Non-negative Adamic–Adar score.
    """
    nodes = set(g.nodes)
    if u not in nodes or v not in nodes:
        return 0.0
    score = 0.0
    for w in _common_neighbors(g, u, v):
        deg = g.degree(w)
        if deg <= 1:
            continue
        score += 1.0 / math.log(float(deg))
    return score


def adamic_adar_scores(
    g: Graph,
    pairs: Iterable[Pair],
) -> List[float]:
    """Compute Adamic–Adar scores for many node pairs.

    Parameters
    ----------
    g:
        Input graph.
    pairs:
        Iterable of ``(u, v)`` node-label pairs.

    Returns
    -------
    list[float]
        Scores in the same order as *pairs* (materialised as a list so
        callers can zip against the input).
    """
    pair_list: List[Pair] = [(int(u), int(v)) for u, v in pairs]
    return [adamic_adar(g, u, v) for u, v in pair_list]


def jaccard(g: Graph, u: int, v: int) -> float:
    """Return the Jaccard coefficient between nodes *u* and *v*.

    The score is ``|N(u) ∩ N(v)| / |N(u) ∪ N(v)|`` where ``N(·)`` is the
    open neighborhood (endpoints themselves are excluded). When the union
    is empty the score is ``0.0``. The score is symmetric:
    ``jaccard(g, u, v) == jaccard(g, v, u)``. Missing nodes also yield
    ``0.0``.

    Parameters
    ----------
    g:
        Input graph. Directed graphs use each node's out-neighborhood.
    u, v:
        Node labels.

    Returns
    -------
    float
        Jaccard coefficient in ``[0, 1]``.
    """
    nodes = set(g.nodes)
    if u not in nodes or v not in nodes:
        return 0.0
    nu, nv = _neighbor_sets(g, u, v)
    union = nu | nv
    if not union:
        return 0.0
    return float(len(nu & nv)) / float(len(union))


def jaccard_scores(
    g: Graph,
    pairs: Iterable[Pair],
) -> List[float]:
    """Compute Jaccard coefficients for many node pairs.

    Parameters
    ----------
    g:
        Input graph.
    pairs:
        Iterable of ``(u, v)`` node-label pairs.

    Returns
    -------
    list[float]
        Scores in the same order as *pairs*.
    """
    pair_list: List[Pair] = [(int(u), int(v)) for u, v in pairs]
    return [jaccard(g, u, v) for u, v in pair_list]


def resource_allocation(g: Graph, u: int, v: int) -> float:
    """Return the Resource Allocation index between nodes *u* and *v*.

    The score sums ``1 / deg(w)`` over common neighbors ``w`` of ``u`` and
    ``v`` (Zhou, Lü & Zhang, 2009). Neighbors with degree ``0`` are skipped.
    Higher scores indicate a stronger predicted link. The score is symmetric:
    ``resource_allocation(g, u, v) == resource_allocation(g, v, u)``. When
    *u* or *v* is missing from *g*, or when the pair shares no eligible
    common neighbors, the score is ``0.0``.

    Parameters
    ----------
    g:
        Input graph. Directed graphs are treated via each node's
        out-neighborhood (same convention as :func:`adamic_adar`).
    u, v:
        Node labels.

    Returns
    -------
    float
        Non-negative Resource Allocation score.
    """
    nodes = set(g.nodes)
    if u not in nodes or v not in nodes:
        return 0.0
    score = 0.0
    for w in _common_neighbors(g, u, v):
        deg = g.degree(w)
        if deg <= 0:
            continue
        score += 1.0 / float(deg)
    return score


def resource_allocation_scores(
    g: Graph,
    pairs: Iterable[Pair],
) -> List[float]:
    """Compute Resource Allocation scores for many node pairs.

    Parameters
    ----------
    g:
        Input graph.
    pairs:
        Iterable of ``(u, v)`` node-label pairs.

    Returns
    -------
    list[float]
        Scores in the same order as *pairs*.
    """
    pair_list: List[Pair] = [(int(u), int(v)) for u, v in pairs]
    return [resource_allocation(g, u, v) for u, v in pair_list]


def preferential_attachment(g: Graph, u: int, v: int) -> float:
    """Return the Preferential Attachment score between nodes *u* and *v*.

    The score is ``deg(u) * deg(v)`` (Barabási–Albert / Newman preferential
    attachment index). Higher scores indicate that high-degree endpoints are
    more likely to form a link. The score is symmetric:
    ``preferential_attachment(g, u, v) == preferential_attachment(g, v, u)``.
    When *u* or *v* is missing from *g*, the score is ``0.0``.

    Parameters
    ----------
    g:
        Input graph. Directed graphs use each node's out-degree (same
        convention as :func:`resource_allocation`).
    u, v:
        Node labels.

    Returns
    -------
    float
        Non-negative Preferential Attachment score.
    """
    nodes = set(g.nodes)
    if u not in nodes or v not in nodes:
        return 0.0
    return float(g.degree(u)) * float(g.degree(v))


def preferential_attachment_scores(
    g: Graph,
    pairs: Iterable[Pair],
) -> List[float]:
    """Compute Preferential Attachment scores for many node pairs.

    Parameters
    ----------
    g:
        Input graph.
    pairs:
        Iterable of ``(u, v)`` node-label pairs.

    Returns
    -------
    list[float]
        Scores in the same order as *pairs*.
    """
    pair_list: List[Pair] = [(int(u), int(v)) for u, v in pairs]
    return [preferential_attachment(g, u, v) for u, v in pair_list]


def common_neighbors(g: Graph, u: int, v: int) -> float:
    """Return the Common Neighbors count between nodes *u* and *v*.

    The score is ``|N(u) ∩ N(v)|`` where ``N(·)`` is the open neighborhood
    (endpoints themselves are excluded). Higher scores indicate a stronger
    predicted link (Liben-Nowell & Kleinberg, 2007). The score is symmetric:
    ``common_neighbors(g, u, v) == common_neighbors(g, v, u)``. When *u* or
    *v* is missing from *g*, or when the pair shares no common neighbors,
    the score is ``0.0``.

    Parameters
    ----------
    g:
        Input graph. Directed graphs use each node's out-neighborhood
        (same convention as :func:`jaccard`).
    u, v:
        Node labels.

    Returns
    -------
    float
        Non-negative Common Neighbors count (as a float for API symmetry
        with the other link-prediction scores).
    """
    nodes = set(g.nodes)
    if u not in nodes or v not in nodes:
        return 0.0
    return float(len(_common_neighbors(g, u, v)))


def common_neighbors_scores(
    g: Graph,
    pairs: Iterable[Pair],
) -> List[float]:
    """Compute Common Neighbors scores for many node pairs.

    Parameters
    ----------
    g:
        Input graph.
    pairs:
        Iterable of ``(u, v)`` node-label pairs.

    Returns
    -------
    list[float]
        Scores in the same order as *pairs*.
    """
    pair_list: List[Pair] = [(int(u), int(v)) for u, v in pairs]
    return [common_neighbors(g, u, v) for u, v in pair_list]


def _adjacency_matrix(g: Graph) -> tuple[list[int], "np.ndarray"]:
    """Return ``(node_order, A)`` for an unweighted adjacency matrix."""
    nodes = sorted(g.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=float)
    for u in nodes:
        for v in g.neighbors(u):
            if v in index:
                A[index[u], index[v]] = 1.0
    # Ensure undirected symmetry if the graph stores one-way undirected edges.
    A = np.maximum(A, A.T)
    return nodes, A


def katz_index(
    g: Graph,
    u: int,
    v: int,
    *,
    beta: float = 0.05,
) -> float:
    """Return the Katz link-prediction index between *u* and *v*.

    Katz (1953) scores a pair by the discounted number of walks of every
    length ``l >= 1`` between them:

    ``score(u, v) = sum_{l=1}^∞ β^l (A^l)_{uv}``

    which equals ``((I - β A)^{-1} - I)_{uv}``. Unlike
    :func:`katz_centrality` (a node prestige score), this is a *pairwise*
    link predictor. Higher scores indicate a stronger predicted link. The
    score is symmetric on undirected graphs. Missing nodes yield ``0.0``.
    ``beta`` must lie in ``(0, 1/ρ(A))``; a conservative default of ``0.05``
    works for sparse social graphs.
    """
    nodes = set(g.nodes)
    if u not in nodes or v not in nodes:
        return 0.0
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie strictly inside (0, 1)")
    order, A = _adjacency_matrix(g)
    index = {node: i for i, node in enumerate(order)}
    n = A.shape[0]
    eye = np.eye(n)
    try:
        resolvent = np.linalg.inv(eye - beta * A)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Katz resolvent is singular; try a smaller beta"
        ) from exc
    S = resolvent - eye
    return float(S[index[u], index[v]])


def katz_index_scores(
    g: Graph,
    pairs: Iterable[Pair],
    *,
    beta: float = 0.05,
) -> List[float]:
    """Compute Katz link-prediction scores for many node pairs.

    The resolvent is factored once and reused for every pair.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie strictly inside (0, 1)")
    pair_list: List[Pair] = [(int(a), int(b)) for a, b in pairs]
    if not pair_list:
        return []
    nodes = set(g.nodes)
    order, A = _adjacency_matrix(g)
    index = {node: i for i, node in enumerate(order)}
    n = A.shape[0]
    eye = np.eye(n)
    try:
        resolvent = np.linalg.inv(eye - beta * A)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Katz resolvent is singular; try a smaller beta"
        ) from exc
    S = resolvent - eye
    out: List[float] = []
    for u, v in pair_list:
        if u not in nodes or v not in nodes:
            out.append(0.0)
        else:
            out.append(float(S[index[u], index[v]]))
    return out


def hub_promoted_index(g: Graph, u: int, v: int) -> float:
    """Return the Hub Promoted Index between nodes *u* and *v*.

    ``HPI(u, v) = |N(u) ∩ N(v)| / min(deg(u), deg(v))`` (Zhou, Lü & Zhang,
    2009). The denominator favours links involving hubs. When either degree
    is zero or a node is missing, the score is ``0.0``. The score is
    symmetric.
    """
    nodes = set(g.nodes)
    if u not in nodes or v not in nodes:
        return 0.0
    du = g.degree(u)
    dv = g.degree(v)
    denom = min(du, dv)
    if denom <= 0:
        return 0.0
    return float(len(_common_neighbors(g, u, v))) / float(denom)


def hub_promoted_index_scores(
    g: Graph,
    pairs: Iterable[Pair],
) -> List[float]:
    """Compute Hub Promoted Index scores for many node pairs."""
    pair_list: List[Pair] = [(int(a), int(b)) for a, b in pairs]
    return [hub_promoted_index(g, u, v) for u, v in pair_list]

