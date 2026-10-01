"""Link-prediction scores for undirected graphs."""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from .graph import Graph

__all__ = ["adamic_adar", "adamic_adar_scores", "jaccard", "jaccard_scores"]

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
