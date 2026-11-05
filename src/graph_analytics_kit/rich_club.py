"""Rich-club coefficient for undirected graphs."""
from __future__ import annotations

from typing import Dict, Optional

from .graph import Graph

__all__ = ["rich_club_coefficient"]


def rich_club_coefficient(g: Graph, k: Optional[int] = None) -> Dict[int, float] | float:
    """Return the (unnormalized) rich-club coefficient of *g*.

    For a degree threshold ``k`` let ``N_k`` be the number of nodes with
    degree strictly greater than ``k`` and ``E_k`` the number of edges among
    them. The rich-club coefficient (Zhou & Mondragón, 2004) is

    .. math::

        \\phi(k) = \\frac{2 E_k}{N_k (N_k - 1)}

    i.e. the edge density of the subgraph induced by the "rich" nodes. Values
    close to ``1`` mean high-degree nodes form a tightly-knit core.

    When *k* is ``None`` a dict mapping every threshold ``0 .. max_degree - 2``
    (those with at least two rich nodes) to ``phi(k)`` is returned; otherwise
    the single float for that threshold. Thresholds with fewer than two rich
    nodes yield ``0.0``. Self-loops are ignored. Only undirected graphs are
    supported.
    """
    if g.directed:
        raise ValueError("rich_club_coefficient requires an undirected graph")

    degree = {n: sum(1 for v in g.neighbors(n) if v != n) for n in g.nodes}
    edges = [(u, v) for u, v, _w in g.edges if u != v]

    def phi(threshold: int) -> float:
        rich = {n for n, d in degree.items() if d > threshold}
        n_k = len(rich)
        if n_k < 2:
            return 0.0
        e_k = sum(1 for u, v in edges if u in rich and v in rich)
        return 2.0 * e_k / (n_k * (n_k - 1))

    if k is not None:
        if int(k) != k or k < 0:
            raise ValueError("k must be a non-negative integer")
        return phi(int(k))

    out: Dict[int, float] = {}
    max_degree = max(degree.values(), default=0)
    for threshold in range(max_degree):
        if sum(1 for d in degree.values() if d > threshold) < 2:
            break
        out[threshold] = phi(threshold)
    return out
