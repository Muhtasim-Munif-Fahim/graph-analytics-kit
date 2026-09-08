"""Clustering coefficients for undirected graphs."""
from __future__ import annotations

from .graph import Graph


def local_clustering(g: Graph) -> dict[int, float]:
    """Return the local clustering coefficient for each node.

    The local clustering coefficient for a node *v* is the fraction of pairs
    of *v*'s neighbors that are themselves connected:

    .. math::
        C_v = \\frac{2 \\cdot |\\{e_{ij}\\}|}{k_v (k_v - 1)}

    where :math:`k_v` is the degree of *v* and the denominator counts the
    number of possible edges among *v*'s neighbors. Nodes with fewer than
    two neighbors receive a coefficient of 0.0.
    """
    result: dict[int, float] = {}
    for node in g.nodes:
        neighbors = g.neighbors(node)
        k = len(neighbors)
        if k < 2:
            result[node] = 0.0
            continue
        links = 0
        for i, n_i in enumerate(neighbors):
            for n_j in neighbors[i + 1:]:
                if g.has_edge(n_i, n_j):
                    links += 1
        possible = k * (k - 1) / 2
        result[node] = links / possible
    return result


def average_clustering(g: Graph) -> float:
    """Return the average local clustering coefficient across all nodes."""
    coefficients = local_clustering(g)
    if not coefficients:
        return 0.0
    return sum(coefficients.values()) / len(coefficients)


__all__ = ["local_clustering", "average_clustering"]
