"""Degree assortativity (Newman) for undirected graphs."""
from __future__ import annotations

from .graph import Graph

__all__ = ["degree_assortativity"]


def degree_assortativity(g: Graph) -> float:
    """Return Newman's degree assortativity coefficient for *g*.

    Assortativity measures the Pearson correlation of degrees across the two
    ends of every edge. Positive values mean high-degree nodes preferentially
    connect to other high-degree nodes (assortative mixing); negative values
    mean high-degree nodes connect to low-degree nodes (disassortative).

    For an undirected graph with ``m`` edges the coefficient is

    .. math::

        r = \\frac{
            \\frac{1}{m}\\sum_{(u,v)\\in E} d_u d_v
            - \\mu^2
        }{
            \\frac{1}{2m}\\sum_{(u,v)\\in E}(d_u^2 + d_v^2)
            - \\mu^2
        }

    where :math:`\\mu = \\frac{1}{2m}\\sum_{(u,v)\\in E}(d_u + d_v)`.

    Returns ``0.0`` when the graph has fewer than one edge or when the
    degree variance across edge ends is zero (undefined correlation).
    Directed graphs are treated via the undirected formula using each node's
    out-neighborhood size as its degree.
    """
    edges = g.edges
    m = len(edges)
    if m < 1:
        return 0.0

    xy = 0.0
    x_plus_y = 0.0
    x2_plus_y2 = 0.0
    for u, v, _w in edges:
        du = float(g.degree(u))
        dv = float(g.degree(v))
        xy += du * dv
        x_plus_y += du + dv
        x2_plus_y2 += du * du + dv * dv

    mu = x_plus_y / (2.0 * m)
    numer = xy / m - mu * mu
    denom = x2_plus_y2 / (2.0 * m) - mu * mu
    if abs(denom) < 1e-15:
        return 0.0
    return numer / denom
