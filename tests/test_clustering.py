"""Tests for clustering coefficients."""
from __future__ import annotations

from graph_analytics_kit import Graph
from graph_analytics_kit.clustering import average_clustering, local_clustering


def _triangle() -> Graph:
    return Graph([(0, 1), (0, 2), (1, 2)])


def _star() -> Graph:
    return Graph([(0, 1), (0, 2), (0, 3)])


def _square() -> Graph:
    return Graph([(0, 1), (1, 2), (2, 3), (3, 0)])


def test_local_clustering_triangle() -> None:
    g = _triangle()
    lc = local_clustering(g)
    assert lc == {0: 1.0, 1: 1.0, 2: 1.0}


def test_local_clustering_star() -> None:
    g = _star()
    lc = local_clustering(g)
    assert lc[0] == 0.0  # center has 3 neighbors but 0 links between them


def test_local_clustering_square() -> None:
    g = _square()
    lc = local_clustering(g)
    # 4-cycle: each node has 2 neighbors, but they are NOT connected -> 0.0
    assert lc == {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}


def test_local_clustering_isolated_node() -> None:
    g = Graph([(0, 1), (0, 2), (1, 2)])
    g.add_node(3)
    lc = local_clustering(g)
    assert lc[3] == 0.0  # isolated node


def test_average_clustering_triangle() -> None:
    g = _triangle()
    assert average_clustering(g) == 1.0


def test_average_clustering_star() -> None:
    g = _star()
    # nodes 1,2,3 have 0.0, node 0 has 0.0 -> avg 0.0
    assert average_clustering(g) == 0.0


def test_average_clustering_square() -> None:
    g = _square()
    # 4-cycle has no triangles -> all coefficients 0
    assert average_clustering(g) == 0.0


def test_local_clustering_path() -> None:
    g = Graph([(0, 1), (1, 2)])
    lc = local_clustering(g)
    # node 0: 1 neighbor -> 0.0
    # node 1: 2 neighbors (0, 2) but no edge 0-2 -> 0.0
    # node 2: 1 neighbor -> 0.0
    assert lc == {0: 0.0, 1: 0.0, 2: 0.0}


def test_karate_club_single_node() -> None:
    g = Graph([(0, 1), (0, 2), (1, 2)])
    # Remove edges to leave isolated node 3
    g.add_node(3)
    lc = local_clustering(g)
    assert lc[3] == 0.0
