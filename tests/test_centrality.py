"""Tests for centrality measures."""
from __future__ import annotations

import numpy as np

from graph_analytics_kit import Graph
from graph_analytics_kit.centrality import (
    betweenness_centrality,
    closeness_centrality,
    degree_centrality,
    degree_centrality_array,
)


def _triangle() -> Graph:
    return Graph([(0, 1), (0, 2), (1, 2)])


def _star() -> Graph:
    return Graph([(0, 1), (0, 2), (0, 3)])


def _path() -> Graph:
    return Graph([(0, 1), (1, 2), (2, 3)])


def test_degree_centrality_triangle() -> None:
    g = _triangle()
    dc = degree_centrality(g)
    assert dc == {0: 2, 1: 2, 2: 2}


def test_degree_centrality_star() -> None:
    g = _star()
    dc = degree_centrality(g)
    assert dc[0] == 3
    assert dc[1] == 1
    assert dc[2] == 1
    assert dc[3] == 1


def test_degree_centrality_array() -> None:
    g = _star()
    arr = degree_centrality_array(g)
    np.testing.assert_allclose(arr, [3, 1, 1, 1])


def test_closeness_centrality_star() -> None:
    g = _star()
    cc = closeness_centrality(g)
    assert cc[0] == 3 / 3  # center: (n-1)/sum_dist = 3/3 = 1.0
    assert cc[1] == 3 / 5  # leaf: dists = 1+2 = 3, wait
    # For star: node 0 has dists {1:1, 2:1, 3:1} sum=3, (n-1)/3 = 1.0
    # node 1 has dists {0:1, 2:2, 3:2} sum=5, (n-1)/5 = 0.6
    assert abs(cc[0] - 1.0) < 1e-9
    assert abs(cc[1] - 0.6) < 1e-9


def test_closeness_centrality_path() -> None:
    g = _path()
    cc = closeness_centrality(g)
    # Node 0: dists {1:1,2:2,3:3} sum=6, (n-1)/6 = 3/6 = 0.5
    assert abs(cc[0] - 0.5) < 1e-9
    # Node 1: dists {0:1,2:1,3:2} sum=4, (n-1)/4 = 3/4 = 0.75
    assert abs(cc[1] - 0.75) < 1e-9


def test_betweenness_centrality_path() -> None:
    g = _path()
    bc = betweenness_centrality(g)
    # In path 0-1-2-3, only nodes 1 and 2 are on shortest paths
    # Node 1: on paths 0->2 (1/1), 0->3 (1/1), so 2.0 unnormalized
    # Normalized: 2.0 * 2/((3*2)) = 2/3
    assert bc[0] == 0.0
    assert bc[3] == 0.0
    assert abs(bc[1] - 2 / 3) < 1e-9
    assert abs(bc[2] - 2 / 3) < 1e-9


def test_betweenness_centrality_star() -> None:
    g = _star()
    bc = betweenness_centrality(g)
    # Center node 0 is on all paths between leaves
    # Leaves: 1,2,3 -> 3 pairs of leaves, each shortest path goes through 0
    # Unnormalized: 3.0 (for center), 0.0 for leaves
    # Normalized: 3.0 * 2/(3*2) = 1.0
    assert abs(bc[0] - 1.0) < 1e-9
    assert bc[1] == 0.0
    assert bc[2] == 0.0
    assert bc[3] == 0.0


def test_betweenness_centrality_triangle() -> None:
    g = _triangle()
    bc = betweenness_centrality(g)
    # In a triangle, no node is on a shortest path through another
    # All betweenness = 0
    assert bc[0] == 0.0
    assert bc[1] == 0.0
    assert bc[2] == 0.0


def test_betweenness_unnormalized() -> None:
    g = _path()
    bc = betweenness_centrality(g, normalized=False)
    assert bc[0] == 0.0
    assert bc[3] == 0.0
    assert bc[1] == 2.0
    assert bc[2] == 2.0


def test_closeness_centrality_single_node() -> None:
    g = Graph()
    cc = closeness_centrality(g)
    assert cc == {}
