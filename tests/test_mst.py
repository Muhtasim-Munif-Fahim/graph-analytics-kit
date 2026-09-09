"""Tests for minimum-spanning-tree algorithms."""
from __future__ import annotations

import pytest

from graph_analytics_kit import Graph
from graph_analytics_kit.mst import kruskal_mst, mst_weight


def _triangle_weighted() -> Graph:
    return Graph([(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)])


def test_mst_returns_n_minus_one_edges() -> None:
    g = _triangle_weighted()
    mst = kruskal_mst(g)
    assert len(mst) == g.number_of_nodes() - 1


def test_mst_selects_smallest_edges() -> None:
    g = _triangle_weighted()
    mst = kruskal_mst(g)
    weights = sorted(w for _, _, w in mst)
    assert weights == [1.0, 2.0]


def test_mst_weight_matches_sum() -> None:
    g = _triangle_weighted()
    assert mst_weight(g) == pytest.approx(3.0)


def test_mst_no_cycle() -> None:
    g = Graph([
        (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0),
        (3, 0, 1.0), (0, 2, 2.0),
    ])
    mst = kruskal_mst(g)
    assert len(mst) == 3
    seen = set()
    for u, v, _ in mst:
        seen.add(u)
        seen.add(v)
    assert len(seen) == 4


def test_mst_disconnected_returns_forest() -> None:
    g = Graph([(0, 1, 1.0), (2, 3, 1.0)])
    mst = kruskal_mst(g)
    assert len(mst) == 2


def test_mst_single_node_returns_empty() -> None:
    g = Graph([(0, 0)])
    mst = kruskal_mst(g)
    assert mst == []


def test_mst_karate_club_weight_positive() -> None:
    from graph_analytics_kit import karate_club
    g = karate_club()
    assert mst_weight(g) > 0.0


def test_mst_empty_graph_returns_empty() -> None:
    g = Graph([])
    assert kruskal_mst(g) == []
    assert mst_weight(g) == 0.0
