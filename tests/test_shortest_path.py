"""Tests for Dijkstra shortest-path algorithms."""
from __future__ import annotations

import pytest

from graph_analytics_kit import Graph
from graph_analytics_kit.shortest_path import (
    dijkstra_shortest_path,
    reconstruct_path,
)


def _weighted_line() -> Graph:
    return Graph([(0, 1, 2.0), (1, 2, 3.0)], directed=False)


def test_distances_on_simple_path() -> None:
    g = _weighted_line()
    dist, _ = dijkstra_shortest_path(g, 0)
    assert dist == {0: 0.0, 1: 2.0, 2: 5.0}


def test_predecessors_reconstruct_path() -> None:
    g = _weighted_line()
    _, prev = dijkstra_shortest_path(g, 0)
    path = reconstruct_path(prev, 0, 2)
    assert path == [0, 1, 2]


def test_target_early_exit() -> None:
    g = _weighted_line()
    dist, _ = dijkstra_shortest_path(g, 0, target=1)
    assert dist == {0: 0.0, 1: 2.0}


def test_unreachable_node_excluded() -> None:
    g = Graph([(0, 1), (2, 3)])
    dist, _ = dijkstra_shortest_path(g, 0)
    assert 0 in dist
    assert 1 in dist
    assert 2 not in dist
    assert 3 not in dist


def test_reconstruct_unreachable_raises() -> None:
    g = Graph([(0, 1), (2, 3)])
    _, prev = dijkstra_shortest_path(g, 0)
    with pytest.raises(KeyError):
        reconstruct_path(prev, 0, 2)


def test_directed_graph_respects_edges() -> None:
    g = Graph([(0, 1, 1.0), (1, 2, 1.0)], directed=True)
    dist, _ = dijkstra_shortest_path(g, 0)
    assert dist == {0: 0.0, 1: 1.0, 2: 2.0}


def test_directed_graph_no_reverse_path() -> None:
    g = Graph([(0, 1, 1.0)], directed=True)
    dist, _ = dijkstra_shortest_path(g, 1)
    assert dist == {1: 0.0}
    assert 0 not in dist


def test_karate_club_source_zero_reaches_all() -> None:
    from graph_analytics_kit import karate_club

    g = karate_club()
    dist, _ = dijkstra_shortest_path(g, 0)
    assert 0 in dist
    assert len(dist) == g.number_of_nodes()


def test_source_distance_is_zero() -> None:
    g = _weighted_line()
    dist, _ = dijkstra_shortest_path(g, 0)
    assert dist[0] == 0.0


def test_path_to_self_is_single_node() -> None:
    g = _weighted_line()
    _, prev = dijkstra_shortest_path(g, 0)
    assert reconstruct_path(prev, 0, 0) == [0]
