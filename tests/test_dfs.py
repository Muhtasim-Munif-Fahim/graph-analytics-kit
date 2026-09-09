"""Tests for depth-first traversal algorithms."""
from __future__ import annotations

from graph_analytics_kit import Graph, karate_club
from graph_analytics_kit.dfs import dfs_path, dfs_traversal


def _triangle() -> Graph:
    return Graph([(0, 1), (0, 2), (1, 2)])


def test_dfs_traversal_visits_all_nodes() -> None:
    g = _triangle()
    visited = dfs_traversal(g, 0)
    assert sorted(visited) == [0, 1, 2]


def test_dfs_traversal_starts_at_given_node() -> None:
    g = _triangle()
    visited = dfs_traversal(g, 1)
    assert visited[0] == 1


def test_dfs_traversal_target_early_exit() -> None:
    g = _triangle()
    visited = dfs_traversal(g, 0, target=2)
    assert visited[-1] == 2


def test_dfs_traversal_max_depth_limits() -> None:
    g = Graph([(0, 1), (1, 2), (2, 3)])
    visited = dfs_traversal(g, 0, max_depth=1)
    assert 3 not in visited


def test_dfs_traversal_disconnected_returns_component() -> None:
    g = Graph([(0, 1), (2, 3)])
    visited = dfs_traversal(g, 0)
    assert visited == [0, 1]


def test_dfs_path_returns_sequence() -> None:
    g = Graph([(0, 1), (1, 2), (0, 2)])
    path = dfs_path(g, 0, 2)
    assert path is not None
    assert path[0] == 0
    assert path[-1] == 2


def test_dfs_path_unreachable_returns_none() -> None:
    g = Graph([(0, 1), (2, 3)])
    assert dfs_path(g, 0, 3) is None


def test_dfs_path_same_node_returns_singleton() -> None:
    g = _triangle()
    assert dfs_path(g, 0, 0) == [0]


def test_dfs_path_max_depth_blocks_deep_path() -> None:
    g = Graph([(0, 1), (1, 2), (2, 3)])
    assert dfs_path(g, 0, 3, max_depth=1) is None


def test_dfs_traversal_karate_club_visits_all() -> None:
    g = karate_club()
    visited = dfs_traversal(g, 0)
    assert len(visited) == g.number_of_nodes()
    assert visited[0] == 0
