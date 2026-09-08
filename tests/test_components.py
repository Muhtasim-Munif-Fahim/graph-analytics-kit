"""Tests for connected components and BFS."""
from __future__ import annotations

from graph_analytics_kit import Graph


def test_single_component() -> None:
    g = Graph([(0, 1), (1, 2), (2, 0)])
    components = g.connected_components()
    assert len(components) == 1
    assert set(components[0]) == {0, 1, 2}


def test_two_components() -> None:
    g = Graph([(0, 1), (2, 3)])
    components = g.connected_components()
    assert len(components) == 2
    comp_sets = [set(c) for c in components]
    assert {0, 1} in comp_sets
    assert {2, 3} in comp_sets


def test_isolated_nodes_are_own_components() -> None:
    g = Graph([(0, 1)])
    g.add_node(2)
    g.add_node(3)
    components = g.connected_components()
    assert len(components) == 3
    comp_sets = [set(c) for c in components]
    assert {0, 1} in comp_sets
    assert {2} in comp_sets
    assert {3} in comp_sets


def test_empty_graph_components() -> None:
    g = Graph()
    assert g.connected_components() == []


def test_bfs_distances() -> None:
    g = Graph([(0, 1), (1, 2), (2, 3)])
    dist = g.bfs_distances(0)
    assert dist == {0: 0, 1: 1, 2: 2, 3: 3}


def test_bfs_distances_disconnected() -> None:
    g = Graph([(0, 1), (2, 3)])
    dist = g.bfs_distances(0)
    assert dist == {0: 0, 1: 1}


def test_bfs_distances_star() -> None:
    g = Graph([(0, 1), (0, 2), (0, 3)])
    dist = g.bfs_distances(1)
    assert dist == {1: 0, 0: 1, 2: 2, 3: 2}


def test_karate_club_single_component() -> None:
    g = Graph.karate_club()
    components = g.connected_components()
    assert len(components) == 1
    assert len(components[0]) == 34
