"""Tests for Girvan–Newman community detection."""
from __future__ import annotations

import pytest

from graph_analytics_kit import Graph, girvan_newman, karate_club, modularity
from graph_analytics_kit.cli import main
from graph_analytics_kit.community import girvan_newman as girvan_newman_impl


def _clique(nodes: list[int]) -> list[tuple[int, int]]:
    return [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]


def _two_cliques(*, bridge: bool) -> Graph:
    edges = _clique([0, 1, 2, 3, 4]) + _clique([5, 6, 7, 8, 9])
    if bridge:
        edges.append((4, 5))
    return Graph(edges)


def test_girvan_newman_exported_from_public_api() -> None:
    assert girvan_newman is girvan_newman_impl
    g = Graph([(0, 1), (1, 2), (2, 0)])
    parts = girvan_newman(g)
    assert set(n for part in parts for n in part) == {0, 1, 2}


def test_empty_graph_returns_empty() -> None:
    assert girvan_newman(Graph([])) == []


def test_single_node_is_its_own_community() -> None:
    g = Graph()
    g.add_node(0)
    assert girvan_newman(g) == [[0]]


def test_disconnected_cliques_are_two_communities() -> None:
    g = _two_cliques(bridge=False)
    parts = girvan_newman(g, n_communities=2)
    assert [set(part) for part in parts] == [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]
    assert modularity(g, parts) > 0.4


def test_bridged_cliques_split_on_bridge() -> None:
    g = _two_cliques(bridge=True)
    parts = girvan_newman(g, n_communities=2)
    assert [set(part) for part in parts] == [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]
    assert modularity(g, parts) > 0.4


def test_best_modularity_recovers_two_blocks() -> None:
    g = _two_cliques(bridge=True)
    parts = girvan_newman(g)
    assert [set(part) for part in parts] == [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]


def test_complete_graph_stays_one_or_splits_late() -> None:
    g = Graph(_clique([0, 1, 2, 3]))
    parts = girvan_newman(g, n_communities=1)
    assert parts == [[0, 1, 2, 3]]


def test_isolated_node_stays_separate() -> None:
    g = Graph([(0, 1), (1, 2), (2, 0)])
    g.add_node(3)
    parts = girvan_newman(g, n_communities=2)
    assert {3} in [set(part) for part in parts]


def test_partition_covers_karate() -> None:
    g = karate_club()
    parts = girvan_newman(g, n_communities=2)
    flat = [node for part in parts for node in part]
    assert sorted(flat) == sorted(g.nodes)
    assert len(flat) == len(set(flat))
    assert len(parts) == 2
    assert modularity(g, parts) > 0.3


def test_deterministic() -> None:
    g = _two_cliques(bridge=True)
    assert girvan_newman(g, n_communities=2) == girvan_newman(g, n_communities=2)


def test_directed_raises() -> None:
    g = Graph([(0, 1), (1, 2)], directed=True)
    with pytest.raises(ValueError, match="undirected"):
        girvan_newman(g)


def test_invalid_n_communities_raises() -> None:
    g = Graph([(0, 1), (1, 2)])
    with pytest.raises(ValueError):
        girvan_newman(g, n_communities=0)
    with pytest.raises(ValueError):
        girvan_newman(g, n_communities=10)


def test_cli_girvan_newman(capsys) -> None:
    assert main(["communities", "--method", "girvan-newman", "--n-communities", "2"]) == 0
    out = capsys.readouterr().out
    assert "girvan-newman" in out
    assert "n_communities:" in out
