"""Tests for spectral clustering community detection."""
from __future__ import annotations

import pytest

from graph_analytics_kit import Graph, karate_club, modularity, spectral_clustering
from graph_analytics_kit.cli import main
from graph_analytics_kit.community import spectral_clustering as spectral_impl


def _clique(nodes: list[int]) -> list[tuple[int, int]]:
    return [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]


def _two_cliques(*, bridge: bool) -> Graph:
    edges = _clique([0, 1, 2, 3, 4]) + _clique([5, 6, 7, 8, 9])
    if bridge:
        edges.append((4, 5))
    return Graph(edges)


def test_spectral_exported_from_public_api() -> None:
    assert spectral_clustering is spectral_impl
    g = Graph([(0, 1), (1, 2), (2, 0)])
    parts = spectral_clustering(g, n_communities=1)
    assert parts == [[0, 1, 2]]


def test_empty_graph_returns_empty() -> None:
    assert spectral_clustering(Graph([]), n_communities=1) == []


def test_single_node_is_its_own_community() -> None:
    g = Graph()
    g.add_node(0)
    assert spectral_clustering(g, n_communities=1) == [[0]]


def test_disconnected_cliques_are_two_communities() -> None:
    g = _two_cliques(bridge=False)
    parts = spectral_clustering(g, n_communities=2)
    assert [set(part) for part in parts] == [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]
    assert modularity(g, parts) > 0.4


def test_bridged_cliques_split() -> None:
    g = _two_cliques(bridge=True)
    parts = spectral_clustering(g, n_communities=2, seed=0)
    assert [set(part) for part in parts] == [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]
    assert modularity(g, parts) > 0.4


def test_unnormalized_laplacian_also_splits() -> None:
    g = _two_cliques(bridge=True)
    parts = spectral_clustering(g, n_communities=2, normalized=False, seed=0)
    assert [set(part) for part in parts] == [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]


def test_partition_format_sorted() -> None:
    g = _two_cliques(bridge=True)
    parts = spectral_clustering(g, n_communities=2, seed=0)
    for part in parts:
        assert part == sorted(part)
    assert parts == sorted(parts, key=lambda p: p[0])


def test_n_communities_equals_nodes() -> None:
    g = Graph([(0, 1), (1, 2)])
    parts = spectral_clustering(g, n_communities=3)
    assert parts == [[0], [1], [2]]


def test_isolated_node_stays_separate() -> None:
    g = Graph([(0, 1), (1, 2), (2, 0)])
    g.add_node(3)
    parts = spectral_clustering(g, n_communities=2, seed=0)
    flat = [node for part in parts for node in part]
    assert sorted(flat) == [0, 1, 2, 3]
    assert len(parts) == 2


def test_partition_covers_karate() -> None:
    g = karate_club()
    parts = spectral_clustering(g, n_communities=2, seed=0)
    flat = [node for part in parts for node in part]
    assert sorted(flat) == sorted(g.nodes)
    assert len(flat) == len(set(flat))
    assert len(parts) == 2
    assert modularity(g, parts) > 0.2


def test_deterministic_without_seed() -> None:
    g = _two_cliques(bridge=True)
    assert spectral_clustering(g, n_communities=2) == spectral_clustering(
        g, n_communities=2
    )


def test_seed_reproducible() -> None:
    g = _two_cliques(bridge=True)
    a = spectral_clustering(g, n_communities=2, seed=7)
    b = spectral_clustering(g, n_communities=2, seed=7)
    assert a == b


def test_directed_raises() -> None:
    g = Graph([(0, 1), (1, 2)], directed=True)
    with pytest.raises(ValueError, match="undirected"):
        spectral_clustering(g, n_communities=2)


def test_invalid_n_communities_raises() -> None:
    g = Graph([(0, 1), (1, 2)])
    with pytest.raises(ValueError):
        spectral_clustering(g, n_communities=0)
    with pytest.raises(ValueError):
        spectral_clustering(g, n_communities=10)


def test_cli_spectral(capsys) -> None:
    assert main(
        ["communities", "--method", "spectral", "--n-communities", "2"]
    ) == 0
    out = capsys.readouterr().out
    assert "spectral" in out
    assert "n_communities:" in out
