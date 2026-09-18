"""Tests for Louvain community detection."""
from __future__ import annotations

import pytest

from graph_analytics_kit import Graph, communities, karate_club, modularity
from graph_analytics_kit.cli import main
from graph_analytics_kit.community import communities as communities_impl


# Zachary's documented faction split (instructor vs. administrator).
# Node 8 is the classic boundary case and is omitted from majority checks.
_INSTRUCTOR = {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21}
_ADMINISTRATOR = {9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}


def _clique(nodes: list[int]) -> list[tuple[int, int]]:
    return [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]


def _two_cliques(*, bridge: bool) -> Graph:
    edges = _clique([0, 1, 2, 3, 4]) + _clique([5, 6, 7, 8, 9])
    if bridge:
        edges.append((4, 5))
    return Graph(edges)


def _community_map(parts: list[list[int]]) -> dict[int, int]:
    return {node: index for index, part in enumerate(parts) for node in part}


def test_communities_exported_from_public_api() -> None:
    assert communities is communities_impl
    g = Graph([(0, 1), (1, 2), (2, 0)])
    parts = communities(g)
    assert set(n for part in parts for n in part) == {0, 1, 2}


def test_empty_graph_returns_empty() -> None:
    assert communities(Graph([])) == []
    assert modularity(Graph([]), []) == 0.0


def test_single_node_is_its_own_community() -> None:
    g = Graph()
    g.add_node(0)
    assert communities(g) == [[0]]
    assert modularity(g, [[0]]) == 0.0


def test_disconnected_cliques_are_two_communities() -> None:
    g = _two_cliques(bridge=False)
    parts = communities(g)
    assert [set(part) for part in parts] == [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]
    assert modularity(g, parts) > 0.4


def test_modular_graph_recovers_two_blocks() -> None:
    g = _two_cliques(bridge=True)
    parts = communities(g)
    assert [set(part) for part in parts] == [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]
    assert modularity(g, parts) > 0.4


def test_complete_graph_is_one_community() -> None:
    g = Graph(_clique([0, 1, 2, 3]))
    parts = communities(g)
    assert parts == [[0, 1, 2, 3]]


def test_isolated_node_stays_separate() -> None:
    g = Graph([(0, 1), (1, 2), (2, 0)])
    g.add_node(3)
    parts = communities(g)
    assert {3} in [set(part) for part in parts]
    assert {0, 1, 2} in [set(part) for part in parts]


def test_communities_are_a_partition() -> None:
    g = karate_club()
    parts = communities(g)
    flat = [node for part in parts for node in part]
    assert sorted(flat) == sorted(g.nodes)
    assert len(flat) == len(set(flat))
    for part in parts:
        assert part == sorted(part)
    assert parts == sorted(parts, key=lambda members: members[0])


def test_karate_club_splits_known_factions() -> None:
    g = karate_club()
    parts = communities(g)
    mapping = _community_map(parts)
    assert 2 <= len(parts) <= 6
    assert mapping[0] != mapping[33]
    # Instructor (0) and administrator (33) hubs stay with close allies.
    assert mapping[1] == mapping[0]
    assert mapping[32] == mapping[33]
    instructor_with_hub = sum(1 for node in _INSTRUCTOR if mapping[node] == mapping[0])
    admin_with_hub = sum(1 for node in _ADMINISTRATOR if mapping[node] == mapping[33])
    assert instructor_with_hub >= 4
    assert admin_with_hub >= 4
    assert modularity(g, parts) > 0.3


def test_deterministic_without_seed() -> None:
    g = karate_club()
    assert communities(g) == communities(g)


def test_seed_is_reproducible() -> None:
    g = karate_club()
    assert communities(g, seed=7) == communities(g, seed=7)


def test_directed_graph_raises() -> None:
    g = Graph([(0, 1)], directed=True)
    with pytest.raises(ValueError, match="undirected"):
        communities(g)
    with pytest.raises(ValueError, match="undirected"):
        modularity(g, [[0, 1]])


def test_invalid_max_iter_raises() -> None:
    g = Graph([(0, 1)])
    with pytest.raises(ValueError, match="max_iter"):
        communities(g, max_iter=0)
    with pytest.raises(ValueError, match="max_iter"):
        communities(g, max_iter=1.5)


def test_modularity_rejects_incomplete_partition() -> None:
    g = Graph([(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="missing"):
        modularity(g, [[0, 1]])
    with pytest.raises(ValueError, match="more than one"):
        modularity(g, [[0, 1], [1, 2]])


def test_modularity_complete_graph_all_together() -> None:
    g = Graph(_clique([0, 1, 2]))
    together = modularity(g, [[0, 1, 2]])
    split = modularity(g, [[0], [1], [2]])
    assert together == pytest.approx(0.0, abs=1e-12)
    assert split < 0.0


def test_cli_communities_prints_karate(capsys) -> None:
    assert main(["communities"]) == 0
    out = capsys.readouterr().out
    assert "n_communities:" in out
    assert "modularity:" in out
    assert "community 0" in out


def test_demo_report_includes_communities(tmp_path) -> None:
    out = tmp_path / "report.md"
    assert main(["demo", "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## Communities" in text
    assert "Louvain" in text
    assert "Modularity:" in text
    assert (
        "| Node | Degree | Closeness | Betweenness | PageRank | Eigenvector | "
        "Clustering | Community |"
        in text
    )
