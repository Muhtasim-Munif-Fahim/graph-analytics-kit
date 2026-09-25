"""Tests for asynchronous label-propagation community detection."""
from __future__ import annotations

import pytest

from graph_analytics_kit import Graph, karate_club, label_propagation, modularity
from graph_analytics_kit.cli import main
from graph_analytics_kit.community import label_propagation as label_propagation_impl


def _clique(nodes: list[int]) -> list[tuple[int, int]]:
    return [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]


def _two_cliques(*, bridge: bool) -> Graph:
    edges = _clique([0, 1, 2, 3, 4]) + _clique([5, 6, 7, 8, 9])
    if bridge:
        edges.append((4, 5))
    return Graph(edges)


def test_label_propagation_exported_from_public_api() -> None:
    assert label_propagation is label_propagation_impl
    g = Graph([(0, 1), (1, 2), (2, 0)])
    parts = label_propagation(g)
    assert isinstance(parts, list)
    assert set(n for part in parts for n in part) == {0, 1, 2}


def test_empty_graph_returns_empty() -> None:
    assert label_propagation(Graph([])) == []
    assert label_propagation(Graph([]), return_labels=True) == ([], {})


def test_single_node_is_its_own_community() -> None:
    g = Graph()
    g.add_node(0)
    assert label_propagation(g) == [[0]]
    parts, labels = label_propagation(g, return_labels=True)
    assert parts == [[0]]
    assert labels == {0: 0}


def test_disconnected_cliques_are_two_communities() -> None:
    g = _two_cliques(bridge=False)
    parts = label_propagation(g)
    assert [set(part) for part in parts] == [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]
    assert modularity(g, parts) > 0.4


def test_bridged_cliques_recover_two_blocks() -> None:
    g = _two_cliques(bridge=True)
    parts, labels = label_propagation(g, return_labels=True)
    assert [set(part) for part in parts] == [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]
    assert modularity(g, parts) > 0.4
    assert len({labels[node] for node in (0, 1, 2, 3, 4)}) == 1
    assert len({labels[node] for node in (5, 6, 7, 8, 9)}) == 1
    assert labels[0] != labels[5]


def test_interleaved_clique_labels_stay_separated() -> None:
    left = [0, 2, 4, 6, 8]
    right = [1, 3, 5, 7, 9]
    edges = _clique(left) + _clique(right) + [(8, 1)]
    parts = label_propagation(Graph(edges))
    assert [set(part) for part in parts] == [set(left), set(right)]


def test_complete_graph_is_one_community() -> None:
    g = Graph(_clique([0, 1, 2, 3]))
    assert label_propagation(g) == [[0, 1, 2, 3]]


def test_isolated_node_stays_separate() -> None:
    g = Graph([(0, 1), (1, 2), (2, 0)])
    g.add_node(3)
    parts = label_propagation(g)
    assert [set(part) for part in parts] == [{0, 1, 2}, {3}]


def test_self_loop_does_not_vote() -> None:
    g = Graph([(0, 0), (1, 1)])
    assert label_propagation(g) == [[0], [1]]


def test_edge_weight_pulls_node_into_heavier_side() -> None:
    # On an unweighted path, node 2 stays with the lower-id side.
    light = Graph([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)])
    assert [set(part) for part in label_propagation(light)] == [{0, 1, 2}, {3, 4}]
    # A heavier 2–3 edge pulls node 2 into the other block.
    heavy = Graph([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 5.0), (3, 4, 1.0)])
    assert [set(part) for part in label_propagation(heavy)] == [{0, 1}, {2, 3, 4}]


def test_partition_is_sorted_and_covers_karate() -> None:
    g = karate_club()
    parts = label_propagation(g)
    flat = [node for part in parts for node in part]
    assert sorted(flat) == sorted(g.nodes)
    assert len(flat) == len(set(flat))
    for part in parts:
        assert part == sorted(part)
    assert parts == sorted(parts, key=lambda members: members[0])
    assert modularity(g, parts) > 0.3


def test_deterministic_without_seed() -> None:
    g = karate_club()
    assert label_propagation(g) == label_propagation(g)
    assert label_propagation(g, return_labels=True) == label_propagation(
        g, return_labels=True
    )


def test_seed_is_reproducible() -> None:
    g = karate_club()
    assert label_propagation(g, seed=7) == label_propagation(g, seed=7)
    assert label_propagation(g, seed=7, return_labels=True) == label_propagation(
        g, seed=7, return_labels=True
    )


def test_seed_changes_sweep_order() -> None:
    g = karate_club()
    assert label_propagation(g, seed=0) == label_propagation(g, seed=0)
    # Degree order (seed unset) is a different schedule from a shuffle.
    assert label_propagation(g) != label_propagation(g, seed=0)


def test_directed_graph_raises() -> None:
    g = Graph([(0, 1)], directed=True)
    with pytest.raises(ValueError, match="undirected"):
        label_propagation(g)


def test_invalid_max_iter_raises() -> None:
    g = Graph([(0, 1)])
    with pytest.raises(ValueError, match="max_iter"):
        label_propagation(g, max_iter=0)
    with pytest.raises(ValueError, match="max_iter"):
        label_propagation(g, max_iter=1.5)
    with pytest.raises(ValueError, match="max_iter"):
        label_propagation(g, max_iter=True)


def test_cli_label_propagation_prints_karate(capsys) -> None:
    assert main(["communities", "--method", "label-propagation", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "method: label propagation" in out
    assert "n_communities:" in out
    assert "modularity:" in out
    assert "community 0" in out


def test_demo_report_includes_label_propagation(tmp_path) -> None:
    out = tmp_path / "report.md"
    assert main(["demo", "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "label propagation" in text
    assert "Label-propagation modularity:" in text
