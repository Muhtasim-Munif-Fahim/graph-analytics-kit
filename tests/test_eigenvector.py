"""Tests for power-iteration eigenvector centrality."""
from __future__ import annotations

import math

import pytest

from graph_analytics_kit import Graph, eigenvector_centrality, karate_club
from graph_analytics_kit.centrality import (
    eigenvector_centrality as eigenvector_from_centrality,
)
from graph_analytics_kit.cli import main


def _star() -> Graph:
    return Graph([(0, 1), (0, 2), (0, 3)])


def _path3() -> Graph:
    return Graph([(0, 1), (1, 2)])


def _triangle() -> Graph:
    return Graph([(0, 1), (0, 2), (1, 2)])


def _l2_norm(scores: dict[int, float]) -> float:
    return math.sqrt(sum(value * value for value in scores.values()))


def test_eigenvector_exported_from_public_api() -> None:
    assert eigenvector_centrality is eigenvector_from_centrality
    g = _triangle()
    scores = eigenvector_centrality(g)
    assert set(scores) == {0, 1, 2}


def test_empty_graph_returns_empty() -> None:
    assert eigenvector_centrality(Graph([])) == {}


def test_single_node_is_unit() -> None:
    g = Graph()
    g.add_node(0)
    assert eigenvector_centrality(g) == {0: pytest.approx(1.0)}


def test_edgeless_graph_is_uniform() -> None:
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    scores = eigenvector_centrality(g)
    expected = 1.0 / math.sqrt(3)
    for score in scores.values():
        assert score == pytest.approx(expected)
    assert _l2_norm(scores) == pytest.approx(1.0)


def test_triangle_is_uniform() -> None:
    scores = eigenvector_centrality(_triangle())
    expected = 1.0 / math.sqrt(3)
    for score in scores.values():
        assert score == pytest.approx(expected, abs=1e-6)
    assert _l2_norm(scores) == pytest.approx(1.0, abs=1e-6)


def test_star_center_outranks_leaves() -> None:
    scores = eigenvector_centrality(_star())
    assert scores[0] == pytest.approx(1.0 / math.sqrt(2), abs=1e-6)
    leaf = 1.0 / math.sqrt(6)
    assert scores[1] == pytest.approx(leaf, abs=1e-6)
    assert scores[2] == pytest.approx(leaf, abs=1e-6)
    assert scores[3] == pytest.approx(leaf, abs=1e-6)
    assert scores[0] > scores[1]
    assert _l2_norm(scores) == pytest.approx(1.0, abs=1e-6)


def test_path_middle_outranks_ends() -> None:
    scores = eigenvector_centrality(_path3())
    assert scores[1] == pytest.approx(math.sqrt(2) / 2, abs=1e-6)
    assert scores[0] == pytest.approx(0.5, abs=1e-6)
    assert scores[2] == pytest.approx(0.5, abs=1e-6)
    assert scores[1] > scores[0]


def test_isolated_node_vanishes_when_edges_exist() -> None:
    g = Graph([(0, 1), (1, 2), (2, 0)])
    g.add_node(3)
    scores = eigenvector_centrality(g)
    assert scores[3] == pytest.approx(0.0, abs=1e-6)
    assert scores[0] == pytest.approx(scores[1], abs=1e-6)
    assert scores[0] > scores[3]
    assert _l2_norm(scores) == pytest.approx(1.0, abs=1e-6)


def test_directed_edge_gives_prestige_to_target() -> None:
    g = Graph([(0, 1, 1.0)], directed=True)
    scores = eigenvector_centrality(g, max_iter=500)
    assert scores[1] > scores[0]
    assert scores[1] == pytest.approx(1.0, abs=1e-2)
    assert scores[0] == pytest.approx(0.0, abs=1e-2)


def test_directed_cycle_is_uniform() -> None:
    g = Graph([(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)], directed=True)
    scores = eigenvector_centrality(g)
    expected = 1.0 / math.sqrt(3)
    for score in scores.values():
        assert score == pytest.approx(expected, abs=1e-6)


def test_heavier_neighbor_raises_score() -> None:
    g = Graph([(0, 1, 1.0), (0, 2, 4.0)])
    scores = eigenvector_centrality(g)
    assert scores[2] > scores[1]


def test_karate_club_returns_all_nodes() -> None:
    g = karate_club()
    scores = eigenvector_centrality(g)
    assert set(scores.keys()) == set(g.nodes)
    assert all(score >= 0.0 for score in scores.values())
    assert _l2_norm(scores) == pytest.approx(1.0, abs=1e-6)


def test_karate_club_hubs_rank_highest() -> None:
    g = karate_club()
    scores = eigenvector_centrality(g)
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert {0, 33}.issubset(set(ranked[:5]))
    assert scores[0] > scores[11]
    assert scores[33] > scores[11]


def test_scores_are_non_negative() -> None:
    scores = eigenvector_centrality(_star())
    assert all(score >= 0.0 for score in scores.values())


def test_invalid_max_iter_raises() -> None:
    g = _star()
    with pytest.raises(ValueError, match="max_iter"):
        eigenvector_centrality(g, max_iter=0)
    with pytest.raises(ValueError, match="max_iter"):
        eigenvector_centrality(g, max_iter=1.5)


def test_cli_eigenvector_prints_karate(capsys) -> None:
    assert main(["centrality", "--measure", "eigenvector", "--top", "3"]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.strip().splitlines() if line]
    assert len(lines) == 3
    assert ":" in lines[0]


def test_demo_report_includes_eigenvector(tmp_path) -> None:
    out = tmp_path / "report.md"
    assert main(["demo", "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Eigenvector" in text
    assert (
        "| Node | Degree | Closeness | Betweenness | Core | PageRank | Eigenvector | "
        "Hub | Authority | Clustering | Community |"
        in text
    )
