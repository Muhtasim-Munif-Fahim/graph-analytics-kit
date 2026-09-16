"""Tests for PageRank centrality."""
from __future__ import annotations

import pytest

from graph_analytics_kit import Graph, karate_club, pagerank
from graph_analytics_kit.centrality import pagerank as pagerank_from_centrality
from graph_analytics_kit.cli import main


def _directed_chain() -> Graph:
    return Graph([(0, 1, 1.0), (1, 2, 1.0)], directed=True)


def test_pagerank_exported_from_public_api() -> None:
    assert pagerank is pagerank_from_centrality
    g = Graph([(0, 1), (1, 2), (2, 0)])
    scores = pagerank(g)
    assert set(scores) == {0, 1, 2}


def test_pagerank_sums_to_one() -> None:
    g = _directed_chain()
    scores = pagerank(g, alpha=0.85, max_iter=100)
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_sink_node_gets_teleport_mass() -> None:
    g = Graph([(0, 1, 1.0)], directed=True)
    scores = pagerank(g, alpha=0.85)
    assert 0 in scores
    assert 1 in scores
    assert scores[0] > 0.0
    assert scores[1] > scores[0]


def test_uniform_rank_for_single_node() -> None:
    g = Graph([(0, 0)])
    scores = pagerank(g)
    assert scores == {0: pytest.approx(1.0)}


def test_directed_chain_ranking() -> None:
    g = _directed_chain()
    scores = pagerank(g, alpha=0.9, max_iter=200)
    assert scores[2] > scores[1] > scores[0]


def test_undirected_symmetric_on_regular_graph() -> None:
    g = Graph([(0, 1), (1, 2), (2, 0)])
    scores = pagerank(g, alpha=0.85, max_iter=100)
    for score in scores.values():
        assert score == pytest.approx(1.0 / 3, abs=1e-6)


def test_star_center_outranks_leaves() -> None:
    g = Graph([(0, 1), (0, 2), (0, 3)])
    scores = pagerank(g, alpha=0.85)
    assert scores[0] > scores[1]
    assert scores[1] == pytest.approx(scores[2], abs=1e-9)
    assert scores[2] == pytest.approx(scores[3], abs=1e-9)
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_isolated_node_receives_teleport_rank() -> None:
    g = Graph([(0, 1)])
    g.add_node(2)
    scores = pagerank(g, alpha=0.85)
    assert scores[2] > 0.0
    assert scores[0] == pytest.approx(scores[1], abs=1e-9)
    assert scores[0] > scores[2]
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_karate_club_returns_all_nodes() -> None:
    g = karate_club()
    scores = pagerank(g, alpha=0.85, max_iter=100)
    assert set(scores.keys()) == set(g.nodes)
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_karate_club_hubs_rank_highest() -> None:
    g = karate_club()
    scores = pagerank(g, alpha=0.85)
    ranked = sorted(scores, key=scores.get, reverse=True)
    # Instructor (0) and president (33) are the two highest-degree hubs.
    assert {0, 33}.issubset(set(ranked[:3]))
    assert scores[0] > scores[11]
    assert scores[33] > scores[11]


def test_invalid_alpha_raises() -> None:
    g = _directed_chain()
    with pytest.raises(ValueError, match="alpha"):
        pagerank(g, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        pagerank(g, alpha=1.0)


def test_invalid_max_iter_raises() -> None:
    g = _directed_chain()
    with pytest.raises(ValueError, match="max_iter"):
        pagerank(g, max_iter=0)
    with pytest.raises(ValueError, match="max_iter"):
        pagerank(g, max_iter=1.5)


def test_empty_graph_returns_empty() -> None:
    g = Graph([])
    assert pagerank(g) == {}


def test_demo_report_includes_pagerank(tmp_path) -> None:
    out = tmp_path / "report.md"
    assert main(["demo", "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "PageRank" in text
    assert "| Node | Degree | Closeness | Betweenness | PageRank | Clustering |" in text
