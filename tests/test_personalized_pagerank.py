"""Tests for Personalized PageRank (random walk with restart)."""
from __future__ import annotations

import pytest

from graph_analytics_kit import Graph, karate_club, pagerank, personalized_pagerank
from graph_analytics_kit.centrality import personalized_pagerank as ppr_from_centrality


def _directed_chain() -> Graph:
    return Graph([(0, 1, 1.0), (1, 2, 1.0)], directed=True)


def test_exported_from_public_api() -> None:
    assert personalized_pagerank is ppr_from_centrality
    g = Graph([(0, 1), (1, 2), (2, 0)])
    scores = personalized_pagerank(g, {0: 1.0})
    assert set(scores) == {0, 1, 2}


def test_none_personalization_matches_pagerank() -> None:
    g = _directed_chain()
    uniform = personalized_pagerank(g, None, alpha=0.85, max_iter=200, tol=1e-12)
    classic = pagerank(g, alpha=0.85, max_iter=200, tol=1e-12)
    for node in g.nodes:
        assert uniform[node] == pytest.approx(classic[node], abs=1e-9)


def test_empty_personalization_matches_pagerank() -> None:
    g = Graph([(0, 1), (1, 2), (2, 0)])
    uniform = personalized_pagerank(g, {}, alpha=0.9, max_iter=200)
    classic = pagerank(g, alpha=0.9, max_iter=200)
    for node in g.nodes:
        assert uniform[node] == pytest.approx(classic[node], abs=1e-8)


def test_sums_to_one() -> None:
    g = _directed_chain()
    scores = personalized_pagerank(g, {0: 1.0}, alpha=0.85)
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_seed_node_outranks_distant_nodes() -> None:
    # Line graph with strong restart: seed dominates the far end.
    g = Graph([(0, 1), (1, 2), (2, 3), (3, 4)])
    scores = personalized_pagerank(g, {0: 1.0}, alpha=0.5, max_iter=200)
    assert scores[0] > scores[1]
    assert scores[0] > scores[4]
    assert scores[1] > scores[4]


def test_two_seed_mixture() -> None:
    g = Graph([(0, 1), (1, 2), (2, 3), (3, 4)])
    scores = personalized_pagerank(g, {0: 1.0, 4: 1.0}, alpha=0.5, max_iter=200)
    assert scores[0] == pytest.approx(scores[4], abs=1e-6)
    assert scores[0] > scores[2]
    assert scores[4] > scores[2]


def test_star_seed_at_leaf() -> None:
    g = Graph([(0, 1), (0, 2), (0, 3)])
    scores = personalized_pagerank(g, {1: 1.0}, alpha=0.85)
    assert scores[1] > scores[2]
    assert scores[1] > scores[3]
    # Center still collects mass from the seed via the edge.
    assert scores[0] > scores[2]


def test_karate_seed_instructor() -> None:
    g = karate_club()
    scores = personalized_pagerank(g, {0: 1.0}, alpha=0.85, max_iter=200)
    assert set(scores.keys()) == set(g.nodes)
    assert abs(sum(scores.values()) - 1.0) < 1e-6
    assert scores[0] == max(scores.values())
    # Neighbors of the instructor outrank a peripheral leaf (node 11).
    assert scores[1] > scores[11] or scores[2] > scores[11]


def test_negative_weight_raises() -> None:
    g = Graph([(0, 1)])
    with pytest.raises(ValueError, match="non-negative"):
        personalized_pagerank(g, {0: -1.0})


def test_all_zero_weights_raise() -> None:
    g = Graph([(0, 1)])
    with pytest.raises(ValueError, match="positive weight"):
        personalized_pagerank(g, {0: 0.0, 1: 0.0})


def test_unknown_only_personalization_raises() -> None:
    g = Graph([(0, 1)])
    with pytest.raises(ValueError, match="positive weight"):
        personalized_pagerank(g, {99: 1.0})


def test_invalid_alpha_raises() -> None:
    g = _directed_chain()
    with pytest.raises(ValueError, match="alpha"):
        personalized_pagerank(g, {0: 1.0}, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        personalized_pagerank(g, {0: 1.0}, alpha=1.0)


def test_invalid_max_iter_raises() -> None:
    g = _directed_chain()
    with pytest.raises(ValueError, match="max_iter"):
        personalized_pagerank(g, {0: 1.0}, max_iter=0)


def test_empty_graph_returns_empty() -> None:
    assert personalized_pagerank(Graph([]), {0: 1.0}) == {}
