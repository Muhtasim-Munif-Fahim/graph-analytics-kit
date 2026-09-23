"""Tests for Katz centrality."""
from __future__ import annotations

import math

import pytest

from graph_analytics_kit import Graph, karate_club, katz_centrality
from graph_analytics_kit.centrality import katz_centrality as katz_from_centrality
from graph_analytics_kit.cli import main


def _path(n: int) -> Graph:
    return Graph([(i, i + 1) for i in range(n - 1)])


def _star(n_leaves: int) -> Graph:
    return Graph([(0, leaf) for leaf in range(1, n_leaves + 1)])


def _complete(nodes: list[int]) -> Graph:
    edges = [
        (nodes[i], nodes[j])
        for i in range(len(nodes))
        for j in range(i + 1, len(nodes))
    ]
    return Graph(edges)


def _l2_norm(scores: dict[int, float]) -> float:
    return math.sqrt(sum(value * value for value in scores.values()))


def test_katz_exported_from_public_api() -> None:
    assert katz_centrality is katz_from_centrality
    scores = katz_centrality(_complete([0, 1, 2]))
    assert set(scores) == {0, 1, 2}


def test_empty_graph_returns_empty() -> None:
    assert katz_centrality(Graph([])) == {}


def test_single_node_is_beta() -> None:
    g = Graph()
    g.add_node(0)
    assert katz_centrality(g, beta=1.0, normalized=False) == {0: pytest.approx(1.0)}
    assert katz_centrality(g, beta=2.5, normalized=False) == {0: pytest.approx(2.5)}
    assert katz_centrality(g) == {0: pytest.approx(1.0)}


def test_path_middle_outranks_ends() -> None:
    alpha = 0.1
    raw = katz_centrality(_path(3), alpha=alpha, normalized=False)
    end = (1.0 + alpha) / (1.0 - 2.0 * alpha * alpha)
    middle = (1.0 + 2.0 * alpha) / (1.0 - 2.0 * alpha * alpha)
    assert raw[0] == pytest.approx(end)
    assert raw[2] == pytest.approx(end)
    assert raw[1] == pytest.approx(middle)
    assert raw[1] > raw[0]
    scores = katz_centrality(_path(3), alpha=alpha)
    assert scores[1] > scores[0]
    assert scores[0] == pytest.approx(scores[2])
    assert _l2_norm(scores) == pytest.approx(1.0)


def test_longer_path_peaks_in_the_middle() -> None:
    raw = katz_centrality(_path(4), alpha=0.1, normalized=False)
    assert raw[0] == pytest.approx(raw[3])
    assert raw[1] == pytest.approx(raw[2])
    assert raw[1] > raw[0]


def test_star_center_outranks_leaves() -> None:
    alpha = 0.1
    n_leaves = 3
    raw = katz_centrality(_star(n_leaves), alpha=alpha, normalized=False)
    leaf = (1.0 + alpha) / (1.0 - n_leaves * alpha * alpha)
    center = (1.0 + n_leaves * alpha) / (1.0 - n_leaves * alpha * alpha)
    assert raw[0] == pytest.approx(center)
    assert raw[1] == pytest.approx(leaf)
    assert raw[2] == pytest.approx(leaf)
    assert raw[3] == pytest.approx(leaf)
    assert raw[0] > raw[1]
    scores = katz_centrality(_star(n_leaves), alpha=alpha)
    assert _l2_norm(scores) == pytest.approx(1.0)


def test_complete_graph_is_uniform() -> None:
    alpha = 0.1
    nodes = [0, 1, 2, 3]
    raw = katz_centrality(_complete(nodes), alpha=alpha, normalized=False)
    expected = 1.0 / (1.0 - alpha * (len(nodes) - 1))
    for node in nodes:
        assert raw[node] == pytest.approx(expected)
    scores = katz_centrality(_complete(nodes), alpha=alpha)
    unit = 1.0 / math.sqrt(len(nodes))
    for score in scores.values():
        assert score == pytest.approx(unit)
    assert _l2_norm(scores) == pytest.approx(1.0)


def test_larger_alpha_widens_path_gap() -> None:
    g = _path(3)
    low = katz_centrality(g, alpha=0.05, normalized=False)
    high = katz_centrality(g, alpha=0.2, normalized=False)
    assert high[1] / high[0] > low[1] / low[0]


def test_beta_scales_unnormalized_scores() -> None:
    g = _star(3)
    base = katz_centrality(g, alpha=0.1, beta=1.0, normalized=False)
    scaled = katz_centrality(g, alpha=0.1, beta=2.0, normalized=False)
    for node in base:
        assert scaled[node] == pytest.approx(2.0 * base[node])


def test_isolated_node_keeps_beta() -> None:
    g = Graph([(0, 1)])
    g.add_node(2)
    raw = katz_centrality(g, alpha=0.1, beta=1.0, normalized=False)
    assert raw[2] == pytest.approx(1.0)
    assert raw[0] == pytest.approx(raw[1])
    assert raw[0] == pytest.approx(1.0 / 0.9)
    assert raw[0] > raw[2]


def test_directed_edge_gives_prestige_to_target() -> None:
    g = Graph([(0, 1)], directed=True)
    raw = katz_centrality(g, alpha=0.1, normalized=False)
    assert raw[0] == pytest.approx(1.0)
    assert raw[1] == pytest.approx(1.1)
    assert raw[1] > raw[0]


def test_heavier_neighbor_raises_score() -> None:
    g = Graph([(0, 1, 1.0), (0, 2, 3.0)])
    raw = katz_centrality(g, alpha=0.1, normalized=False)
    assert raw[2] > raw[1]
    assert raw[0] > raw[2]


def test_edgeless_graph_is_uniform() -> None:
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    raw = katz_centrality(g, beta=1.0, normalized=False)
    assert raw == {0: pytest.approx(1.0), 1: pytest.approx(1.0), 2: pytest.approx(1.0)}
    scores = katz_centrality(g)
    expected = 1.0 / math.sqrt(3)
    for score in scores.values():
        assert score == pytest.approx(expected)


def test_karate_club_returns_all_nodes() -> None:
    g = karate_club()
    scores = katz_centrality(g, alpha=0.1)
    assert set(scores.keys()) == set(g.nodes)
    assert all(score > 0.0 for score in scores.values())
    assert _l2_norm(scores) == pytest.approx(1.0)


def test_karate_club_hubs_rank_highest() -> None:
    g = karate_club()
    scores = katz_centrality(g, alpha=0.1)
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert {0, 33}.issubset(set(ranked[:5]))
    assert scores[0] > scores[11]
    assert scores[33] > scores[11]


def test_invalid_alpha_raises() -> None:
    g = _path(3)
    with pytest.raises(ValueError, match="alpha"):
        katz_centrality(g, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        katz_centrality(g, alpha=-0.2)
    with pytest.raises(ValueError, match="alpha"):
        katz_centrality(g, alpha=float("nan"))


def test_alpha_above_spectral_radius_raises() -> None:
    # K3 has lambda_max = 2, so alpha must be strictly less than 0.5.
    g = _complete([0, 1, 2])
    with pytest.raises(ValueError, match="converge"):
        katz_centrality(g, alpha=0.6, max_iter=30)


def test_invalid_max_iter_raises() -> None:
    g = _star(3)
    with pytest.raises(ValueError, match="max_iter"):
        katz_centrality(g, max_iter=0)
    with pytest.raises(ValueError, match="max_iter"):
        katz_centrality(g, max_iter=1.5)


def test_cli_katz_prints_karate(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["centrality", "--measure", "katz", "--top", "3"]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.strip().splitlines() if line]
    assert len(lines) == 3
    assert ":" in lines[0]


def test_cli_katz_alpha_changes_scores(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["centrality", "--measure", "katz", "--alpha", "0.05", "--top", "34"]) == 0
    low = capsys.readouterr().out
    assert main(["centrality", "--measure", "katz", "--alpha", "0.12", "--top", "34"]) == 0
    high = capsys.readouterr().out
    assert low != high


def test_demo_report_includes_katz(tmp_path) -> None:
    out = tmp_path / "report.md"
    assert main(["demo", "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Katz" in text
    assert (
        "| Node | Degree | Closeness | Betweenness | Core | PageRank | Eigenvector | "
        "Katz | Hub | Authority | Clustering | Community |"
        in text
    )
