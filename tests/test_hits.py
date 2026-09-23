"""Tests for Kleinberg HITS (hubs and authorities)."""
from __future__ import annotations

import math

import pytest

from graph_analytics_kit import Graph, hits, karate_club
from graph_analytics_kit.centrality import hits as hits_from_centrality
from graph_analytics_kit.cli import main


def _star() -> Graph:
    return Graph([(0, 1), (0, 2), (0, 3)])


def _triangle() -> Graph:
    return Graph([(0, 1), (0, 2), (1, 2)])


def _path3() -> Graph:
    return Graph([(0, 1), (1, 2)])


def _l2_norm(scores: dict[int, float]) -> float:
    return math.sqrt(sum(value * value for value in scores.values()))


def test_hits_exported_from_public_api() -> None:
    assert hits is hits_from_centrality
    g = _triangle()
    hubs, authorities = hits(g)
    assert set(hubs) == {0, 1, 2}
    assert set(authorities) == {0, 1, 2}


def test_empty_graph_returns_empty() -> None:
    assert hits(Graph([])) == ({}, {})


def test_single_node_is_unit() -> None:
    g = Graph()
    g.add_node(0)
    assert hits(g) == ({0: pytest.approx(1.0)}, {0: pytest.approx(1.0)})


def test_edgeless_graph_is_uniform() -> None:
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    hubs, authorities = hits(g)
    expected = 1.0 / math.sqrt(3)
    for score in hubs.values():
        assert score == pytest.approx(expected)
    for score in authorities.values():
        assert score == pytest.approx(expected)
    assert _l2_norm(hubs) == pytest.approx(1.0)
    assert _l2_norm(authorities) == pytest.approx(1.0)


def test_triangle_is_uniform() -> None:
    hubs, authorities = hits(_triangle())
    expected = 1.0 / math.sqrt(3)
    for score in hubs.values():
        assert score == pytest.approx(expected, abs=1e-6)
    for score in authorities.values():
        assert score == pytest.approx(expected, abs=1e-6)
    assert hubs == pytest.approx(authorities)
    assert _l2_norm(hubs) == pytest.approx(1.0, abs=1e-6)
    assert _l2_norm(authorities) == pytest.approx(1.0, abs=1e-6)


def test_star_hubs_uniform_authorities_center() -> None:
    hubs, authorities = hits(_star())
    # Undirected star: A^2 has a 2-D leading eigenspace. Uniform init yields
    # uniform hubs and a center-heavy authority vector [3, 1, 1, 1].
    for score in hubs.values():
        assert score == pytest.approx(0.5, abs=1e-6)
    assert authorities[0] == pytest.approx(3.0 / math.sqrt(12), abs=1e-6)
    leaf = 1.0 / math.sqrt(12)
    assert authorities[1] == pytest.approx(leaf, abs=1e-6)
    assert authorities[2] == pytest.approx(leaf, abs=1e-6)
    assert authorities[3] == pytest.approx(leaf, abs=1e-6)
    assert authorities[0] > authorities[1]
    assert _l2_norm(hubs) == pytest.approx(1.0, abs=1e-6)
    assert _l2_norm(authorities) == pytest.approx(1.0, abs=1e-6)


def test_path3_middle_is_strongest_authority() -> None:
    hubs, authorities = hits(_path3())
    expected_hub = 1.0 / math.sqrt(3)
    for score in hubs.values():
        assert score == pytest.approx(expected_hub, abs=1e-6)
    assert authorities[1] == pytest.approx(2.0 / math.sqrt(6), abs=1e-6)
    assert authorities[0] == pytest.approx(1.0 / math.sqrt(6), abs=1e-6)
    assert authorities[2] == pytest.approx(1.0 / math.sqrt(6), abs=1e-6)
    assert authorities[1] > authorities[0]


def test_directed_out_star() -> None:
    g = Graph([(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)], directed=True)
    hubs, authorities = hits(g)
    assert hubs[0] == pytest.approx(1.0, abs=1e-6)
    assert hubs[1] == pytest.approx(0.0, abs=1e-6)
    assert hubs[2] == pytest.approx(0.0, abs=1e-6)
    assert hubs[3] == pytest.approx(0.0, abs=1e-6)
    assert authorities[0] == pytest.approx(0.0, abs=1e-6)
    leaf = 1.0 / math.sqrt(3)
    assert authorities[1] == pytest.approx(leaf, abs=1e-6)
    assert authorities[2] == pytest.approx(leaf, abs=1e-6)
    assert authorities[3] == pytest.approx(leaf, abs=1e-6)


def test_directed_in_star() -> None:
    g = Graph([(1, 0, 1.0), (2, 0, 1.0), (3, 0, 1.0)], directed=True)
    hubs, authorities = hits(g)
    leaf = 1.0 / math.sqrt(3)
    assert hubs[0] == pytest.approx(0.0, abs=1e-6)
    assert hubs[1] == pytest.approx(leaf, abs=1e-6)
    assert hubs[2] == pytest.approx(leaf, abs=1e-6)
    assert hubs[3] == pytest.approx(leaf, abs=1e-6)
    assert authorities[0] == pytest.approx(1.0, abs=1e-6)
    assert authorities[1] == pytest.approx(0.0, abs=1e-6)
    assert authorities[2] == pytest.approx(0.0, abs=1e-6)
    assert authorities[3] == pytest.approx(0.0, abs=1e-6)


def test_directed_chain() -> None:
    g = Graph([(0, 1, 1.0), (1, 2, 1.0)], directed=True)
    hubs, authorities = hits(g)
    half = 1.0 / math.sqrt(2)
    assert hubs[0] == pytest.approx(half, abs=1e-6)
    assert hubs[1] == pytest.approx(half, abs=1e-6)
    assert hubs[2] == pytest.approx(0.0, abs=1e-6)
    assert authorities[0] == pytest.approx(0.0, abs=1e-6)
    assert authorities[1] == pytest.approx(half, abs=1e-6)
    assert authorities[2] == pytest.approx(half, abs=1e-6)


def test_directed_cycle_is_uniform() -> None:
    g = Graph([(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)], directed=True)
    hubs, authorities = hits(g)
    expected = 1.0 / math.sqrt(3)
    for score in hubs.values():
        assert score == pytest.approx(expected, abs=1e-6)
    for score in authorities.values():
        assert score == pytest.approx(expected, abs=1e-6)


def test_isolated_node_vanishes_when_edges_exist() -> None:
    g = Graph([(0, 1), (1, 2), (2, 0)])
    g.add_node(3)
    hubs, authorities = hits(g)
    assert hubs[3] == pytest.approx(0.0, abs=1e-6)
    assert authorities[3] == pytest.approx(0.0, abs=1e-6)
    assert hubs[0] == pytest.approx(hubs[1], abs=1e-6)
    assert authorities[0] == pytest.approx(authorities[1], abs=1e-6)
    assert hubs[0] > hubs[3]
    assert _l2_norm(hubs) == pytest.approx(1.0, abs=1e-6)
    assert _l2_norm(authorities) == pytest.approx(1.0, abs=1e-6)


def test_heavier_out_edge_raises_target_authority() -> None:
    g = Graph([(0, 1, 1.0), (0, 2, 4.0)], directed=True)
    _hubs, authorities = hits(g)
    assert authorities[2] > authorities[1]
    assert authorities[0] == pytest.approx(0.0, abs=1e-6)
    assert authorities[2] == pytest.approx(4.0 / math.sqrt(17), abs=1e-6)
    assert authorities[1] == pytest.approx(1.0 / math.sqrt(17), abs=1e-6)


def test_scores_are_non_negative() -> None:
    hubs, authorities = hits(_star())
    assert all(score >= 0.0 for score in hubs.values())
    assert all(score >= 0.0 for score in authorities.values())


def test_karate_club_returns_all_nodes() -> None:
    g = karate_club()
    hubs, authorities = hits(g)
    assert set(hubs.keys()) == set(g.nodes)
    assert set(authorities.keys()) == set(g.nodes)
    assert all(score >= 0.0 for score in hubs.values())
    assert all(score >= 0.0 for score in authorities.values())
    assert _l2_norm(hubs) == pytest.approx(1.0, abs=1e-6)
    assert _l2_norm(authorities) == pytest.approx(1.0, abs=1e-6)


def test_karate_club_hubs_rank_highest() -> None:
    g = karate_club()
    hubs, authorities = hits(g)
    ranked_hubs = sorted(hubs, key=hubs.get, reverse=True)
    ranked_auth = sorted(authorities, key=authorities.get, reverse=True)
    assert {0, 33}.issubset(set(ranked_hubs[:5]))
    assert {0, 33}.issubset(set(ranked_auth[:5]))
    assert hubs[0] > hubs[11]
    assert hubs[33] > hubs[11]
    assert authorities[0] > authorities[11]
    assert authorities[33] > authorities[11]


def test_invalid_max_iter_raises() -> None:
    g = _star()
    with pytest.raises(ValueError, match="max_iter"):
        hits(g, max_iter=0)
    with pytest.raises(ValueError, match="max_iter"):
        hits(g, max_iter=1.5)


def test_cli_hubs_prints_karate(capsys) -> None:
    assert main(["centrality", "--measure", "hubs", "--top", "3"]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.strip().splitlines() if line]
    assert len(lines) == 3
    assert ":" in lines[0]


def test_cli_authorities_prints_karate(capsys) -> None:
    assert main(["centrality", "--measure", "authorities", "--top", "3"]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.strip().splitlines() if line]
    assert len(lines) == 3
    assert ":" in lines[0]


def test_demo_report_includes_hits(tmp_path) -> None:
    out = tmp_path / "report.md"
    assert main(["demo", "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Hub" in text
    assert "Authority" in text
    assert (
        "| Node | Degree | Closeness | Betweenness | Core | PageRank | Eigenvector | "
        "Katz | Hub | Authority | Clustering | Community |"
        in text
    )
