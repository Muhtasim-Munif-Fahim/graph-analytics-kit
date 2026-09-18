"""Tests for Dijkstra shortest-path algorithms."""
from __future__ import annotations

import pytest

from graph_analytics_kit import (
    Graph,
    dijkstra_shortest_path,
    karate_club,
    reconstruct_path,
)
from graph_analytics_kit.cli import main
from graph_analytics_kit.shortest_path import (
    dijkstra_shortest_path as dijkstra_impl,
    reconstruct_path as reconstruct_impl,
)


def _weighted_line() -> Graph:
    return Graph([(0, 1, 2.0), (1, 2, 3.0)], directed=False)


def _star() -> Graph:
    return Graph([(0, 1), (0, 2), (0, 3)])


def _weighted_star() -> Graph:
    return Graph([(0, 1, 1.0), (0, 2, 2.0), (0, 3, 3.0)])


def test_dijkstra_exported_from_public_api() -> None:
    assert dijkstra_shortest_path is dijkstra_impl
    assert reconstruct_path is reconstruct_impl
    g = _weighted_line()
    dist, prev = dijkstra_shortest_path(g, 0)
    assert dist[2] == 5.0
    assert reconstruct_path(prev, 0, 2) == [0, 1, 2]


def test_distances_on_simple_path() -> None:
    g = _weighted_line()
    dist, _ = dijkstra_shortest_path(g, 0)
    assert dist == {0: 0.0, 1: 2.0, 2: 5.0}


def test_predecessors_reconstruct_path() -> None:
    g = _weighted_line()
    _, prev = dijkstra_shortest_path(g, 0)
    path = reconstruct_path(prev, 0, 2)
    assert path == [0, 1, 2]


def test_target_early_exit() -> None:
    g = _weighted_line()
    dist, _ = dijkstra_shortest_path(g, 0, target=1)
    assert dist == {0: 0.0, 1: 2.0}


def test_unreachable_node_excluded() -> None:
    g = Graph([(0, 1), (2, 3)])
    dist, _ = dijkstra_shortest_path(g, 0)
    assert 0 in dist
    assert 1 in dist
    assert 2 not in dist
    assert 3 not in dist


def test_reconstruct_unreachable_raises() -> None:
    g = Graph([(0, 1), (2, 3)])
    _, prev = dijkstra_shortest_path(g, 0)
    with pytest.raises(KeyError):
        reconstruct_path(prev, 0, 2)


def test_directed_graph_respects_edges() -> None:
    g = Graph([(0, 1, 1.0), (1, 2, 1.0)], directed=True)
    dist, _ = dijkstra_shortest_path(g, 0)
    assert dist == {0: 0.0, 1: 1.0, 2: 2.0}


def test_directed_graph_no_reverse_path() -> None:
    g = Graph([(0, 1, 1.0)], directed=True)
    dist, _ = dijkstra_shortest_path(g, 1)
    assert dist == {1: 0.0}
    assert 0 not in dist


def test_karate_club_source_zero_reaches_all() -> None:
    g = karate_club()
    dist, _ = dijkstra_shortest_path(g, 0)
    assert 0 in dist
    assert len(dist) == g.number_of_nodes()


def test_source_distance_is_zero() -> None:
    g = _weighted_line()
    dist, _ = dijkstra_shortest_path(g, 0)
    assert dist[0] == 0.0


def test_path_to_self_is_single_node() -> None:
    g = _weighted_line()
    _, prev = dijkstra_shortest_path(g, 0)
    assert reconstruct_path(prev, 0, 0) == [0]


def test_star_center_reaches_leaves_in_one_hop() -> None:
    g = _star()
    dist, prev = dijkstra_shortest_path(g, 0)
    assert dist == {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0}
    assert reconstruct_path(prev, 0, 2) == [0, 2]


def test_star_leaf_to_leaf_goes_through_center() -> None:
    g = _star()
    dist, prev = dijkstra_shortest_path(g, 1)
    assert dist[1] == 0.0
    assert dist[0] == 1.0
    assert dist[2] == 2.0
    assert dist[3] == 2.0
    assert reconstruct_path(prev, 1, 3) == [1, 0, 3]


def test_weighted_star_uses_edge_weights() -> None:
    g = _weighted_star()
    dist, prev = dijkstra_shortest_path(g, 1)
    assert dist[0] == pytest.approx(1.0)
    assert dist[2] == pytest.approx(3.0)
    assert dist[3] == pytest.approx(4.0)
    assert reconstruct_path(prev, 1, 3) == [1, 0, 3]


def test_unknown_source_raises() -> None:
    g = _star()
    with pytest.raises(KeyError, match="not a node"):
        dijkstra_shortest_path(g, 99)


def test_negative_weight_raises() -> None:
    g = Graph([(0, 1, -1.0)])
    with pytest.raises(ValueError, match="non-negative"):
        dijkstra_shortest_path(g, 0)


def test_cli_shortest_path_prints_target(capsys) -> None:
    assert main(["shortest-path", "--source", "0", "--target", "33"]) == 0
    out = capsys.readouterr().out
    assert "distance:" in out
    assert "path:" in out
    assert "0 ->" in out
    assert "33" in out


def test_cli_shortest_path_lists_closest(capsys) -> None:
    assert main(["shortest-path", "--source", "0", "--top", "3"]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.strip().splitlines() if line]
    assert lines[0].startswith("0: 0.000000")
    assert len(lines) == 3


def test_demo_report_includes_shortest_paths(tmp_path) -> None:
    out = tmp_path / "report.md"
    assert main(["demo", "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## Shortest paths" in text
    assert "Dijkstra" in text
    assert "node 33" in text
    assert "Path:" in text
