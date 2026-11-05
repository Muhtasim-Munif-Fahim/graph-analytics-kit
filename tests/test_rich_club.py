"""Tests for the rich-club coefficient."""
from __future__ import annotations

import pytest

from graph_analytics_kit import Graph, karate_club, rich_club_coefficient


def test_complete_graph_is_fully_rich() -> None:
    g = Graph([(i, j) for i in range(5) for j in range(i + 1, 5)])
    coeffs = rich_club_coefficient(g)
    assert set(coeffs) == {0, 1, 2, 3}
    assert all(v == pytest.approx(1.0) for v in coeffs.values())


def test_star_has_no_rich_club() -> None:
    g = Graph([(0, i) for i in range(1, 6)])
    # k=0: all 6 nodes are rich, 5 edges out of 15 possible.
    assert rich_club_coefficient(g, 0) == pytest.approx(5 / 15)
    # k=1: only the hub survives -> undefined -> 0.0
    assert rich_club_coefficient(g, 1) == 0.0
    assert rich_club_coefficient(g) == {0: pytest.approx(1 / 3)}


def test_core_periphery() -> None:
    # Triangle core 0-1-2, each with two pendant leaves.
    edges = [(0, 1), (1, 2), (0, 2)]
    leaf = 3
    for hub in (0, 1, 2):
        for _ in range(2):
            edges.append((hub, leaf))
            leaf += 1
    g = Graph(edges)
    assert rich_club_coefficient(g, 1) == pytest.approx(1.0)
    assert rich_club_coefficient(g, 0) == pytest.approx(2 * 9 / (9 * 8))


def test_self_loops_ignored_and_validation() -> None:
    g = Graph([(0, 1), (1, 1)])
    assert rich_club_coefficient(g, 0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rich_club_coefficient(Graph([(0, 1)], directed=True))
    with pytest.raises(ValueError):
        rich_club_coefficient(g, -1)


def test_karate_club_values_in_range() -> None:
    coeffs = rich_club_coefficient(karate_club())
    assert coeffs[0] == pytest.approx(78 * 2 / (34 * 33))
    assert all(0.0 <= v <= 1.0 for v in coeffs.values())
    assert max(coeffs) < 16
