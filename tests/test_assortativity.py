"""Tests for Newman degree assortativity."""
from __future__ import annotations

import math

import pytest

from graph_analytics_kit import Graph, degree_assortativity, karate_club


def test_empty_and_edgeless() -> None:
    assert degree_assortativity(Graph()) == 0.0
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    assert degree_assortativity(g) == 0.0


def test_single_edge() -> None:
    # Both ends have degree 1 -> zero variance -> 0.0
    assert degree_assortativity(Graph([(0, 1)])) == 0.0


def test_complete_triangle_zero_variance() -> None:
    # Every node degree 2, every edge joins 2-2 -> undefined -> 0.0
    g = Graph([(0, 1), (0, 2), (1, 2)])
    assert degree_assortativity(g) == 0.0


def test_star_is_disassortative() -> None:
    # Center (deg 4) connected to four leaves (deg 1) -> strongly negative.
    g = Graph([(0, 1), (0, 2), (0, 3), (0, 4)])
    r = degree_assortativity(g)
    assert r < -0.5
    # Exact: edges all join deg 4 and deg 1.
    # mu = (4+1)/2 = 2.5, xy/m = 4*1 = 4, x2+y2/(2m) = (16+1)/2 = 8.5
    # r = (4 - 6.25) / (8.5 - 6.25) = (-2.25) / 2.25 = -1.0
    assert r == pytest.approx(-1.0)


def test_assortative_k4_plus_edge() -> None:
    # A 3-regular K4 plus a disjoint 1-regular edge: every edge joins
    # equal-degree endpoints, so Newman assortativity is exactly +1.
    edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (4, 5),
    ]
    assert degree_assortativity(Graph(edges)) == pytest.approx(1.0)


def test_path_of_three() -> None:
    # Degrees: 1-2-1. One edge joins 1-2, one joins 2-1.
    # mu = (1+2)/2 = 1.5 (same both edges), xy/m = 2, x2+y2/(2m) = (1+4)/2 = 2.5
    # r = (2 - 2.25) / (2.5 - 2.25) = (-0.25)/0.25 = -1.0
    g = Graph([(0, 1), (1, 2)])
    assert degree_assortativity(g) == pytest.approx(-1.0)


def test_karate_club_in_expected_range() -> None:
    r = degree_assortativity(karate_club())
    # Zachary's Karate Club is mildly disassortative.
    assert -1.0 <= r <= 1.0
    assert r < 0.0
    assert math.isfinite(r)


def test_export_available() -> None:
    from graph_analytics_kit import degree_assortativity as exported

    assert callable(exported)
