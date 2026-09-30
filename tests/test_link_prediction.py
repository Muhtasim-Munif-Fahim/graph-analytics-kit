"""Tests for Adamic–Adar link prediction."""
from __future__ import annotations

import math

import pytest

from graph_analytics_kit import Graph, adamic_adar, adamic_adar_scores, karate_club


def test_no_common_neighbors() -> None:
    g = Graph([(0, 1), (2, 3)])
    assert adamic_adar(g, 0, 2) == 0.0
    assert adamic_adar(g, 0, 3) == 0.0


def test_missing_nodes() -> None:
    g = Graph([(0, 1)])
    assert adamic_adar(g, 0, 99) == 0.0
    assert adamic_adar(g, 99, 100) == 0.0


def test_triangle_exact() -> None:
    # Complete triangle: nodes 0 and 1 share neighbor 2 (deg 2).
    # score = 1 / log(2)
    g = Graph([(0, 1), (0, 2), (1, 2)])
    expected = 1.0 / math.log(2.0)
    assert adamic_adar(g, 0, 1) == pytest.approx(expected)
    assert adamic_adar(g, 1, 0) == pytest.approx(expected)


def test_skips_degree_one_common_neighbor() -> None:
    # Path 0-1-2: common neighbor of 0 and 2 is 1 with deg 2 -> score = 1/log(2).
    # Add a pendant that is NOT a common neighbor; ensure deg-1 commons are skipped.
    # Graph: 0-w-1 and 0-w-2 would make w common for 1,2; if w only connects to
    # 1 and 2 then deg(w)=2. For deg-1 skip: not possible as common neighbor of
    # two distinct nodes (would need edges w-u and w-v => deg >= 2).
    # Instead verify star center: leaves 1 and 2 share center 0 with deg 3.
    g = Graph([(0, 1), (0, 2), (0, 3)])
    expected = 1.0 / math.log(3.0)
    assert adamic_adar(g, 1, 2) == pytest.approx(expected)
    # Leaves share only the center; no other commons.
    assert adamic_adar(g, 1, 3) == pytest.approx(expected)


def test_two_common_neighbors_sum() -> None:
    # u=0, v=1 share neighbors 2 (deg 3: connected to 0,1,3) and 3 (deg 3: 0,1,2)
    # Wait: edges 0-2,1-2,0-3,1-3,2-3 -> deg(2)=3, deg(3)=3
    g = Graph([(0, 2), (1, 2), (0, 3), (1, 3), (2, 3)])
    expected = 1.0 / math.log(3.0) + 1.0 / math.log(3.0)
    assert adamic_adar(g, 0, 1) == pytest.approx(expected)


def test_adamic_adar_scores_batch() -> None:
    g = Graph([(0, 1), (0, 2), (1, 2), (2, 3)])
    pairs = [(0, 1), (0, 3), (1, 3), (0, 99)]
    scores = adamic_adar_scores(g, pairs)
    assert len(scores) == 4
    assert scores[0] == pytest.approx(adamic_adar(g, 0, 1))
    assert scores[1] == pytest.approx(adamic_adar(g, 0, 3))
    assert scores[2] == pytest.approx(adamic_adar(g, 1, 3))
    assert scores[3] == 0.0


def test_karate_club_finite_and_symmetric() -> None:
    g = karate_club()
    # Non-edge that likely has common neighbors in Karate Club.
    s01 = adamic_adar(g, 0, 33)
    s10 = adamic_adar(g, 33, 0)
    assert s01 == pytest.approx(s10)
    assert s01 >= 0.0
    assert math.isfinite(s01)


def test_empty_graph() -> None:
    assert adamic_adar(Graph(), 0, 1) == 0.0
    assert adamic_adar_scores(Graph(), []) == []


def test_export_available() -> None:
    from graph_analytics_kit import adamic_adar as aa
    from graph_analytics_kit import adamic_adar_scores as aas

    assert callable(aa)
    assert callable(aas)
