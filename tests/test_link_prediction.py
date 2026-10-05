"""Tests for Adamic–Adar, Jaccard, Resource Allocation, Preferential Attachment, and Common Neighbors link prediction."""
from __future__ import annotations

import math

import pytest

from graph_analytics_kit import (
    Graph,
    adamic_adar,
    adamic_adar_scores,
    common_neighbors,
    common_neighbors_scores,
    jaccard,
    jaccard_scores,
    karate_club,
    preferential_attachment,
    preferential_attachment_scores,
    resource_allocation,
    resource_allocation_scores,
)


def test_no_common_neighbors() -> None:
    g = Graph([(0, 1), (2, 3)])
    assert adamic_adar(g, 0, 2) == 0.0
    assert adamic_adar(g, 0, 3) == 0.0
    assert jaccard(g, 0, 2) == 0.0
    assert jaccard(g, 0, 3) == 0.0
    assert resource_allocation(g, 0, 2) == 0.0
    assert resource_allocation(g, 0, 3) == 0.0


def test_missing_nodes() -> None:
    g = Graph([(0, 1)])
    assert adamic_adar(g, 0, 99) == 0.0
    assert adamic_adar(g, 99, 100) == 0.0
    assert jaccard(g, 0, 99) == 0.0
    assert jaccard(g, 99, 100) == 0.0
    assert resource_allocation(g, 0, 99) == 0.0
    assert resource_allocation(g, 99, 100) == 0.0


def test_triangle_exact() -> None:
    # Complete triangle: nodes 0 and 1 share neighbor 2 (deg 2).
    # Adamic–Adar score = 1 / log(2)
    # Jaccard: N(0)={2}, N(1)={2} after excluding endpoints → |∩|=1, |∪|=1 → 1.0
    g = Graph([(0, 1), (0, 2), (1, 2)])
    expected = 1.0 / math.log(2.0)
    assert adamic_adar(g, 0, 1) == pytest.approx(expected)
    assert adamic_adar(g, 1, 0) == pytest.approx(expected)
    assert jaccard(g, 0, 1) == pytest.approx(1.0)
    assert jaccard(g, 1, 0) == pytest.approx(1.0)


def test_skips_degree_one_common_neighbor() -> None:
    # Star: leaves 1 and 2 share center 0 with deg 3.
    g = Graph([(0, 1), (0, 2), (0, 3)])
    expected = 1.0 / math.log(3.0)
    assert adamic_adar(g, 1, 2) == pytest.approx(expected)
    assert adamic_adar(g, 1, 3) == pytest.approx(expected)
    # Jaccard: N(1)={0}, N(2)={0} → 1.0
    assert jaccard(g, 1, 2) == pytest.approx(1.0)
    assert jaccard(g, 1, 3) == pytest.approx(1.0)


def test_two_common_neighbors_sum() -> None:
    # u=0, v=1 share neighbors 2 and 3 (both deg 3)
    g = Graph([(0, 2), (1, 2), (0, 3), (1, 3), (2, 3)])
    expected = 1.0 / math.log(3.0) + 1.0 / math.log(3.0)
    assert adamic_adar(g, 0, 1) == pytest.approx(expected)
    # Jaccard: N(0)={2,3}, N(1)={2,3} → 1.0
    assert jaccard(g, 0, 1) == pytest.approx(1.0)


def test_jaccard_partial_overlap() -> None:
    # 0 neighbors {1,2}; 3 neighbors {2,4} → ∩={2}, ∪={1,2,4} → 1/3
    g = Graph([(0, 1), (0, 2), (3, 2), (3, 4)])
    assert jaccard(g, 0, 3) == pytest.approx(1.0 / 3.0)
    assert jaccard(g, 3, 0) == pytest.approx(1.0 / 3.0)


def test_jaccard_empty_union() -> None:
    # Isolated nodes have empty neighborhoods → empty union → 0.0
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    assert jaccard(g, 0, 1) == 0.0


def test_adamic_adar_scores_batch() -> None:
    g = Graph([(0, 1), (0, 2), (1, 2), (2, 3)])
    pairs = [(0, 1), (0, 3), (1, 3), (0, 99)]
    scores = adamic_adar_scores(g, pairs)
    assert len(scores) == 4
    assert scores[0] == pytest.approx(adamic_adar(g, 0, 1))
    assert scores[1] == pytest.approx(adamic_adar(g, 0, 3))
    assert scores[2] == pytest.approx(adamic_adar(g, 1, 3))
    assert scores[3] == 0.0


def test_jaccard_scores_batch() -> None:
    g = Graph([(0, 1), (0, 2), (1, 2), (2, 3)])
    pairs = [(0, 1), (0, 3), (1, 3), (0, 99)]
    scores = jaccard_scores(g, pairs)
    assert len(scores) == 4
    assert scores[0] == pytest.approx(jaccard(g, 0, 1))
    assert scores[1] == pytest.approx(jaccard(g, 0, 3))
    assert scores[2] == pytest.approx(jaccard(g, 1, 3))
    assert scores[3] == 0.0


def test_karate_club_finite_and_symmetric() -> None:
    g = karate_club()
    s01 = adamic_adar(g, 0, 33)
    s10 = adamic_adar(g, 33, 0)
    assert s01 == pytest.approx(s10)
    assert s01 >= 0.0
    assert math.isfinite(s01)
    j01 = jaccard(g, 0, 33)
    j10 = jaccard(g, 33, 0)
    assert j01 == pytest.approx(j10)
    assert 0.0 <= j01 <= 1.0
    assert math.isfinite(j01)


def test_empty_graph() -> None:
    assert adamic_adar(Graph(), 0, 1) == 0.0
    assert adamic_adar_scores(Graph(), []) == []
    assert jaccard(Graph(), 0, 1) == 0.0
    assert jaccard_scores(Graph(), []) == []


def test_export_available() -> None:
    from graph_analytics_kit import adamic_adar as aa
    from graph_analytics_kit import adamic_adar_scores as aas
    from graph_analytics_kit import jaccard as jc
    from graph_analytics_kit import jaccard_scores as jcs

    assert callable(aa)
    assert callable(aas)
    assert callable(jc)
    assert callable(jcs)


def test_resource_allocation_triangle_exact() -> None:
    # Complete triangle: 0 and 1 share neighbor 2 (deg 2) → RA = 1/2
    g = Graph([(0, 1), (0, 2), (1, 2)])
    assert resource_allocation(g, 0, 1) == pytest.approx(0.5)
    assert resource_allocation(g, 1, 0) == pytest.approx(0.5)


def test_resource_allocation_star() -> None:
    # Leaves share center deg 3 → RA = 1/3
    g = Graph([(0, 1), (0, 2), (0, 3)])
    assert resource_allocation(g, 1, 2) == pytest.approx(1.0 / 3.0)
    assert resource_allocation(g, 1, 3) == pytest.approx(1.0 / 3.0)


def test_resource_allocation_two_common() -> None:
    # 0,1 share 2 and 3 (both deg 3) → RA = 1/3 + 1/3
    g = Graph([(0, 2), (1, 2), (0, 3), (1, 3), (2, 3)])
    assert resource_allocation(g, 0, 1) == pytest.approx(2.0 / 3.0)


def test_resource_allocation_no_common_and_missing() -> None:
    g = Graph([(0, 1), (2, 3)])
    assert resource_allocation(g, 0, 2) == 0.0
    assert resource_allocation(g, 0, 99) == 0.0
    assert resource_allocation(Graph(), 0, 1) == 0.0


def test_resource_allocation_scores_batch() -> None:
    g = Graph([(0, 1), (0, 2), (1, 2), (2, 3)])
    pairs = [(0, 1), (0, 3), (1, 3), (0, 99)]
    scores = resource_allocation_scores(g, pairs)
    assert len(scores) == 4
    assert scores[0] == pytest.approx(resource_allocation(g, 0, 1))
    assert scores[1] == pytest.approx(resource_allocation(g, 0, 3))
    assert scores[2] == pytest.approx(resource_allocation(g, 1, 3))
    assert scores[3] == 0.0
    assert resource_allocation_scores(Graph(), []) == []


def test_resource_allocation_karate_finite_and_symmetric() -> None:
    g = karate_club()
    s01 = resource_allocation(g, 0, 33)
    s10 = resource_allocation(g, 33, 0)
    assert s01 == pytest.approx(s10)
    assert s01 >= 0.0
    assert math.isfinite(s01)


def test_resource_allocation_export_available() -> None:
    from graph_analytics_kit import resource_allocation as ra
    from graph_analytics_kit import resource_allocation_scores as ras

    assert callable(ra)
    assert callable(ras)


def test_preferential_attachment_star() -> None:
    # Star: center deg 3, leaves deg 1 → PA(1,2) = 1*1 = 1; PA(0,1) = 3*1 = 3
    g = Graph([(0, 1), (0, 2), (0, 3)])
    assert preferential_attachment(g, 1, 2) == pytest.approx(1.0)
    assert preferential_attachment(g, 2, 1) == pytest.approx(1.0)
    assert preferential_attachment(g, 0, 1) == pytest.approx(3.0)
    assert preferential_attachment(g, 1, 0) == pytest.approx(3.0)


def test_preferential_attachment_triangle() -> None:
    # Complete triangle: every node deg 2 → PA = 4
    g = Graph([(0, 1), (0, 2), (1, 2)])
    assert preferential_attachment(g, 0, 1) == pytest.approx(4.0)
    assert preferential_attachment(g, 1, 0) == pytest.approx(4.0)


def test_preferential_attachment_path() -> None:
    # Path 0-1-2-3: deg(0)=1, deg(1)=2, deg(2)=2, deg(3)=1
    g = Graph([(0, 1), (1, 2), (2, 3)])
    assert preferential_attachment(g, 0, 3) == pytest.approx(1.0)
    assert preferential_attachment(g, 0, 2) == pytest.approx(2.0)
    assert preferential_attachment(g, 1, 2) == pytest.approx(4.0)


def test_preferential_attachment_missing_and_empty() -> None:
    g = Graph([(0, 1)])
    assert preferential_attachment(g, 0, 99) == 0.0
    assert preferential_attachment(g, 99, 100) == 0.0
    assert preferential_attachment(Graph(), 0, 1) == 0.0


def test_preferential_attachment_isolated() -> None:
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    assert preferential_attachment(g, 0, 1) == 0.0


def test_preferential_attachment_scores_batch() -> None:
    g = Graph([(0, 1), (0, 2), (1, 2), (2, 3)])
    pairs = [(0, 1), (0, 3), (1, 3), (0, 99)]
    scores = preferential_attachment_scores(g, pairs)
    assert len(scores) == 4
    assert scores[0] == pytest.approx(preferential_attachment(g, 0, 1))
    assert scores[1] == pytest.approx(preferential_attachment(g, 0, 3))
    assert scores[2] == pytest.approx(preferential_attachment(g, 1, 3))
    assert scores[3] == 0.0
    assert preferential_attachment_scores(Graph(), []) == []


def test_preferential_attachment_karate_finite_and_symmetric() -> None:
    g = karate_club()
    s01 = preferential_attachment(g, 0, 33)
    s10 = preferential_attachment(g, 33, 0)
    assert s01 == pytest.approx(s10)
    assert s01 >= 0.0
    assert math.isfinite(s01)
    # High-degree hubs should score higher than two leaves if degrees differ.
    assert preferential_attachment(g, 0, 33) == float(g.degree(0)) * float(g.degree(33))


def test_preferential_attachment_export_available() -> None:
    from graph_analytics_kit import preferential_attachment as pa
    from graph_analytics_kit import preferential_attachment_scores as pas

    assert callable(pa)
    assert callable(pas)


def test_common_neighbors_triangle_exact() -> None:
    # Complete triangle: 0 and 1 share neighbor 2 → CN = 1
    g = Graph([(0, 1), (0, 2), (1, 2)])
    assert common_neighbors(g, 0, 1) == pytest.approx(1.0)
    assert common_neighbors(g, 1, 0) == pytest.approx(1.0)


def test_common_neighbors_star() -> None:
    # Leaves share center → CN = 1
    g = Graph([(0, 1), (0, 2), (0, 3)])
    assert common_neighbors(g, 1, 2) == pytest.approx(1.0)
    assert common_neighbors(g, 1, 3) == pytest.approx(1.0)


def test_common_neighbors_two_common() -> None:
    g = Graph([(0, 2), (1, 2), (0, 3), (1, 3), (2, 3)])
    assert common_neighbors(g, 0, 1) == pytest.approx(2.0)


def test_common_neighbors_no_common_and_missing() -> None:
    g = Graph([(0, 1), (2, 3)])
    assert common_neighbors(g, 0, 2) == 0.0
    assert common_neighbors(g, 0, 99) == 0.0
    assert common_neighbors(Graph(), 0, 1) == 0.0


def test_common_neighbors_partial_overlap() -> None:
    # 0 neighbors {1,2}; 3 neighbors {2,4} → ∩={2} → 1
    g = Graph([(0, 1), (0, 2), (3, 2), (3, 4)])
    assert common_neighbors(g, 0, 3) == pytest.approx(1.0)
    assert common_neighbors(g, 3, 0) == pytest.approx(1.0)


def test_common_neighbors_scores_batch() -> None:
    g = Graph([(0, 1), (0, 2), (1, 2), (2, 3)])
    pairs = [(0, 1), (0, 3), (1, 3), (0, 99)]
    scores = common_neighbors_scores(g, pairs)
    assert len(scores) == 4
    assert scores[0] == pytest.approx(common_neighbors(g, 0, 1))
    assert scores[1] == pytest.approx(common_neighbors(g, 0, 3))
    assert scores[2] == pytest.approx(common_neighbors(g, 1, 3))
    assert scores[3] == 0.0
    assert common_neighbors_scores(Graph(), []) == []


def test_common_neighbors_karate_finite_and_symmetric() -> None:
    g = karate_club()
    s01 = common_neighbors(g, 0, 33)
    s10 = common_neighbors(g, 33, 0)
    assert s01 == pytest.approx(s10)
    assert s01 >= 0.0
    assert math.isfinite(s01)
    # Count must be an integer-valued float.
    assert s01 == float(int(s01))


def test_common_neighbors_export_available() -> None:
    from graph_analytics_kit import common_neighbors as cn
    from graph_analytics_kit import common_neighbors_scores as cns

    assert callable(cn)
    assert callable(cns)

