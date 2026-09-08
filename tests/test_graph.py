"""Graph data model and built-in datasets."""
from __future__ import annotations


import numpy as np
import pytest

from graph_analytics_kit import Graph, karate_club


def _triangle() -> Graph:
    return Graph([(0, 1), (0, 2), (1, 2)])


def _star() -> Graph:
    return Graph([(0, 1), (0, 2), (0, 3)])


def test_triangle_counts() -> None:
    g = _triangle()
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 3
    assert set(g.nodes) == {0, 1, 2}


def test_star_degrees() -> None:
    g = _star()
    assert g.degree(0) == 3
    assert g.degree(1) == 1
    assert g.degree(2) == 1
    assert g.degree(3) == 1
    assert g.density() == pytest.approx(0.5, abs=1e-12)


def test_edges_deduplicated() -> None:
    g = Graph([(0, 1), (1, 0), (0, 2)])
    assert g.number_of_edges() == 2
    assert set(g.edges) == {(0, 1, 1.0), (0, 2, 1.0)}


def test_has_edge_and_neighbors() -> None:
    g = _triangle()
    assert g.has_edge(0, 2)
    assert not g.has_edge(0, 3)
    assert set(g.neighbors(0)) == {1, 2}


def test_adjacency_matrix_triangle() -> None:
    g = _triangle()
    A = g.adjacency_matrix()
    expected = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    np.testing.assert_allclose(A, expected)


def test_adjacency_matrix_symmetric() -> None:
    g = _star()
    A = g.adjacency_matrix()
    np.testing.assert_allclose(A, A.T)


def test_karate_club_dataset() -> None:
    g = karate_club()
    assert g.number_of_nodes() == 34
    assert g.number_of_edges() == 78
    assert g.is_connected()
    assert g.degree(0) == 16
    assert set(g.nodes) == set(range(34))


def test_from_edge_list(tmp_path) -> None:
    path = tmp_path / "edges.tsv"
    path.write_text("0 1\n0 2\n1 2\n", encoding="utf-8")
    g = Graph.from_edge_list(path, delimiter=" ")
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 3
    assert g.has_edge(0, 2)


def test_from_edge_list_weighted(tmp_path) -> None:
    path = tmp_path / "weighted.tsv"
    path.write_text("0 1 0.5\n1 2 1.5\n", encoding="utf-8")
    g = Graph.from_edge_list(path, delimiter=" ", weighted=True)
    assert g.has_edge(0, 1)
    _, _, w = next(e for e in g.edges if {e[0], e[1]} == {0, 1})
    assert pytest.approx(w, abs=1e-12) == 0.5


def test_without_edges_preserves_structure() -> None:
    g = Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
    reduced = g.without_edges([(0, 1)])
    assert reduced.number_of_nodes() == 4
    assert reduced.number_of_edges() == 3
    assert not reduced.has_edge(0, 1)
    assert reduced.has_edge(1, 2)


def test_summary_dict() -> None:
    g = _star()
    s = g.summary()
    assert s["n_nodes"] == 4
    assert s["n_edges"] == 3
    assert s["directed"] is False
    assert s["connected"] is True
    assert s["density"] == pytest.approx(0.5, abs=1e-12)
