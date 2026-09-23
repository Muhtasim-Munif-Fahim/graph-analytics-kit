"""Tests for k-core numbers.

Closeness centrality already exists, so these tests cover core decomposition
(the pivot measure) on the same path, star, and complete graphs.
"""
from __future__ import annotations

import pytest

from graph_analytics_kit import Graph, core_number, karate_club
from graph_analytics_kit.centrality import core_number as core_number_from_centrality
from graph_analytics_kit.cli import main


def _path(n: int, *, start: int = 0) -> Graph:
    nodes = [start + i for i in range(n)]
    return Graph([(nodes[i], nodes[i + 1]) for i in range(n - 1)])


def _star(n_leaves: int) -> Graph:
    return Graph([(0, leaf) for leaf in range(1, n_leaves + 1)])


def _complete(nodes: list[int]) -> Graph:
    edges = [
        (nodes[i], nodes[j])
        for i in range(len(nodes))
        for j in range(i + 1, len(nodes))
    ]
    return Graph(edges)


def _cycle(n: int) -> Graph:
    edges = [(i, (i + 1) % n) for i in range(n)]
    return Graph(edges)


def _induced_min_degree(g: Graph, members: set[int]) -> int:
    if not members:
        return 0

    def _linked(node: int, nbr: int) -> bool:
        if nbr == node or nbr not in members:
            return False
        if g.directed:
            return g.has_edge(node, nbr) or g.has_edge(nbr, node)
        return g.has_edge(node, nbr)

    return min(sum(1 for nbr in members if _linked(node, nbr)) for node in members)


def _assert_k_core_property(g: Graph, cores: dict[int, int]) -> None:
    """Every {v | core(v) >= k} induces a subgraph of minimum degree >= k."""
    if not cores:
        return
    for k in range(1, max(cores.values()) + 1):
        members = {node for node, value in cores.items() if value >= k}
        assert _induced_min_degree(g, members) >= k


def _peel_core_numbers(g: Graph) -> dict[int, int]:
    """Independent oracle: repeated deletion of minimum-degree nodes."""
    adj = {node: set() for node in g.nodes}
    for u, v, _weight in g.edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    remaining = set(g.nodes)
    cores: dict[int, int] = {}
    while remaining:
        threshold = min(len(adj[node] & remaining) for node in remaining)
        layer: list[int] = []
        progressed = True
        while progressed:
            progressed = False
            for node in list(remaining):
                if len(adj[node] & remaining) <= threshold:
                    remaining.remove(node)
                    layer.append(node)
                    progressed = True
        for node in layer:
            cores[node] = threshold
    return cores


def _assert_matches_oracle(g: Graph) -> dict[int, int]:
    cores = core_number(g)
    assert cores == _peel_core_numbers(g)
    assert set(cores) == set(g.nodes)
    for value in cores.values():
        assert isinstance(value, int)
        assert value >= 0
    _assert_k_core_property(g, cores)
    return cores


def test_core_number_exported_from_public_api() -> None:
    assert core_number is core_number_from_centrality
    g = _complete([0, 1, 2])
    assert core_number(g) == {0: 2, 1: 2, 2: 2}


def test_empty_graph_returns_empty() -> None:
    assert core_number(Graph([])) == {}


def test_single_node_and_edgeless_are_zero() -> None:
    alone = Graph()
    alone.add_node(4)
    assert core_number(alone) == {4: 0}

    edgeless = Graph()
    for node in (1, 2, 3):
        edgeless.add_node(node)
    assert core_number(edgeless) == {1: 0, 2: 0, 3: 0}


def test_self_loop_is_ignored() -> None:
    loop = Graph([(0, 0)])
    assert core_number(loop) == {0: 0}
    with_edge = Graph([(0, 0), (0, 1), (1, 1)])
    assert core_number(with_edge) == {0: 1, 1: 1}


def test_path_all_core_one() -> None:
    g = _path(4)
    assert core_number(g) == {0: 1, 1: 1, 2: 1, 3: 1}
    assert _assert_matches_oracle(_path(6)) == {i: 1 for i in range(6)}
    # Non-contiguous labels stay attached to the right nodes.
    labeled = _path(3, start=10)
    assert core_number(labeled) == {10: 1, 11: 1, 12: 1}


def test_star_all_core_one() -> None:
    g = _star(3)
    assert core_number(g) == {0: 1, 1: 1, 2: 1, 3: 1}
    larger = _star(8)
    assert core_number(larger) == {node: 1 for node in range(9)}
    assert core_number(larger)[0] == 1


def test_complete_graph_core_is_n_minus_one() -> None:
    k1 = Graph()
    k1.add_node(0)
    assert core_number(k1) == {0: 0}
    assert core_number(_complete([0, 1])) == {0: 1, 1: 1}
    triangle = _complete([0, 1, 2])
    assert core_number(triangle) == {0: 2, 1: 2, 2: 2}
    for n in (4, 5, 6):
        nodes = list(range(n))
        cores = core_number(_complete(nodes))
        assert cores == {node: n - 1 for node in nodes}


def test_complete_graph_noncontiguous_labels() -> None:
    nodes = [10, 20, 30, 40]
    assert core_number(_complete(nodes)) == {node: 3 for node in nodes}


def test_cycle_is_two_core() -> None:
    g = _cycle(6)
    assert core_number(g) == {i: 2 for i in range(6)}


def test_triangle_with_pendant() -> None:
    g = Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
    assert core_number(g) == {0: 2, 1: 2, 2: 2, 3: 1}


def test_nested_clique_and_triangle() -> None:
    # K4 on {0,1,2,3} plus a triangle {3,4,5}: clique vertices are 3-core,
    # the two outer triangle vertices are 2-core.
    k4 = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    g = Graph(k4 + [(3, 4), (3, 5), (4, 5)])
    assert core_number(g) == {0: 3, 1: 3, 2: 3, 3: 3, 4: 2, 5: 2}


def test_disconnected_components_keep_their_cores() -> None:
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6)]
    g = Graph(edges)
    g.add_node(7)
    assert core_number(g) == {
        0: 2,
        1: 2,
        2: 2,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 0,
    }


def test_weights_do_not_change_core_numbers() -> None:
    unweighted = _complete([0, 1, 2, 3])
    weighted = Graph(
        [(0, 1, 0.1), (0, 2, 9.0), (0, 3, 4.0), (1, 2, 2.0), (1, 3, 8.0), (2, 3, 1.0)]
    )
    assert core_number(weighted) == core_number(unweighted)


def test_directed_graphs_use_underlying_undirected_graph() -> None:
    cycle = Graph([(0, 1), (1, 2), (2, 0)], directed=True)
    assert core_number(cycle) == {0: 2, 1: 2, 2: 2}
    star = Graph([(0, 1), (0, 2), (0, 3)], directed=True)
    assert core_number(star) == {0: 1, 1: 1, 2: 1, 3: 1}
    mutual = Graph([(0, 1), (1, 0)], directed=True)
    assert core_number(mutual) == {0: 1, 1: 1}
    _assert_matches_oracle(cycle)
    _assert_matches_oracle(star)
    _assert_matches_oracle(mutual)


def test_matches_peeling_oracle_on_mixed_graph() -> None:
    g = Graph(
        [
            (0, 1), (1, 2), (2, 0),
            (0, 3), (1, 3), (2, 3),
            (3, 4),
            (4, 5), (5, 6), (6, 4),
            (7, 8),
        ]
    )
    g.add_node(9)
    cores = _assert_matches_oracle(g)
    assert cores[9] == 0
    assert cores[7] == 1
    assert cores[0] == 3
    assert cores[4] == 2


def test_does_not_mutate_graph() -> None:
    g = _star(4)
    before = list(g.edges)
    core_number(g)
    assert list(g.edges) == before
    assert g.degree(0) == 4


def test_karate_club_core_numbers() -> None:
    g = karate_club()
    cores = _assert_matches_oracle(g)
    # Zachary's Karate Club has degeneracy 4; the hubs sit in that core.
    assert max(cores.values()) == 4
    assert cores[0] == 4
    assert cores[33] == 4
    assert min(cores.values()) >= 1


def test_cli_core_prints_karate(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["centrality", "--measure", "core", "--top", "3"]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.strip().splitlines() if line]
    assert len(lines) == 3
    for line in lines:
        node_text, score_text = line.split(":")
        assert int(node_text.strip()) >= 0
        assert float(score_text) == pytest.approx(4.0)


def test_demo_report_includes_core(tmp_path) -> None:
    out = tmp_path / "report.md"
    assert main(["demo", "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Core" in text
    assert (
        "| Node | Degree | Closeness | Betweenness | Core | PageRank | Eigenvector | "
        "Katz | Hub | Authority | Clustering | Community |"
        in text
    )
