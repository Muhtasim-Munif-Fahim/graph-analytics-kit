"""Tests for vertex colouring and Eulerian circuits/paths."""

from __future__ import annotations

import itertools
import random

import pytest

from graph_analytics_kit import (
    COLORING_STRATEGIES,
    Graph,
    chromatic_number,
    color_classes,
    coloring_summary,
    eulerian_circuit,
    eulerian_path,
    greedy_color,
    has_eulerian_path,
    is_eulerian,
    is_proper_coloring,
    karate_club,
    odd_degree_nodes,
)
from graph_analytics_kit.cli import main


def _brute_force_chi(g: Graph) -> int:
    nodes = g.nodes
    idx = {u: i for i, u in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v, _ in g.edges]
    for k in range(1, len(nodes) + 1):
        for c in itertools.product(range(k), repeat=len(nodes) - 1):
            col = (0,) + c
            if all(col[a] != col[b] for a, b in edges):
                return k
    return len(nodes)


def _random_graph(rng: random.Random, n: int, p: float) -> Graph:
    g = Graph([(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p])
    for u in range(n):
        g.add_node(u)
    return g


def test_every_strategy_gives_proper_coloring_on_karate():
    g = karate_club()
    for strategy in COLORING_STRATEGIES:
        col = greedy_color(g, strategy, seed=3)
        assert is_proper_coloring(g, col)
        assert set(col) == set(g.nodes)
        assert set(col.values()) == set(range(max(col.values()) + 1))


def test_karate_reference_values_match_networkx():
    # networkx 3.x greedy_color: largest_first=5, smallest_last=5, DSATUR=5, bfs=6
    summary = coloring_summary(karate_club())
    assert summary["largest_first"] == 5
    assert summary["smallest_last"] == 5
    assert summary["dsatur"] == 5
    assert summary["connected_sequential_bfs"] == 6
    assert chromatic_number(karate_club()) == 5


def test_largest_first_matches_networkx_rule():
    # Stable sort by decreasing degree, then smallest free colour.
    g = Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (4, 0)])
    col = greedy_color(g, "largest_first")
    assert col == {0: 0, 2: 1, 1: 2, 3: 2, 4: 1}


def test_dsatur_is_optimal_on_bipartite_and_cycles():
    even_cycle = Graph([(i, (i + 1) % 8) for i in range(8)])
    odd_cycle = Graph([(i, (i + 1) % 7) for i in range(7)])
    grid = Graph([(r * 4 + c, r * 4 + c + 1) for r in range(4) for c in range(3)]
                 + [(r * 4 + c, (r + 1) * 4 + c) for r in range(3) for c in range(4)])
    for g, k in [(even_cycle, 2), (odd_cycle, 3), (grid, 2)]:
        col = greedy_color(g, "dsatur")
        assert max(col.values()) + 1 == k
        assert chromatic_number(g) == k


def test_smallest_last_respects_degeneracy_bound():
    rng = random.Random(0)
    from graph_analytics_kit import core_number

    for _ in range(30):
        g = _random_graph(rng, rng.randint(2, 25), rng.random())
        if g.number_of_edges() == 0:
            continue
        degeneracy = max(core_number(g).values())
        col = greedy_color(g, "smallest_last")
        assert max(col.values()) + 1 <= degeneracy + 1


def test_chromatic_number_matches_brute_force():
    rng = random.Random(1)
    for _ in range(60):
        g = _random_graph(rng, rng.randint(1, 8), rng.random())
        chi, col = chromatic_number(g, return_coloring=True)
        assert chi == _brute_force_chi(g)
        assert is_proper_coloring(g, col)
        assert max(col.values()) + 1 == chi


def test_known_chromatic_numbers():
    petersen = Graph([(i, (i + 1) % 5) for i in range(5)]
                     + [(i, i + 5) for i in range(5)]
                     + [(5 + i, 5 + (i + 2) % 5) for i in range(5)])
    k5 = Graph([(i, j) for i in range(5) for j in range(i + 1, 5)])
    wheel = Graph([(0, i) for i in range(1, 6)] + [(i, i % 5 + 1) for i in range(1, 6)])
    assert chromatic_number(petersen) == 3
    assert chromatic_number(k5) == 5
    assert chromatic_number(wheel) == 4  # odd rim -> 4


def test_coloring_helpers_and_validation():
    g = Graph([(0, 1), (1, 2)])
    col = {0: 0, 1: 1, 2: 0}
    assert is_proper_coloring(g, col)
    assert not is_proper_coloring(g, {0: 0, 1: 0, 2: 1})
    assert not is_proper_coloring(g, {0: 0, 1: 1})
    assert color_classes(col) == [[0, 2], [1]]
    with pytest.raises(ValueError, match="unknown strategy"):
        greedy_color(g, "nope")
    with pytest.raises(ValueError, match="self-loop"):
        greedy_color(Graph([(0, 0), (0, 1)]))
    with pytest.raises(ValueError, match="max_nodes"):
        chromatic_number(g, max_nodes=2)
    assert greedy_color(Graph()) == {}
    assert chromatic_number(Graph()) == 0
    # Directed graphs colour the underlying undirected graph.
    d = Graph([(0, 1), (1, 2), (2, 0)], directed=True)
    assert chromatic_number(d) == 3


def test_random_sequential_is_seeded():
    g = karate_club()
    assert greedy_color(g, "random_sequential", seed=7) == greedy_color(g, "random_sequential", seed=7)


def _check_walk(edges, expected, directed):
    assert all(edges[i][1] == edges[i + 1][0] for i in range(len(edges) - 1))
    if directed:
        assert sorted(edges) == sorted(expected)
    else:
        assert sorted(tuple(sorted(e)) for e in edges) == sorted(tuple(sorted(e)) for e in expected)


def test_undirected_circuit_and_path():
    square_diag = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    g = Graph(square_diag)
    assert not is_eulerian(g)
    assert has_eulerian_path(g)
    assert sorted(odd_degree_nodes(g)) == [0, 2]
    path = eulerian_path(g)
    _check_walk(path, square_diag, directed=False)
    assert {path[0][0], path[-1][1]} == {0, 2}
    assert eulerian_path(g, source=2)[0][0] == 2
    with pytest.raises(ValueError, match="must start"):
        eulerian_path(g, source=1)
    with pytest.raises(ValueError, match="not Eulerian"):
        eulerian_circuit(g)

    bowtie = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]
    h = Graph(bowtie)
    assert is_eulerian(h)
    circ = eulerian_circuit(h, source=3)
    _check_walk(circ, bowtie, directed=False)
    assert circ[0][0] == 3 and circ[-1][1] == 3


def test_directed_conditions():
    cyc = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 2)]
    g = Graph(cyc, directed=True)
    assert is_eulerian(g)
    _check_walk(eulerian_circuit(g), cyc, directed=True)
    path_edges = [(0, 1), (1, 2), (2, 0), (0, 3)]
    p = Graph(path_edges, directed=True)
    assert not is_eulerian(p)
    assert has_eulerian_path(p)
    walk = eulerian_path(p)
    _check_walk(walk, path_edges, directed=True)
    assert walk[0][0] == 0 and walk[-1][1] == 3
    with pytest.raises(ValueError):
        eulerian_path(p, source=3)
    assert not has_eulerian_path(Graph([(0, 1), (0, 2)], directed=True))


def test_disconnected_isolated_and_self_loops():
    g = Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    assert not is_eulerian(g) and not has_eulerian_path(g)
    h = Graph([(0, 1), (1, 2), (2, 0)])
    h.add_node(9)  # isolated nodes are ignored
    assert is_eulerian(h)
    loop = Graph([(0, 1), (1, 2), (2, 0), (1, 1)])
    assert is_eulerian(loop)
    walk = eulerian_circuit(loop)
    assert len(walk) == 4 and (1, 1) in walk
    assert not is_eulerian(Graph()) and not has_eulerian_path(Graph())


def test_random_walks_cover_every_edge_once():
    rng = random.Random(5)
    for _ in range(80):
        n = rng.randint(2, 10)
        edges = list({(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5})
        if not edges:
            continue
        g = Graph(edges)
        if has_eulerian_path(g):
            _check_walk(eulerian_path(g), edges, directed=False)


def test_cli_coloring_and_eulerian(capsys):
    assert main(["coloring", "--strategy", "largest_first", "--exact"]) == 0
    out = capsys.readouterr().out
    assert "dsatur: 5 colours" in out
    assert "chromatic number: 5" in out
    assert main(["eulerian"]) == 0
    out = capsys.readouterr().out
    assert "eulerian circuit: no" in out
    assert "odd-degree nodes (12)" in out
