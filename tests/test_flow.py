"""Tests for max-flow / min-cut, edge connectivity, and bipartite matching."""
from __future__ import annotations

import itertools
import random

import pytest

from graph_analytics_kit import (
    Graph,
    edge_connectivity,
    karate_club,
    maximum_bipartite_matching,
    maximum_flow,
    minimum_cut,
)
from graph_analytics_kit.cli import main


def _clrs_network() -> Graph:
    # CLRS Figure 26.1: max flow 23 from 0 (s) to 5 (t)
    edges = [
        (0, 1, 16), (0, 2, 13), (1, 3, 12), (2, 1, 4), (2, 4, 14),
        (3, 2, 9), (3, 5, 20), (4, 3, 7), (4, 5, 4),
    ]
    return Graph(edges, directed=True)


def _check_flow(g: Graph, res, s: int, t: int) -> None:
    caps = {}
    for u, v, w in g.edges:
        caps[(u, v)] = caps.get((u, v), 0) + w
        if not g.directed:
            caps[(v, u)] = caps.get((v, u), 0) + w
    for (u, v), f in res.flow.items():
        assert f <= caps[(u, v)] + 1e-9
    net = {n: 0.0 for n in g.nodes}
    for (u, v), f in res.flow.items():
        net[u] -= f
        net[v] += f
    for n in g.nodes:
        if n == s:
            assert net[n] == pytest.approx(-res.value)
        elif n == t:
            assert net[n] == pytest.approx(res.value)
        else:
            assert net[n] == pytest.approx(0.0, abs=1e-9)
    assert res.cut_capacity == pytest.approx(res.value)
    assert s in res.source_side and t in res.sink_side


def test_clrs_example():
    g = _clrs_network()
    res = maximum_flow(g, 0, 5)
    assert res.value == pytest.approx(23)
    _check_flow(g, res, 0, 5)
    assert res.source_side == [0, 1, 2, 4]
    assert res.cut_edges == [(1, 3), (4, 3), (4, 5)]
    value, S, T, cut = minimum_cut(g, 0, 5)
    assert value == pytest.approx(23) and S == [0, 1, 2, 4] and T == [3, 5]


def test_undirected_edges_carry_flow_either_way():
    g = Graph([(0, 1, 3.0), (1, 2, 2.0), (0, 2, 1.0), (2, 3, 5.0)])
    res = maximum_flow(g, 0, 3)
    assert res.value == pytest.approx(3.0)
    _check_flow(g, res, 0, 3)
    # never both orientations of one undirected edge
    for u, v in res.flow:
        assert (v, u) not in res.flow
    rev = maximum_flow(g, 3, 0)
    assert rev.value == pytest.approx(3.0)


def test_disconnected_and_validation():
    g = Graph([(0, 1, 1.0), (2, 3, 1.0)])
    res = maximum_flow(g, 0, 3)
    assert res.value == 0.0 and res.cut_edges == []
    with pytest.raises(ValueError):
        maximum_flow(g, 0, 0)
    with pytest.raises(ValueError):
        maximum_flow(g, 0, 99)
    with pytest.raises(ValueError):
        maximum_flow(Graph([(0, 1, -1.0)]), 0, 1)


def test_matches_networkx_on_random_graphs():
    nx = pytest.importorskip("networkx")
    rng = random.Random(0)
    for trial in range(25):
        directed = trial % 2 == 0
        n = rng.randint(4, 12)
        edges = []
        for u, v in itertools.permutations(range(n), 2):
            if (directed or u < v) and rng.random() < 0.35:
                edges.append((u, v, float(rng.randint(1, 9))))
        g = Graph(edges, directed=directed)
        for i in range(n):
            g.add_node(i)
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(range(n))
        for u, v, w in edges:
            G.add_edge(u, v, capacity=w)
        s, t = 0, n - 1
        res = maximum_flow(g, s, t)
        assert res.value == pytest.approx(nx.maximum_flow_value(G, s, t))
        _check_flow(g, res, s, t)


def test_edge_connectivity_local_and_global():
    g = karate_club()
    # node 11 hangs off node 0 by a bridge
    assert edge_connectivity(g) == 1
    assert edge_connectivity(g, 0, 33) == 10
    assert edge_connectivity(g, 0, 33) <= min(g.degree(0), g.degree(33))
    cycle = Graph([(i, (i + 1) % 6) for i in range(6)])
    assert edge_connectivity(cycle) == 2
    k5 = Graph(list(itertools.combinations(range(5), 2)))
    assert edge_connectivity(k5) == 4
    assert edge_connectivity(Graph([(0, 1), (2, 3)])) == 0
    with pytest.raises(ValueError):
        edge_connectivity(k5, 0)
    nx = pytest.importorskip("networkx")
    G = nx.karate_club_graph()
    assert edge_connectivity(g, 0, 33) == nx.edge_connectivity(G, 0, 33)


def test_directed_global_edge_connectivity():
    ring = Graph([(i, (i + 1) % 5) for i in range(5)], directed=True)
    assert edge_connectivity(ring) == 1
    one_way = Graph([(0, 1), (1, 2)], directed=True)
    assert edge_connectivity(one_way) == 0


def test_bipartite_matching_known_cases():
    # perfect matching on K_{3,3}
    k33 = Graph([(u, v) for u in range(3) for v in range(3, 6)])
    m = maximum_bipartite_matching(k33)
    assert len(m) // 2 == 3
    for u, v in m.items():
        assert m[v] == u and k33.has_edge(u, v)
    # augmenting path required: greedy would pick (0,3) and block 1
    g = Graph([(0, 3), (0, 4), (1, 3), (2, 4), (2, 5)])
    m = maximum_bipartite_matching(g, left=[0, 1, 2])
    assert len(m) // 2 == 3
    # star: only one edge can be matched
    star = Graph([(0, i) for i in range(1, 6)])
    assert len(maximum_bipartite_matching(star)) // 2 == 1
    with pytest.raises(ValueError):
        maximum_bipartite_matching(Graph([(0, 1), (1, 2), (2, 0)]))
    with pytest.raises(ValueError):
        maximum_bipartite_matching(g, left=[0, 3])


def test_bipartite_matching_equals_flow_value():
    rng = random.Random(1)
    for _ in range(20):
        nl, nr = rng.randint(1, 8), rng.randint(1, 8)
        edges = [(u, nl + v) for u in range(nl) for v in range(nr) if rng.random() < 0.3]
        if not edges:
            continue
        g = Graph(edges)
        left = [u for u in range(nl) if u in set(g.nodes)]
        m = maximum_bipartite_matching(g, left=left)
        # reduce to unit-capacity flow: super source -> left -> right -> super sink
        S, T = 1000, 1001
        fg = Graph(directed=True)
        for u, v in edges:
            fg.add_edge(u, v)
        for u in left:
            fg.add_edge(S, u)
        for v in {v for _, v in edges}:
            fg.add_edge(v, T)
        assert len(m) // 2 == int(maximum_flow(fg, S, T).value)


def test_cli_flow(capsys):
    assert main(["flow", "--source", "0", "--sink", "33"]) == 0
    out = capsys.readouterr().out
    assert "max flow: 10" in out
    assert "edge connectivity (global): 1" in out
    assert main(["flow", "--source", "0", "--sink", "0"]) == 2
