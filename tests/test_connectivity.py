import random

import pytest

from graph_analytics_kit import (
    Graph,
    articulation_points,
    biconnected_components,
    bridges,
    cut_structure,
    karate_club,
    two_edge_connected_components,
)
from graph_analytics_kit.cli import main


def _n_components(nodes, edges):
    adj = {u: set() for u in nodes}
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    seen, count = set(), 0
    for s in nodes:
        if s in seen:
            continue
        count += 1
        seen.add(s)
        stack = [s]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
    return count


def _random_graph(rng, n, p):
    g = Graph()
    for u in range(n):
        g.add_node(u)
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                g.add_edge(u, v)
    return g


def _brute_force(g):
    nodes = g.nodes
    edges = [(u, v) for u, v, _ in g.edges]
    base = _n_components(nodes, edges)
    br = sorted(
        (min(u, v), max(u, v))
        for u, v in edges
        if u != v and _n_components(nodes, [e for e in edges if e != (u, v)]) > base
    )
    cuts = sorted(
        x for x in nodes
        if _n_components(
            [n for n in nodes if n != x],
            [e for e in edges if x not in e],
        ) > base - (1 if not any(x in e and e[0] != e[1] for e in edges) else 0)
    )
    return br, cuts


def test_karate_club_cut_structure():
    g = karate_club()
    assert bridges(g) == [(0, 11)]
    assert articulation_points(g) == [0]
    comps = biconnected_components(g)
    assert [len(c) for c in comps] == [28, 6, 2]
    assert comps[1] == [0, 4, 5, 6, 10, 16]
    assert comps[2] == [0, 11]
    info = cut_structure(g)
    assert info["n_biconnected_components"] == 3
    assert info["largest_biconnected_component"] == 28


@pytest.mark.parametrize("seed", range(25))
def test_matches_brute_force_on_random_graphs(seed):
    rng = random.Random(seed)
    g = _random_graph(rng, rng.randint(2, 14), rng.choice([0.12, 0.2, 0.35]))
    br, cuts = _brute_force(g)
    assert bridges(g) == br
    assert articulation_points(g) == cuts
    # Articulation points are exactly the nodes shared by several components.
    comps = biconnected_components(g)
    counts = {}
    for comp in comps:
        for node in comp:
            counts[node] = counts.get(node, 0) + 1
    assert sorted(n for n, c in counts.items() if c > 1) == cuts
    # Every edge lies in exactly one biconnected component.
    for u, v, _ in g.edges:
        assert sum(1 for c in comps if u in c and v in c) == 1


@pytest.mark.parametrize("seed", range(10))
def test_matches_networkx(seed):
    nx = pytest.importorskip("networkx")
    rng = random.Random(100 + seed)
    g = _random_graph(rng, 30, 0.08)
    h = nx.Graph()
    h.add_nodes_from(g.nodes)
    h.add_edges_from((u, v) for u, v, _ in g.edges)
    assert bridges(g) == sorted(tuple(sorted(e)) for e in nx.bridges(h))
    assert articulation_points(g) == sorted(nx.articulation_points(h))
    ours = sorted(biconnected_components(g))
    theirs = sorted(sorted(c) for c in nx.biconnected_components(h))
    assert ours == theirs
    ours2 = sorted(two_edge_connected_components(g))
    theirs2 = sorted(sorted(c) for c in nx.k_edge_components(h, k=2))
    assert ours2 == theirs2


def test_cycle_and_path():
    cycle = Graph([(i, (i + 1) % 6) for i in range(6)])
    assert bridges(cycle) == []
    assert articulation_points(cycle) == []
    assert biconnected_components(cycle) == [list(range(6))]
    path = Graph([(0, 1), (1, 2), (2, 3)])
    assert bridges(path) == [(0, 1), (1, 2), (2, 3)]
    assert articulation_points(path) == [1, 2]
    assert biconnected_components(path) == [[0, 1], [1, 2], [2, 3]]
    assert two_edge_connected_components(path) == [[0], [1], [2], [3]]


def test_two_triangles_joined_by_bridge():
    g = Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)])
    assert bridges(g) == [(2, 3)]
    assert articulation_points(g) == [2, 3]
    assert two_edge_connected_components(g) == [[0, 1, 2], [3, 4, 5]]


def test_long_path_does_not_recurse():
    n = 20000
    g = Graph([(i, i + 1) for i in range(n - 1)])
    assert len(bridges(g)) == n - 1
    assert len(articulation_points(g)) == n - 2


def test_isolated_nodes_self_loops_and_directed():
    g = Graph([(0, 1), (1, 1)])
    g.add_node(7)
    assert bridges(g) == [(0, 1)]
    assert articulation_points(g) == []
    assert biconnected_components(g) == [[0, 1]]
    assert [7] in two_edge_connected_components(g)
    # Directed graphs use the underlying undirected graph.
    d = Graph([(0, 1), (1, 2), (2, 0), (2, 3)], directed=True)
    assert bridges(d) == [(2, 3)]
    assert articulation_points(d) == [2]
    assert cut_structure(Graph())["largest_biconnected_component"] == 0


def test_cli_connectivity(capsys):
    assert main(["connectivity", "--components"]) == 0
    out = capsys.readouterr().out
    assert "bridges (1): 0-11" in out
    assert "articulation_points (1): 0" in out
    assert "biconnected_components: 3" in out
    assert "size=6: 0 4 5 6 10 16" in out


def test_demo_report_includes_cut_structure(tmp_path):
    out = tmp_path / "report.md"
    assert main(["demo", "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## Cut structure" in text
    assert "- Bridges: 0-11" in text
    assert "- Articulation points: 0" in text
    assert "- Biconnected components: 3 (largest has 28 nodes)" in text
