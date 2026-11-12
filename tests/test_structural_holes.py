import math

import pytest

from graph_analytics_kit import (
    Graph,
    burt_efficiency,
    burt_hierarchy,
    constraint,
    effective_size,
    local_constraint,
    structural_holes,
)


def test_star_center_spans_all_holes():
    g = Graph([(0, i) for i in range(1, 6)])
    c = constraint(g)
    assert c[0] == pytest.approx(1 / 5)
    assert c[1] == pytest.approx(1.0)
    es = effective_size(g)
    assert es[0] == pytest.approx(5.0)
    assert burt_efficiency(g)[0] == pytest.approx(1.0)
    assert burt_hierarchy(g)[0] == pytest.approx(0.0)
    assert burt_hierarchy(g)[1] == 0.0


def test_triangle_is_fully_redundant():
    g = Graph([(0, 1), (1, 2), (0, 2)])
    assert local_constraint(g, 0, 1) == pytest.approx((0.5 + 0.25) ** 2)
    assert constraint(g)[0] == pytest.approx(2 * 0.5625)
    assert effective_size(g)[0] == pytest.approx(1.0)


def test_unweighted_effective_size_identity_on_karate():
    g = Graph.karate_club()
    es = effective_size(g)
    for v in g.nodes:
        nb = set(g.neighbors(v)) - {v}
        n = len(nb)
        t = sum(1 for a in nb for b in nb if a < b and g.has_edge(a, b))
        assert es[v] == pytest.approx(n - 2 * t / n)


def test_broker_has_lower_constraint_than_clique_member():
    # two triangles bridged by node 6
    g = Graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (6, 0), (6, 3)])
    c = constraint(g)
    assert c[6] < c[1]
    assert c[6] < c[4]
    h = burt_hierarchy(g)
    assert h[0] > 0.0  # node 0's constraint concentrates on its clique mates


def test_isolated_node_and_subset_and_bundle():
    g = Graph([(0, 1)])
    g.add_node(9)
    assert math.isnan(constraint(g)[9])
    assert math.isnan(effective_size(g)[9])
    assert math.isnan(burt_efficiency(g)[9])
    assert set(constraint(g, nodes=[0])) == {0}
    bundle = structural_holes(g, nodes=[0, 1])
    assert bundle[0]["constraint"] == pytest.approx(1.0)
    assert bundle[0]["effective_size"] == pytest.approx(1.0)
    with pytest.raises(KeyError):
        constraint(g, nodes=[42])
    with pytest.raises(ValueError):
        constraint(Graph([(0, 1, -1.0)]))


@pytest.mark.parametrize("directed", [False, True])
def test_matches_networkx(directed):
    nx = pytest.importorskip("networkx")
    import random

    rng = random.Random(7 if directed else 3)
    edges = set()
    while len(edges) < 40:
        u, v = rng.randrange(15), rng.randrange(15)
        if u != v and (directed or (v, u) not in edges):
            edges.add((u, v))
    weighted = [(u, v, rng.choice([1.0, 2.0, 3.5])) for u, v in sorted(edges)]
    g = Graph(weighted, directed=directed)
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_weighted_edges_from(weighted)
    ours_c = constraint(g)
    ours_e = effective_size(g)
    ref_c = nx.constraint(G, weight="weight")
    # per-node reference path (the definition in networkx's ``redundancy``)
    ref_e = nx.effective_size(G, nodes=list(G), weight="weight")
    for v in G.nodes:
        assert ours_c[v] == pytest.approx(ref_c[v], rel=1e-9)
        assert ours_e[v] == pytest.approx(ref_e[v], rel=1e-9)
