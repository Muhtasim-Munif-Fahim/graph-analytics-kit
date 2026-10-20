"""Tests for Bellman-Ford, Floyd-Warshall and graph distance measures."""
from __future__ import annotations

import math
import random

import pytest

from graph_analytics_kit import (
    Graph,
    NegativeCycleError,
    average_shortest_path_length,
    bellman_ford,
    center,
    diameter,
    dijkstra_shortest_path,
    distance_summary,
    eccentricity,
    floyd_warshall,
    floyd_warshall_path,
    global_efficiency,
    karate_club,
    periphery,
    radius,
    wiener_index,
)
from graph_analytics_kit.cli import main


def _path(n):
    return Graph([(i, i + 1) for i in range(n - 1)])


def _cycle(n):
    return Graph([(i, (i + 1) % n) for i in range(n)])


def _complete(n):
    return Graph([(i, j) for i in range(n) for j in range(i + 1, n)])


def test_closed_forms_on_paths_cycles_stars_and_complete_graphs():
    n = 7
    p = _path(n)
    assert diameter(p) == n - 1
    assert radius(p) == 3
    assert center(p) == [3]
    assert sorted(periphery(p)) == [0, n - 1]
    assert wiener_index(p) == (n - 1) * n * (n + 1) / 6  # binom(n+1, 3)
    assert average_shortest_path_length(p) == pytest.approx((n + 1) / 3)

    c = _cycle(8)
    assert diameter(c) == radius(c) == 4
    assert len(center(c)) == 8
    assert wiener_index(c) == 8 * (8**2) / 8  # n^3 / 8 for even n

    star = Graph([(0, i) for i in range(1, 6)])
    assert radius(star) == 1 and diameter(star) == 2
    assert center(star) == [0]
    assert eccentricity(star)[3] == 2

    k = _complete(6)
    assert global_efficiency(k) == pytest.approx(1.0)
    assert average_shortest_path_length(k) == pytest.approx(1.0)
    assert wiener_index(k) == 15


def test_karate_club_matches_reference_values():
    s = distance_summary(karate_club())
    assert s["diameter"] == 5 and s["radius"] == 3
    assert sorted(s["center"]) == [0, 1, 2, 3, 8, 13, 19, 31]
    assert sorted(s["periphery"]) == [14, 15, 16, 18, 20, 22, 23, 26, 29]
    assert s["average_shortest_path_length"] == pytest.approx(2.408199643493761)
    assert s["wiener_index"] == 1351
    assert s["global_efficiency"] == pytest.approx(0.4920083184789052)


def test_floyd_warshall_matches_bfs_and_dijkstra():
    g = karate_club()
    nodes, D, nxt = floyd_warshall(g)
    for i, u in enumerate(nodes):
        bfs = g.bfs_distances(u)
        for j, v in enumerate(nodes):
            assert D[i, j] == bfs[v]
    rng = random.Random(0)
    wg = Graph([(u, v, rng.uniform(0.1, 3.0)) for u, v, _ in g.edges])
    nodes, D, nxt = floyd_warshall(wg, weighted=True)
    dist, _ = dijkstra_shortest_path(wg, 0)
    for j, v in enumerate(nodes):
        assert D[nodes.index(0), j] == pytest.approx(dist[v])
    path = floyd_warshall_path(nodes, nxt, 16, 26)
    assert path[0] == 16 and path[-1] == 26
    length = sum(wg.get_weight(a, b) for a, b in zip(path, path[1:]))
    assert length == pytest.approx(D[nodes.index(16), nodes.index(26)])


def _potential_digraph(seed, n=10, p=0.3):
    """Random digraph with negative edges but no negative cycles (reweighting)."""
    rng = random.Random(seed)
    h = [rng.uniform(-4, 4) for _ in range(n)]
    edges = [(u, v, rng.uniform(0, 2) + h[u] - h[v])
             for u in range(n) for v in range(n) if u != v and rng.random() < p]
    return Graph(edges, directed=True), h


def _brute_force_apsp(g):
    nodes = g.nodes
    n = len(nodes)
    dist = {(u, v): (0.0 if u == v else math.inf) for u in nodes for v in nodes}
    for u, v, w in g.edges:
        dist[(u, v)] = min(dist[(u, v)], w)
    for _ in range(n):  # repeated relaxation to a fixed point
        for (u, v), d in list(dist.items()):
            for w_ in nodes:
                alt = dist[(u, w_)] + dist[(w_, v)]
                if alt < dist[(u, v)]:
                    dist[(u, v)] = alt
    return dist


def test_negative_weights_floyd_warshall_and_bellman_ford_agree():
    for seed in range(6):
        g, _ = _potential_digraph(seed)
        ref = _brute_force_apsp(g)
        nodes, D, nxt = floyd_warshall(g, weighted=True)
        for i, u in enumerate(nodes):
            bf, prev = bellman_ford(g, u)
            for j, v in enumerate(nodes):
                if math.isinf(ref[(u, v)]):
                    assert math.isinf(D[i, j]) and v not in bf
                else:
                    assert D[i, j] == pytest.approx(ref[(u, v)])
                    assert bf[v] == pytest.approx(ref[(u, v)])
                    path = floyd_warshall_path(nodes, nxt, u, v)
                    assert sum(g.get_weight(a, b) for a, b in zip(path, path[1:])) == pytest.approx(D[i, j])
            assert prev[u] is None


def test_bellman_ford_small_example_and_unweighted_mode():
    dg = Graph([(0, 1, 4.0), (0, 2, 1.0), (2, 1, -2.0), (1, 3, 1.0)], directed=True)
    dist, prev = bellman_ford(dg, 0)
    assert dist == {0: 0.0, 1: -1.0, 2: 1.0, 3: 0.0}
    assert prev[1] == 2 and prev[3] == 1
    hops, _ = bellman_ford(dg, 0, weighted=False)
    assert hops[3] == 2


def test_negative_cycles_are_reported():
    g = Graph([(0, 1, 1.0), (1, 2, -3.0), (2, 0, 1.0), (2, 3, 1.0)], directed=True)
    with pytest.raises(NegativeCycleError) as bf:
        bellman_ford(g, 0)
    cyc = bf.value.cycle
    assert cyc[0] == cyc[-1] and set(cyc) == {0, 1, 2}
    assert sum(g.get_weight(a, b) for a, b in zip(cyc, cyc[1:])) < 0
    with pytest.raises(NegativeCycleError) as fw:
        floyd_warshall(g, weighted=True)
    assert set(fw.value.cycle) == {0, 1, 2}
    # a negative undirected edge is a 2-cycle
    with pytest.raises(NegativeCycleError):
        bellman_ford(Graph([(0, 1, -1.0)]), 0)
    # unreachable negative cycle does not affect Bellman-Ford
    far = Graph([(0, 1, 1.0), (2, 3, -1.0), (3, 2, -1.0)], directed=True)
    assert bellman_ford(far, 0)[0] == {0: 0.0, 1: 1.0}


def test_directed_eccentricity_and_disconnected_graphs():
    dc = Graph([(0, 1), (1, 2), (2, 0)], directed=True)
    assert eccentricity(dc) == {0: 2.0, 1: 2.0, 2: 2.0}
    assert wiener_index(dc) == 9  # ordered pairs: 1 + 2 per node
    with pytest.raises(ValueError, match="strongly connected"):
        diameter(Graph([(0, 1), (1, 2)], directed=True))
    split = Graph([(0, 1), (2, 3)])
    with pytest.raises(ValueError, match="not connected"):
        radius(split)
    assert global_efficiency(split) == pytest.approx(2 / 12 * 2)  # 4 reachable ordered pairs at d=1
    lone = Graph()
    lone.add_node(5)
    assert global_efficiency(lone) == 0.0
    assert average_shortest_path_length(lone) == 0.0
    with pytest.raises(ValueError, match="no nodes"):
        diameter(Graph())


def test_weighted_measures():
    g = Graph([(0, 1, 2.0), (1, 2, 3.0), (0, 2, 10.0)])
    assert diameter(g, weighted=True) == 5.0
    assert diameter(g) == 1.0
    assert radius(g, weighted=True) == 3.0
    assert center(g, weighted=True) == [1]
    assert global_efficiency(g, weighted=True) == pytest.approx((1 / 2 + 1 / 3 + 1 / 5) * 2 / 6)


def test_path_reconstruction_errors_and_self_loops():
    g = Graph([(0, 1), (1, 2)], directed=True)
    nodes, D, nxt = floyd_warshall(g)
    assert floyd_warshall_path(nodes, nxt, 0, 2) == [0, 1, 2]
    assert floyd_warshall_path(nodes, nxt, 1, 1) == [1]
    with pytest.raises(KeyError, match="not reachable"):
        floyd_warshall_path(nodes, nxt, 2, 0)
    with pytest.raises(KeyError):
        floyd_warshall_path(nodes, nxt, 0, 99)
    loop = Graph([(0, 0, 5.0), (0, 1, 1.0)], directed=True)
    _, D, _ = floyd_warshall(loop, weighted=True)
    assert D[0, 0] == 0.0
    with pytest.raises(NegativeCycleError):
        floyd_warshall(Graph([(0, 0, -1.0)], directed=True), weighted=True)
    with pytest.raises(KeyError):
        bellman_ford(g, 42)


def test_cli_distances(capsys):
    assert main(["distances", "--eccentricity"]) == 0
    out = capsys.readouterr().out
    assert "diameter: 5" in out and "radius: 3" in out
    assert "center: 0 1 2 3 8 13 19 31" in out
    assert "average shortest path length: 2.408200" in out
    assert "Wiener index: 1351" in out
    assert "  33: 4" in out
