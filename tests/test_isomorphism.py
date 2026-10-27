"""Tests for WL refinement/hashing/kernels and exact graph isomorphism."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from graph_analytics_kit import Graph, karate_club
from graph_analytics_kit.cli import main
from graph_analytics_kit.isomorphism import (
    automorphism_count,
    automorphism_orbits,
    color_refinement,
    could_be_isomorphic,
    find_isomorphism,
    is_isomorphic,
    iter_isomorphisms,
    wl_colors,
    wl_graph_hash,
    wl_subtree_kernel,
)


def cycle(n, offset=0):
    return Graph([(offset + i, offset + (i + 1) % n) for i in range(n)])


def complete(n):
    return Graph([(i, j) for i in range(n) for j in range(i + 1, n)])


def petersen():
    return Graph(
        [(i, (i + 1) % 5) for i in range(5)]
        + [(i, i + 5) for i in range(5)]
        + [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    )


def relabel(g, seed=0):
    rng = random.Random(seed)
    nodes = g.nodes
    perm = nodes[:]
    rng.shuffle(perm)
    m = dict(zip(nodes, [p + 100 for p in perm]))
    edges = [(m[u], m[v]) for u, v, _ in g.edges]
    rng.shuffle(edges)
    return Graph(edges, directed=g.directed), m


def _is_iso_map(g1, g2, phi):
    if sorted(phi) != sorted(g1.nodes) or sorted(phi.values()) != sorted(g2.nodes):
        return False
    return all(g1.has_edge(u, v) == g2.has_edge(phi[u], phi[v]) for u in g1.nodes for v in g1.nodes)


@pytest.mark.parametrize(
    "g, expected",
    [
        (cycle(7), 14),
        (complete(5), 120),
        (Graph([(i, i + 1) for i in range(5)]), 2),
        (Graph([(0, i) for i in range(1, 5)]), 24),
        (petersen(), 120),
        (karate_club(), 480),
    ],
)
def test_automorphism_counts(g, expected):
    assert automorphism_count(g) == expected


def test_hypercube_q3_has_48_automorphisms():
    q3 = Graph([(u, u ^ (1 << b)) for u in range(8) for b in range(3) if u < u ^ (1 << b)])
    assert automorphism_count(q3) == 48
    assert automorphism_count(q3, limit=5) == 5


def test_relabelled_graph_is_isomorphic_with_valid_mapping():
    g = karate_club()
    h, _ = relabel(g, seed=3)
    assert wl_graph_hash(g) == wl_graph_hash(h)
    phi = find_isomorphism(g, h)
    assert phi is not None and _is_iso_map(g, h, phi)


def test_wl_failure_case_c6_vs_two_triangles():
    c6 = cycle(6)
    tt = Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    assert wl_graph_hash(c6) == wl_graph_hash(tt)
    assert could_be_isomorphic(c6, tt)
    assert not is_isomorphic(c6, tt)
    assert find_isomorphism(c6, tt) is None


def test_non_isomorphic_same_degree_sequence_distinguished():
    # path 0-1-2-3-4 plus chord vs. different chord: same degrees, different structure
    a = Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 4)])
    b = Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)])
    assert not is_isomorphic(a, b)
    assert not is_isomorphic(cycle(5), cycle(6))


def test_directed_graphs_respect_direction():
    d1 = Graph([(0, 1), (1, 2), (2, 0)], directed=True)
    d2 = Graph([(0, 2), (2, 1), (1, 0)], directed=True)
    d3 = Graph([(0, 1), (1, 2), (0, 2)], directed=True)
    assert is_isomorphic(d1, d2)
    assert not is_isomorphic(d1, d3)
    assert automorphism_count(d1) == 3  # rotations only, no reflections
    assert not is_isomorphic(d1, cycle(3))


def test_self_loops_and_node_labels():
    a = Graph([(0, 0), (0, 1), (1, 2)])
    b = Graph([(2, 2), (2, 1), (1, 0)])
    c = Graph([(1, 1), (0, 1), (1, 2)])
    assert is_isomorphic(a, b) and not is_isomorphic(a, c)
    p = Graph([(0, 1), (1, 2)])
    assert wl_graph_hash(p, node_labels={0: "x", 1: "y", 2: "z"}) != wl_graph_hash(p, node_labels={0: "x", 1: "y", 2: "x"})
    with pytest.raises(ValueError):
        wl_colors(p, node_labels={0: "x"})


def test_color_refinement_and_orbits():
    star = Graph([(0, i) for i in range(1, 5)])
    assert color_refinement(star) == [[1, 2, 3, 4], [0]]
    orbits = automorphism_orbits(karate_club())
    assert sum(len(o) for o in orbits) == 34
    assert [14, 15, 18, 20, 22] in orbits
    # orbits refine the WL partition
    cls = {u: i for i, c in enumerate(color_refinement(karate_club())) for u in c}
    for o in orbits:
        assert len({cls[u] for u in o}) == 1
    rounds = wl_colors(Graph([(i, i + 1) for i in range(6)]))
    assert len(set(rounds[-1].values())) == 4  # path P7 splits into 4 symmetric classes


def test_iter_isomorphisms_all_valid_and_distinct():
    g = cycle(5)
    h, _ = relabel(g, seed=1)
    maps = list(iter_isomorphisms(g, h))
    assert len(maps) == 10
    assert len({tuple(sorted(m.items())) for m in maps}) == 10
    assert all(_is_iso_map(g, h, m) for m in maps)


def test_wl_subtree_kernel():
    graphs = [cycle(6), Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]), complete(4), relabel(complete(4))[0]]
    K = wl_subtree_kernel(graphs, iterations=2)
    assert K.shape == (4, 4) and np.allclose(K, K.T)
    assert np.all(np.linalg.eigvalsh(K) > -1e-9)
    assert K[0, 0] == 108  # 3 rounds x (6 nodes sharing one label)^2
    assert K[0, 1] == K[0, 0]  # 1-WL cannot separate them
    Kn = wl_subtree_kernel(graphs, iterations=2, normalize=True)
    np.testing.assert_allclose(np.diag(Kn), 1.0)
    assert Kn[2, 3] == pytest.approx(1.0) and Kn[0, 2] == 0.0
    with pytest.raises(ValueError):
        wl_subtree_kernel([])
    with pytest.raises(ValueError):
        wl_colors(cycle(3), iterations=-1)
    with pytest.raises(ValueError):
        automorphism_count(cycle(3), limit=0)


def test_cli_isomorphism(tmp_path, capsys):
    assert main(["isomorphism"]) == 0
    out = capsys.readouterr().out
    assert "automorphisms: 480" in out and "14 15 18 20 22" in out
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("0 1\n1 2\n2 3\n3 0\n")
    b.write_text("10 30\n30 20\n20 40\n40 10\n")
    assert main(["isomorphism", "--edges", str(a), "--compare", str(b)]) == 0
    out = capsys.readouterr().out
    assert "isomorphic: yes" in out and "automorphisms: 8" in out and "mapping:" in out
    assert main(["isomorphism", "--limit", "10"]) == 0
    assert "limit reached" in capsys.readouterr().out
