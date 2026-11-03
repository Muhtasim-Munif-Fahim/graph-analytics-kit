"""Tests for dominator trees, dominance frontiers and post-dominators."""

from __future__ import annotations

import random

import pytest

from graph_analytics_kit import (
    Graph,
    dominance_frontiers,
    dominates,
    dominator_tree,
    dominators,
    immediate_dominators,
    post_dominators,
)
from graph_analytics_kit.cli import main


def _reachable(g: Graph, root: int, removed: int | None = None) -> set:
    if root == removed:
        return set()
    seen = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in g.neighbors(u):
            if v != removed and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def _brute_dominators(g: Graph, root: int) -> dict:
    reach = _reachable(g, root)
    doms = {v: {v} for v in reach}
    for d in reach:
        without = _reachable(g, root, removed=d)
        for v in reach:
            if v not in without:
                doms[v].add(d)
    return doms


def _random_digraph(n: int, p: float, seed: int) -> Graph:
    rng = random.Random(seed)
    g = Graph(directed=True)
    for v in range(n):
        g.add_node(v)
    for u in range(n):
        for v in range(n):
            if u != v and rng.random() < p:
                g.add_edge(u, v)
    return g


def _lengauer_tarjan_example() -> Graph:
    # Figure from Lengauer & Tarjan (1979), letters R..L mapped to 0..12.
    R, A, B, C, D, E, F, G, H, I, J, K, L = range(13)
    edges = [
        (R, A), (R, B), (R, C), (A, D), (B, A), (B, D), (B, E), (C, F), (C, G),
        (D, L), (E, H), (F, I), (G, I), (G, J), (H, E), (H, K), (I, K), (J, I),
        (K, I), (K, R), (L, H),
    ]
    return Graph(edges, directed=True)


def test_lengauer_tarjan_reference_idoms():
    g = _lengauer_tarjan_example()
    idom = immediate_dominators(g, 0)
    R, A, B, C, D, E, F, G, H, I, J, K, L = range(13)
    expected = {R: R, A: R, B: R, C: R, D: R, E: R, F: C, G: C, H: R, I: R, J: G, K: R, L: D}
    assert idom == expected


@pytest.mark.parametrize("seed", range(25))
def test_matches_brute_force_on_random_graphs(seed):
    g = _random_digraph(9, 0.22, seed)
    assert dominators(g, 0) == _brute_dominators(g, 0)


def test_dominator_tree_and_dominates_consistent():
    g = _random_digraph(12, 0.2, 7)
    doms = dominators(g, 0)
    tree = dominator_tree(g, 0)
    assert set(tree) == set(doms)
    assert sum(len(c) for c in tree.values()) == len(doms) - 1
    for v, ds in doms.items():
        for d in doms:
            assert dominates(g, 0, d, v) == (d in ds)


def test_unreachable_nodes_excluded():
    g = Graph([(0, 1), (2, 1), (3, 3)], directed=True)
    idom = immediate_dominators(g, 0)
    assert idom == {0: 0, 1: 0}
    assert not dominates(g, 0, 2, 1)


def test_diamond_frontiers():
    #   0 -> 1 -> 3, 0 -> 2 -> 3, 3 -> 4
    g = Graph([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], directed=True)
    df = dominance_frontiers(g, 0)
    assert df == {0: set(), 1: {3}, 2: {3}, 3: set(), 4: set()}


def test_frontier_definition_on_random_graphs():
    for seed in range(15):
        g = _random_digraph(10, 0.2, 100 + seed)
        doms = dominators(g, 0)
        df = dominance_frontiers(g, 0)
        preds = {v: [u for u in doms if v in g.neighbors(u)] for v in doms}
        for d in doms:
            expected = {
                w for w in doms
                if any(d in doms[p] for p in preds[w]) and not (d in doms[w] and d != w)
            }
            assert df[d] == expected


def test_loop_back_edge_frontier():
    # 0 -> 1 -> 2 -> 1 (loop), 2 -> 3: the loop header is in its own frontier.
    g = Graph([(0, 1), (1, 2), (2, 1), (2, 3)], directed=True)
    df = dominance_frontiers(g, 0)
    assert df[2] == {1}
    assert df[1] == {1}


def test_post_dominators():
    g = Graph([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], directed=True)
    ipdom = post_dominators(g, 4)
    assert ipdom == {4: 4, 3: 4, 1: 3, 2: 3, 0: 3}


def test_deep_chain_no_recursion():
    n = 5000
    g = Graph([(i, i + 1) for i in range(n)], directed=True)
    idom = immediate_dominators(g, 0)
    assert idom[n] == n - 1
    assert dominates(g, 0, 10, n)


def test_validation():
    with pytest.raises(ValueError):
        immediate_dominators(Graph([(0, 1)]), 0)
    with pytest.raises(ValueError):
        immediate_dominators(Graph([(0, 1)], directed=True), 9)


def test_cli(tmp_path, capsys):
    path = tmp_path / "flow.txt"
    path.write_text("0 1\n0 2\n1 3\n2 3\n3 4\n")
    assert main(["dominators", "--edges", str(path), "--root", "0", "--post", "4"]) == 0
    out = capsys.readouterr().out
    assert "node 3: idom=0" in out
    assert "frontier=3" in out
    assert "node 0: ipdom=3" in out
    assert main(["dominators", "--edges", str(path), "--root", "42"]) == 2
