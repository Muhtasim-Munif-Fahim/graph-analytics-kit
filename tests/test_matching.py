"""Tests for Edmonds blossom matching, the Hungarian method and edge covers."""

import itertools

import numpy as np
import pytest

from graph_analytics_kit import (
    Graph,
    is_matching,
    is_perfect_matching,
    karate_club,
    linear_sum_assignment,
    matching_edges,
    maximum_bipartite_matching,
    maximum_matching,
    maximum_weight_bipartite_matching,
    minimum_edge_cover,
)
from graph_analytics_kit.cli import main


def _brute_matching_number(edges):
    for k in range(len(edges), 0, -1):
        for subset in itertools.combinations(edges, k):
            vs = [x for e in subset for x in e]
            if len(set(vs)) == len(vs):
                return k
    return 0


def _brute_assignment(C, maximize=False):
    C = np.asarray(C, dtype=float)
    r, c = C.shape
    best = None
    if r <= c:
        for perm in itertools.permutations(range(c), r):
            val = sum(C[i, perm[i]] for i in range(r))
            if best is None or (val > best if maximize else val < best):
                best = val
    else:
        return _brute_assignment(C.T, maximize)
    return best


def test_blossom_on_odd_cycle_structure():
    # Two triangles joined by a path: needs blossom contraction for a perfect matching.
    g = Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 5)])
    m = maximum_matching(g)
    assert len(m) // 2 == 4
    assert is_perfect_matching(g, m)


def test_petersen_has_perfect_matching():
    g = Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7),
               (3, 8), (4, 9), (5, 7), (7, 9), (9, 6), (6, 8), (8, 5)])
    m = maximum_matching(g)
    assert is_perfect_matching(g, m)
    assert len(matching_edges(m)) == 5


def test_odd_cycle_and_star():
    assert len(maximum_matching(Graph([(i, (i + 1) % 7) for i in range(7)]))) // 2 == 3
    assert len(maximum_matching(Graph([(0, i) for i in range(1, 6)]))) // 2 == 1


def test_matches_bruteforce_on_random_graphs():
    rng = np.random.default_rng(0)
    for _ in range(150):
        n = int(rng.integers(2, 10))
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.35]
        g = Graph(edges)
        for i in range(n):
            g.add_node(i)
        m = maximum_matching(g)
        assert is_matching(g, m)
        assert len(m) // 2 == _brute_matching_number(edges)


def test_agrees_with_hopcroft_karp_on_bipartite():
    rng = np.random.default_rng(1)
    for _ in range(30):
        edges = [(i, 10 + j) for i in range(8) for j in range(8) if rng.random() < 0.25]
        if not edges:
            continue
        g = Graph(edges)
        assert len(maximum_matching(g)) == len(maximum_bipartite_matching(g))


def test_karate_matching_and_edge_cover():
    g = karate_club()
    m = maximum_matching(g)
    assert is_matching(g, m)
    nu = len(m) // 2
    cover = minimum_edge_cover(g)
    assert len(cover) == g.number_of_nodes() - nu  # Gallai
    covered = {x for e in cover for x in e}
    assert covered == set(g.nodes)


def test_matching_ignores_direction_and_self_loops():
    g = Graph([(0, 0), (0, 1), (2, 1), (3, 2)], directed=True)
    m = maximum_matching(g)
    assert len(m) // 2 == 2 and is_matching(g, m)


def test_is_matching_rejects_bad_sets():
    g = Graph([(0, 1), (1, 2), (2, 3)])
    assert not is_matching(g, [(0, 1), (1, 2)])
    assert not is_matching(g, [(0, 2)])
    assert not is_matching(g, {0: 1, 1: 2})
    assert is_matching(g, [(0, 1), (2, 3)]) and is_perfect_matching(g, [(0, 1), (2, 3)])


def test_edge_cover_requires_no_isolated_nodes():
    g = Graph([(0, 1)])
    g.add_node(5)
    with pytest.raises(ValueError, match="isolated"):
        minimum_edge_cover(g)


def test_linear_sum_assignment_known_example():
    C = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    rows, cols = linear_sum_assignment(C)
    np.testing.assert_array_equal(rows, [0, 1, 2])
    assert C[rows, cols].sum() == 5


def test_linear_sum_assignment_matches_bruteforce_rectangular():
    rng = np.random.default_rng(2)
    for _ in range(200):
        r, c = rng.integers(1, 6, 2)
        C = rng.integers(-5, 20, (r, c)).astype(float)
        for maximize in (False, True):
            rows, cols = linear_sum_assignment(C, maximize=maximize)
            assert len(rows) == min(r, c)
            assert len(set(rows)) == len(rows) and len(set(cols)) == len(cols)
            assert C[rows, cols].sum() == pytest.approx(_brute_assignment(C, maximize))


def test_linear_sum_assignment_matches_scipy():
    scipy_opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(3)
    for _ in range(50):
        r, c = rng.integers(1, 25, 2)
        C = rng.normal(size=(r, c))
        rows, cols = linear_sum_assignment(C)
        r2, c2 = scipy_opt.linear_sum_assignment(C)
        assert C[rows, cols].sum() == pytest.approx(C[r2, c2].sum())


def test_linear_sum_assignment_forbidden_pairs():
    inf = np.inf
    C = np.array([[1.0, inf], [inf, 2.0]])
    rows, cols = linear_sum_assignment(C)
    np.testing.assert_array_equal(cols, [0, 1])
    with pytest.raises(ValueError, match="infeasible"):
        linear_sum_assignment([[1.0, inf], [2.0, inf]])
    with pytest.raises(ValueError):
        linear_sum_assignment([1.0, 2.0])
    with pytest.raises(ValueError, match="NaN"):
        linear_sum_assignment([[np.nan]])
    r, c = linear_sum_assignment(np.zeros((0, 3)))
    assert r.size == 0 and c.size == 0


def test_maximum_weight_bipartite_matching():
    # weights favour a smaller matching unless maximum_cardinality is requested
    g = Graph([(0, 10, 10.0), (0, 11, 1.0), (1, 10, 1.0)])
    m = maximum_weight_bipartite_matching(g, left=[0, 1])
    assert matching_edges(m) == [(0, 10)]
    m2 = maximum_weight_bipartite_matching(g, left=[0, 1], maximum_cardinality=True)
    assert matching_edges(m2) == [(0, 11), (1, 10)]


def test_weighted_bipartite_matches_bruteforce():
    rng = np.random.default_rng(4)
    for _ in range(40):
        edges = [(i, 10 + j, float(rng.integers(-3, 10))) for i in range(4) for j in range(5)
                 if rng.random() < 0.6]
        if not edges:
            continue
        g = Graph(edges)
        left = [u for u in g.nodes if u < 10]
        m = maximum_weight_bipartite_matching(g, left=left)
        assert is_matching(g, m)
        got = sum(g.get_weight(u, v) for u, v in matching_edges(m))
        best = 0.0
        E = [(u, v, w) for u, v, w in g.edges]
        for k in range(1, len(E) + 1):
            for subset in itertools.combinations(E, k):
                vs = [x for e in subset for x in e[:2]]
                if len(set(vs)) == len(vs):
                    best = max(best, sum(e[2] for e in subset))
        assert got == pytest.approx(best)


def test_weighted_bipartite_rejects_non_bipartite_split():
    g = Graph([(0, 1, 1.0), (1, 2, 1.0)])
    with pytest.raises(ValueError, match="internal edge"):
        maximum_weight_bipartite_matching(g, left=[0, 1])


def test_cli_matching(tmp_path, capsys):
    assert main(["matching"]) == 0
    out = capsys.readouterr().out
    assert "Edmonds blossom" in out and "minimum edge cover" in out
    f = tmp_path / "jobs.txt"
    f.write_text("0 10 5\n0 11 2\n1 10 4\n1 11 1\n")
    assert main(["matching", "--edges", str(f), "--weighted", "--bipartite-weight"]) == 0
    out = capsys.readouterr().out
    assert "total weight 6" in out
