import itertools

import numpy as np
import pytest

from graph_analytics_kit.graph import Graph
from graph_analytics_kit.tsp import (
    double_tree_tour,
    held_karp,
    nearest_neighbor_tour,
    one_tree_bound,
    or_opt,
    solve_tsp,
    tour_cost,
    tsp_distance_matrix,
    two_opt,
)


def _euclid(n, seed):
    pts = np.random.default_rng(seed).random((n, 2))
    return np.linalg.norm(pts[:, None] - pts[None], axis=-1)


def _brute(D):
    n = D.shape[0]
    best = np.inf
    for perm in itertools.permutations(range(1, n)):
        best = min(best, tour_cost(D, (0,) + perm))
    return best


@pytest.mark.parametrize("seed", range(4))
def test_held_karp_matches_brute_force_symmetric_and_asymmetric(seed):
    D = _euclid(8, seed)
    cost, tour = held_karp(D)
    assert cost == pytest.approx(_brute(D))
    assert sorted(tour) == list(range(8)) and tour[0] == 0
    assert tour_cost(D, tour) == pytest.approx(cost)
    A = np.random.default_rng(seed).random((7, 7)) * 10
    np.fill_diagonal(A, 0)
    ca, ta = held_karp(A)
    assert ca == pytest.approx(_brute(A))
    assert tour_cost(A, ta) == pytest.approx(ca)


def test_held_karp_limits_and_tiny_instances():
    assert held_karp(np.array([[0, 2.0], [3.0, 0]]))[0] == 5.0
    with pytest.raises(ValueError, match="max_n"):
        held_karp(np.zeros((20, 20)))
    D = np.full((4, 4), np.inf)
    np.fill_diagonal(D, 0)
    D[0, 1] = D[1, 0] = 1
    with pytest.raises(ValueError, match="Hamiltonian"):
        held_karp(D)


@pytest.mark.parametrize("seed", range(5))
def test_heuristics_bounds_and_local_search(seed):
    D = _euclid(10, seed)
    opt, _ = held_karp(D)
    lb = one_tree_bound(D)
    assert lb <= opt + 1e-9
    assert lb >= 0.7 * opt
    dt = double_tree_tour(D)
    assert sorted(dt) == list(range(10))
    assert tour_cost(D, dt) <= 2 * opt + 1e-9  # metric 2-approximation
    nn = nearest_neighbor_tour(D, start=3)
    assert nn[0] == 3 and sorted(nn) == list(range(10))
    improved = two_opt(D, nn)
    assert tour_cost(D, improved) <= tour_cost(D, nn) + 1e-12
    oo = or_opt(D, improved)
    assert tour_cost(D, oo) <= tour_cost(D, improved) + 1e-12
    # a 2-opt local optimum on Euclidean points has no crossing edges -> close to opt
    assert tour_cost(D, oo) <= 1.15 * opt


def test_two_opt_untangles_crossing_square():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    D = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    crossed = [0, 2, 1, 3]
    assert tour_cost(D, two_opt(D, crossed)) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="exactly once"):
        two_opt(D, [0, 1, 1, 3])


def test_solve_tsp_exact_and_heuristic_agree_on_medium_instance():
    D = _euclid(11, 7)
    exact = solve_tsp(D)
    assert exact.method == "exact" and exact.optimal and exact.gap == 0
    heur = solve_tsp(D, method="heuristic")
    assert not heur.optimal and heur.lower_bound <= exact.cost + 1e-9
    assert heur.cost <= 1.05 * exact.cost
    big = solve_tsp(_euclid(40, 1), n_starts=5)
    assert big.method == "heuristic" and sorted(big.tour) == list(range(40))
    assert 0 <= big.gap < 0.35
    with pytest.raises(ValueError, match="unknown method"):
        solve_tsp(D, method="magic")


def test_asymmetric_heuristic_uses_exact_reevaluation():
    rng = np.random.default_rng(3)
    A = rng.random((9, 9)) * 10
    np.fill_diagonal(A, 0)
    opt, _ = held_karp(A)
    res = solve_tsp(A, method="heuristic")
    assert res.lower_bound is None
    assert res.cost >= opt - 1e-9
    assert res.cost == pytest.approx(tour_cost(A, res.tour))


def test_graph_input_labels_and_metric_closure_path():
    # a weighted ring 10-11-12-13-14 with a heavy chord; not complete
    g = Graph([(10, 11, 1.0), (11, 12, 1.0), (12, 13, 1.0), (13, 14, 1.0), (14, 10, 1.0), (10, 12, 5.0)])
    nodes, D = tsp_distance_matrix(g)
    assert np.isinf(D[nodes.index(10), nodes.index(13)])
    res = solve_tsp(g)
    assert res.cost == pytest.approx(5.0) and set(res.tour) == {10, 11, 12, 13, 14}
    # star graph has no Hamiltonian cycle; the metric closure gives a closed walk
    star = Graph([(0, 1, 1.0), (0, 2, 2.0), (0, 3, 3.0)])
    res = solve_tsp(star, metric_closure=True, expand_path=True)
    assert res.cost == pytest.approx(12.0)
    assert res.path[0] == res.path[-1]
    for a, b in zip(res.path, res.path[1:]):
        assert star.has_edge(a, b)
    assert sum(star.get_weight(a, b) for a, b in zip(res.path, res.path[1:])) == pytest.approx(res.cost)
    with pytest.raises(ValueError, match="Graph"):
        solve_tsp(np.ones((3, 3)), expand_path=True)


def test_cli_tsp(tmp_path, capsys):
    from graph_analytics_kit.cli import main

    path = tmp_path / "e.txt"
    path.write_text("0 1 1\n1 2 1\n2 3 1\n3 0 1\n0 2 1.5\n")
    assert main(["tsp", "--edges", str(path), "--weighted"]) == 0
    out = capsys.readouterr().out
    assert "tour cost: 4" in out and "optimal" in out
    assert main(["tsp", "--edges", str(path), "--weighted", "--method", "heuristic"]) == 0
    assert "lower bound" in capsys.readouterr().out
