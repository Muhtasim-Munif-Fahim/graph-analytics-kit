import itertools
import random

import pytest

from graph_analytics_kit import (
    Graph,
    clique_number,
    core_number,
    find_cliques,
    karate_club,
    max_clique,
    node_clique_number,
    number_of_maximal_cliques,
    total_triangles,
    triangles,
)
from graph_analytics_kit.cli import main


def _brute_maximal_cliques(nodes, edges):
    adj = {u: set() for u in nodes}
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    cliques = []
    for r in range(1, len(nodes) + 1):
        for combo in itertools.combinations(nodes, r):
            if all(b in adj[a] for a, b in itertools.combinations(combo, 2)):
                cliques.append(set(combo))
    maximal = [c for c in cliques if not any(c < d for d in cliques)]
    return sorted(sorted(c) for c in maximal)


def _random_graph(seed, n_max=11):
    rng = random.Random(seed)
    n = rng.randint(1, n_max)
    p = rng.random()
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    g = Graph(edges)
    for i in range(n):
        g.add_node(i)
    return g, list(range(n)), edges


@pytest.mark.parametrize("seed", range(25))
def test_matches_brute_force(seed):
    g, nodes, edges = _random_graph(seed)
    assert sorted(find_cliques(g)) == _brute_maximal_cliques(nodes, edges)


def test_karate_club_known_values():
    g = karate_club()
    cliques = find_cliques(g)
    assert number_of_maximal_cliques(g) == 36
    assert clique_number(g) == 5
    assert cliques[:2] == [[0, 1, 2, 3, 7], [0, 1, 2, 3, 13]]
    assert max_clique(g) == [0, 1, 2, 3, 7]
    assert total_triangles(g) == 45
    tri = triangles(g)
    assert tri[0] == 18 and tri[33] == 15 and tri[11] == 0


def test_ordering_and_min_size():
    g = karate_club()
    big = find_cliques(g, min_size=4)
    sizes = [len(c) for c in big]
    assert sizes == sorted(sizes, reverse=True) and min(sizes) >= 4
    assert all(c == sorted(c) for c in big)
    with pytest.raises(ValueError, match="min_size"):
        find_cliques(g, min_size=0)


def test_complete_graph_path_and_isolated_nodes():
    k6 = Graph([(i, j) for i in range(6) for j in range(i + 1, 6)])
    assert find_cliques(k6) == [list(range(6))]
    assert total_triangles(k6) == 20
    path = Graph([(0, 1), (1, 2), (2, 3)])
    assert find_cliques(path) == [[0, 1], [1, 2], [2, 3]]
    assert clique_number(path) == 2
    g = Graph([(0, 1)])
    g.add_node(7)
    assert find_cliques(g) == [[0, 1], [7]]
    assert node_clique_number(g) == {0: 2, 1: 2, 7: 1}
    assert clique_number(Graph()) == 0 and max_clique(Graph()) == []


def test_directed_and_self_loops_use_simple_undirected_graph():
    d = Graph([(0, 1), (1, 2), (2, 0), (2, 2), (3, 2)], directed=True)
    assert find_cliques(d) == [[0, 1, 2], [2, 3]]
    assert triangles(d) == {0: 1, 1: 1, 2: 1, 3: 0}


def test_clique_members_have_high_core_number():
    g = karate_club()
    core = core_number(g)
    for clique in find_cliques(g):
        assert all(core[v] >= len(clique) - 1 for v in clique)


def test_large_clique_no_recursion_issue():
    n = 60
    g = Graph([(i, j) for i in range(n) for j in range(i + 1, n)])
    assert clique_number(g) == n


def test_cli(capsys):
    assert main(["cliques", "--top", "2"]) == 0
    out = capsys.readouterr().out
    assert "maximal_cliques: 36" in out and "clique_number: 5" in out
    assert "size=5: 0 1 2 3 7" in out and "triangles: 45" in out
    assert main(["cliques", "--min-size", "9"]) == 0
    assert "clique_number: 5" in capsys.readouterr().out
    assert main(["cliques", "--min-size", "0"]) == 2
