"""Tests for DAG algorithms: topological order, longest path, reduction, CPM."""

from __future__ import annotations

import itertools
import random

import pytest

from graph_analytics_kit import (
    CycleError,
    Graph,
    ancestors,
    count_paths,
    critical_path,
    dag_longest_path,
    descendants,
    find_cycle,
    is_dag,
    topological_generations,
    topological_sort,
    transitive_closure,
    transitive_reduction,
)
from graph_analytics_kit.cli import main


def _random_dag(n: int, p: float, seed: int, weighted: bool = False) -> Graph:
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    g = Graph(directed=True)
    for u in perm:
        g.add_node(u)
    for i, j in itertools.combinations(range(n), 2):
        if rng.random() < p:
            w = rng.uniform(-1.0, 3.0) if weighted else 1.0
            g.add_edge(perm[i], perm[j], weight=w)
    return g


def _all_paths(g: Graph, s: int, t: int):
    if s == t:
        yield [s]
        return
    for v in g.neighbors(s):
        for rest in _all_paths(g, v, t):
            yield [s] + rest


def _reachable(g: Graph, u: int):
    seen, stack = set(), [u]
    while stack:
        x = stack.pop()
        for v in g.neighbors(x):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


@pytest.mark.parametrize("seed", range(8))
def test_topological_sort_respects_every_edge(seed):
    g = _random_dag(15, 0.3, seed)
    order = topological_sort(g)
    pos = {u: i for i, u in enumerate(order)}
    assert sorted(order) == sorted(g.nodes)
    assert all(pos[u] < pos[v] for u, v, _ in g.edges)
    assert is_dag(g) and find_cycle(g) is None


def test_topological_sort_is_deterministic_by_insertion_order():
    g = Graph([(3, 1), (2, 1), (1, 0)], directed=True)
    assert topological_sort(g) == [3, 2, 1, 0]
    assert topological_generations(g) == [[3, 2], [1], [0]]


def test_generations_equal_longest_incoming_path_length():
    g = _random_dag(14, 0.25, 4)
    gens = topological_generations(g)
    level = {u: k for k, layer in enumerate(gens) for u in layer}
    for u in g.nodes:
        preds = [p for p in g.nodes if g.has_edge(p, u)]
        expected = 0 if not preds else 1 + max(level[p] for p in preds)
        assert level[u] == expected


def test_cycle_detection_and_errors():
    g = Graph([(0, 1), (1, 2), (2, 3), (3, 1)], directed=True)
    cycle = find_cycle(g)
    assert cycle is not None and sorted(cycle) == [1, 2, 3]
    for a, b in zip(cycle, cycle[1:] + cycle[:1]):
        assert g.has_edge(a, b)
    assert not is_dag(g)
    with pytest.raises(CycleError) as info:
        topological_sort(g)
    assert sorted(info.value.cycle) == [1, 2, 3]
    with pytest.raises(CycleError):
        topological_generations(g)
    assert find_cycle(Graph([(5, 5)], directed=True)) == [5]
    with pytest.raises(ValueError):
        topological_sort(Graph([(0, 1)]))


def test_ancestors_and_descendants():
    g = _random_dag(12, 0.3, 7)
    for u in g.nodes:
        assert descendants(g, u) == _reachable(g, u)
        assert ancestors(g, u) == {v for v in g.nodes if u in _reachable(g, v)}
    with pytest.raises(KeyError):
        descendants(g, 999)


@pytest.mark.parametrize("seed", range(6))
def test_longest_path_matches_brute_force(seed):
    g = _random_dag(9, 0.4, seed, weighted=True)
    length, path = dag_longest_path(g)
    best = 0.0
    for s in g.nodes:
        for t in g.nodes:
            for p in _all_paths(g, s, t):
                best = max(best, sum(g.get_weight(a, b) for a, b in zip(p, p[1:])))
    assert length == pytest.approx(best)
    assert sum(g.get_weight(a, b) for a, b in zip(path, path[1:])) == pytest.approx(length)
    unweighted, upath = dag_longest_path(g, weighted=False)
    assert unweighted == len(topological_generations(g)) - 1 == len(upath) - 1


def test_longest_path_edge_cases():
    assert dag_longest_path(Graph(directed=True)) == (0.0, [])
    g = Graph(directed=True)
    g.add_node(4)
    assert dag_longest_path(g) == (0.0, [4])


@pytest.mark.parametrize("seed", range(4))
def test_count_paths_matches_enumeration(seed):
    g = _random_dag(10, 0.45, seed)
    for s, t in itertools.product(g.nodes, repeat=2):
        assert count_paths(g, s, t) == len(list(_all_paths(g, s, t)))


@pytest.mark.parametrize("seed", range(6))
def test_transitive_reduction_and_closure(seed):
    g = _random_dag(12, 0.4, seed)
    red = transitive_reduction(g)
    clo = transitive_closure(g)
    for u in g.nodes:
        reach = _reachable(g, u)
        assert _reachable(red, u) == reach
        assert set(clo.neighbors(u)) == reach
    # minimality: dropping any kept edge changes reachability
    for u, v, _ in red.edges:
        kept = Graph([(a, b) for a, b, _ in red.edges if (a, b) != (u, v)], directed=True)
        for x in red.nodes:
            kept.add_node(x)
        assert v not in _reachable(kept, u)
    assert set((u, v) for u, v, _ in red.edges) <= set((u, v) for u, v, _ in g.edges)


def test_transitive_reduction_of_tournament_is_a_chain():
    g = Graph([(i, j) for i in range(5) for j in range(i + 1, 5)], directed=True)
    red = transitive_reduction(g)
    assert sorted((u, v) for u, v, _ in red.edges) == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_critical_path_textbook_example():
    # Tasks: A(3) -> C(2) -> E(4); A -> D(5) -> E; B(2) -> D
    a, b, c, d, e = range(5)
    g = Graph([(a, c), (a, d), (b, d), (c, e), (d, e)], directed=True)
    cpm = critical_path(g, {a: 3, b: 2, c: 2, d: 5, e: 4})
    assert cpm["duration"] == 12
    assert cpm["path"] == [a, d, e]
    assert cpm["critical"] == [a, d, e]
    assert cpm["slack"][c] == 3 and cpm["slack"][b] == 1
    assert cpm["es"][e] == 8 and cpm["lf"][c] == 8
    for u in g.nodes:
        assert cpm["ef"][u] - cpm["es"][u] == cpm["lf"][u] - cpm["ls"][u]


def test_critical_path_duration_equals_node_weighted_longest_path():
    g = _random_dag(12, 0.3, 3)
    rng = random.Random(1)
    dur = {u: rng.randint(1, 9) for u in g.nodes}
    cpm = critical_path(g, dur)
    best = max(
        sum(dur[x] for x in p)
        for s in g.nodes for t in g.nodes for p in _all_paths(g, s, t)
    )
    assert cpm["duration"] == best
    assert sum(dur[x] for x in cpm["path"]) == best
    assert all(cpm["slack"][u] >= 0 for u in g.nodes)
    with pytest.raises(KeyError):
        critical_path(g, {})
    with pytest.raises(ValueError):
        critical_path(g, {u: -1 for u in g.nodes})


def test_cli_dag(tmp_path, capsys):
    path = tmp_path / "edges.txt"
    path.write_text("0 2\n0 3\n1 3\n2 4\n3 4\n0 4\n", encoding="utf-8")
    assert main(["dag", "--edges", str(path), "--durations", "0=3,1=2,2=2,3=5,4=4"]) == 0
    out = capsys.readouterr().out
    assert "topological order: 0 2 1 3 4" in out
    assert "transitive reduction: 5 of 6 edges kept" in out
    assert "project duration: 12" in out and "critical path: 0 -> 3 -> 4" in out
    cyc = tmp_path / "cyc.txt"
    cyc.write_text("0 1\n1 0\n", encoding="utf-8")
    assert main(["dag", "--edges", str(cyc)]) == 1
    assert "not a DAG" in capsys.readouterr().out


def test_from_edge_list_default_whitespace_delimiter(tmp_path):
    path = tmp_path / "w.txt"
    path.write_text("# comment\n0  1\t2.5\n1 2 1.0\n\n", encoding="utf-8")
    g = Graph.from_edge_list(path, directed=True, weighted=True)
    assert g.get_weight(0, 1) == 2.5 and g.has_edge(1, 2)
    csv_path = tmp_path / "c.csv"
    csv_path.write_text("0,1\n1,2\n", encoding="utf-8")
    assert Graph.from_edge_list(csv_path, delimiter=",").number_of_edges() == 2
