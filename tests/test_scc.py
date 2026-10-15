"""Tests for strongly connected components."""

from __future__ import annotations

import pytest

from graph_analytics_kit import Graph
from graph_analytics_kit.scc import (
    condensation_dag,
    is_strongly_connected,
    strongly_connected_components,
)


def test_scc_classic_example():
    # Two SCCs: {0,1,2} and {3,4}, plus edge 2->3
    g = Graph(
        [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3)],
        directed=True,
    )
    comps = strongly_connected_components(g)
    sets = {frozenset(c) for c in comps}
    assert frozenset([0, 1, 2]) in sets
    assert frozenset([3, 4]) in sets
    assert len(comps) == 2


def test_single_node_and_cycle():
    g = Graph([(0, 0)], directed=True)  # may or may not allow self-loop
    # Prefer a 3-cycle
    g = Graph([(0, 1), (1, 2), (2, 0)], directed=True)
    assert is_strongly_connected(g)
    assert len(strongly_connected_components(g)) == 1


def test_condensation_is_dag():
    g = Graph(
        [(0, 1), (1, 0), (1, 2), (2, 3), (3, 2), (3, 4)],
        directed=True,
    )
    dag, comps, mapping = condensation_dag(g)
    assert dag.directed
    assert len(comps) == len(dag.nodes)
    # No cycles in condensation: Tarjan finish order => edges go forward
    # Check acyclicity via DFS
    def has_cycle(graph):
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in graph.nodes}

        def dfs(u):
            color[u] = GRAY
            for v in graph.neighbors(u):
                if color[v] == GRAY:
                    return True
                if color[v] == WHITE and dfs(v):
                    return True
            color[u] = BLACK
            return False

        return any(dfs(n) for n in graph.nodes if color[n] == WHITE)

    assert not has_cycle(dag)
    assert mapping[0] == mapping[1]


def test_undirected_rejected():
    g = Graph([(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="directed"):
        strongly_connected_components(g)


def test_isolated_nodes():
    g = Graph(directed=True)
    g.add_node(0)
    g.add_node(1)
    g.add_edge(0, 1)
    comps = strongly_connected_components(g)
    assert len(comps) == 2
