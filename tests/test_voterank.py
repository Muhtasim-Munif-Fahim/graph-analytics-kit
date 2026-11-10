"""Tests for VoteRank influential-spreader selection."""
from __future__ import annotations

import pytest

from graph_analytics_kit import Graph, karate_club, voterank


def test_star_elects_hub_only():
    g = Graph([(0, leaf) for leaf in range(1, 6)])
    # After the hub is elected every leaf loses its whole vote (avg degree < 2).
    assert voterank(g) == [0]


def test_two_stars_elects_both_hubs():
    edges = [(0, i) for i in range(1, 6)] + [(10, i) for i in range(11, 16)] + [(5, 11)]
    g = Graph(edges)
    top = voterank(g, 2)
    assert set(top) == {0, 10}


def test_number_of_nodes_limits_and_zero():
    g = karate_club()
    assert len(voterank(g, 3)) == 3
    assert voterank(g, 0) == []
    assert voterank(g, 3) == voterank(g)[:3]


def test_karate_matches_reference_prefix():
    # Reference ordering from NetworkX's voterank on Zachary's karate club.
    assert voterank(karate_club(), 4) == [33, 0, 32, 2]


def test_elected_nodes_unique():
    out = voterank(karate_club())
    assert len(out) == len(set(out))


def test_directed_votes_flow_to_targets():
    g = Graph([(1, 0), (2, 0), (3, 0), (0, 4)], directed=True)
    assert voterank(g, 1) == [0]


def test_empty_and_edgeless():
    assert voterank(Graph()) == []
    g = Graph()
    g.add_node(1)
    g.add_node(2)
    assert voterank(g) == []


@pytest.mark.parametrize("bad", [-1, 1.5, True, "3"])
def test_invalid_number_of_nodes(bad):
    with pytest.raises(ValueError):
        voterank(karate_club(), bad)
