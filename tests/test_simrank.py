"""Tests for SimRank structural similarity."""
from __future__ import annotations

import numpy as np
import pytest

from graph_analytics_kit import (
    Graph,
    karate_club,
    most_similar,
    simrank_matrix,
    simrank_scores,
    simrank_similarity,
)
from graph_analytics_kit.cli import main


def _fixed_point_residual(g: Graph, decay: float) -> float:
    nodes, S = simrank_matrix(g, decay=decay, tol=1e-12, max_iter=2000)
    index = {n: i for i, n in enumerate(nodes)}
    A = g.adjacency_matrix() != 0
    in_nbrs = {n: [nodes[i] for i in np.flatnonzero(A[:, index[n]])] for n in nodes}
    worst = 0.0
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            Ia, Ib = in_nbrs[a], in_nbrs[b]
            want = 0.0
            if Ia and Ib:
                want = decay / (len(Ia) * len(Ib)) * sum(S[index[i], index[j]] for i in Ia for j in Ib)
            worst = max(worst, abs(want - S[index[a], index[b]]))
    return worst


def test_karate_satisfies_simrank_equations():
    assert _fixed_point_residual(karate_club(), 0.8) < 1e-9


def test_directed_graph_satisfies_simrank_equations():
    g = Graph([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 0), (5, 4)], directed=True)
    assert _fixed_point_residual(g, 0.6) < 1e-10
    # node 5 has no in-neighbours, so it is similar only to itself
    row = simrank_similarity(g, 5)
    assert row[5] == 1.0 and all(score == 0.0 for node, score in row.items() if node != 5)


def test_star_leaves_have_similarity_c():
    g = Graph([(0, leaf) for leaf in range(1, 6)])
    for decay in (0.5, 0.8):
        assert simrank_similarity(g, 1, 2, decay=decay) == pytest.approx(decay, abs=1e-6)
        # centre vs leaf: their in-neighbours never overlap structurally
        assert simrank_similarity(g, 0, 3, decay=decay) == pytest.approx(0.0, abs=1e-12)


def test_matrix_properties_and_monotone_in_decay():
    nodes, S = simrank_matrix(karate_club())
    assert S.shape == (len(nodes), len(nodes))
    assert np.allclose(S, S.T)
    assert np.allclose(np.diag(S), 1.0)
    assert S.min() >= 0.0 and S.max() <= 1.0 + 1e-12
    _, S_low = simrank_matrix(karate_club(), decay=0.5)
    off = ~np.eye(len(nodes), dtype=bool)
    assert np.all(S_low[off] <= S[off] + 1e-12)


def test_scores_pairs_without_common_neighbours():
    # path 0-1-2-3-4: 0 and 4 share no neighbour, yet SimRank is positive
    # through the chain (0,4) <- (1,3) <- (2,2) in the even-distance pairing.
    g = Graph([(0, 1), (1, 2), (2, 3), (3, 4)])
    assert simrank_similarity(g, 0, 4) > 0.0
    # odd distance on a bipartite path is always zero
    assert simrank_similarity(g, 0, 1) == pytest.approx(0.0, abs=1e-12)


def test_batch_scores_and_missing_nodes():
    g = karate_club()
    pairs = [(0, 33), (5, 6), (0, 999)]
    scores = simrank_scores(g, pairs)
    assert scores[0] == pytest.approx(simrank_similarity(g, 0, 33))
    assert scores[1] == pytest.approx(simrank_similarity(g, 5, 6))
    assert scores[2] == 0.0
    assert simrank_scores(g, []) == []
    with pytest.raises(KeyError):
        simrank_similarity(g, 999)


def test_most_similar_ranking_and_bipartite_roles():
    # two users who rated the same items are each other's best match
    g = Graph([(1, 10), (1, 11), (2, 10), (2, 11), (3, 12), (3, 13), (4, 12)])
    top = most_similar(g, 1, k=3)
    assert top[0][0] == 2
    scores = [s for _, s in top]
    assert scores == sorted(scores, reverse=True)
    assert len(most_similar(g, 1, k=50)) == len(g.nodes) - 1


def test_validation():
    g = karate_club()
    with pytest.raises(ValueError, match="decay"):
        simrank_matrix(g, decay=1.0)
    with pytest.raises(ValueError, match="max_iter"):
        simrank_matrix(g, max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        simrank_matrix(g, tol=0.0)
    with pytest.raises(ValueError, match="k"):
        most_similar(g, 0, k=0)
    assert simrank_matrix(Graph())[1].shape == (0, 0)


def test_cli_similar_and_link_prediction(capsys):
    assert main(["similar", "--node", "33", "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert "most SimRank-similar to 33" in out and len(out.strip().splitlines()) == 4
    assert main(["link-prediction", "--u", "0", "--v", "33", "--metric", "simrank", "--top-pairs", "2"]) == 0
    out = capsys.readouterr().out
    assert "simrank(0, 33):" in out and "top non-edges by SimRank" in out
    assert main(["similar", "--node", "999"]) == 2
