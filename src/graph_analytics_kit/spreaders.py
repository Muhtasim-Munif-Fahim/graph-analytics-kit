"""Influential-spreader selection with VoteRank (Zhang et al., 2016).

VoteRank picks a set of decentralised spreaders by running repeated voting
rounds. Every node starts with voting ability ``1``. In each round a node's
score is the sum of the voting abilities of the nodes that vote for it (its
in-neighbours on a directed graph, its neighbours otherwise). The top scorer
is elected, loses its own vote, and every neighbour of the elected node has
its voting ability reduced by ``1 / <k>`` (``<k>`` = average degree), so the
next spreaders tend to sit away from the ones already chosen.

Reference: J.-X. Zhang, D.-B. Chen, Q. Dong, Z.-D. Zhao, "Identifying a set
of influential spreaders in complex networks", Scientific Reports 6, 27823
(2016).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from .graph import Graph


def _predecessors(g: Graph) -> Dict[int, Set[int]]:
    preds: Dict[int, Set[int]] = {node: set() for node in g.nodes}
    for u in g.nodes:
        for v in g.neighbors(u):
            if v != u:
                preds[v].add(u)
    return preds


def voterank(g: Graph, number_of_nodes: Optional[int] = None) -> List[int]:
    """Return influential spreaders of *g* in election order using VoteRank.

    Parameters
    ----------
    g:
        Input graph (directed or undirected). Edge weights are ignored and
        self-loops never vote.
    number_of_nodes:
        Maximum number of spreaders to elect. ``None`` (default) keeps
        electing until no remaining node receives a positive vote.

    Returns
    -------
    list[int]
        Elected node labels, most influential first. Ties are broken by node
        insertion order. Fewer than ``number_of_nodes`` labels are returned
        when the remaining nodes cannot collect any votes.
    """
    if number_of_nodes is not None:
        if isinstance(number_of_nodes, bool) or not isinstance(number_of_nodes, int) or number_of_nodes < 0:
            raise ValueError("number_of_nodes must be a non-negative integer or None")
    nodes = g.nodes
    if not nodes or number_of_nodes == 0:
        return []
    limit = len(nodes) if number_of_nodes is None else min(number_of_nodes, len(nodes))

    preds = _predecessors(g)
    succs: Dict[int, Set[int]] = {u: {v for v in g.neighbors(u) if v != u} for u in nodes}
    # Average degree: out-degree for directed graphs, plain degree otherwise.
    avg_degree = sum(len(succs[u]) for u in nodes) / len(nodes)
    if avg_degree == 0:
        return []
    decrement = 1.0 / avg_degree

    ability: Dict[int, float] = {node: 1.0 for node in nodes}
    elected: List[int] = []
    elected_set: Set[int] = set()
    while len(elected) < limit:
        best: Optional[int] = None
        best_score = 0.0
        for node in nodes:
            if node in elected_set:
                continue
            score = sum(ability[voter] for voter in preds[node])
            if score > best_score + 1e-12:
                best, best_score = node, score
        if best is None:
            break
        elected.append(best)
        elected_set.add(best)
        ability[best] = 0.0
        for nbr in preds[best] | succs[best]:
            ability[nbr] = max(0.0, ability[nbr] - decrement)
    return elected


__all__ = ["voterank"]
