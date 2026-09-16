"""Community detection via the Louvain method (modularity maximization)."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .graph import Graph


def communities(
    g: Graph,
    *,
    max_iter: int = 100,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """Detect communities with the Louvain method.

    Louvain greedily maximizes Newman modularity in two repeating phases
    (Blondel et al., 2008): local moves of nodes into neighboring
    communities, then aggregation of each community into a supernode. The
    hierarchy stops when a pass no longer merges communities. Isolated
    nodes remain their own community. Directed graphs are not supported.

    When *seed* is ``None`` the scan order is ``g.nodes`` and the result is
    deterministic. When *seed* is set, each local-moving sweep shuffles
    that order with the given RNG.

    Parameters
    ----------
    g:
        Undirected input graph.
    max_iter:
        Maximum local-moving sweeps per hierarchy level. A level also
        stops as soon as a sweep moves no nodes.
    seed:
        Optional RNG seed for shuffled local-moving sweeps.

    Returns
    -------
    list[list[int]]
        Communities as lists of node labels. Nodes within a community are
        sorted, and communities are ordered by their smallest node label.
    """
    if g.directed:
        raise ValueError("communities() requires an undirected graph")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    nodes = g.nodes
    if not nodes:
        return []

    adj = _undirected_adj(g)
    rng = random.Random(seed) if seed is not None else None
    assignment = {node: node for node in nodes}
    current_nodes = list(nodes)
    current_adj = adj

    for _ in range(len(nodes)):
        membership = _one_level(current_adj, current_nodes, max_iter, rng)
        assignment = {node: membership[assignment[node]] for node in assignment}
        n_communities = len(set(membership.values()))
        if n_communities == len(current_nodes):
            break
        current_adj, current_nodes = _aggregate(current_adj, current_nodes, membership)
        if len(current_nodes) <= 1:
            break

    return _labels_to_communities(assignment)


def modularity(g: Graph, parts: Sequence[Iterable[int]]) -> float:
    """Return Newman modularity of *parts* on undirected graph *g*.

    .. math::
        Q = \\sum_c \\left(
            \\frac{L_c}{m} - \\left(\\frac{k_c}{2m}\\right)^2
        \\right)

    where :math:`L_c` is the total weight of edges inside community *c*
    and :math:`k_c` is the total degree of nodes in *c*. Every node of
    *g* must appear in exactly one part. Graphs with no edges score
    ``0.0``.
    """
    if g.directed:
        raise ValueError("modularity() requires an undirected graph")

    nodes = g.nodes
    if not nodes:
        return 0.0

    comm = _partition_map(nodes, parts)
    strength = {node: 0.0 for node in nodes}
    internal: Dict[int, float] = defaultdict(float)
    for u, v, weight in g.edges:
        strength[u] += weight
        if u != v:
            strength[v] += weight
        if comm[u] == comm[v]:
            internal[comm[u]] += weight

    two_m = sum(strength.values())
    if two_m == 0.0:
        return 0.0
    m = two_m / 2.0

    degree_sum: Dict[int, float] = defaultdict(float)
    for node in nodes:
        degree_sum[comm[node]] += strength[node]

    quality = 0.0
    for community_id in set(comm.values()):
        quality += internal[community_id] / m - (degree_sum[community_id] / two_m) ** 2
    return quality


def _undirected_adj(g: Graph) -> Dict[int, Dict[int, float]]:
    adj: Dict[int, Dict[int, float]] = {node: {} for node in g.nodes}
    for u, v, weight in g.edges:
        adj[u][v] = adj[u].get(v, 0.0) + weight
        if u != v:
            adj[v][u] = adj[v].get(u, 0.0) + weight
    return adj


def _one_level(
    adj: Dict[int, Dict[int, float]],
    nodes: Sequence[int],
    max_iter: int,
    rng: Optional[random.Random],
) -> Dict[int, int]:
    strength = {node: sum(adj.get(node, {}).values()) for node in nodes}
    two_m = sum(strength.values())
    community = {node: node for node in nodes}
    if two_m == 0.0:
        return community

    sigma_tot = dict(strength)
    order = list(nodes)
    for _ in range(max_iter):
        moved = False
        if rng is not None:
            rng.shuffle(order)
        for node in order:
            c_old = community[node]
            k_i = strength[node]
            sigma_tot[c_old] -= k_i

            weights: Dict[int, float] = defaultdict(float)
            for nbr, weight in adj.get(node, {}).items():
                if nbr == node:
                    continue
                weights[community[nbr]] += weight

            best_comm = c_old
            best_gain = 0.0
            for comm_id, k_in in weights.items():
                gain = k_in - sigma_tot[comm_id] * k_i / two_m
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_comm = comm_id
                elif (
                    gain > 1e-12
                    and abs(gain - best_gain) <= 1e-12
                    and comm_id < best_comm
                ):
                    best_comm = comm_id

            community[node] = best_comm
            sigma_tot[best_comm] += k_i
            if best_comm != c_old:
                moved = True
        if not moved:
            break
    return community


def _aggregate(
    adj: Dict[int, Dict[int, float]],
    nodes: Sequence[int],
    membership: Dict[int, int],
) -> Tuple[Dict[int, Dict[int, float]], List[int]]:
    new_adj: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for u in nodes:
        c_u = membership[u]
        for v, weight in adj.get(u, {}).items():
            new_adj[c_u][membership[v]] += weight
    new_nodes = sorted(set(membership.values()))
    return {node: dict(new_adj[node]) for node in new_nodes}, new_nodes


def _labels_to_communities(labels: Dict[int, int]) -> List[List[int]]:
    groups: Dict[int, List[int]] = defaultdict(list)
    for node, label in labels.items():
        groups[label].append(node)
    parts = [sorted(members) for members in groups.values()]
    parts.sort(key=lambda members: members[0])
    return parts


def _partition_map(
    nodes: Sequence[int],
    parts: Sequence[Iterable[int]],
) -> Dict[int, int]:
    comm: Dict[int, int] = {}
    node_set = set(nodes)
    for index, part in enumerate(parts):
        for node in part:
            if node in comm:
                raise ValueError(f"node {node} appears in more than one community")
            comm[node] = index
    missing = [node for node in nodes if node not in comm]
    if missing:
        raise ValueError(f"partition is missing nodes: {missing}")
    extra = [node for node in comm if node not in node_set]
    if extra:
        raise ValueError(f"partition contains unknown nodes: {extra}")
    return comm


__all__ = ["communities", "modularity"]
