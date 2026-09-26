"""Community detection: Louvain, Girvan–Newman, and label propagation."""
from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

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



def girvan_newman(
    g: Graph,
    *,
    n_communities: Optional[int] = None,
) -> List[List[int]]:
    """Detect communities with the Girvan–Newman edge-betweenness method.

    Iteratively removes the edge with the highest betweenness centrality
    (Girvan & Newman, 2002). Betweenness is recomputed after every removal.
    When *n_communities* is ``None`` the dendrogram level with the highest
    Newman modularity is returned; otherwise removal stops as soon as that
    many connected components exist. Isolated nodes remain their own
    community. Directed graphs are not supported. Ties among equal
    betweenness edges break by ascending ``(min(u, v), max(u, v))``.

    Parameters
    ----------
    g:
        Undirected input graph.
    n_communities:
        Optional target community count. Must be at least 1 and at most
        the number of nodes when set.

    Returns
    -------
    list[list[int]]
        Communities as lists of node labels. Nodes within a community are
        sorted, and communities are ordered by their smallest node label.
    """
    if g.directed:
        raise ValueError("girvan_newman() requires an undirected graph")
    nodes = g.nodes
    if not nodes:
        return []
    if n_communities is not None:
        if (
            not isinstance(n_communities, int)
            or isinstance(n_communities, bool)
            or n_communities < 1
            or n_communities > len(nodes)
        ):
            raise ValueError(
                "n_communities must be an integer between 1 and the number of nodes"
            )

    # Work on a mutable adjacency copy (ignore self-loops for splitting).
    adj: Dict[int, Dict[int, float]] = {node: {} for node in nodes}
    for u, v, weight in g.edges:
        if u == v:
            continue
        adj[u][v] = adj[u].get(v, 0.0) + weight
        adj[v][u] = adj[v].get(u, 0.0) + weight

    def components() -> List[List[int]]:
        seen = set()
        parts: List[List[int]] = []
        for start in sorted(nodes):
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            comp = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for nbr in adj[node]:
                    if nbr not in seen:
                        seen.add(nbr)
                        stack.append(nbr)
            parts.append(sorted(comp))
        parts.sort(key=lambda part: part[0])
        return parts

    # Original Graph for modularity scoring (edge set does not change).
    best_parts = components()
    best_q = modularity(g, best_parts)

    while True:
        parts = components()
        if n_communities is not None and len(parts) >= n_communities:
            return parts
        q = modularity(g, parts)
        if q >= best_q:
            best_q = q
            best_parts = parts

        # Count remaining undirected edges.
        n_edges = sum(len(nbrs) for nbrs in adj.values()) // 2
        if n_edges == 0:
            break

        betweenness = _edge_betweenness(adj)
        # Highest betweenness; ties break by ascending endpoint pair.
        edge, _score = max(
            betweenness.items(),
            key=lambda item: (item[1], -item[0][0], -item[0][1]),
        )
        u, v = edge
        adj[u].pop(v, None)
        adj[v].pop(u, None)

    if n_communities is not None:
        return components()
    return best_parts


def _edge_betweenness(adj: Dict[int, Dict[int, float]]) -> Dict[Tuple[int, int], float]:
    """Brandes edge betweenness on an unweighted view of *adj*.

    Edge weights are ignored for path length (hop count), matching the
    classic Girvan–Newman formulation. Each undirected edge is keyed as
    ``(min(u, v), max(u, v))``.
    """
    nodes = list(adj.keys())
    betweenness: Dict[Tuple[int, int], float] = defaultdict(float)
    for source in nodes:
        # BFS layering
        S: list[int] = []
        P: Dict[int, list[int]] = {node: [] for node in nodes}
        sigma: Dict[int, float] = {node: 0.0 for node in nodes}
        sigma[source] = 1.0
        dist: Dict[int, int] = {node: -1 for node in nodes}
        dist[source] = 0
        queue: deque[int] = deque([source])
        while queue:
            v = queue.popleft()
            S.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        delta: Dict[int, float] = {node: 0.0 for node in nodes}
        while S:
            w = S.pop()
            for v in P[w]:
                if sigma[w] == 0.0:
                    continue
                c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                edge = (v, w) if v < w else (w, v)
                betweenness[edge] += c
                delta[v] += c

    # Undirected: Brandes counts each edge once per directed traversal pair;
    # divide by 2 to match the undirected convention.
    return {edge: value / 2.0 for edge, value in betweenness.items()}



def label_propagation(
    g: Graph,
    *,
    max_iter: int = 100,
    seed: Optional[int] = None,
    return_labels: bool = False,
) -> Union[List[List[int]], Tuple[List[List[int]], Dict[int, int]]]:
    """Detect communities with asynchronous label propagation.

    Each node starts with its own label. In a sweep, every node adopts the
    label with the greatest total incident weight among its neighbors
    (Raghavan, Albert, and Kumara, 2007). The current label is kept when
    it is already one of those maxima. Otherwise a tie uses the smallest
    label when *seed* is ``None``, and a uniform draw from *seed* when it
    is set. Self-loops are ignored. Isolated nodes stay singletons.
    Directed graphs are not supported.

    When *seed* is ``None`` each sweep visits nodes in increasing neighbor
    count, then node label, so the partition is deterministic and low-degree
    nodes consolidate before hubs. When *seed* is set, each sweep shuffles
    that order with the given RNG. Sweeps stop when a pass changes no label
    or *max_iter* is reached.

    Parameters
    ----------
    g:
        Undirected input graph. Edge weights vote for the neighbor's label;
        omitted weights count as ``1``.
    max_iter:
        Maximum sweeps. A run also stops as soon as a sweep changes nothing.
    seed:
        Optional RNG seed for shuffled sweeps and random tie breaks.
    return_labels:
        When ``True``, also return the node-to-label map. Label values are
        the surviving propagated ids (original node labels), not renumbered
        community indexes.

    Returns
    -------
    list[list[int]] or tuple[list[list[int]], dict[int, int]]
        Communities as lists of node labels, in the same order as
        :func:`communities`: members sorted, communities ordered by their
        smallest node label. With ``return_labels=True``, a ``(communities,
        labels)`` pair.
    """
    if g.directed:
        raise ValueError("label_propagation() requires an undirected graph")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    nodes = g.nodes
    if not nodes:
        parts: List[List[int]] = []
        return (parts, {}) if return_labels else parts

    adj = _undirected_adj(g)
    neighbors = {
        node: {nbr: weight for nbr, weight in adj[node].items() if nbr != node}
        for node in nodes
    }
    labels = {node: node for node in nodes}
    rng = random.Random(seed) if seed is not None else None
    if rng is None:
        base_order = sorted(nodes, key=lambda node: (len(neighbors[node]), node))
    else:
        base_order = list(nodes)

    for _ in range(max_iter):
        order = list(base_order)
        if rng is not None:
            rng.shuffle(order)
        changed = False
        for node in order:
            votes: Dict[int, float] = defaultdict(float)
            for nbr, weight in neighbors[node].items():
                votes[labels[nbr]] += weight
            if not votes:
                continue
            best_score = max(votes.values())
            candidates = sorted(
                label for label, score in votes.items() if score >= best_score - 1e-12
            )
            current = labels[node]
            if current in candidates:
                chosen = current
            elif rng is not None:
                chosen = rng.choice(candidates)
            else:
                chosen = candidates[0]
            if chosen != current:
                labels[node] = chosen
                changed = True
        if not changed:
            break

    parts = _labels_to_communities(labels)
    if return_labels:
        return parts, labels
    return parts


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


__all__ = ["communities", "girvan_newman", "label_propagation", "modularity"]
