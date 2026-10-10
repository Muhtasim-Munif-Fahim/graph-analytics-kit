"""Cut structure: bridges, articulation points, and biconnected components.

All functions run one iterative Tarjan/Hopcroft depth-first search, which is
``O(n + m)`` and safe on long paths (it does not use Python recursion). Edge
weights and self-loops are ignored. A directed graph is analyzed as its
underlying simple undirected graph, matching :func:`core_number`.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .graph import Graph

Edge = Tuple[int, int]


def _undirected_neighbors(g: Graph) -> Dict[int, List[int]]:
    nbrs: Dict[int, Set[int]] = {u: set() for u in g.nodes}
    for u, v, _ in g.edges:
        if u == v:
            continue
        nbrs[u].add(v)
        nbrs[v].add(u)
    # Deterministic traversal order: node insertion order, then sorted labels.
    return {u: sorted(vs) for u, vs in nbrs.items()}


def _tarjan(g: Graph) -> Tuple[List[Edge], Set[int], List[List[int]]]:
    """Return ``(bridges, articulation_points, biconnected_components)``."""
    adj = _undirected_neighbors(g)
    disc: Dict[int, int] = {}
    low: Dict[int, int] = {}
    bridges_out: List[Edge] = []
    cut_nodes: Set[int] = set()
    components: List[List[int]] = []
    edge_stack: List[Edge] = []
    clock = 0

    for root in g.nodes:
        if root in disc:
            continue
        disc[root] = low[root] = clock
        clock += 1
        root_children = 0
        # Frame: (node, parent, index of next neighbor to explore).
        stack: List[List[int]] = [[root, -1, 0]]
        parent_of: Dict[int, object] = {root: None}
        while stack:
            frame = stack[-1]
            u, idx = frame[0], frame[2]
            nbrs = adj[u]
            if idx < len(nbrs):
                frame[2] += 1
                v = nbrs[idx]
                if v not in disc:
                    parent_of[v] = u
                    disc[v] = low[v] = clock
                    clock += 1
                    edge_stack.append((u, v))
                    if u == root:
                        root_children += 1
                    stack.append([v, u, 0])
                elif v != parent_of[u] and disc[v] < disc[u]:
                    # Back edge to an ancestor.
                    low[u] = min(low[u], disc[v])
                    edge_stack.append((u, v))
                continue

            # All neighbors of u explored: retreat to its parent.
            stack.pop()
            p = parent_of[u]
            if p is None:
                continue
            low[p] = min(low[p], low[u])
            if low[u] > disc[p]:
                bridges_out.append((min(p, u), max(p, u)))
            if low[u] >= disc[p]:
                if p != root:
                    cut_nodes.add(p)
                nodes: Set[int] = set()
                while edge_stack:
                    a, b = edge_stack.pop()
                    nodes.update((a, b))
                    if (a, b) == (p, u):
                        break
                components.append(sorted(nodes))
        if root_children > 1:
            cut_nodes.add(root)

    bridges_out.sort()
    components.sort(key=lambda c: (-len(c), c))
    return bridges_out, cut_nodes, components


def bridges(g: Graph) -> List[Edge]:
    """Return the bridges (cut edges) of *g* as sorted ``(u, v)`` pairs, ``u < v``.

    A bridge is an edge whose removal increases the number of connected
    components. Edges on a cycle are never bridges.
    """
    return _tarjan(g)[0]


def articulation_points(g: Graph) -> List[int]:
    """Return the articulation points (cut vertices) of *g*, sorted.

    An articulation point is a node whose removal (with its edges) increases
    the number of connected components.
    """
    return sorted(_tarjan(g)[1])


def biconnected_components(g: Graph) -> List[List[int]]:
    """Return the biconnected components of *g* as sorted node lists.

    Each component is a maximal set of nodes that stays connected after
    removing any single node; a bridge forms a two-node component. Isolated
    nodes belong to no component. Components are ordered largest first, then
    by their smallest labels. Articulation points are exactly the nodes that
    appear in more than one component.
    """
    return _tarjan(g)[2]


def two_edge_connected_components(g: Graph) -> List[List[int]]:
    """Return the 2-edge-connected components of *g* (connected after deleting bridges).

    Every node, including isolated ones, belongs to exactly one component.
    Components are ordered largest first, then by their smallest labels.
    """
    cut = set(_tarjan(g)[0])
    adj = _undirected_neighbors(g)
    seen: Set[int] = set()
    out: List[List[int]] = []
    for start in g.nodes:
        if start in seen:
            continue
        seen.add(start)
        comp = [start]
        frontier = [start]
        while frontier:
            u = frontier.pop()
            for v in adj[u]:
                if v in seen or (min(u, v), max(u, v)) in cut:
                    continue
                seen.add(v)
                comp.append(v)
                frontier.append(v)
        out.append(sorted(comp))
    out.sort(key=lambda c: (-len(c), c))
    return out


def cut_structure(g: Graph) -> Dict[str, object]:
    """Summarize bridges, articulation points, and biconnected components in one pass."""
    br, cuts, comps = _tarjan(g)
    return {
        "bridges": br,
        "articulation_points": sorted(cuts),
        "biconnected_components": comps,
        "n_bridges": len(br),
        "n_articulation_points": len(cuts),
        "n_biconnected_components": len(comps),
        "largest_biconnected_component": len(comps[0]) if comps else 0,
    }
