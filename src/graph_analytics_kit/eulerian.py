"""Eulerian circuits and paths (Hierholzer's algorithm).

An *Eulerian circuit* traverses every edge exactly once and returns to its
start; an *Eulerian path* traverses every edge exactly once but may end
elsewhere. Euler (1736) / Hierholzer (1873) characterisation:

* undirected — all nodes with edges lie in one connected component and
  every degree is even (circuit), or exactly two degrees are odd (path,
  running between the two odd nodes);
* directed — all nodes with edges lie in one weakly connected component and
  ``in == out`` everywhere (circuit), or exactly one node has
  ``out - in = 1`` (the start) and one has ``in - out = 1`` (the end).

Isolated nodes are ignored. A self-loop counts twice towards an undirected
degree and once towards both in- and out-degree of a directed node. Weights
are ignored. Hierholzer's algorithm runs in ``O(V + E)`` and is implemented
iteratively (no recursion limit).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .graph import Graph

__all__ = [
    "is_eulerian",
    "has_eulerian_path",
    "eulerian_circuit",
    "eulerian_path",
    "odd_degree_nodes",
]


def _undirected_degree(g: Graph, u: int) -> int:
    return sum(2 if v == u else 1 for v in g.neighbors(u))


def _in_out(g: Graph) -> Tuple[Dict[int, int], Dict[int, int]]:
    indeg = {u: 0 for u in g.nodes}
    outdeg = {u: 0 for u in g.nodes}
    for u in g.nodes:
        for v in g.neighbors(u):
            outdeg[u] += 1
            indeg[v] += 1
    return indeg, outdeg


def _edge_nodes_connected(g: Graph) -> bool:
    """All nodes incident to an edge are in one (weakly) connected component."""
    adj: Dict[int, Set[int]] = {u: set() for u in g.nodes}
    for u in g.nodes:
        for v in g.neighbors(u):
            adj[u].add(v)
            adj[v].add(u)
    active = [u for u in g.nodes if adj[u]]
    if not active:
        return True
    seen = {active[0]}
    queue = deque([active[0]])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return all(u in seen for u in active)


def odd_degree_nodes(g: Graph) -> List[int]:
    """Nodes with odd degree (undirected graphs only)."""
    if g.directed:
        raise ValueError("odd_degree_nodes is defined for undirected graphs")
    return [u for u in g.nodes if _undirected_degree(g, u) % 2 == 1]


def is_eulerian(g: Graph) -> bool:
    """True iff ``g`` has an Eulerian circuit (and at least one edge)."""
    if g.number_of_edges() == 0 or not _edge_nodes_connected(g):
        return False
    if g.directed:
        indeg, outdeg = _in_out(g)
        return all(indeg[u] == outdeg[u] for u in g.nodes)
    return not odd_degree_nodes(g)


def _path_endpoints(g: Graph) -> Optional[Tuple[int, Optional[int]]]:
    """Return ``(start, end)`` of an Eulerian path, ``end=None`` for a circuit."""
    if g.number_of_edges() == 0 or not _edge_nodes_connected(g):
        return None
    if g.directed:
        indeg, outdeg = _in_out(g)
        starts = [u for u in g.nodes if outdeg[u] - indeg[u] == 1]
        ends = [u for u in g.nodes if indeg[u] - outdeg[u] == 1]
        bad = [u for u in g.nodes if abs(outdeg[u] - indeg[u]) > 1]
        if bad:
            return None
        if not starts and not ends:
            return next(u for u in g.nodes if outdeg[u] > 0), None
        if len(starts) == 1 and len(ends) == 1:
            return starts[0], ends[0]
        return None
    odd = odd_degree_nodes(g)
    if not odd:
        return next(u for u in g.nodes if g.neighbors(u)), None
    if len(odd) == 2:
        return odd[0], odd[1]
    return None


def has_eulerian_path(g: Graph) -> bool:
    """True iff ``g`` has an Eulerian path (every circuit is also a path)."""
    return _path_endpoints(g) is not None


def _hierholzer(g: Graph, start: int) -> List[Tuple[int, int]]:
    # Remaining-edge adjacency (lists so traversal order is deterministic).
    remaining: Dict[int, List[int]] = {u: [] for u in g.nodes}
    for u in g.nodes:
        remaining[u] = list(reversed(g.neighbors(u)))
    used: Set[Tuple[int, int]] = set()
    stack: List[int] = [start]
    circuit: List[int] = []
    while stack:
        u = stack[-1]
        nbrs = remaining[u]
        advanced = False
        while nbrs:
            v = nbrs.pop()
            key = (u, v) if g.directed else (min(u, v), max(u, v))
            if key in used:
                continue
            used.add(key)
            stack.append(v)
            advanced = True
            break
        if not advanced:
            circuit.append(stack.pop())
    circuit.reverse()
    return list(zip(circuit[:-1], circuit[1:]))


def eulerian_circuit(g: Graph, source: Optional[int] = None) -> List[Tuple[int, int]]:
    """Return an Eulerian circuit as a list of ``(u, v)`` edges.

    The circuit starts and ends at ``source`` (default: first node with an
    edge). Raises ``ValueError`` if ``g`` is not Eulerian.
    """
    if not is_eulerian(g):
        raise ValueError("graph is not Eulerian (no circuit uses every edge once)")
    if source is None:
        source = next(u for u in g.nodes if g.neighbors(u))
    elif source not in set(g.nodes) or not g.neighbors(source):
        raise ValueError(f"source {source} is not incident to any edge")
    return _hierholzer(g, source)


def eulerian_path(g: Graph, source: Optional[int] = None) -> List[Tuple[int, int]]:
    """Return an Eulerian path as a list of ``(u, v)`` edges.

    If the graph has odd-degree nodes (or unbalanced directed nodes) the path
    must start at one of them; ``source`` may pick which (undirected) and is
    otherwise validated. Raises ``ValueError`` if no Eulerian path exists.
    """
    ends = _path_endpoints(g)
    if ends is None:
        raise ValueError("graph has no Eulerian path")
    start, end = ends
    if source is not None:
        if end is None:
            if source not in set(g.nodes) or not g.neighbors(source):
                raise ValueError(f"source {source} is not incident to any edge")
            start = source
        elif source == start:
            pass
        elif not g.directed and source == end:
            start = end
        else:
            raise ValueError(
                f"an Eulerian path must start at {start}"
                + ("" if g.directed else f" or {end}")
            )
    return _hierholzer(g, start)
