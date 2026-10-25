"""Directed acyclic graph (DAG) algorithms.

Topological ordering (Kahn, 1962) underpins scheduling, build systems and
causal / Bayesian-network work. This module provides:

* :func:`topological_sort` / :func:`topological_generations` – deterministic
  Kahn ordering (ties broken by node insertion order) and its layering;
* :func:`find_cycle` / :func:`is_dag` – a witness cycle when the graph is not
  acyclic;
* :func:`ancestors` / :func:`descendants` – reachability sets;
* :func:`dag_longest_path` – the maximum-weight path by dynamic programming
  over a topological order;
* :func:`count_paths` – the number of distinct source-target paths;
* :func:`transitive_closure` / :func:`transitive_reduction` – the unique
  minimal DAG with the same reachability (Aho, Garey & Ullman, 1972);
* :func:`critical_path` – the Critical Path Method (Kelley & Walker, 1959)
  with earliest/latest start and finish times and total float per task.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Mapping, Optional, Set, Tuple

from .graph import Graph

__all__ = [
    "CycleError",
    "find_cycle",
    "is_dag",
    "topological_sort",
    "topological_generations",
    "ancestors",
    "descendants",
    "dag_longest_path",
    "count_paths",
    "transitive_closure",
    "transitive_reduction",
    "critical_path",
]


class CycleError(ValueError):
    """Raised when a DAG algorithm is given a graph that contains a cycle."""

    def __init__(self, cycle: List[int]) -> None:
        self.cycle = cycle
        super().__init__("graph contains a cycle: " + " -> ".join(map(str, cycle + cycle[:1])))


def _require_directed(g: Graph, name: str) -> None:
    if not g.directed:
        raise ValueError(f"{name} requires a directed graph")


def find_cycle(g: Graph) -> Optional[List[int]]:
    """Return the nodes of one directed cycle, or ``None`` if ``g`` is acyclic.

    Uses an iterative three-colour DFS, so deep graphs do not hit the
    recursion limit. A self-loop is reported as the one-node cycle ``[u]``.
    """
    _require_directed(g, "find_cycle")
    white, grey, black = 0, 1, 2
    colour: Dict[int, int] = {u: white for u in g.nodes}
    parent: Dict[int, int] = {}
    for root in g.nodes:
        if colour[root] != white:
            continue
        colour[root] = grey
        stack: List[Tuple[int, int]] = [(root, 0)]
        while stack:
            u, i = stack[-1]
            nbrs = g.neighbors(u)
            if i < len(nbrs):
                stack[-1] = (u, i + 1)
                v = nbrs[i]
                if colour[v] == white:
                    colour[v] = grey
                    parent[v] = u
                    stack.append((v, 0))
                elif colour[v] == grey:
                    cycle = [u]
                    while cycle[-1] != v:
                        cycle.append(parent[cycle[-1]])
                    cycle.reverse()
                    return cycle
            else:
                colour[u] = black
                stack.pop()
    return None


def is_dag(g: Graph) -> bool:
    """Return ``True`` iff ``g`` is directed and has no directed cycle."""
    return g.directed and find_cycle(g) is None


def _in_degrees(g: Graph) -> Dict[int, int]:
    indeg = {u: 0 for u in g.nodes}
    for u in g.nodes:
        for v in g.neighbors(u):
            indeg[v] += 1
    return indeg


def topological_generations(g: Graph) -> List[List[int]]:
    """Return the Kahn layering of ``g``.

    Generation 0 holds the sources; generation ``k`` holds the nodes whose
    longest incoming path has ``k`` edges. Nodes inside a generation keep
    their insertion order. Raises :class:`CycleError` on cyclic input.
    """
    _require_directed(g, "topological_generations")
    indeg = _in_degrees(g)
    current = [u for u in g.nodes if indeg[u] == 0]
    order = {u: i for i, u in enumerate(g.nodes)}
    generations: List[List[int]] = []
    seen = 0
    while current:
        generations.append(current)
        seen += len(current)
        nxt: List[int] = []
        for u in current:
            for v in g.neighbors(u):
                indeg[v] -= 1
                if indeg[v] == 0:
                    nxt.append(v)
        current = sorted(nxt, key=order.__getitem__)
    if seen != g.number_of_nodes():
        raise CycleError(find_cycle(g) or [])
    return generations


def topological_sort(g: Graph) -> List[int]:
    """Return a topological order of ``g`` (Kahn's algorithm, O(V + E log V)).

    Among the nodes that are ready at each step the one inserted first is
    emitted, so the result is deterministic and is the lexicographically
    smallest order with respect to insertion order. Raises
    :class:`CycleError` (carrying a witness ``cycle``) if ``g`` is cyclic.
    """
    _require_directed(g, "topological_sort")
    indeg = _in_degrees(g)
    order = {u: i for i, u in enumerate(g.nodes)}
    nodes = g.nodes
    heap = [order[u] for u in nodes if indeg[u] == 0]
    heapq.heapify(heap)
    out: List[int] = []
    while heap:
        u = nodes[heapq.heappop(heap)]
        out.append(u)
        for v in g.neighbors(u):
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, order[v])
    if len(out) != len(nodes):
        raise CycleError(find_cycle(g) or [])
    return out


def _reach(g: Graph, node: int, forward: bool) -> Set[int]:
    if node not in set(g.nodes):
        raise KeyError(f"node {node} is not in the graph")
    if forward:
        adj = {u: g.neighbors(u) for u in g.nodes}
    else:
        adj = {u: [] for u in g.nodes}
        for u in g.nodes:
            for v in g.neighbors(u):
                adj[v].append(u)
    seen: Set[int] = set()
    stack = [node]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    seen.discard(node)
    return seen


def descendants(g: Graph, node: int) -> Set[int]:
    """Return every node reachable from ``node`` (excluding ``node`` itself)."""
    _require_directed(g, "descendants")
    return _reach(g, node, forward=True)


def ancestors(g: Graph, node: int) -> Set[int]:
    """Return every node that can reach ``node`` (excluding ``node`` itself)."""
    _require_directed(g, "ancestors")
    return _reach(g, node, forward=False)


def dag_longest_path(g: Graph, *, weighted: bool = True) -> Tuple[float, List[int]]:
    """Return ``(length, path)`` of a maximum-weight path in a DAG.

    Dynamic programming over :func:`topological_sort` in O(V + E). With
    ``weighted=False`` every edge counts 1, giving the number of edges on a
    longest path (the DAG's depth). A path may start at any node, so
    negative-weight edges are only used when they still lead to a larger total. An empty
    graph returns ``(0.0, [])``; a graph without edges returns a single node.
    """
    order = topological_sort(g)
    if not order:
        return 0.0, []
    best: Dict[int, float] = {u: 0.0 for u in order}
    pred: Dict[int, Optional[int]] = {u: None for u in order}
    for u in order:
        for v in g.neighbors(u):
            w = g.get_weight(u, v) if weighted else 1.0
            if best[u] + w > best[v]:
                best[v], pred[v] = best[u] + w, u
    rank = {u: i for i, u in enumerate(order)}
    end = max(order, key=lambda u: (best[u], -rank[u]))
    path = [end]
    while pred[path[-1]] is not None:
        path.append(pred[path[-1]])  # type: ignore[arg-type]
    path.reverse()
    return float(best[end]), path


def count_paths(g: Graph, source: int, target: int) -> int:
    """Return the number of distinct directed paths from ``source`` to ``target``.

    Exact integer DP over a topological order; ``count_paths(g, u, u) == 1``.
    """
    order = topological_sort(g)
    if source not in set(order) or target not in set(order):
        raise KeyError("source and target must be nodes of the graph")
    ways: Dict[int, int] = {u: 0 for u in order}
    ways[source] = 1
    start = order.index(source)
    for u in order[start:]:
        if ways[u]:
            for v in g.neighbors(u):
                ways[v] += ways[u]
    return ways[target]


def transitive_closure(g: Graph) -> Graph:
    """Return the DAG with an edge ``u -> v`` whenever ``v`` is reachable from ``u``.

    Built in reverse topological order with set unions; edges have weight 1.
    """
    order = topological_sort(g)
    reach: Dict[int, Set[int]] = {}
    for u in reversed(order):
        r: Set[int] = set()
        for v in g.neighbors(u):
            r.add(v)
            r |= reach[v]
        reach[u] = r
    out = Graph(directed=True)
    position = {u: i for i, u in enumerate(g.nodes)}
    for u in g.nodes:
        out.add_node(u)
    for u in g.nodes:
        for v in sorted(reach[u], key=position.__getitem__):
            out.add_edge(u, v)
    return out


def transitive_reduction(g: Graph) -> Graph:
    """Return the transitive reduction of a DAG.

    The result keeps edge ``u -> v`` (with its original weight) only when no
    other path leads from ``u`` to ``v``. For a DAG it is unique and has the
    same reachability as ``g`` with the fewest edges. Raises
    :class:`CycleError` on cyclic input.
    """
    order = topological_sort(g)
    reach: Dict[int, Set[int]] = {}
    for u in reversed(order):
        r: Set[int] = set()
        for v in g.neighbors(u):
            r.add(v)
            r |= reach[v]
        reach[u] = r
    out = Graph(directed=True)
    for u in g.nodes:
        out.add_node(u)
    for u in g.nodes:
        children = g.neighbors(u)
        covered: Set[int] = set()
        for v in children:
            covered |= reach[v]
        for v in children:
            if v not in covered:
                out.add_edge(u, v, weight=g.get_weight(u, v))
    return out


def critical_path(g: Graph, durations: Mapping[int, float]) -> Dict[str, object]:
    """Run the Critical Path Method on an activity-on-node DAG.

    ``durations[u]`` is the (non-negative) duration of task ``u``; an edge
    ``u -> v`` means ``v`` cannot start before ``u`` finishes. The forward
    pass gives earliest start/finish (``es``/``ef``), the backward pass
    latest start/finish (``ls``/``lf``) for the minimum project duration,
    and ``slack = ls - es`` is each task's total float. ``critical`` lists
    the zero-slack tasks in topological order and ``path`` is one critical
    chain from a source to a sink.
    """
    order = topological_sort(g)
    missing = [u for u in order if u not in durations]
    if missing:
        raise KeyError(f"missing durations for nodes: {missing}")
    dur = {u: float(durations[u]) for u in order}
    if any(d < 0 for d in dur.values()):
        raise ValueError("durations must be non-negative")
    preds: Dict[int, List[int]] = {u: [] for u in order}
    for u in order:
        for v in g.neighbors(u):
            preds[v].append(u)
    es: Dict[int, float] = {}
    ef: Dict[int, float] = {}
    for u in order:
        es[u] = max((ef[p] for p in preds[u]), default=0.0)
        ef[u] = es[u] + dur[u]
    makespan = max(ef.values(), default=0.0)
    lf: Dict[int, float] = {}
    ls: Dict[int, float] = {}
    for u in reversed(order):
        lf[u] = min((ls[s] for s in g.neighbors(u)), default=makespan)
        ls[u] = lf[u] - dur[u]
    tol = 1e-9 * max(1.0, makespan)
    slack = {u: ls[u] - es[u] for u in order}
    critical = [u for u in order if abs(slack[u]) <= tol]
    path: List[int] = []
    if order:
        current = next(u for u in order if not preds[u] and abs(slack[u]) <= tol)
        path.append(current)
        while True:
            nxt = [
                v for v in g.neighbors(current)
                if abs(slack[v]) <= tol and abs(es[v] - ef[current]) <= tol
            ]
            if not nxt:
                break
            current = nxt[0]
            path.append(current)
    return {
        "duration": makespan,
        "es": es,
        "ef": ef,
        "ls": ls,
        "lf": lf,
        "slack": slack,
        "critical": critical,
        "path": path,
    }
