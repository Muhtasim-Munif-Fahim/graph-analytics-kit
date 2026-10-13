"""Maximum flow, minimum s-t cut, edge connectivity, and bipartite matching.

``maximum_flow`` runs Dinic's algorithm (BFS level graph + blocking flow
with current-arc pointers), ``O(V^2 E)`` in general and ``O(E sqrt(V))`` on
unit-capacity graphs. Edge weights are capacities. In an undirected graph
every edge ``{u, v}`` with weight ``c`` can carry up to ``c`` units in
either direction, the standard convention. The max-flow/min-cut theorem
then gives the minimum s-t cut: the source side is everything reachable
from ``source`` in the final residual graph.

``edge_connectivity`` uses unit capacities: the local version is the
number of edge-disjoint s-t paths (Menger), the global version the size of
the smallest edge set whose removal disconnects the graph (Even-Tarjan
reduction: fix any node ``s`` and minimise over ``t``; for directed graphs
both directions are tried).

``maximum_bipartite_matching`` is Hopcroft-Karp on a bipartite graph
whose left side is given (or 2-coloured automatically).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .graph import Graph

Edge = Tuple[int, int]


@dataclass
class FlowResult:
    """Outcome of :func:`maximum_flow`."""

    value: float
    flow: Dict[Edge, float]
    source_side: List[int]
    sink_side: List[int]
    cut_edges: List[Edge]
    cut_capacity: float
    meta: Dict[str, object] = field(default_factory=dict)


class _Residual:
    """Arc-list residual network for Dinic's algorithm."""

    def __init__(self, n: int) -> None:
        self.head: List[List[int]] = [[] for _ in range(n)]
        self.to: List[int] = []
        self.cap: List[float] = []

    def add(self, u: int, v: int, c_uv: float, c_vu: float) -> int:
        idx = len(self.to)
        self.to.append(v)
        self.cap.append(c_uv)
        self.head[u].append(idx)
        self.to.append(u)
        self.cap.append(c_vu)
        self.head[v].append(idx + 1)
        return idx

    def bfs(self, s: int, t: int, eps: float) -> Optional[List[int]]:
        level = [-1] * len(self.head)
        level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for e in self.head[u]:
                v = self.to[e]
                if level[v] < 0 and self.cap[e] > eps:
                    level[v] = level[u] + 1
                    q.append(v)
        return level if level[t] >= 0 else None

    def blocking_flow(self, s: int, t: int, level: List[int], eps: float) -> float:
        it = [0] * len(self.head)
        total = 0.0
        while True:
            # iterative DFS for one augmenting path in the level graph
            path: List[int] = []
            u = s
            while u != t:
                advanced = False
                arcs = self.head[u]
                while it[u] < len(arcs):
                    e = arcs[it[u]]
                    v = self.to[e]
                    if self.cap[e] > eps and level[v] == level[u] + 1:
                        path.append(e)
                        u = v
                        advanced = True
                        break
                    it[u] += 1
                if not advanced:
                    if u == s:
                        return total
                    level[u] = -1  # dead end: prune node
                    e = path.pop()
                    u = self.to[e ^ 1]
                    it[u] += 1
            push = min(self.cap[e] for e in path)
            for e in path:
                self.cap[e] -= push
                self.cap[e ^ 1] += push
            total += push

    def reachable(self, s: int, eps: float) -> Set[int]:
        seen = {s}
        q = deque([s])
        while q:
            u = q.popleft()
            for e in self.head[u]:
                v = self.to[e]
                if v not in seen and self.cap[e] > eps:
                    seen.add(v)
                    q.append(v)
        return seen


def _check_nodes(g: Graph, source: int, sink: int) -> None:
    nodes = set(g.nodes)
    if source not in nodes:
        raise ValueError(f"source {source} is not in the graph")
    if sink not in nodes:
        raise ValueError(f"sink {sink} is not in the graph")
    if source == sink:
        raise ValueError("source and sink must differ")


def _capacity_pairs(g: Graph, unit: bool) -> Dict[Edge, float]:
    """Directed capacity map; undirected edges contribute both directions."""
    caps: Dict[Edge, float] = {}
    for u, v, w in g.edges:
        if u == v:
            continue
        c = 1.0 if unit else float(w)
        if c < 0:
            raise ValueError(f"edge ({u}, {v}) has negative capacity {c}")
        caps[(u, v)] = caps.get((u, v), 0.0) + c
        if not g.directed:
            caps[(v, u)] = caps.get((v, u), 0.0) + c
    return caps


def maximum_flow(
    g: Graph,
    source: int,
    sink: int,
    *,
    unit_capacity: bool = False,
    eps: float = 1e-12,
) -> FlowResult:
    """Maximum ``source`` -> ``sink`` flow with Dinic's algorithm.

    Capacities are edge weights (``unit_capacity=True`` treats every edge
    as capacity 1). ``flow`` maps each directed arc ``(u, v)`` carrying
    positive net flow to that amount (for undirected graphs at most one
    orientation of an edge appears). The minimum cut is returned too:
    ``source_side`` / ``sink_side`` node lists (sorted), the saturated
    ``cut_edges`` crossing from source to sink side, and their total
    ``cut_capacity`` (equal to ``value``).
    """
    _check_nodes(g, source, sink)
    nodes = g.nodes
    index = {node: i for i, node in enumerate(nodes)}
    caps = _capacity_pairs(g, unit_capacity)
    R = _Residual(len(nodes))
    arc_of: Dict[Edge, int] = {}
    done: Set[Edge] = set()
    for (u, v), c in caps.items():
        if (u, v) in done:
            continue
        back = caps.get((v, u), 0.0)
        arc_of[(u, v)] = R.add(index[u], index[v], c, back)
        done.add((u, v))
        done.add((v, u))
    s, t = index[source], index[sink]
    value = 0.0
    phases = 0
    while True:
        level = R.bfs(s, t, eps)
        if level is None:
            break
        phases += 1
        value += R.blocking_flow(s, t, level, eps)

    flow: Dict[Edge, float] = {}
    for (u, v), e in arc_of.items():
        original = caps[(u, v)]
        net = original - R.cap[e]  # positive => u->v, negative => v->u
        if net > eps:
            flow[(u, v)] = net
        elif net < -eps:
            flow[(v, u)] = -net
    reach = R.reachable(s, eps)
    src_side = sorted(nodes[i] for i in reach)
    src_set = set(src_side)
    sink_side = sorted(n for n in nodes if n not in src_set)
    cut_edges = sorted(
        (u, v) for (u, v), c in caps.items() if c > 0 and u in src_set and v not in src_set
    )
    cut_capacity = float(sum(caps[e] for e in cut_edges))
    return FlowResult(
        value=float(value),
        flow=flow,
        source_side=src_side,
        sink_side=sink_side,
        cut_edges=cut_edges,
        cut_capacity=cut_capacity,
        meta={"algorithm": "dinic", "phases": phases, "unit_capacity": unit_capacity},
    )


def minimum_cut(g: Graph, source: int, sink: int, *, unit_capacity: bool = False) -> Tuple[float, List[int], List[int], List[Edge]]:
    """``(cut_value, source_side, sink_side, cut_edges)`` of a minimum s-t cut."""
    res = maximum_flow(g, source, sink, unit_capacity=unit_capacity)
    return res.value, res.source_side, res.sink_side, res.cut_edges


def edge_connectivity(g: Graph, source: Optional[int] = None, sink: Optional[int] = None) -> int:
    """Local (s-t) or global edge connectivity with unit capacities.

    With both ``source`` and ``sink`` it is the number of edge-disjoint
    paths between them. With neither it is the global edge connectivity
    ``lambda(G)`` (0 for a disconnected graph, and for graphs with < 2 nodes).
    """
    if (source is None) != (sink is None):
        raise ValueError("pass both source and sink, or neither")
    if source is not None:
        return int(round(maximum_flow(g, source, sink, unit_capacity=True).value))
    nodes = g.nodes
    if len(nodes) < 2:
        return 0
    s = nodes[0]
    best: Optional[int] = None
    for t in nodes[1:]:
        k = int(round(maximum_flow(g, s, t, unit_capacity=True).value))
        if g.directed:
            k = min(k, int(round(maximum_flow(g, t, s, unit_capacity=True).value)))
        best = k if best is None else min(best, k)
        if best == 0:
            break
    return int(best or 0)


def _two_colour(g: Graph) -> Set[int]:
    left: Set[int] = set()
    colour: Dict[int, int] = {}
    for start in g.nodes:
        if start in colour:
            continue
        colour[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if colour[u] == 0:
                left.add(u)
            for v in _undirected_neighbors(g, u):
                if v not in colour:
                    colour[v] = 1 - colour[u]
                    q.append(v)
                elif colour[v] == colour[u]:
                    raise ValueError("graph is not bipartite")
    return left


def _undirected_neighbors(g: Graph, u: int) -> Iterable[int]:
    if not g.directed:
        return g.neighbors(u)
    out = set(g.neighbors(u))
    for x in g.nodes:
        if g.has_edge(x, u):
            out.add(x)
    return out


def maximum_bipartite_matching(g: Graph, left: Optional[Iterable[int]] = None) -> Dict[int, int]:
    """Maximum-cardinality matching of a bipartite graph (Hopcroft-Karp).

    ``left`` lists the nodes of one side; when omitted the graph is
    2-coloured (``ValueError`` if it is not bipartite). Returns a symmetric
    dict ``{u: v, v: u}`` over matched pairs; ``len(result) // 2`` is the
    matching size. Edge direction and weights are ignored.
    """
    left_set = _two_colour(g) if left is None else set(left)
    unknown = left_set - set(g.nodes)
    if unknown:
        raise ValueError(f"left contains nodes not in the graph: {sorted(unknown)}")
    L = sorted(left_set)
    adj: Dict[int, List[int]] = {}
    for u in L:
        nbrs = sorted(v for v in _undirected_neighbors(g, u) if v != u)
        if any(v in left_set for v in nbrs):
            raise ValueError("left side has an internal edge; graph is not bipartite on this split")
        adj[u] = nbrs
    match_l: Dict[int, Optional[int]] = {u: None for u in L}
    match_r: Dict[int, Optional[int]] = {}
    INF = float("inf")

    def bfs() -> Tuple[bool, Dict[int, float]]:
        dist: Dict[int, float] = {}
        q: deque = deque()
        for u in L:
            if match_l[u] is None:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                w = match_r.get(v)
                if w is None:
                    found = True
                elif dist[w] == INF:
                    dist[w] = dist[u] + 1
                    q.append(w)
        return found, dist

    def dfs(root: int, dist: Dict[int, float]) -> bool:
        # iterative augmenting-path search along BFS layers
        stack = [(root, iter(adj[root]))]
        path: List[Tuple[int, int]] = []
        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                w = match_r.get(v)
                if w is None:
                    path.append((u, v))
                    for a, b in path:
                        match_l[a] = b
                        match_r[b] = a
                    return True
                if dist.get(w, INF) == dist[u] + 1:
                    path.append((u, v))
                    stack.append((w, iter(adj[w])))
                    advanced = True
                    break
            if not advanced:
                dist[u] = INF
                stack.pop()
                if path:
                    path.pop()
        return False

    while True:
        found, dist = bfs()
        if not found:
            break
        for u in L:
            if match_l[u] is None:
                dfs(u, dist)
    out: Dict[int, int] = {}
    for u, v in match_l.items():
        if v is not None:
            out[u] = v
            out[v] = u
    return out


__all__ = [
    "FlowResult",
    "edge_connectivity",
    "maximum_bipartite_matching",
    "maximum_flow",
    "minimum_cut",
]
