"""Dominator analysis for directed flow graphs.

In a directed graph with a distinguished entry node ``root``, node ``d``
*dominates* node ``v`` when every path from ``root`` to ``v`` passes through
``d``. Dominators are the backbone of control-flow analysis (SSA
construction, loop detection), but they are just as useful for dependency
graphs, supply chains and attack graphs: the immediate dominator of a node is
its single point of failure closest to it.

This module provides:

* :func:`immediate_dominators` – the iterative algorithm of Cooper, Harvey &
  Kennedy (2001), "A Simple, Fast Dominance Algorithm", over a reverse
  post-order computed with an iterative DFS (no recursion limits);
* :func:`dominator_tree` – children lists of the dominator tree;
* :func:`dominators` – the full dominator set of every reachable node;
* :func:`dominates` – a constant-time-per-query test via tree DFS intervals;
* :func:`dominance_frontiers` – Cytron et al. (1991) dominance frontiers,
  computed with the Cooper–Harvey–Kennedy runner walk;
* :func:`post_dominators` – immediate post-dominators (dominators of the
  reversed graph with respect to an exit node).

Only nodes reachable from ``root`` are reported; unreachable nodes have no
dominators. Edge weights are ignored.
"""

from __future__ import annotations

from typing import Dict, List, Set

from .graph import Graph

__all__ = [
    "immediate_dominators",
    "dominator_tree",
    "dominators",
    "dominates",
    "dominance_frontiers",
    "post_dominators",
]


def _require(g: Graph, root: int, name: str) -> None:
    if not g.directed:
        raise ValueError(f"{name} requires a directed graph")
    if root not in g._index:
        raise ValueError(f"root {root!r} is not a node of the graph")


def _reverse_postorder(g: Graph, root: int) -> List[int]:
    """Reverse post-order of the nodes reachable from ``root`` (iterative DFS)."""
    seen = {root}
    order: List[int] = []
    stack = [(root, iter(g.neighbors(root)))]
    while stack:
        node, it = stack[-1]
        advanced = False
        for nxt in it:
            if nxt not in seen:
                seen.add(nxt)
                stack.append((nxt, iter(g.neighbors(nxt))))
                advanced = True
                break
        if not advanced:
            order.append(node)
            stack.pop()
    order.reverse()
    return order


def _predecessors(g: Graph, reachable: Set[int]) -> Dict[int, List[int]]:
    preds: Dict[int, List[int]] = {v: [] for v in reachable}
    for u in reachable:
        for v in g.neighbors(u):
            if v in reachable:
                preds[v].append(u)
    return preds


def _idom_core(g: Graph, root: int) -> Dict[int, int]:
    rpo = _reverse_postorder(g, root)
    pos = {v: i for i, v in enumerate(rpo)}
    preds = _predecessors(g, set(rpo))
    idom: Dict[int, int] = {root: root}

    def intersect(a: int, b: int) -> int:
        while a != b:
            while pos[a] > pos[b]:
                a = idom[a]
            while pos[b] > pos[a]:
                b = idom[b]
        return a

    changed = True
    while changed:
        changed = False
        for v in rpo[1:]:
            new = None
            for p in preds[v]:
                if p in idom:
                    new = p if new is None else intersect(p, new)
            if new is not None and idom.get(v) != new:
                idom[v] = new
                changed = True
    return idom


def immediate_dominators(g: Graph, root: int) -> Dict[int, int]:
    """Map every node reachable from ``root`` to its immediate dominator.

    ``root`` maps to itself. Runs the Cooper–Harvey–Kennedy fixed-point
    iteration, which converges in a couple of passes on typical graphs and is
    ``O(n * m)`` in the worst case.

    Raises ``ValueError`` for undirected graphs or an unknown ``root``.
    """
    _require(g, root, "immediate_dominators")
    return _idom_core(g, root)


def dominator_tree(g: Graph, root: int) -> Dict[int, List[int]]:
    """Children lists of the dominator tree rooted at ``root``.

    Every reachable node appears as a key; children are listed in graph
    insertion order so the output is deterministic.
    """
    idom = immediate_dominators(g, root)
    tree: Dict[int, List[int]] = {v: [] for v in idom}
    for v in g.nodes:
        if v in idom and v != root:
            tree[idom[v]].append(v)
    return tree


def dominators(g: Graph, root: int) -> Dict[int, Set[int]]:
    """Full dominator set of every reachable node (each node dominates itself)."""
    idom = immediate_dominators(g, root)
    out: Dict[int, Set[int]] = {}
    for v in idom:
        doms = {v}
        u = v
        while u != root:
            u = idom[u]
            doms.add(u)
        out[v] = doms
    return out


def dominates(g: Graph, root: int, d: int, v: int) -> bool:
    """Return ``True`` when ``d`` dominates ``v`` (reflexive: ``d`` dominates itself).

    Unreachable nodes are dominated by nothing and dominate nothing.
    """
    idom = immediate_dominators(g, root)
    if d not in idom or v not in idom:
        return False
    tree = {u: [] for u in idom}
    for u in idom:
        if u != root:
            tree[idom[u]].append(u)
    # Pre/post intervals: d dominates v iff pre[d] <= pre[v] and post[v] <= post[d].
    pre: Dict[int, int] = {}
    post: Dict[int, int] = {}
    clock = 0
    stack = [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            post[node] = clock
            clock += 1
            continue
        pre[node] = clock
        clock += 1
        stack.append((node, True))
        for child in reversed(tree[node]):
            stack.append((child, False))
    return pre[d] <= pre[v] and post[v] <= post[d]


def dominance_frontiers(g: Graph, root: int) -> Dict[int, Set[int]]:
    """Dominance frontier of every reachable node.

    ``DF(d)`` is the set of nodes ``w`` such that ``d`` dominates a predecessor
    of ``w`` but does not strictly dominate ``w`` – exactly where SSA phi
    functions are placed.
    """
    idom = immediate_dominators(g, root)
    preds = _predecessors(g, set(idom))
    df: Dict[int, Set[int]] = {v: set() for v in idom}
    for w in idom:
        ps = preds[w]
        # The root has an implicit entry edge, so any predecessor makes it a join.
        if len(ps) < 2 and not (w == root and ps):
            continue
        for p in ps:
            runner = p
            # For w == root nothing strictly dominates w, so walk up to the root inclusive.
            while w == root or runner != idom[w]:
                df[runner].add(w)
                if runner == root:
                    break
                runner = idom[runner]
    return df


def post_dominators(g: Graph, exit_node: int) -> Dict[int, int]:
    """Immediate post-dominators with respect to ``exit_node``.

    Node ``p`` post-dominates ``v`` when every path from ``v`` to
    ``exit_node`` passes through ``p``. Computed as the immediate dominators
    of the reversed graph; only nodes that can reach ``exit_node`` appear.
    """
    _require(g, exit_node, "post_dominators")
    rev = Graph(directed=True)
    for v in g.nodes:
        rev.add_node(v)
    for u, v, _w in g.edges:
        rev.add_edge(v, u)
    return _idom_core(rev, exit_node)
