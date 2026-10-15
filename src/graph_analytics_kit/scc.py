"""Strongly connected components (Tarjan) and condensation DAG.

Tarjan's algorithm computes SCCs of a directed graph in a single DFS
pass using discovery times and a low-link stack. The condensation of the
SCC DAG is always a directed acyclic graph: each component becomes a
supernode, and an edge ``A -> B`` exists when some original edge goes
from a node in ``A`` to a node in ``B``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from .graph import Graph

__all__ = [
    "strongly_connected_components",
    "condensation_dag",
    "is_strongly_connected",
]


def strongly_connected_components(g: Graph) -> List[List[int]]:
    """Return the SCCs of a directed graph (Tarjan, 1972).

    Each component is a list of node labels. Components are ordered by
    finishing time (later components appear first in a reverse-topological
    sense of the condensation). Isolated nodes form singleton components.
    Undirected graphs are rejected — convert or use connected components.
    """
    if not g.directed:
        raise ValueError("strongly_connected_components requires a directed graph")
    nodes = g.nodes
    index_of: Dict[int, int] = {}
    lowlink: Dict[int, int] = {}
    on_stack: Set[int] = set()
    stack: List[int] = []
    components: List[List[int]] = []
    index = 0

    def strongconnect(v: int) -> None:
        nonlocal index
        index_of[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in g.neighbors(v):
            if w not in index_of:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index_of[w])
        if lowlink[v] == index_of[v]:
            comp: List[int] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            components.append(comp)

    for v in nodes:
        if v not in index_of:
            strongconnect(v)
    return components


def is_strongly_connected(g: Graph) -> bool:
    """Return True iff ``g`` has a single strongly connected component."""
    comps = strongly_connected_components(g)
    return len(comps) == 1 and len(comps[0]) == g.number_of_nodes()


def condensation_dag(g: Graph) -> Tuple[Graph, List[List[int]], Dict[int, int]]:
    """Build the condensation DAG of ``g``.

    Returns ``(dag, components, node_to_component)`` where ``dag`` is a
    directed acyclic :class:`Graph` whose nodes are component indices
    ``0 .. C-1``, ``components[i]`` lists the original nodes in component
    ``i``, and ``node_to_component`` maps each original node to its
    component index.
    """
    comps = strongly_connected_components(g)
    # Re-index so component 0..C-1 matches comps order
    node_to_comp: Dict[int, int] = {}
    for i, comp in enumerate(comps):
        for v in comp:
            node_to_comp[v] = i
    dag = Graph(directed=True)
    for i in range(len(comps)):
        dag.add_node(i)
    seen_edges: Set[Tuple[int, int]] = set()
    for u, v, _w in g.edges:
        cu, cv = node_to_comp[u], node_to_comp[v]
        if cu != cv and (cu, cv) not in seen_edges:
            dag.add_edge(cu, cv)
            seen_edges.add((cu, cv))
    return dag, comps, node_to_comp
