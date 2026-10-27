"""Weisfeiler-Lehman colour refinement, WL hashing/kernels and exact isomorphism.

**1-WL colour refinement** gives every node a label (by default its degree;
for directed graphs the pair ``(out-degree, in-degree)``). It then
repeatedly replaces each label with a hash of ``(label, sorted multiset of
neighbour labels)``. Directed graphs keep out- and in-neighbours separate.
Refinement stops once the partition into colour classes stops splitting,
which takes at most ``n`` rounds. The labels are content hashes and do not
depend on node order or labels, so they can be compared *across* graphs.

* :func:`wl_colors` returns the per-round labels. :func:`color_refinement`
  returns the stable (coarsest equitable) partition.
* :func:`wl_graph_hash` hashes the histogram of labels over all rounds.
  Isomorphic graphs always get the same hash. The converse fails: 1-WL
  cannot tell apart e.g. ``C6`` and two disjoint triangles, or any two
  regular graphs of the same degree and order.
* :func:`wl_subtree_kernel` gives the Weisfeiler-Lehman subtree kernel
  (Shervashidze et al., JMLR 2011). It is the dot product of label-count
  vectors over ``h`` rounds and can be cosine-normalised.

**Exact isomorphism.** :func:`iter_isomorphisms` runs a backtracking search
that matches nodes only within equal stable WL colours. Nodes are visited in
connectivity order, so each new node has an already-mapped neighbour when
possible, and every partial map is checked against the adjacency structure
(including direction and self-loops). :func:`find_isomorphism`,
:func:`is_isomorphic`, :func:`automorphism_count` and
:func:`automorphism_orbits` are built on it. The search is exponential in the
worst case but fast on typical sparse graphs. Edge weights are ignored.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Dict, Hashable, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from .graph import Graph

__all__ = [
    "wl_colors",
    "color_refinement",
    "wl_graph_hash",
    "wl_subtree_kernel",
    "could_be_isomorphic",
    "iter_isomorphisms",
    "find_isomorphism",
    "is_isomorphic",
    "automorphism_count",
    "automorphism_orbits",
]


def _h(text: str, digest_size: int = 16) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=digest_size).hexdigest()


def _in_neighbors(g: Graph) -> Dict[int, List[int]]:
    inn: Dict[int, List[int]] = {u: [] for u in g.nodes}
    for u in g.nodes:
        for v in g.neighbors(u):
            inn[v].append(u)
    return inn


def _initial(g: Graph, node_labels, inn) -> Dict[int, str]:
    out = {}
    for u in g.nodes:
        base = f"d{g.degree(u)}"
        if g.directed:
            base += f"/i{len(inn[u])}"
        if g.has_edge(u, u):
            base += "/loop"
        if node_labels is not None:
            if u not in node_labels:
                raise ValueError(f"node_labels is missing node {u}")
            base += f"|{node_labels[u]!r}"
        out[u] = _h(base)
    return out


def _n_classes(labels: Dict[int, str]) -> int:
    return len(set(labels.values()))


def wl_colors(
    g: Graph,
    iterations: Optional[int] = None,
    node_labels: Optional[Mapping[int, Hashable]] = None,
) -> List[Dict[int, str]]:
    """Per-round WL labels, starting with round 0 (the initial labels).

    With ``iterations=None``, refinement runs until the partition is stable
    and the last element is the stable colouring. Otherwise exactly
    ``iterations`` rounds are run.
    """
    if iterations is not None and (
        isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 0
    ):
        raise ValueError("iterations must be a non-negative integer or None")
    inn = _in_neighbors(g) if g.directed else None
    labels = _initial(g, node_labels, inn)
    rounds = [labels]
    limit = len(g.nodes) if iterations is None else int(iterations)
    for _ in range(limit):
        new = {}
        for u in g.nodes:
            sig = labels[u] + "|o:" + ",".join(sorted(labels[v] for v in g.neighbors(u)))
            if inn is not None:
                sig += "|i:" + ",".join(sorted(labels[v] for v in inn[u]))
            new[u] = _h(sig)
        if iterations is None and _n_classes(new) == _n_classes(labels):
            break
        rounds.append(new)
        labels = new
    return rounds


def color_refinement(g: Graph, node_labels: Optional[Mapping[int, Hashable]] = None) -> List[List[int]]:
    """Stable 1-WL partition: colour classes sorted by size, then smallest node."""
    final = wl_colors(g, None, node_labels)[-1]
    groups: Dict[str, List[int]] = {}
    for u in g.nodes:
        groups.setdefault(final[u], []).append(u)
    return sorted((sorted(c) for c in groups.values()), key=lambda c: (-len(c), c[0]))


def wl_graph_hash(
    g: Graph,
    iterations: int = 3,
    node_labels: Optional[Mapping[int, Hashable]] = None,
    digest_size: int = 16,
) -> str:
    """Isomorphism-invariant hex hash of the WL label histograms over ``iterations`` rounds."""
    rounds = wl_colors(g, iterations, node_labels)
    parts = [f"n{len(g.nodes)}m{g.number_of_edges()}d{int(g.directed)}"]
    for r, labels in enumerate(rounds):
        hist = sorted(Counter(labels.values()).items())
        parts.append(f"{r}:" + ";".join(f"{k}x{c}" for k, c in hist))
    return _h("#".join(parts), digest_size)


def wl_subtree_kernel(
    graphs: Sequence[Graph],
    iterations: int = 3,
    normalize: bool = False,
    node_labels: Optional[Sequence[Optional[Mapping[int, Hashable]]]] = None,
) -> np.ndarray:
    """Gram matrix ``K[i, j] = <phi(G_i), phi(G_j)>`` of WL label counts over rounds ``0..h``."""
    if len(graphs) == 0:
        raise ValueError("graphs must be non-empty")
    if node_labels is not None and len(node_labels) != len(graphs):
        raise ValueError("node_labels must have one entry per graph")
    feats: List[Counter] = []
    for i, g in enumerate(graphs):
        labs = None if node_labels is None else node_labels[i]
        c: Counter = Counter()
        for r, labels in enumerate(wl_colors(g, iterations, labs)):
            c.update(f"{r}:{v}" for v in labels.values())
        feats.append(c)
    vocab = {k: j for j, k in enumerate(sorted(set().union(*feats)))}
    Phi = np.zeros((len(graphs), len(vocab)))
    for i, c in enumerate(feats):
        for k, v in c.items():
            Phi[i, vocab[k]] = v
    K = Phi @ Phi.T
    if normalize:
        d = np.sqrt(np.diag(K))
        d[d == 0] = 1.0
        K = K / np.outer(d, d)
    return K


def _rounds(g1: Graph, g2: Graph):
    # Same number of rounds for both graphs; labels are content hashes, so
    # round-r labels are directly comparable across graphs.
    a, b = wl_colors(g1), wl_colors(g2)
    k = max(len(a), len(b)) - 1
    if len(a) - 1 < k:
        a = wl_colors(g1, k)
    if len(b) - 1 < k:
        b = wl_colors(g2, k)
    return a, b


def could_be_isomorphic(g1: Graph, g2: Graph) -> bool:
    """Necessary conditions: same size, directedness and 1-WL label histograms."""
    if g1.directed != g2.directed:
        return False
    if len(g1.nodes) != len(g2.nodes) or g1.number_of_edges() != g2.number_of_edges():
        return False
    r1, r2 = _rounds(g1, g2)
    return Counter(r1[-1].values()) == Counter(r2[-1].values())


def _search_order(g: Graph, colors: Dict[int, str]) -> List[int]:
    size = Counter(colors.values())
    pos = {u: i for i, u in enumerate(g.nodes)}
    und = {u: set(g.neighbors(u)) for u in g.nodes}
    if g.directed:
        for u in g.nodes:
            for v in g.neighbors(u):
                und[v].add(u)
    order: List[int] = []
    placed: set = set()
    remaining = set(g.nodes)
    while remaining:
        start = min(remaining, key=lambda u: (size[colors[u]], -len(und[u]), pos[u]))
        frontier = {start}
        while frontier:
            u = min(frontier, key=lambda x: (-len(und[x] & placed), size[colors[x]], pos[x]))
            frontier.discard(u)
            order.append(u)
            placed.add(u)
            remaining.discard(u)
            frontier |= und[u] & remaining
    return order


def iter_isomorphisms(g1: Graph, g2: Graph) -> Iterator[Dict[int, int]]:
    """Yield every isomorphism ``g1 -> g2`` as a node mapping dict."""
    if not could_be_isomorphic(g1, g2):
        return
    r1, r2 = _rounds(g1, g2)
    c1, c2 = r1[-1], r2[-1]
    by_color: Dict[str, List[int]] = {}
    for v in g2.nodes:
        by_color.setdefault(c2[v], []).append(v)
    order = _search_order(g1, c1)
    adj1 = {u: set(g1.neighbors(u)) for u in g1.nodes}
    adj2 = {u: set(g2.neighbors(u)) for u in g2.nodes}
    mapping: Dict[int, int] = {}
    used: set = set()

    def consistent(u: int, v: int) -> bool:
        if (u in adj1[u]) != (v in adj2[v]):
            return False
        for x, y in mapping.items():
            if (x in adj1[u]) != (y in adj2[v]):
                return False
            if g1.directed and (u in adj1[x]) != (v in adj2[y]):
                return False
        return True

    def backtrack(i: int):
        if i == len(order):
            yield dict(mapping)
            return
        u = order[i]
        for v in by_color.get(c1[u], ()):
            if v in used or not consistent(u, v):
                continue
            mapping[u] = v
            used.add(v)
            yield from backtrack(i + 1)
            del mapping[u]
            used.discard(v)

    yield from backtrack(0)


def find_isomorphism(g1: Graph, g2: Graph) -> Optional[Dict[int, int]]:
    """One isomorphism ``g1 -> g2``, or ``None`` if the graphs are not isomorphic."""
    return next(iter_isomorphisms(g1, g2), None)


def is_isomorphic(g1: Graph, g2: Graph) -> bool:
    """Exact isomorphism test (edge weights ignored)."""
    return find_isomorphism(g1, g2) is not None


def automorphism_count(g: Graph, limit: Optional[int] = None) -> int:
    """Size of the automorphism group (stops counting at ``limit`` if given)."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    count = 0
    for _ in iter_isomorphisms(g, g):
        count += 1
        if limit is not None and count >= limit:
            break
    return count


def automorphism_orbits(g: Graph) -> List[List[int]]:
    """Orbits of the automorphism group, sorted by size then smallest node."""
    parent = {u: u for u in g.nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for phi in iter_isomorphisms(g, g):
        for u, v in phi.items():
            a, b = find(u), find(v)
            if a != b:
                parent[b] = a
    groups: Dict[int, List[int]] = {}
    for u in g.nodes:
        groups.setdefault(find(u), []).append(u)
    return sorted((sorted(c) for c in groups.values()), key=lambda c: (-len(c), c[0]))
