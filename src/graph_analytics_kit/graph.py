"""Graph data model, edge-list I/O, and built-in datasets."""
from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

WeightedEdge = Tuple[int, int, float]
Edge = Tuple[int, int]


class Graph:
    """An undirected or directed graph with integer node labels.

    The graph is stored as an adjacency map keyed by node label, so neighbor
    lookups and matrix construction are cheap. Nodes keep their original integer
    labels and insertion order; algorithms operate on labels rather than on
    positions, which keeps results interpretable when loading real datasets.
    """

    def __init__(
        self,
        edges: Optional[Iterable[Edge] | Iterable[WeightedEdge]] = None,
        *,
        directed: bool = False,
    ) -> None:
        self.directed = directed
        self._adj: Dict[int, Dict[int, float]] = defaultdict(dict)
        self._nodes: List[int] = []
        self._index: Dict[int, int] = {}
        if edges is not None:
            for edge in edges:
                u, v = int(edge[0]), int(edge[1])
                weight = float(edge[2]) if len(edge) > 2 else 1.0
                self.add_edge(u, v, weight=weight)

    def add_node(self, node: int) -> None:
        if node not in self._index:
            self._index[node] = len(self._nodes)
            self._nodes.append(node)

    def add_edge(self, u: int, v: int, *, weight: float = 1.0) -> None:
        self.add_node(u)
        self.add_node(v)
        self._adj[u][v] = weight
        if not self.directed:
            self._adj[v][u] = weight

    @classmethod
    def from_edge_list(
        cls,
        path: str | Path,
        *,
        directed: bool = False,
        delimiter: Optional[str] = None,
        weighted: bool = False,
        comments: str = "#",
    ) -> "Graph":
        """Build a :class:`Graph` from a whitespace/CSV delimited edge list file.

        Each non-comment line is ``u v`` (or ``u v w`` when ``weighted=True``).
        """
        path = Path(path)
        edges: List[WeightedEdge] = []
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            for row in reader:
                if not row or row[0].lstrip().startswith(comments):
                    continue
                if weighted:
                    edges.append((int(row[0]), int(row[1]), float(row[2])))
                else:
                    edges.append((int(row[0]), int(row[1]), 1.0))
        return cls(edges, directed=directed)

    @classmethod
    def karate_club(cls) -> "Graph":
        """Zachary's Karate Club (34 nodes, 78 undirected edges)."""
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
            (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21),
            (0, 31),
            (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21),
            (1, 30),
            (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28),
            (2, 32),
            (3, 7), (3, 12), (3, 13),
            (4, 6), (4, 10),
            (5, 6), (5, 10), (5, 16),
            (6, 16),
            (8, 30), (8, 32), (8, 33),
            (9, 33),
            (13, 33),
            (14, 32), (14, 33),
            (15, 32), (15, 33),
            (18, 32), (18, 33),
            (19, 33),
            (20, 32), (20, 33),
            (22, 32), (22, 33),
            (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
            (24, 25), (24, 27), (24, 31),
            (25, 31),
            (26, 29), (26, 33),
            (27, 33),
            (28, 31), (28, 33),
            (29, 32), (29, 33),
            (30, 32), (30, 33),
            (31, 32), (31, 33),
            (32, 33),
        ]
        return cls(edges, directed=False)

    @property
    def nodes(self) -> List[int]:
        return list(self._nodes)

    def neighbors(self, node: int) -> List[int]:
        return list(self._adj[node].keys())

    def has_edge(self, u: int, v: int) -> bool:
        return v in self._adj.get(u, {})

    def degree(self, node: int) -> int:
        if not self.directed:
            return len(self._adj[node])
        return len(self._adj[node])

    @property
    def edges(self) -> List[WeightedEdge]:
        seen = set()
        out: List[WeightedEdge] = []
        for u in self._nodes:
            for v, w in self._adj[u].items():
                if self.directed:
                    out.append((u, v, w))
                else:
                    key = (min(u, v), max(u, v))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((u, v, w))
        return out

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def number_of_edges(self) -> int:
        if self.directed:
            return sum(len(neigh) for neigh in self._adj.values())
        return sum(len(neigh) for neigh in self._adj.values()) // 2

    def adjacency_matrix(self) -> np.ndarray:
        n = self.number_of_nodes()
        matrix = np.zeros((n, n), dtype=float)
        for i, u in enumerate(self._nodes):
            for v, w in self._adj[u].items():
                if v in self._index:
                    matrix[i, self._index[v]] = w
        return matrix

    def node_labels(self) -> List[int]:
        return list(self._nodes)

    def density(self) -> float:
        n = self.number_of_nodes()
        if n < 2:
            return 0.0
        m = self.number_of_edges()
        denom = n * (n - 1) / 2 if not self.directed else n * (n - 1)
        return m / denom if denom else 0.0

    def without_edges(self, edge_set: Iterable[Edge]) -> "Graph":
        """Return a copy of this graph with the given undirected edges removed."""
        removed = {tuple(sorted((u, v))) for u, v in edge_set}
        kept = [
            (u, v, w)
            for u, v, w in self.edges
            if tuple(sorted((u, v))) not in removed
        ]
        return Graph(kept, directed=self.directed)

    def is_connected(self) -> bool:
        if self.number_of_nodes() == 0:
            return True
        start = self._nodes[0]
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in self._adj[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return len(seen) == self.number_of_nodes()

    def summary(self) -> Dict[str, object]:
        return {
            "n_nodes": self.number_of_nodes(),
            "n_edges": self.number_of_edges(),
            "directed": self.directed,
            "density": self.density(),
            "connected": self.is_connected(),
        }

    def connected_components(self) -> List[List[int]]:
        """Return all connected components as lists of node labels.

        Each component is a list of node labels. The order of components and
        nodes within a component is not guaranteed to be deterministic.
        """
        seen: set[int] = set()
        components: List[List[int]] = []
        for start in self._nodes:
            if start in seen:
                continue
            component: List[int] = []
            stack = [start]
            seen.add(start)
            while stack:
                node = stack.pop()
                component.append(node)
                for nxt in self._adj[node]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            components.append(component)
        return components

    def bfs_distances(self, source: int) -> Dict[int, int]:
        """Return shortest-path distances from *source* via BFS.

        Only reachable nodes appear in the dict; unreachable nodes are omitted.
        """
        dist: Dict[int, int] = {source: 0}
        queue: deque[int] = deque([source])
        while queue:
            v = queue.popleft()
            for w in self._adj[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
        return dist


def karate_club() -> "Graph":
    """Zachary's Karate Club (34 nodes, 78 undirected edges).

    Convenience module-level wrapper around :meth:`Graph.karate_club`.
    """
    return Graph.karate_club()


def _karate_club_edges() -> List[WeightedEdge]:
    """Return the edge list for Zachary's Karate Club graph."""
    return [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
        (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21),
        (0, 31),
        (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21),
        (1, 30),
        (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28),
        (2, 32),
        (3, 7), (3, 12), (3, 13),
        (4, 6), (4, 10),
        (5, 6), (5, 10), (5, 16),
        (6, 16),
        (8, 30), (8, 32), (8, 33),
        (9, 33),
        (13, 33),
        (14, 32), (14, 33),
        (15, 32), (15, 33),
        (18, 32), (18, 33),
        (19, 33),
        (20, 32), (20, 33),
        (22, 32), (22, 33),
        (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
        (24, 25), (24, 27), (24, 31),
        (25, 31),
        (26, 29), (26, 33),
        (27, 33),
        (28, 31), (28, 33),
        (29, 32), (29, 33),
        (30, 32), (30, 33),
        (31, 32), (31, 33),
        (32, 33),
    ]
