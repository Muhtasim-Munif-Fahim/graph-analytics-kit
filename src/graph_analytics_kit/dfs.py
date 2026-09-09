"""Depth-first traversal for graphs."""
from __future__ import annotations

from typing import Dict, List, Optional


def dfs_traversal(
    g,
    start: int,
    *,
    target: Optional[int] = None,
    max_depth: Optional[int] = None,
) -> List[int]:
    """Return nodes visited in depth-first order starting from *start*.

    The traversal follows the adjacency order of the underlying graph.
    When *target* is provided, the search terminates as soon as *target*
    is popped from the stack and the visited prefix up to and including
    *target* is returned. When *max_depth* is provided, nodes deeper than
    *max_depth* edges from *start* are not enqueued.
    """
    visited: List[int] = []
    seen: set[int] = set()
    stack: List[tuple[int, int]] = [(start, 0)]

    while stack:
        node, depth = stack.pop()
        if node in seen:
            continue
        if max_depth is not None and depth > max_depth:
            continue
        seen.add(node)
        visited.append(node)
        if node == target:
            break
        neighbors = g.neighbors(node)
        stack.extend((neighbor, depth + 1) for neighbor in reversed(neighbors) if neighbor not in seen)

    return visited


def dfs_path(
    g,
    start: int,
    target: int,
    *,
    max_depth: Optional[int] = None,
) -> Optional[List[int]]:
    """Return a DFS path from *start* to *target*, or ``None`` if unreachable."""
    if start == target:
        return [start]
    visited: Dict[int, int] = {}
    stack: List[int] = [start]
    visited[start] = -1

    while stack:
        node = stack.pop()
        for neighbor in g.neighbors(node):
            if neighbor in visited:
                continue
            if max_depth is not None and visited[node] + 1 > max_depth:
                continue
            visited[neighbor] = node
            if neighbor == target:
                path: List[int] = [target]
                cur: int = target
                while cur != start:
                    cur = visited[cur]
                    path.append(cur)
                path.reverse()
                return path
            stack.append(neighbor)
    return None


__all__ = ["dfs_traversal", "dfs_path"]
