"""Burt's structural-hole measures: constraint, effective size, efficiency, hierarchy.

A node spans a *structural hole* when its contacts are not connected to
each other, so it brokers between otherwise separate groups. With the
normalised mutual tie strength

    p_ij = (w_ij + w_ji) / sum_k (w_ik + w_ki)

(summing over all in- and out-neighbours ``k`` of ``i``, self-loops ignored)
Burt (1992) defines

* **local constraint** ``c_ij = (p_ij + sum_{q != i, j} p_iq p_qj)^2``,
* **constraint** ``C_i = sum_j c_ij`` -- high when ``i``'s contacts are
  themselves tightly connected (few holes),
* **effective size** ``E_i = sum_j (1 - sum_{q != i, j} p_iq m_jq)`` with
  ``m_jq = (w_jq + w_qj) / max_k (w_jk + w_kj)`` -- the number of
  non-redundant contacts (``n - 2t/n`` for an unweighted undirected graph
  with ``t`` ties among the ``n`` neighbours),
* **efficiency** ``E_i / n_i``,
* **hierarchy** ``H_i = sum_j r_j ln r_j / (n ln n)`` with
  ``r_j = c_ij / (C_i / n)`` -- how concentrated the constraint is on a
  single contact (0 when evenly spread).

Isolated nodes get ``nan`` for every measure (``hierarchy`` is ``0`` for a
single contact). Definitions follow Burt (1992) and match NetworkX.

Reference: R. S. Burt, *Structural Holes: The Social Structure of
Competition*, Harvard University Press (1992); Borgatti (1997) for the
unweighted effective-size identity.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, Optional

from .graph import Graph

__all__ = [
    "constraint",
    "local_constraint",
    "effective_size",
    "burt_efficiency",
    "burt_hierarchy",
    "structural_holes",
]


def _mutual_weights(g: Graph) -> Dict[int, Dict[int, float]]:
    mw: Dict[int, Dict[int, float]] = {u: defaultdict(float) for u in g.nodes}
    for u in g.nodes:
        for v in g.neighbors(u):
            if v == u:
                continue
            w = float(g.get_weight(u, v))
            if w < 0:
                raise ValueError("structural-hole measures need non-negative edge weights")
            mw[u][v] += w
            mw[v][u] += w
    return mw


class _Ctx:
    def __init__(self, g: Graph) -> None:
        self.mw = _mutual_weights(g)
        self.p: Dict[int, Dict[int, float]] = {}
        self.m: Dict[int, Dict[int, float]] = {}
        for u, nbrs in self.mw.items():
            total = sum(nbrs.values())
            biggest = max(nbrs.values()) if nbrs else 0.0
            self.p[u] = {v: (w / total if total > 0 else 0.0) for v, w in nbrs.items()}
            self.m[u] = {v: (w / biggest if biggest > 0 else 0.0) for v, w in nbrs.items()}

    def local(self, i: int, j: int) -> float:
        pi = self.p[i]
        direct = pi.get(j, 0.0)
        indirect = sum(pi[q] * self.p[q].get(j, 0.0) for q in pi if q != j)
        return (direct + indirect) ** 2


def _nodes(g: Graph, nodes: Optional[Iterable[int]]):
    if nodes is None:
        return g.nodes
    out = list(nodes)
    known = set(g.nodes)
    for n in out:
        if n not in known:
            raise KeyError(f"node {n} is not in the graph")
    return out


def local_constraint(g: Graph, i: int, j: int) -> float:
    """Burt's local constraint ``c_ij`` of ``i`` with respect to ``j``."""
    ctx = _Ctx(g)
    if i not in ctx.p or j not in ctx.p:
        raise KeyError("both nodes must be in the graph")
    return ctx.local(i, j)


def _constraint(ctx: _Ctx, i: int) -> float:
    nbrs = ctx.p[i]
    if not nbrs:
        return math.nan
    return sum(ctx.local(i, j) for j in nbrs)


def constraint(g: Graph, nodes: Optional[Iterable[int]] = None) -> Dict[int, float]:
    """Burt's constraint ``C_i`` for every node (or the given subset)."""
    ctx = _Ctx(g)
    return {i: _constraint(ctx, i) for i in _nodes(g, nodes)}


def _effective_size(ctx: _Ctx, i: int) -> float:
    nbrs = ctx.p[i]
    if not nbrs:
        return math.nan
    total = 0.0
    for j in nbrs:
        mj = ctx.m[j]
        redundancy = sum(nbrs[q] * mj.get(q, 0.0) for q in nbrs if q != j)
        total += 1.0 - redundancy
    return total


def effective_size(g: Graph, nodes: Optional[Iterable[int]] = None) -> Dict[int, float]:
    """Burt's effective size (non-redundant contacts) of every node."""
    ctx = _Ctx(g)
    return {i: _effective_size(ctx, i) for i in _nodes(g, nodes)}


def burt_efficiency(g: Graph, nodes: Optional[Iterable[int]] = None) -> Dict[int, float]:
    """Effective size divided by the number of distinct contacts."""
    ctx = _Ctx(g)
    out = {}
    for i in _nodes(g, nodes):
        n = len(ctx.p[i])
        out[i] = _effective_size(ctx, i) / n if n else math.nan
    return out


def _hierarchy(ctx: _Ctx, i: int) -> float:
    nbrs = ctx.p[i]
    n = len(nbrs)
    if n == 0:
        return math.nan
    if n == 1:
        return 0.0
    locals_ = [ctx.local(i, j) for j in nbrs]
    c = sum(locals_)
    if c <= 0:
        return 0.0
    mean = c / n
    acc = 0.0
    for v in locals_:
        r = v / mean
        if r > 0:
            acc += r * math.log(r)
    return acc / (n * math.log(n))


def burt_hierarchy(g: Graph, nodes: Optional[Iterable[int]] = None) -> Dict[int, float]:
    """Burt's hierarchy: concentration of a node's constraint on few contacts."""
    ctx = _Ctx(g)
    return {i: _hierarchy(ctx, i) for i in _nodes(g, nodes)}


def structural_holes(g: Graph, nodes: Optional[Iterable[int]] = None) -> Dict[int, Dict[str, float]]:
    """All four measures per node: ``constraint``, ``effective_size``, ``efficiency``, ``hierarchy``."""
    ctx = _Ctx(g)
    out = {}
    for i in _nodes(g, nodes):
        n = len(ctx.p[i])
        es = _effective_size(ctx, i)
        out[i] = {
            "constraint": _constraint(ctx, i),
            "effective_size": es,
            "efficiency": es / n if n else math.nan,
            "hierarchy": _hierarchy(ctx, i),
        }
    return out
