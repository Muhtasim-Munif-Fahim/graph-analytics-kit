# graph-analytics-kit

A small, dependency-light Python toolkit for graph analytics with
reproducible Markdown reporting. It implements centrality measures
(degree, closeness, harmonic, betweenness, PageRank, eigenvector, Katz, and HITS
hubs/authorities), k-core numbers, Dijkstra shortest paths, clustering
coefficients, community detection (Louvain and label propagation),
link-prediction scores, AUC evaluation, and a command-line entry point
that runs an end-to-end
analysis on the classic Zachary's Karate Club network.

## Install

```bash
pip install -e .
```

## Library quick start

```python
from graph_analytics_kit import (
    Graph,
    communities,
    core_number,
    degree_centrality,
    dijkstra_shortest_path,
    eigenvector_centrality,
    harmonic_centrality,
    hits,
    katz_centrality,
    label_propagation,
    local_clustering,
    pagerank,
    reconstruct_path,
)

g = Graph.karate_club()
print(degree_centrality(g)[0])
print(local_clustering(g)[0])
print(pagerank(g)[0])
print(eigenvector_centrality(g)[0])
print(harmonic_centrality(g)[0])
print(katz_centrality(g, alpha=0.1)[0])
hubs, authorities = hits(g)
print(hubs[0], authorities[0])
print(core_number(g)[0])
print(communities(g))
print(label_propagation(g))
dist, prev = dijkstra_shortest_path(g, 0)
print(dist[33], reconstruct_path(prev, 0, 33))
```

`communities(g)` runs Louvain modularity maximization and returns a
partition (lists of node labels). `modularity(g, parts)` scores that
partition. `label_propagation(g, max_iter=100, seed=None)` runs
asynchronous label propagation and returns communities in that same
format: members sorted, communities ordered by their smallest node
label. Pass `return_labels=True` to also receive the node-to-label map
(surviving propagated ids, not renumbered community indexes). With
`seed` unset, each sweep visits nodes in increasing neighbor count,
then node label, and a tie keeps the current label or else takes the
smallest label, so the result is deterministic. A seed shuffles each
sweep and breaks remaining ties at random, reproducibly. Edge weights
vote for the neighbor's label; self-loops are ignored. Isolated nodes
stay singletons. `dijkstra_shortest_path(g, source)` returns distances and
predecessors; `reconstruct_path` rebuilds a concrete path.
`eigenvector_centrality(g)` returns the L2-normalized principal
eigenvector of the adjacency matrix (power iteration).
`katz_centrality(g, alpha=0.1)` solves `x = alpha * A^T x + beta`
and returns one score per node. The attenuation factor `alpha`
(default `0.1`) must be positive and strictly less than `1 / lambda_max`
of the adjacency matrix, so longer walks count for less. Scores are
L2-normalized by default; pass `normalized=False` for the raw solution.
`beta` (default `1.0`) is the constant term.
`hits(g)` returns Kleinberg hub and authority scores (L2-normalized);
betweenness already ships via Brandes, so HITS is the additional
link-analysis measure.

`harmonic_centrality(g)` returns the sum of reciprocal shortest-path
distances `sum 1/d(u,v)` over reachable nodes `v ≠ u`. Isolated nodes
score `0.0`. Unlike closeness, harmonic centrality is well-defined on
disconnected graphs because unreachable pairs simply contribute nothing.

`core_number(g)` returns the k-core number of each node (the largest `k`
such that the node sits in a subgraph of minimum degree `k`).

## k-core decomposition

`core_number` uses the Batagelj–Zaversnik bin-sort. Edge weights and
self-loops are ignored. On a directed graph the core numbers are those of
the underlying simple undirected graph. Harmonic centrality is provided
separately as `harmonic_centrality` (sum of inverse distances).

## Console script

```bash
graph-analytics demo -o report.md
graph-analytics centrality --measure harmonic
graph-analytics centrality --measure eigenvector
graph-analytics centrality --measure katz --alpha 0.1
graph-analytics centrality --measure hubs
graph-analytics centrality --measure authorities
graph-analytics centrality --measure core
graph-analytics communities
graph-analytics communities --method label-propagation --seed 0
graph-analytics shortest-path --source 0 --target 33
```

See `examples/run_demo.py` for a complete end-to-end demo that writes a
Markdown report to `examples/output/demo_report.md`, and `tests/` for the
unit-test contract.
