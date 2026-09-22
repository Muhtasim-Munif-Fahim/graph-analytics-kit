# graph-analytics-kit

A small, dependency-light Python toolkit for graph analytics with
reproducible Markdown reporting. It implements centrality measures
(degree, closeness, betweenness, PageRank, eigenvector, and HITS
hubs/authorities), k-core numbers, Dijkstra shortest paths, clustering
coefficients, community detection (Louvain), link-prediction scores, AUC
evaluation, and a command-line entry point that runs an end-to-end
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
    hits,
    local_clustering,
    pagerank,
    reconstruct_path,
)

g = Graph.karate_club()
print(degree_centrality(g)[0])
print(local_clustering(g)[0])
print(pagerank(g)[0])
print(eigenvector_centrality(g)[0])
hubs, authorities = hits(g)
print(hubs[0], authorities[0])
print(core_number(g)[0])
print(communities(g))
dist, prev = dijkstra_shortest_path(g, 0)
print(dist[33], reconstruct_path(prev, 0, 33))
```

`communities(g)` runs Louvain modularity maximization and returns a
partition (lists of node labels). `modularity(g, parts)` scores that
partition. `dijkstra_shortest_path(g, source)` returns distances and
predecessors; `reconstruct_path` rebuilds a concrete path.
`eigenvector_centrality(g)` returns the L2-normalized principal
eigenvector of the adjacency matrix (power iteration).
`hits(g)` returns Kleinberg hub and authority scores (L2-normalized);
betweenness already ships via Brandes, so HITS is the additional
link-analysis measure.
`core_number(g)` returns the k-core number of each node (the largest `k`
such that the node sits in a subgraph of minimum degree `k`).

## Pivot: k-core instead of another closeness

Closeness centrality is already implemented as `closeness_centrality`:
Wasserman–Faust closeness, `(n - 1) / sum of shortest-path distances`,
with isolated nodes (and nodes that reach nobody) scoring `0.0`. Brandes
betweenness is also already implemented and was not rewritten. This change
adds k-core decomposition instead of a second closeness (or harmonic
closeness) implementation. `core_number` uses the Batagelj–Zaversnik
bin-sort. Edge weights and self-loops are ignored. On a directed graph the
core numbers are those of the underlying simple undirected graph.

## Console script

```bash
graph-analytics demo -o report.md
graph-analytics centrality --measure eigenvector
graph-analytics centrality --measure hubs
graph-analytics centrality --measure authorities
graph-analytics centrality --measure core
graph-analytics communities
graph-analytics shortest-path --source 0 --target 33
```

See `examples/run_demo.py` for a complete end-to-end demo that writes a
Markdown report to `examples/output/demo_report.md`, and `tests/` for the
unit-test contract.
