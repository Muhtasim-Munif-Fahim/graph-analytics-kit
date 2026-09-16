# graph-analytics-kit

A small, dependency-light Python toolkit for graph analytics with
reproducible Markdown reporting. It implements centrality measures
(degree, closeness, betweenness, and PageRank), clustering coefficients,
community detection (Louvain), link-prediction scores, AUC
evaluation, and a command-line entry point that runs an end-to-end
analysis on the classic Zachary's Karate Club network.

## Install

```bash
pip install -e .
```

## Library quick start

```python
from graph_analytics_kit import Graph, communities, degree_centrality, local_clustering, pagerank

g = Graph.karate_club()
print(degree_centrality(g)[0])
print(local_clustering(g)[0])
print(pagerank(g)[0])
print(communities(g))
```

`communities(g)` runs Louvain modularity maximization and returns a
partition (lists of node labels). `modularity(g, parts)` scores that
partition.

## Console script

```bash
graph-analytics demo -o report.md
graph-analytics communities
```

See `examples/run_demo.py` for a complete end-to-end demo that writes a
Markdown report to `examples/output/demo_report.md`, and `tests/` for the
unit-test contract.
