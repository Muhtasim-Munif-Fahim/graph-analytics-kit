"""Command-line interface for graph-analytics-kit."""
from __future__ import annotations

import argparse
from pathlib import Path

from .centrality import (
    betweenness_centrality,
    closeness_centrality,
    degree_centrality,
    eigenvector_centrality,
    pagerank,
)
from .clustering import average_clustering, local_clustering
from .community import communities, modularity
from .graph import Graph, karate_club
from .shortest_path import dijkstra_shortest_path, reconstruct_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graph-analytics")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run an end-to-end analysis on Karate Club")
    demo.add_argument(
        "--output", "-o", default="examples/output/demo_report.md",
        help="Output path for the Markdown report",
    )

    stats = sub.add_parser("stats", help="Print summary statistics for a graph")
    _add_graph_args(stats)

    cent = sub.add_parser("centrality", help="Print centrality rankings")
    _add_graph_args(cent)
    cent.add_argument(
        "--measure",
        choices=["degree", "closeness", "betweenness", "pagerank", "eigenvector"],
        default="degree",
    )
    cent.add_argument("--top", type=int, default=5, help="Show top N nodes")

    clust = sub.add_parser("clustering", help="Print clustering coefficients")
    _add_graph_args(clust)

    comm = sub.add_parser("communities", help="Print detected communities")
    _add_graph_args(comm)

    sp = sub.add_parser("shortest-path", help="Print Dijkstra shortest paths")
    _add_graph_args(sp)
    sp.add_argument("--source", type=int, default=0, help="Source node label")
    sp.add_argument("--target", type=int, default=None, help="Optional target node")
    sp.add_argument("--top", type=int, default=5, help="Show closest N nodes")

    return parser


def _add_graph_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--graph", choices=["karate"], default="karate",
        help="Built-in dataset to use (default: karate)",
    )


def _load_graph(args: argparse.Namespace) -> Graph:
    graph_name = getattr(args, "graph", "karate")
    if graph_name == "karate":
        return karate_club()
    raise ValueError(f"Unknown dataset: {graph_name}")


def cmd_demo(args: argparse.Namespace) -> int:
    g = _load_graph(args)
    report = _compose_report(g)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    print(f"Wrote {target}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    g = _load_graph(args)
    s = g.summary()
    for key, value in s.items():
        print(f"{key}: {value}")
    return 0


def cmd_centrality(args: argparse.Namespace) -> int:
    g = _load_graph(args)
    measures = {
        "degree": degree_centrality,
        "closeness": closeness_centrality,
        "betweenness": betweenness_centrality,
        "pagerank": pagerank,
        "eigenvector": eigenvector_centrality,
    }
    scores = measures[args.measure](g)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for node, score in ranked[:args.top]:
        print(f"{node}: {score:.6f}")
    return 0


def cmd_clustering(args: argparse.Namespace) -> int:
    g = _load_graph(args)
    avg = average_clustering(g)
    print(f"average_clustering: {avg:.6f}")
    lc = local_clustering(g)
    ranked = sorted(lc.items(), key=lambda x: x[1], reverse=True)
    for node, score in ranked[:5]:
        print(f"  node {node}: {score:.6f}")
    return 0


def cmd_communities(args: argparse.Namespace) -> int:
    g = _load_graph(args)
    parts = communities(g)
    print(f"n_communities: {len(parts)}")
    print(f"modularity: {modularity(g, parts):.6f}")
    for index, part in enumerate(parts):
        members = ", ".join(str(node) for node in part)
        print(f"  community {index} (n={len(part)}): {members}")
    return 0


def cmd_shortest_path(args: argparse.Namespace) -> int:
    g = _load_graph(args)
    dist, prev = dijkstra_shortest_path(g, args.source, target=args.target)
    if args.target is not None:
        if args.target not in dist:
            print(f"{args.target} is not reachable from {args.source}")
            return 1
        path = reconstruct_path(prev, args.source, args.target)
        print(f"distance: {dist[args.target]:.6f}")
        print("path: " + " -> ".join(str(node) for node in path))
        return 0
    ranked = sorted(dist.items(), key=lambda item: (item[1], item[0]))
    for node, score in ranked[: args.top]:
        print(f"{node}: {score:.6f}")
    return 0


def _compose_report(g: Graph) -> str:
    dc = degree_centrality(g)
    cc = closeness_centrality(g)
    bc = betweenness_centrality(g)
    pr = pagerank(g)
    ev = eigenvector_centrality(g)
    lc = local_clustering(g)
    parts = communities(g)
    dist, prev = dijkstra_shortest_path(g, 0, target=33)
    path_to_33 = reconstruct_path(prev, 0, 33)
    comm_of = {node: index for index, part in enumerate(parts) for node in part}
    lines = [
        "# Graph Analytics Report",
        "",
        "Source: Zachary's Karate Club",
        "",
        "## Summary",
        "",
        f"- Nodes: {g.number_of_nodes()}",
        f"- Edges: {g.number_of_edges()}",
        f"- Density: {g.density():.4f}",
        f"- Connected: {g.is_connected()}",
        f"- Average clustering: {average_clustering(g):.6f}",
        f"- Communities (Louvain): {len(parts)}",
        f"- Modularity: {modularity(g, parts):.6f}",
        "",
        "## Centrality (top 5)",
        "",
        "| Node | Degree | Closeness | Betweenness | PageRank | Eigenvector | Clustering | Community |",
        "|------|--------|-----------|-------------|----------|-------------|------------|-----------|",
    ]
    dc_ranked = sorted(dc.items(), key=lambda x: x[1], reverse=True)
    for node, score in dc_ranked[:5]:
        lines.append(
            f"| {node} | {dc[node]:.4f} | {cc[node]:.6f} | {bc[node]:.6f} | "
            f"{pr[node]:.6f} | {ev[node]:.6f} | {lc[node]:.6f} | {comm_of[node]} |"
        )
    lines.extend(
        [
            "",
            "## Communities",
            "",
            "Louvain modularity maximization on Zachary's Karate Club. "
            "Community ids are ordered by the smallest node label in each group.",
            "",
            "| Community | Size | Nodes |",
            "|-----------|------|-------|",
        ]
    )
    for index, part in enumerate(parts):
        members = ", ".join(str(node) for node in part)
        lines.append(f"| {index} | {len(part)} | {members} |")
    hop_path = " -> ".join(str(node) for node in path_to_33)
    lines.extend(
        [
            "",
            "## Shortest paths",
            "",
            "Dijkstra distances from the instructor (node 0). Karate Club "
            "edges are unweighted, so distance is hop count.",
            "",
            f"- Distance to administrator (node 33): {dist[33]:.0f}",
            f"- Path: {hop_path}",
        ]
    )
    return "\n".join(lines) + "\n"


_COMMANDS = {
    "demo": cmd_demo,
    "stats": cmd_stats,
    "centrality": cmd_centrality,
    "clustering": cmd_clustering,
    "communities": cmd_communities,
    "shortest-path": cmd_shortest_path,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
