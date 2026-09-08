"""Command-line interface for graph-analytics-kit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .centrality import betweenness_centrality, closeness_centrality, degree_centrality
from .clustering import average_clustering, local_clustering
from .graph import Graph, karate_club


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
        "--measure", choices=["degree", "closeness", "betweenness"],
        default="degree",
    )
    cent.add_argument("--top", type=int, default=5, help="Show top N nodes")

    clust = sub.add_parser("clustering", help="Print clustering coefficients")
    _add_graph_args(clust)

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
    if args.measure == "degree":
        scores = degree_centrality(g)
    elif args.measure == "closeness":
        scores = closeness_centrality(g)
    else:
        scores = betweenness_centrality(g)
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


def _compose_report(g: Graph) -> str:
    dc = degree_centrality(g)
    cc = closeness_centrality(g)
    bc = betweenness_centrality(g)
    lc = local_clustering(g)
    lines = [
        f"# Graph Analytics Report",
        f"",
        f"Source: Zachary's Karate Club",
        f"",
        f"## Summary",
        f"",
        f"- Nodes: {g.number_of_nodes()}",
        f"- Edges: {g.number_of_edges()}",
        f"- Density: {g.density():.4f}",
        f"- Connected: {g.is_connected()}",
        f"- Average clustering: {average_clustering(g):.6f}",
        f"",
        f"## Centrality (top 5)",
        f"",
        f"| Node | Degree | Closeness | Betweenness | Clustering |",
        f"|------|--------|-----------|-------------|------------|",
    ]
    dc_ranked = sorted(dc.items(), key=lambda x: x[1], reverse=True)
    for node, score in dc_ranked[:5]:
        lines.append(
            f"| {node} | {dc[node]:.4f} | {cc[node]:.6f} | {bc[node]:.6f} | {lc[node]:.6f} |"
        )
    return "\n".join(lines) + "\n"


_COMMANDS = {
    "demo": cmd_demo,
    "stats": cmd_stats,
    "centrality": cmd_centrality,
    "clustering": cmd_clustering,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
