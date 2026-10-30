"""Command-line interface for graph-analytics-kit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .centrality import (
    betweenness_centrality,
    closeness_centrality,
    core_number,
    degree_centrality,
    eigenvector_centrality,
    harmonic_centrality,
    hits,
    katz_centrality,
    pagerank,
    personalized_pagerank,
)
from .assortativity import degree_assortativity
from .link_prediction import (
    adamic_adar,
    adamic_adar_scores,
    common_neighbors,
    common_neighbors_scores,
    hub_promoted_index,
    hub_promoted_index_scores,
    katz_index,
    katz_index_scores,
    jaccard,
    jaccard_scores,
    preferential_attachment,
    preferential_attachment_scores,
    resource_allocation,
    resource_allocation_scores,
)
from .clustering import average_clustering, local_clustering
from .community import communities, girvan_newman, label_propagation, modularity, spectral_clustering
from .graph import Graph, karate_club
from .simrank import most_similar, simrank_scores, simrank_similarity
from .shortest_path import dijkstra_shortest_path, reconstruct_path
from .distance import distance_summary


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
        choices=[
            "degree",
            "closeness",
            "harmonic",
            "betweenness",
            "pagerank",
            "personalized-pagerank",
            "eigenvector",
            "katz",
            "hubs",
            "authorities",
            "core",
        ],
        default="degree",
    )
    cent.add_argument("--top", type=int, default=5, help="Show top N nodes")
    cent.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Attenuation factor for Katz centrality (default: 0.1)",
    )

    clust = sub.add_parser("clustering", help="Print clustering coefficients")
    _add_graph_args(clust)

    comm = sub.add_parser("communities", help="Print detected communities")
    _add_graph_args(comm)
    comm.add_argument(
        "--method",
        choices=["louvain", "label-propagation", "girvan-newman", "spectral"],
        default="louvain",
        help="Community algorithm (default: louvain)",
    )
    comm.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Maximum sweeps (default: 100)",
    )
    comm.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for shuffled sweeps (label propagation and Louvain)",
    )
    comm.add_argument(
        "--n-communities",
        type=int,
        default=None,
        help="Target community count for Girvan–Newman / spectral (default: best modularity for GN; required for spectral)",
    )
    comm.add_argument(
        "--normalized",
        action="store_true",
        default=True,
        help="Use normalized Laplacian for spectral clustering (default)",
    )
    comm.add_argument(
        "--unnormalized",
        action="store_true",
        default=False,
        help="Use unnormalized Laplacian L=D-A for spectral clustering",
    )

    sp = sub.add_parser("shortest-path", help="Print Dijkstra shortest paths")
    _add_graph_args(sp)
    sp.add_argument("--source", type=int, default=0, help="Source node label")
    sp.add_argument("--target", type=int, default=None, help="Optional target node")
    sp.add_argument("--top", type=int, default=5, help="Show closest N nodes")

    dist = sub.add_parser(
        "distances",
        help="Diameter, radius, center, periphery, average path length, Wiener index, efficiency",
    )
    _add_graph_args(dist)
    dist.add_argument("--weighted", action="store_true", help="Use edge weights instead of hop counts")
    dist.add_argument("--eccentricity", action="store_true", help="Also print every node's eccentricity")


    conn = sub.add_parser(
        "connectivity",
        help="Print bridges, articulation points, and biconnected components",
    )
    _add_graph_args(conn)
    conn.add_argument(
        "--components", action="store_true",
        help="Also list every biconnected component's nodes",
    )

    cq = sub.add_parser(
        "cliques",
        help="Print maximal cliques (Bron–Kerbosch with pivoting), clique number, and triangles",
    )
    _add_graph_args(cq)
    cq.add_argument("--min-size", type=int, default=3, help="Only list cliques with at least this many nodes (default: 3)")
    cq.add_argument("--top", type=int, default=10, help="List at most this many cliques (default: 10; 0 = all)")

    fl = sub.add_parser(
        "flow",
        help="Print the max s-t flow (Dinic), the minimum cut, and edge connectivity",
    )
    _add_graph_args(fl)
    fl.add_argument("--source", type=int, default=0, help="Source node label")
    fl.add_argument("--sink", type=int, default=33, help="Sink node label")
    fl.add_argument(
        "--unit", action="store_true",
        help="Treat every edge as capacity 1 (edge-disjoint paths)",
    )

    lp = sub.add_parser(
        "link-prediction",
        help="Print Adamic–Adar / Jaccard / Resource Allocation / Preferential Attachment / Common Neighbors / Katz Index / Hub Promoted Index / SimRank link-prediction scores",
    )
    _add_graph_args(lp)
    lp.add_argument("--u", type=int, required=True, help="First node label")
    lp.add_argument("--v", type=int, required=True, help="Second node label")
    lp.add_argument(
        "--top-pairs",
        type=int,
        default=0,
        help="If >0, also rank this many non-edges by the chosen metric (default: 0)",
    )
    lp.add_argument(
        "--metric",
        choices=["adamic-adar", "jaccard", "resource-allocation", "preferential-attachment", "common-neighbors", "katz-index", "hub-promoted", "simrank", "both"],
        default="both",
        help="Score to print / rank by (default: both AA+Jaccard for the pair; top-pairs uses adamic-adar unless jaccard, resource-allocation, preferential-attachment, or common-neighbors)",
    )
    lp.add_argument(
        "--decay",
        type=float,
        default=0.8,
        help="SimRank decay factor C in (0, 1) (default: 0.8; used with --metric simrank)",
    )

    sim = sub.add_parser("similar", help="Rank the nodes most SimRank-similar to a node")
    _add_graph_args(sim)
    sim.add_argument("--node", type=int, required=True, help="Query node label")
    sim.add_argument("--top", type=int, default=5, help="How many similar nodes to list (default: 5)")
    sim.add_argument("--decay", type=float, default=0.8, help="SimRank decay factor C (default: 0.8)")

    col = sub.add_parser(
        "coloring",
        help="Greedy vertex colouring (largest-first, smallest-last, DSATUR, ...) and exact chromatic number",
    )
    _add_graph_args(col)
    col.add_argument(
        "--strategy",
        choices=["largest_first", "smallest_last", "dsatur", "connected_sequential_bfs",
                 "connected_sequential_dfs", "random_sequential"],
        default="dsatur",
        help="Greedy order used for the printed colour classes (default: dsatur)",
    )
    col.add_argument("--seed", type=int, default=0, help="Seed for random_sequential (default: 0)")
    col.add_argument("--exact", action="store_true", help="Also compute the exact chromatic number")

    eu = sub.add_parser("eulerian", help="Check for an Eulerian circuit/path and print it (Hierholzer)")
    _add_graph_args(eu)
    eu.add_argument("--source", type=int, default=None, help="Optional start node")

    iso = sub.add_parser(
        "isomorphism",
        help="WL hash, colour refinement, automorphism group/orbits, and exact isomorphism test",
    )
    _add_graph_args(iso)
    iso.add_argument("--edges", default=None, help="Analyse this edge-list file instead of the built-in graph")
    iso.add_argument("--directed", action="store_true", help="Treat edge-list files as directed")
    iso.add_argument("--compare", default=None, help="Edge-list file to test for isomorphism against")
    iso.add_argument("--iterations", type=int, default=3, help="WL rounds for the hash (default: 3)")
    iso.add_argument("--limit", type=int, default=None, help="Stop counting automorphisms at this many")

    dg = sub.add_parser(
        "dag",
        help="Topological order, longest path, transitive reduction and critical path of a DAG",
    )
    dg.add_argument("--edges", required=True, help="Directed edge-list file ('u v' or 'u v w' per line)")
    dg.add_argument("--weighted", action="store_true", help="Read a third column as edge weight")
    dg.add_argument(
        "--durations", default=None,
        help="Task durations 'node=dur,node=dur,...' to run the Critical Path Method",
    )

    ts = sub.add_parser(
        "tsp",
        help="Travelling salesman tour (Held-Karp exact, or NN/double-tree + 2-opt/Or-opt)",
    )
    ts.add_argument("--edges", default=None, help="Edge-list file ('u v' or 'u v w'); default: karate club")
    ts.add_argument("--weighted", action="store_true", help="Read a third column as edge weight")
    ts.add_argument("--directed", action="store_true", help="Treat the edge list as directed (asymmetric TSP)")
    ts.add_argument(
        "--method", choices=["auto", "exact", "heuristic", "nearest-neighbor", "double-tree"], default="auto",
    )
    ts.add_argument("--no-closure", action="store_true",
                    help="Use only direct edges (default: metric closure via shortest paths)")
    ts.add_argument("--starts", type=int, default=None, help="Nearest-neighbour start cities for the heuristic")

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
    print(f"degree_assortativity: {degree_assortativity(g):.6f}")
    return 0


def cmd_centrality(args: argparse.Namespace) -> int:
    g = _load_graph(args)
    measures = {
        "degree": degree_centrality,
        "closeness": closeness_centrality,
        "harmonic": harmonic_centrality,
        "betweenness": betweenness_centrality,
        "pagerank": pagerank,
        "personalized-pagerank": lambda g: personalized_pagerank(g, {g.nodes[0]: 1.0} if g.nodes else {}),
        "eigenvector": eigenvector_centrality,
        "katz": katz_centrality,
        "hubs": lambda graph: hits(graph)[0],
        "authorities": lambda graph: hits(graph)[1],
        "core": core_number,
    }
    measure = measures[args.measure]
    if args.measure == "katz" and args.alpha is not None:
        scores = measure(g, alpha=args.alpha)
    else:
        scores = measure(g)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for node, score in ranked[:args.top]:
        print(f"{node}: {score:.6f}")
    return 0


def cmd_clustering(args: argparse.Namespace) -> int:
    g = _load_graph(args)
    avg = average_clustering(g)
    print(f"average_clustering: {avg:.6f}")
    print(f"degree_assortativity: {degree_assortativity(g):.6f}")
    lc = local_clustering(g)
    ranked = sorted(lc.items(), key=lambda x: x[1], reverse=True)
    for node, score in ranked[:5]:
        print(f"  node {node}: {score:.6f}")
    return 0


def cmd_communities(args: argparse.Namespace) -> int:
    g = _load_graph(args)
    kwargs = {}
    if args.max_iter is not None:
        kwargs["max_iter"] = args.max_iter
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.method == "label-propagation":
        parts = label_propagation(g, **kwargs)
        method_name = "label propagation"
    elif args.method == "girvan-newman":
        gn_kwargs = {}
        if args.n_communities is not None:
            gn_kwargs["n_communities"] = args.n_communities
        parts = girvan_newman(g, **gn_kwargs)
        method_name = "girvan-newman"
    elif args.method == "spectral":
        n_comm = args.n_communities if args.n_communities is not None else 2
        sc_kwargs = {"n_communities": n_comm, "normalized": not args.unnormalized}
        if args.seed is not None:
            sc_kwargs["seed"] = args.seed
        if args.max_iter is not None:
            sc_kwargs["max_iter"] = args.max_iter
        parts = spectral_clustering(g, **sc_kwargs)
        method_name = "spectral"
    else:
        parts = communities(g, **kwargs)
        method_name = "louvain"
    print(f"method: {method_name}")
    print(f"n_communities: {len(parts)}")
    print(f"modularity: {modularity(g, parts):.6f}")
    for index, part in enumerate(parts):
        members = ", ".join(str(node) for node in part)
        print(f"  community {index} (n={len(part)}): {members}")
    return 0



def cmd_link_prediction(args: argparse.Namespace) -> int:
    g = _load_graph(args)
    metric = getattr(args, "metric", "both")
    if metric in ("adamic-adar", "both"):
        aa = adamic_adar(g, args.u, args.v)
        print(f"adamic_adar({args.u}, {args.v}): {aa:.6f}")
    if metric in ("jaccard", "both"):
        jc = jaccard(g, args.u, args.v)
        print(f"jaccard({args.u}, {args.v}): {jc:.6f}")
    if metric == "resource-allocation":
        ra = resource_allocation(g, args.u, args.v)
        print(f"resource_allocation({args.u}, {args.v}): {ra:.6f}")
    if metric == "preferential-attachment":
        pa = preferential_attachment(g, args.u, args.v)
        print(f"preferential_attachment({args.u}, {args.v}): {pa:.6f}")
    if metric == "common-neighbors":
        cn = common_neighbors(g, args.u, args.v)
        print(f"common_neighbors({args.u}, {args.v}): {cn:.6f}")
    if metric == "katz-index":
        kz = katz_index(g, args.u, args.v)
        print(f"katz_index({args.u}, {args.v}): {kz:.6f}")
    if metric == "hub-promoted":
        hpi = hub_promoted_index(g, args.u, args.v)
        print(f"hub_promoted_index({args.u}, {args.v}): {hpi:.6f}")
    if metric == "simrank":
        sr = simrank_similarity(g, args.u, args.v, decay=args.decay)
        print(f"simrank({args.u}, {args.v}): {sr:.6f}")
    if args.top_pairs and args.top_pairs > 0:
        nodes = g.nodes
        existing = {(min(u, v), max(u, v)) for u, v, _w in g.edges}
        candidates = []
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                key = (min(u, v), max(u, v))
                if key in existing or u == v:
                    continue
                candidates.append((u, v))
        if metric == "jaccard":
            scores = jaccard_scores(g, candidates)
            label = "Jaccard"
        elif metric == "resource-allocation":
            scores = resource_allocation_scores(g, candidates)
            label = "Resource Allocation"
        elif metric == "preferential-attachment":
            scores = preferential_attachment_scores(g, candidates)
            label = "Preferential Attachment"
        elif metric == "common-neighbors":
            scores = common_neighbors_scores(g, candidates)
            label = "Common Neighbors"
        elif metric == "katz-index":
            scores = katz_index_scores(g, candidates)
            label = "Katz Index"
        elif metric == "hub-promoted":
            scores = hub_promoted_index_scores(g, candidates)
            label = "Hub Promoted Index"
        elif metric == "simrank":
            scores = simrank_scores(g, candidates, decay=args.decay)
            label = "SimRank"
        else:
            scores = adamic_adar_scores(g, candidates)
            label = "Adamic–Adar"
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
        print(f"top non-edges by {label} (n={min(args.top_pairs, len(ranked))}):")
        for (u, v), s in ranked[: args.top_pairs]:
            print(f"  ({u}, {v}): {s:.6f}")
    return 0


def cmd_similar(args: argparse.Namespace) -> int:
    g = _load_graph(args)
    if args.node not in set(g.nodes):
        print(f"graph-analytics: error: node {args.node} is not in the graph", file=sys.stderr)
        return 2
    ranked = most_similar(g, args.node, k=args.top, decay=args.decay)
    print(f"nodes most SimRank-similar to {args.node} (C={args.decay:g}):")
    for other, score in ranked:
        print(f"  {other}: {score:.6f}")
    return 0


def cmd_connectivity(args: argparse.Namespace) -> int:
    from .connectivity import cut_structure

    g = _load_graph(args)
    info = cut_structure(g)
    bridges_txt = ", ".join(f"{u}-{v}" for u, v in info["bridges"]) or "none"
    cuts_txt = ", ".join(str(n) for n in info["articulation_points"]) or "none"
    print(f"bridges ({info['n_bridges']}): {bridges_txt}")
    print(f"articulation_points ({info['n_articulation_points']}): {cuts_txt}")
    print(f"biconnected_components: {info['n_biconnected_components']}")
    print(f"largest_biconnected_component: {info['largest_biconnected_component']}")
    if args.components:
        for i, comp in enumerate(info["biconnected_components"]):
            print(f"  [{i}] size={len(comp)}: {' '.join(map(str, comp))}")
    return 0


def cmd_cliques(args: argparse.Namespace) -> int:
    from .cliques import clique_number, find_cliques, number_of_maximal_cliques, total_triangles, triangles

    if args.min_size < 1:
        print("cliques: --min-size must be >= 1", file=sys.stderr)
        return 2
    g = _load_graph(args)
    cliques = find_cliques(g, min_size=args.min_size)
    omega = clique_number(g)
    print(f"maximal_cliques: {number_of_maximal_cliques(g)}")
    print(f"clique_number: {omega}")
    print(f"maximal_cliques_with_size>={args.min_size}: {len(cliques)}")
    print(f"triangles: {total_triangles(g)}")
    shown = cliques if args.top == 0 else cliques[: args.top]
    for clique in shown:
        print(f"  size={len(clique)}: {' '.join(map(str, clique))}")
    tri = triangles(g)
    top_nodes = sorted(tri, key=lambda n: (-tri[n], n))[:5]
    print("most_triangles: " + ", ".join(f"{n}({tri[n]})" for n in top_nodes))
    return 0


def cmd_flow(args: argparse.Namespace) -> int:
    from .flow import edge_connectivity, maximum_flow

    g = _load_graph(args)
    try:
        res = maximum_flow(g, args.source, args.sink, unit_capacity=args.unit)
    except ValueError as exc:
        print(f"flow: {exc}", file=sys.stderr)
        return 2
    print(f"max flow: {res.value:g}  ({args.source} -> {args.sink}, {res.meta['phases']} Dinic phases)")
    print(f"min cut capacity: {res.cut_capacity:g}")
    print("cut edges: " + (", ".join(f"{u}-{v}" for u, v in res.cut_edges) or "none"))
    print(f"source side ({len(res.source_side)}): {' '.join(map(str, res.source_side))}")
    print(f"sink side ({len(res.sink_side)}): {' '.join(map(str, res.sink_side))}")
    print(f"edge connectivity ({args.source}-{args.sink}): {edge_connectivity(g, args.source, args.sink)}")
    print(f"edge connectivity (global): {edge_connectivity(g)}")
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


def cmd_distances(args: argparse.Namespace) -> int:
    g = _load_graph(args)
    try:
        summary = distance_summary(g, weighted=args.weighted)
    except ValueError as exc:
        print(f"distances: {exc}", file=sys.stderr)
        return 2
    print(f"diameter: {summary['diameter']:g}")
    print(f"radius: {summary['radius']:g}")
    print("center: " + " ".join(map(str, summary["center"])))
    print("periphery: " + " ".join(map(str, summary["periphery"])))
    print(f"average shortest path length: {summary['average_shortest_path_length']:.6f}")
    print(f"Wiener index: {summary['wiener_index']:g}")
    print(f"global efficiency: {summary['global_efficiency']:.6f}")
    if args.eccentricity:
        for node, ecc in summary["eccentricity"].items():
            print(f"  {node}: {ecc:g}")
    return 0


def cmd_coloring(args: argparse.Namespace) -> int:
    from .coloring import chromatic_number, color_classes, coloring_summary, greedy_color

    g = _load_graph(args)
    for name, k in coloring_summary(g, seed=args.seed).items():
        print(f"{name}: {k} colours")
    coloring = greedy_color(g, args.strategy, seed=args.seed)
    classes = color_classes(coloring)
    print(f"{args.strategy} colour classes ({len(classes)}):")
    for c, members in enumerate(classes):
        print(f"  [{c}] size={len(members)}: {' '.join(map(str, members))}")
    if args.exact:
        print(f"chromatic number: {chromatic_number(g)}")
    return 0


def cmd_eulerian(args: argparse.Namespace) -> int:
    from .eulerian import eulerian_path, has_eulerian_path, is_eulerian, odd_degree_nodes

    g = _load_graph(args)
    circuit = is_eulerian(g)
    path = has_eulerian_path(g)
    print(f"eulerian circuit: {'yes' if circuit else 'no'}")
    print(f"eulerian path: {'yes' if path else 'no'}")
    if not g.directed:
        odd = odd_degree_nodes(g)
        print(f"odd-degree nodes ({len(odd)}): {' '.join(map(str, odd)) or 'none'}")
    if path:
        try:
            edges = eulerian_path(g, source=args.source)
        except ValueError as exc:
            print(f"eulerian: {exc}", file=sys.stderr)
            return 2
        nodes = [edges[0][0]] + [v for _, v in edges]
        print("walk: " + " -> ".join(map(str, nodes)))
    return 0


def cmd_isomorphism(args: argparse.Namespace) -> int:
    from .isomorphism import (
        automorphism_count,
        automorphism_orbits,
        color_refinement,
        find_isomorphism,
        wl_colors,
        wl_graph_hash,
    )

    if args.edges:
        g = Graph.from_edge_list(args.edges, directed=args.directed)
    else:
        g = _load_graph(args)
    rounds = len(wl_colors(g)) - 1
    classes = color_refinement(g)
    print(f"WL hash ({args.iterations} rounds): {wl_graph_hash(g, iterations=args.iterations)}")
    print(f"colour refinement: {len(classes)} stable classes after {rounds} rounds")
    n_aut = automorphism_count(g, limit=args.limit)
    reached = args.limit is not None and n_aut >= args.limit
    print(f"automorphisms: {n_aut}" + (" (limit reached)" if reached else ""))
    nontrivial = [o for o in automorphism_orbits(g) if len(o) > 1] if not args.limit else []
    if nontrivial:
        print(f"non-trivial orbits ({len(nontrivial)}):")
        for orbit in nontrivial:
            print("  " + " ".join(map(str, orbit)))
    if args.compare:
        other = Graph.from_edge_list(args.compare, directed=args.directed)
        phi = find_isomorphism(g, other)
        same_hash = wl_graph_hash(g, iterations=args.iterations) == wl_graph_hash(other, iterations=args.iterations)
        print(f"WL hashes match: {'yes' if same_hash else 'no'}")
        print(f"isomorphic: {'yes' if phi is not None else 'no'}")
        if phi is not None:
            print("mapping: " + " ".join(f"{u}->{v}" for u, v in sorted(phi.items())))
    return 0


def cmd_dag(args: argparse.Namespace) -> int:
    from .dag import (
        critical_path,
        dag_longest_path,
        find_cycle,
        topological_generations,
        topological_sort,
        transitive_reduction,
    )

    g = Graph.from_edge_list(args.edges, directed=True, weighted=args.weighted)
    cycle = find_cycle(g)
    if cycle is not None:
        print("not a DAG; cycle: " + " -> ".join(map(str, cycle + cycle[:1])))
        return 1
    print("topological order: " + " ".join(map(str, topological_sort(g))))
    for k, layer in enumerate(topological_generations(g)):
        print(f"  generation {k}: {' '.join(map(str, layer))}")
    length, path = dag_longest_path(g, weighted=args.weighted)
    print(f"longest path ({length:g}): " + " -> ".join(map(str, path)))
    red = transitive_reduction(g)
    print(f"transitive reduction: {red.number_of_edges()} of {g.number_of_edges()} edges kept")
    if args.durations:
        durations = {}
        for item in args.durations.split(","):
            node, _, value = item.partition("=")
            durations[int(node)] = float(value)
        cpm = critical_path(g, durations)
        print(f"project duration: {cpm['duration']:g}")
        print("critical path: " + " -> ".join(map(str, cpm["path"])))
        for u in topological_sort(g):
            print(
                f"  task {u}: ES={cpm['es'][u]:g} EF={cpm['ef'][u]:g} "
                f"LS={cpm['ls'][u]:g} LF={cpm['lf'][u]:g} slack={cpm['slack'][u]:g}"
            )
    return 0


def cmd_tsp(args: argparse.Namespace) -> int:
    from .tsp import solve_tsp

    if args.edges:
        g = Graph.from_edge_list(args.edges, directed=args.directed, weighted=args.weighted)
    else:
        g = karate_club()
    closure = not args.no_closure
    try:
        res = solve_tsp(g, method=args.method, metric_closure=closure, expand_path=closure,
                        n_starts=args.starts)
    except ValueError as exc:
        print(f"tsp: {exc}", file=sys.stderr)
        return 2
    status = "optimal" if res.optimal else "heuristic"
    print(f"tour cost: {res.cost:g}  ({status}, method={res.method}, {len(res.tour)} cities)")
    print("tour: " + " -> ".join(map(str, res.tour + res.tour[:1])))
    if res.path is not None and len(res.path) != len(res.tour) + 1:
        print("walk over graph edges: " + " -> ".join(map(str, res.path)))
    if not res.optimal and res.lower_bound is not None:
        print(f"1-tree lower bound: {res.lower_bound:g}  gap <= {res.gap:.2%}")
    return 0


def _compose_report(g: Graph) -> str:
    dc = degree_centrality(g)
    cc = closeness_centrality(g)
    hc = harmonic_centrality(g)
    bc = betweenness_centrality(g)
    pr = pagerank(g)
    ev = eigenvector_centrality(g)
    kz = katz_centrality(g)
    hubs, authorities = hits(g)
    cores = core_number(g)
    lc = local_clustering(g)
    parts = communities(g)
    lpa_parts = label_propagation(g)
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
        f"- Degree assortativity: {degree_assortativity(g):.6f}",
        f"- Communities (Louvain): {len(parts)}",
        f"- Modularity: {modularity(g, parts):.6f}",
        f"- Communities (label propagation): {len(lpa_parts)}",
        f"- Label-propagation modularity: {modularity(g, lpa_parts):.6f}",
        "",
        "## Centrality (top 5)",
        "",
        "| Node | Degree | Closeness | Harmonic | Betweenness | Core | PageRank | Eigenvector | Katz | Hub | Authority | Clustering | Community |",
        "|------|--------|-----------|----------|-------------|------|----------|-------------|------|-----|-----------|------------|-----------|",
    ]
    dc_ranked = sorted(dc.items(), key=lambda x: x[1], reverse=True)
    for node, score in dc_ranked[:5]:
        lines.append(
            f"| {node} | {dc[node]:.4f} | {cc[node]:.6f} | {hc[node]:.6f} | {bc[node]:.6f} | "
            f"{cores[node]} | {pr[node]:.6f} | {ev[node]:.6f} | {kz[node]:.6f} | "
            f"{hubs[node]:.6f} | {authorities[node]:.6f} | {lc[node]:.6f} | "
            f"{comm_of[node]} |"
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
    lines.extend(
        [
            "",
            "### Label propagation",
            "",
            "Asynchronous label propagation on the same graph. With no seed, "
            "sweeps follow increasing degree then node label.",
            "",
            "| Community | Size | Nodes |",
            "|-----------|------|-------|",
        ]
    )
    for index, part in enumerate(lpa_parts):
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
    from .connectivity import cut_structure

    cuts = cut_structure(g)
    bridge_txt = ", ".join(f"{u}-{v}" for u, v in cuts["bridges"]) or "none"
    cut_txt = ", ".join(str(n) for n in cuts["articulation_points"]) or "none"
    lines.extend(
        [
            "",
            "## Cut structure",
            "",
            "Bridges and articulation points are single edges and nodes whose "
            "removal disconnects the graph (Tarjan/Hopcroft DFS).",
            "",
            f"- Bridges: {bridge_txt}",
            f"- Articulation points: {cut_txt}",
            f"- Biconnected components: {cuts['n_biconnected_components']} "
            f"(largest has {cuts['largest_biconnected_component']} nodes)",
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
    "distances": cmd_distances,
    "link-prediction": cmd_link_prediction,
    "similar": cmd_similar,
    "connectivity": cmd_connectivity,
    "flow": cmd_flow,
    "cliques": cmd_cliques,
    "coloring": cmd_coloring,
    "eulerian": cmd_eulerian,
    "dag": cmd_dag,
    "isomorphism": cmd_isomorphism,
    "tsp": cmd_tsp,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
