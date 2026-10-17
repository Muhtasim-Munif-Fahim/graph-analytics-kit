"""Graph analytics: centrality, clustering, cliques, communities, link prediction, and flows."""

from __future__ import annotations

__version__ = "0.1.0"

from .graph import Graph, karate_club
from .centrality import (
    betweenness_centrality,
    closeness_centrality,
    degree_centrality,
    degree_centrality_array,
    eigenvector_centrality,
    harmonic_centrality,
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
from .connectivity import (
    articulation_points,
    biconnected_components,
    bridges,
    cut_structure,
    two_edge_connected_components,
)
from .clustering import average_clustering, local_clustering
from .community import communities, girvan_newman, label_propagation, modularity, spectral_clustering
from .dfs import dfs_path, dfs_traversal
from .flow import FlowResult, edge_connectivity, maximum_bipartite_matching, maximum_flow, minimum_cut
from .mst import kruskal_mst, mst_weight
from .hits import hits
from .katz import katz_centrality
from .kcore import core_number
from .cliques import (
    clique_number,
    find_cliques,
    max_clique,
    node_clique_number,
    number_of_maximal_cliques,
    total_triangles,
    triangles,
)
from .pagerank import pagerank, personalized_pagerank
from .shortest_path import dijkstra_shortest_path, reconstruct_path
from .scc import condensation_dag, is_strongly_connected, strongly_connected_components

__all__ = [
    "__version__",
    "Graph",
    "karate_club",
    "degree_centrality",
    "degree_centrality_array",
    "closeness_centrality",
    "harmonic_centrality",
    "betweenness_centrality",
    "pagerank",
    "personalized_pagerank",
    "eigenvector_centrality",
    "katz_centrality",
    "hits",
    "core_number",
    "find_cliques",
    "max_clique",
    "clique_number",
    "node_clique_number",
    "number_of_maximal_cliques",
    "triangles",
    "total_triangles",
    "bridges",
    "strongly_connected_components",
    "condensation_dag",
    "is_strongly_connected",
    "articulation_points",
    "biconnected_components",
    "two_edge_connected_components",
    "cut_structure",
    "local_clustering",
    "average_clustering",
    "degree_assortativity",
    "adamic_adar",
    "adamic_adar_scores",
    "jaccard",
    "jaccard_scores",
    "resource_allocation",
    "resource_allocation_scores",
    "preferential_attachment",
    "preferential_attachment_scores",
    "common_neighbors",
    "common_neighbors_scores",
    "katz_index_scores",
    "katz_index",
    "hub_promoted_index_scores",
    "hub_promoted_index",
    "communities",
    "girvan_newman",
    "label_propagation",
    "modularity",
    "spectral_clustering",
    "dfs_traversal",
    "dfs_path",
    "kruskal_mst",
    "mst_weight",
    "dijkstra_shortest_path",
    "reconstruct_path",
    "FlowResult",
    "maximum_flow",
    "minimum_cut",
    "edge_connectivity",
    "maximum_bipartite_matching",
]
