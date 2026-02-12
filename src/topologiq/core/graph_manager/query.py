"""Critical query and creation utilities to assist the primary graph managemer BFS.

Usage:
    Call any function/class from a separate script.

"""

import random

import networkx as nx


def get_first_id(nx_g: nx.Graph, first_id_strategy: str = "centrality_random") -> int:
    """Pick a node for use as starting point by outer graph manager BFS.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        first_id_strategy (optional): Enables switch between available strategies for choosing first node.

    Returns:
        first_id: ID of node with highest closeness centrality or random ID from list of highest centrality.

    """

    # Terminate if graph is empty
    if not nx_g.nodes:
        raise ValueError("ERROR: nx_g.nodes() empty. Graph appears empty.")

    # ID of first non-boundary node
    if first_id_strategy == "first_spider":

        # Sort all IDS in graph excluding boundaries
        all_node_ids = sorted(
            [node_id for node_id, node_info in nx_g.nodes(data=True) if node_info["type"] != "O"]
        )

        # Pick first
        first_id = all_node_ids[0]

    # Majority vote from applicable centrality measures
    elif first_id_strategy == "centrality_majority":

        # Append ID determined as central by several centrality measures to a single array
        central_nodes = []

        degree_centrality = nx.degree_centrality(nx_g)
        central_nodes.append(sorted(degree_centrality, key=degree_centrality.get, reverse=True)[0])

        closeness_centrality = nx.closeness_centrality(nx_g)
        central_nodes.append(sorted(closeness_centrality, key=closeness_centrality.get, reverse=True)[0])

        info_centrality = nx.current_flow_closeness_centrality(nx_g, weight=None, solver='lu')
        central_nodes.append(sorted(info_centrality, key=info_centrality.get, reverse=True)[0])

        betweenness_centrality = nx.betweenness_centrality(nx_g, normalized=True, endpoints=True)
        central_nodes.append(sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)[0])

        harmonic_centrality = nx.harmonic_centrality(nx_g, nbunch=None, distance=None, sources=None)
        central_nodes.append(sorted(harmonic_centrality, key=harmonic_centrality.get, reverse=True)[0])

        laplacian = nx.laplacian_centrality(nx_g, normalized=True, nodelist=None, weight='weight', walk_type=None, alpha=0.95)
        central_nodes.append(sorted(laplacian, key=laplacian.get, reverse=True)[0])

        eigen_centrality = nx.eigenvector_centrality_numpy(nx_g)
        central_nodes.append(sorted(eigen_centrality, key=eigen_centrality.get, reverse=True)[0])

        # Choose most common
        first_id = max(set(central_nodes), key=central_nodes.count)

    # Random choice from central spiders
    elif first_id_strategy == "centrality_random":

        # Loose build a list of central spiders
        max_degree = -1
        central_nodes: list[int] = []
        node_degrees = nx_g.degree

        if isinstance(node_degrees, int):
            raise ValueError("ERROR: nx_g.degree() returned int. Cannot determine first ID.")

        for node, degree in node_degrees:
            if degree > max_degree:
                max_degree = degree
                central_nodes = [node]
            elif degree == max_degree:
                central_nodes.append(node)

        # Randomly pick a spider from list of central spiders
        first_id: int = random.choice(central_nodes)

    else:
        raise ValueError("ERROR @ get_first_id. Invalid selection strategy.")

    return first_id


def get_node_degree(g: nx.Graph, node: int) -> int:
    """Get the degree (# of edges) of a given node.

    Args:
        g: an nx Graph.
        node: the node of interest.

    Returns:
        int: the degree for the node of interest, or 0 if graph has no edges.

    """

    # GET DEGREES FOR THE ENTIRE GRAPH
    degrees = g.degree

    # GET DEGREE FOR NODE OF INTEREST
    if not isinstance(degrees, int) and hasattr(degrees, "__getitem__"):
        return degrees[node]

    # IF DEGREES NOT A LIST, RETURN 0 (SINGLE NODE WON'T HAVE EDGES)
    return 0
