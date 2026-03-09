"""Query and creation utilities used to define and place the first cube.

Usage:
    Call any function/class from a separate script.

"""

import random

import networkx as nx

from topologiq.core.pathfinder.symbolic import check_exits
from topologiq.utils.classes import StandardBlock, StandardCoord


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
        central_nodes.append(
            sorted(closeness_centrality, key=closeness_centrality.get, reverse=True)[0]
        )

        info_centrality = nx.current_flow_closeness_centrality(nx_g, weight=None, solver="lu")
        central_nodes.append(sorted(info_centrality, key=info_centrality.get, reverse=True)[0])

        betweenness_centrality = nx.betweenness_centrality(nx_g, normalized=True, endpoints=True)
        central_nodes.append(
            sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)[0]
        )

        harmonic_centrality = nx.harmonic_centrality(nx_g, nbunch=None, distance=None, sources=None)
        central_nodes.append(
            sorted(harmonic_centrality, key=harmonic_centrality.get, reverse=True)[0]
        )

        laplacian = nx.laplacian_centrality(
            nx_g, normalized=True, nodelist=None, weight="weight", walk_type=None, alpha=0.95
        )
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


def get_first_cube(
    nx_g: nx.Graph, # TODO-ANG: replace with ang
    first_cube: tuple[int | None, str | None] = (None, None),
    first_id_strategy: str = "centrality_random",
    random_seed: int | None = None,
) -> tuple[int, str]:
    """Determine the iID and kind of the first block to place in 3D space.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        first_cube (optional): Override ID and kind (used to replicate specific cases).
        first_id_strategy (optional): Strategy for selecting the ID of the first spider processed by the algorithm.
            centrality_majority: Use a majority vote from several centrality measures (deterministic).
            centrality_random: Pick randomly from a list of central spiders (probabilistic).
            first_spider: Select lowest ID non-boundary spider, typically 1st spider on 1st qubit (deterministic).
        random_seed: Typically `None`, but can be used to pass a specific seed across the entire algorithm.

    Returns:
        first_id: ID of the first block to place in 3D space
        first_kind: Kind of the first block to place in 3D space

    """

    first_id, first_kind = first_cube

    if (not first_id or not first_kind) and random_seed:
        random.seed(random_seed)

    if not first_id:
        first_id = get_first_id(nx_g, first_id_strategy=first_id_strategy)

    if not first_kind:
        deterministic = False if first_id_strategy == "centrality_random" else True
        tentative_kinds = nx_g.nodes[first_id].get("type_fam")
        first_kind = tentative_kinds[0] if deterministic else random.choice(tentative_kinds)

    return first_id, first_kind


def place_first_cube(
    nx_g: nx.Graph, # TODO-ANG: replace with ang
    taken: list[StandardCoord],
    first_cube: StandardBlock,
    log_stats_id: int | None = None,
    debug: int = 0,
) -> tuple[list[StandardCoord], nx.Graph]:
    """Place the first cube in the 3D space.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        first_cube: ID and kind for the very first spider/cube to place in 3D space.
        log_stats_id (optional): A unique datetime-based identifier for the purposes of logging stats for an specific run.
        debug (optional): Debug mode (0: off, 1: graph manager, 2: pathfinder, 3: pathfinder w. discarded paths).

    Returns:
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.

    """

    # Update taken
    taken.append((0, 0, 0))

    # Get beams
    first_id, first_kind = first_cube
    _, src_beams, src_beams_short = check_exits((0, 0, 0), first_kind, taken, [(0, 0, 0)])

    # Write info to nx_g
    nx_g.nodes[first_id]["coords"] = (0, 0, 0)
    nx_g.nodes[first_id]["kind"] = first_kind
    nx_g.nodes[first_id]["beams"] = src_beams
    nx_g.nodes[first_id]["beams_short"] = src_beams_short

    if log_stats_id or debug > 0:
        print(f"First cube ID: {first_id} ({first_kind}).")

    return nx_g, taken
