"""Query and creation utilities used to define and place the first cube.

Usage:
    Call any function/class from a separate script.

"""

import random

import networkx as nx

from topologiq.core.blocks import ZXBlock, ZXBlockRegistry
from topologiq.core.pathfinder.symbolic import check_exits_add_beams
from topologiq.utils.classes import CubeBeams, StandardBlock


###########
# MANAGER #
###########
def get_first_cube_data(
    bgraph: nx.Graph,
    first_id_strategy: str,
    inputs: list[int],
    override_first_cube: tuple[int | None, str | None] = (None, None),
    random_seed: int | None = None,
) -> tuple[int, ZXBlock, CubeBeams, CubeBeams]:
    """Define and place the very first cube of the blockgraph.

    Args:
        bgraph: The BlockGraph being built.
        first_id_strategy (optional): Strategy for selecting the ID of the first spider processed by the algorithm.
            centrality_majority: Use a majority vote from several centrality measures (deterministic).
            centrality_random: Pick randomly from a list of central spiders (probabilistic).
            first_spider: Select lowest ID non-boundary spider, typically 1st spider on 1st qubit (deterministic).
        inputs: A list of spiders formally declared as graph inputs.
        override_first_cube: Override ID and kind (used to replicate specific cases).
        random_seed: random_seed: Typically `None`, but can be used to pass a specific seed across the entire algorithm.

    Returns:
        first_id: The ID of the first cube.
        zx_block: The ZX block for the node corresponding to the first cube.
        src_beams: The beams for the first node.
        src_beams_short: The short beams for the first node.


    """

    # Pick and ID and kind for first cube
    first_cube: StandardBlock = pick_id_and_kind(
        bgraph,
        first_id_strategy=first_id_strategy,
        inputs=inputs,
        override_first_cube=override_first_cube,
        random_seed=random_seed,
    )
    first_id, first_kind = first_cube

    # Create ZXBlock corresponding to chosen kind
    zx_block = ZXBlockRegistry.get_create(kind=first_kind)

    # Get and write beams for corresponding node
    _, src_beams, src_beams_short = check_exits_add_beams(zx_block, (0, 0, 0), [], [(0, 0, 0)])

    return first_id, zx_block, src_beams, src_beams_short


###############
# ID/KIND GEN #
###############
def pick_first_id(bgraph: nx.Graph, first_id_strategy: str, inputs: list[int]) -> int:
    """Pick a node for use as starting point by outer graph manager BFS.

    Args:
        bgraph: The BlockGraph being built.
        first_id_strategy (optional): Strategy for selecting the ID of the first spider processed by the algorithm.
            centrality_majority: Use a majority vote from several centrality measures (deterministic).
            centrality_random: Pick randomly from a list of central spiders (probabilistic).
            first_spider: Select lowest ID non-boundary spider, typically 1st spider on 1st qubit (deterministic).
        inputs: A list of spiders formally declared as graph inputs.

    Returns:
        first_id: ID of node with highest closeness centrality or random ID from list of highest centrality.

    """

    # Terminate if graph is empty
    if not bgraph.nodes:
        raise ValueError("ERROR: bgraph.nodes() empty. Graph appears empty.")

    # ID of first non-boundary node
    if first_id_strategy == "first_spider":
        # Try getting the lowest formally declared input ID.
        if inputs:
            all_node_ids = sorted(inputs)

        # If no inputs are formally declared try fetching boundary spider with lowest ID
        else:
            all_node_ids = sorted(
                [
                    n_id
                    for n_id, attrs in bgraph.nodes(data=True)
                    if attrs["zx_block"].zx_type == "O"
                ]
            )

        # Worse case scenario, fetch smallest ID in graph.
        if not all_node_ids:
            all_node_ids = sorted([n_id for n_id in bgraph.nodes()])

        # Pick first
        first_id = all_node_ids[0]

    # Majority vote from applicable centrality measures
    elif first_id_strategy == "centrality_majority":
        # Append ID determined as central by several centrality measures to a single array
        central_nodes = []

        degree_centrality = nx.degree_centrality(bgraph)
        central_nodes.append(sorted(degree_centrality, key=degree_centrality.get, reverse=True)[0])

        closeness_centrality = nx.closeness_centrality(bgraph)
        central_nodes.append(
            sorted(closeness_centrality, key=closeness_centrality.get, reverse=True)[0]
        )

        info_centrality = nx.current_flow_closeness_centrality(bgraph, weight=None, solver="lu")
        central_nodes.append(sorted(info_centrality, key=info_centrality.get, reverse=True)[0])

        betweenness_centrality = nx.betweenness_centrality(bgraph, normalized=True, endpoints=True)
        central_nodes.append(
            sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)[0]
        )

        harmonic_centrality = nx.harmonic_centrality(
            bgraph, nbunch=None, distance=None, sources=None
        )
        central_nodes.append(
            sorted(harmonic_centrality, key=harmonic_centrality.get, reverse=True)[0]
        )

        laplacian = nx.laplacian_centrality(
            bgraph, normalized=True, nodelist=None, weight="weight", walk_type=None, alpha=0.95
        )
        central_nodes.append(sorted(laplacian, key=laplacian.get, reverse=True)[0])

        eigen_centrality = nx.eigenvector_centrality_numpy(bgraph)
        central_nodes.append(sorted(eigen_centrality, key=eigen_centrality.get, reverse=True)[0])

        # Choose most common
        first_id = max(set(central_nodes), key=central_nodes.count)

    # Random choice from central spiders
    elif first_id_strategy == "centrality_random":
        # Loose build a list of central spiders
        max_degree = -1
        central_nodes: list[int] = []
        node_degrees = bgraph.degree

        if isinstance(node_degrees, int):
            raise ValueError("ERROR: bgraph.degree() returned int. Cannot determine first ID.")

        for node, degree in node_degrees:
            if degree > max_degree:
                max_degree = degree
                central_nodes = [node]
            elif degree == max_degree:
                central_nodes.append(node)

        # Randomly pick a spider from list of central spiders
        first_id: int = random.choice(central_nodes)

    else:
        raise ValueError("ERROR @ pick_first_id. Invalid selection strategy.")

    return first_id


def pick_id_and_kind(
    bgraph: nx.Graph,
    first_id_strategy: str,
    inputs: list[int],
    override_first_cube: tuple[int | None, str | None] = (None, None),
    random_seed: int | None = None,
) -> tuple[int, str]:
    """Determine the iID and kind of the first block to place in 3D space.

    Args:
        bgraph: The BlockGraph being built.
        first_id_strategy (optional): Strategy for selecting the ID of the first spider processed by the algorithm.
            centrality_majority: Use a majority vote from several centrality measures (deterministic).
            centrality_random: Pick randomly from a list of central spiders (probabilistic).
            first_spider: Select lowest ID non-boundary spider, typically 1st spider on 1st qubit (deterministic).
        inputs: A list of spiders formally declared as graph inputs.
        override_first_cube: Override ID and kind (used to replicate specific cases).
        random_seed: Typically `None`, but can be used to pass a specific seed across the entire algorithm.

    Returns:
        first_id: ID of the first block to place in 3D space
        first_kind: Kind of the first block to place in 3D space

    """

    first_id, first_kind = override_first_cube

    if (not first_id or not first_kind) and random_seed:
        random.seed(random_seed)

    if not first_id:
        first_id = pick_first_id(bgraph, first_id_strategy, inputs)

    if not first_kind:
        deterministic = False if first_id_strategy == "centrality_random" else True
        tentative_kinds = bgraph.nodes[first_id].get("zx_block").get_kind_family
        first_kind = tentative_kinds[0] if deterministic else random.choice(tentative_kinds)

    return first_id, first_kind
