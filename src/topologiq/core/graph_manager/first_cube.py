"""Query and creation utilities used to define and place the first cube.

Usage:
    Call any function/class from a separate script.

"""

import random

import networkx as nx

from topologiq.core.beams import CubeBeams
from topologiq.core.blocks import ZXBlock, ZXBlockRegistry
from topologiq.core.pathfinder.symbolic import check_exits_add_beams
from topologiq.utils.classes import StandardCoord


###########
# MANAGER #
###########
def get_first_cube_data(
    bgraph: nx.Graph,
    first_id_strategy: str,
    qubits: dict[int, int],
    inputs: list[int],
    beams_len_short: int,
    first_coords: StandardCoord = (0, 0, 0),
    override_first_cube: tuple[int | None, str | None] = (None, None),
    random_seed: int | None = None,
) -> tuple[int, ZXBlock, CubeBeams, CubeBeams]:
    """Define and place the very first cube of the blockgraph.

    Args:
        bgraph: The BlockGraph being built.
        first_id_strategy (optional): Strategy for selecting the ID of the first spider processed by the algorithm.
            centrality-majority: Use a majority vote from several centrality measures (deterministic).
            centrality-random: Pick randomly from a list of central spiders (probabilistic).
            first_spider: Select lowest ID non-boundary spider, typically 1st spider on 1st qubit (deterministic).
        qubits: Spiders organised by qubit.
        inputs: A list of spiders formally declared as graph inputs.
        beams_len_short: The length of any short beams.
        first_coords (optional): First coords are (0,0,0) unless a specific value is given.
        override_first_cube: Override ID and kind (used to replicate specific cases).
        random_seed: random_seed (optional): Typically `None`, but used to pass a seed across the entire algorithm.

    Returns:
        first_id: The ID of the first cube.
        other_ids: A list of other IDs relevant to the type of first spider strategy used, None if such list does not exist
        zx_block: The ZX block for the node corresponding to the first cube.
        src_beams: The beams for the first node.
        src_beams_short: The short beams for the first node.


    """

    # Pick and ID and kind for first cube
    first_id, first_kind, other_ids = pick_id_and_kind(
        bgraph,
        first_id_strategy=first_id_strategy,
        qubits=qubits,
        inputs=inputs,
        override_first_cube=override_first_cube,
        random_seed=random_seed,
    )

    # Create ZXBlock corresponding to chosen kind
    zx_block = ZXBlockRegistry.get_create(kind=first_kind)

    # Get and write beams for corresponding node
    _, src_beams, src_beams_short = check_exits_add_beams(
        zx_block, first_coords, [], [first_coords], beams_len_short
    )

    return first_id, other_ids, zx_block, src_beams, src_beams_short


###############
# ID/KIND GEN #
###############


def pick_id_and_kind(
    bgraph: nx.Graph,
    first_id_strategy: str,
    qubits: dict[int, int],
    inputs: list[int],
    override_first_cube: tuple[int | None, str | None] = (None, None),
    random_seed: int | None = None,
) -> tuple[int, str, list[int]]:
    """Determine the iID and kind of the first block to place in 3D space.

    Args:
        bgraph: The BlockGraph being built.
        first_id_strategy (optional): Strategy for selecting the ID of the first spider processed by the algorithm.
            centrality-majority: Use a majority vote from several centrality measures (deterministic).
            centrality-random: Pick randomly from a list of central spiders (probabilistic).
            first_spider: Select lowest ID non-boundary spider, typically 1st spider on 1st qubit (deterministic).
        qubits: Spiders organised by qubit.
        inputs: A list of spiders formally declared as graph inputs.
        override_first_cube: Override ID and kind (used to replicate specific cases).
        random_seed: Typically `None`, but can be used to pass a specific seed across the entire algorithm.

    Returns:
        first_id: ID of the first block to place in 3D space
        first_kind: Kind of the first block to place in 3D space
        other_ids: A list of other IDs relevant to the type of first spider strategy used, None if such list does not exist

    """

    first_id, first_kind = override_first_cube

    if (not first_id or not first_kind) and random_seed:
        random.seed(random_seed)

    if override_first_cube:
        other_ids = []

    if not first_id:
        first_id, other_ids = pick_first_id(bgraph, first_id_strategy, qubits, inputs)

    if not first_kind:
        deterministic = False if first_id_strategy == "centrality-random" else True
        tentative_kinds = bgraph.nodes[first_id].get("zx_block").get_kind_family
        first_kind = tentative_kinds[0] if deterministic else random.choice(tentative_kinds)

    return first_id, first_kind, other_ids


def pick_first_id(
    bgraph: nx.Graph,
    first_id_strategy: str,
    qubits: dict[int, int],
    inputs: list[int],
) -> int:
    """Pick a node for use as starting point by outer graph manager BFS.

    Args:
        bgraph: The BlockGraph being built.
        first_id_strategy (optional): Strategy for selecting the ID of the first spider processed by the algorithm.
            centrality-majority: Use a majority vote from several centrality measures (deterministic).
            centrality-random: Pick randomly from a list of central spiders (probabilistic).
            first_spider: Select lowest ID non-boundary spider, typically 1st spider on 1st qubit (deterministic).
        qubits: Spiders organised by qubit.
        inputs: A list of spiders formally declared as graph inputs.

    Returns:
        first_id: ID of node with highest closeness centrality or random ID from list of highest centrality.
        other_ids: A list of other IDs relevant to the type of first spider strategy used, None if such list does not exist

    """

    # Terminate if graph is empty
    if not bgraph.nodes:
        raise ValueError("ERROR: bgraph.nodes() empty. Graph appears empty.")

    # Default other_ids to None
    other_ids = None

    # ID of first non-boundary node
    if first_id_strategy == "first-spider":
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
    elif first_id_strategy == "centrality-majority":
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

    elif first_id_strategy == "central-in-first-cycle":
        try:
            # Try determining central spider in first cycle
            # Will fail when the circuit has no cycles
            first_cycle = nx.cycle_basis(bgraph)[0]
            central_id, max_degree = (None, 0)
            for spider_id in first_cycle:
                degree = bgraph.degree[spider_id]
                if degree > max_degree:
                    max_degree = degree
                    central_id = spider_id
            first_id = central_id
        except IndexError:
            # If circuit has no cycles use a centrality measure
            degree_centrality = nx.betweenness_centrality(bgraph)
            first_id = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[0]

    # Random choice from central spiders
    elif first_id_strategy in ["centrality-random", "random"]:
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
        if first_id_strategy == "random":
            first_id: int = random.choice(
                [
                    n_id
                    for n_id, b in bgraph.nodes(data="zx_block")
                    if b.zx_type not in ["O", "Y", "XZ", "T"]
                ]
            )
        else:
            first_id: int = random.choice(central_nodes)

    elif first_id_strategy == "central-qubit":
        q_to_q_counts: dict[int, dict[int, list[int]]] = {}
        for node_id in bgraph.nodes():
            if qubits[node_id] < 0:
                continue
            if qubits[node_id] not in q_to_q_counts:
                q_to_q_counts[qubits[node_id]] = {}
            for neigh_id in bgraph.neighbors(node_id):
                if qubits[neigh_id] < 0 or qubits[node_id] == qubits[neigh_id]:
                    continue
                if qubits[neigh_id] not in q_to_q_counts[qubits[node_id]]:
                    q_to_q_counts[qubits[node_id]][qubits[neigh_id]] = 0
                q_to_q_counts[qubits[node_id]][qubits[neigh_id]] += 1

        q_to_q_means = {k: (sum(v.values()) / len(v)) for k, v in q_to_q_counts.items()}
        q_to_q_means_sorted = sorted(q_to_q_means.items(), key=lambda item: item[1], reverse=True)
        first_id = q_to_q_means_sorted[0][0]
        other_ids = [k for k, _ in q_to_q_means_sorted]

    else:
        raise ValueError("ERROR @ pick_first_id. Invalid selection strategy.")

    return first_id, other_ids
