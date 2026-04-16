"""Utilities to assist the management of the graph manager BFS.

Usage:
    Call any function/class from a separate script.

"""

import random
import shutil
from collections import deque
from typing import Iterable

import networkx as nx

from topologiq.core.pathfinder.spatial import get_taken_coords
from topologiq.dzw.common.components_zx import EdgeType
from topologiq.input.simple_graphs import check_zx_types, get_zx_type_fam
from topologiq.utils.classes import (
    Colors,
    PathBetweenNodes,
    SimpleDictGraph,
    StandardBlock,
    StandardCoord,
)


from topologiq.dzw.augmented_nx_graph import AugmentedNxGraph
from topologiq.dzw.common.components_bg import CubeKind
from topologiq.dzw.common.path import Path


#################
# HEALTH CHECKS #
#################
def validity_checks(simple_graph: SimpleDictGraph, first_cube: tuple[int, str]) -> bool:
    """Check validity of key non-optional BFS parameters.

    Args:
        simple_graph: The `simple_graph` form of an arbitrary ZX circuit.
        first_cube: ID and kind for the very first spider/cube to place in 3D space

    Returns:
        bool: True if all parameters are valid, otherwise False.

    """

    if not check_zx_types(simple_graph):
        print(Colors.RED + "Graph validity checks failed. Aborting." + Colors.RESET)
        return False

    first_id, first_kind = first_cube
    if first_id is None:
        print(
            Colors.RED + "Invalid first type. Input graph might not have any nodes." + Colors.RESET
        )
        return False

    if first_kind not in ["zxz", "zzx", "xzz", "yyy", "xzx", "xxz", "zxx", "ooo"]:
        print(Colors.RED + "Invalid first kind. Input graph might be malformed" + Colors.RESET)
        return False

    return True


######################
# GRAPH MANAGER INIT #
######################
def prep_3d_g(simple_graph: SimpleDictGraph) -> nx.Graph:
    """Convert a `simple_graph` into an NX graph with syntax and structure amicable to 3D transformations.

    This function takes a `simple_graph` containing the spiders and edges of a ZX graph and converts it into
    an NX graph. The resulting NX graph contains the same information as the `simple_graph` but has a number
    of placeholders that enable the algorithm to overwrite the NX graph with 3D information as the algorithm
    traverses the graph making 3D placements.

    Args:
        simple_graph: The `simple_graph` form of an arbitrary ZX circuit.

    Returns:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.

    """

    # Prepare an empty NX graph
    nx_g = nx.Graph()

    # Get the spiders and edges of incoming `simple_graph`
    nodes: list[tuple[int, str]] = simple_graph.get("nodes", [])
    edges: list[tuple[tuple[int, int], str]] = simple_graph.get("edges", [])

    # Add the spiders to the NX graph
    for n_id, n_type in nodes:
        nx_g.add_node(
            n_id,
            type=n_type,
            type_fam=get_zx_type_fam(n_type),
            kind=None,
            coords=None,
            beams=None,
            beams_short=None,
            completed=0,
        )

    # Add the edges to the NX graph
    for (src_id, tgt_id), e_type in edges:
        nx_g.add_edge(src_id, tgt_id, type=e_type)

    # Break any spiders with mode than 4 edges/neigbours
    # Note. Backup feature. Ideally, the incoming graph will have been pre-processed.
    nodes_with_more_than_four_edges = [
        n for n in list(nx_g.nodes()) if get_node_degree(nx_g, n) > 4
    ]
    if nodes_with_more_than_four_edges:
        nx_g = enforce_max_four_legs_per_spider(nx_g)

    return nx_g


def enforce_max_four_legs_per_spider(nx_g: nx.Graph) -> nx.Graph:
    """Ensure that all spiders in input graph have at most four legs/edges.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.

    Return:
        nx_g: The updated version of the incoming graph.

    """

    # Determine max degree
    max_id = max(nx_g.nodes) if nx_g.nodes else 0

    # Loop over max nodes and break as appropriate
    i = 0
    while i < 100:
        # List of high degree nodes
        all_nodes_loop = list(nx_g.nodes())
        nodes_with_more_than_four_edges = [
            n for n in all_nodes_loop if get_node_degree(nx_g, n) > 4
        ]

        # Exit loop when no nodes with more than 4 edges
        if nodes_with_more_than_four_edges:
            # Pick a high degree node
            node_to_sanitise = random.choice(nodes_with_more_than_four_edges)
            orig_node_type = nx_g.nodes[node_to_sanitise]["type"]

            # Add a twin
            max_id += 1
            twin_node_id = max_id
            nx_g.add_node(
                twin_node_id,
                type=orig_node_type,
                type_fam=get_zx_type_fam(orig_node_type),
                kind=None,
                coords=None,
                beams=None,
                completed=0,
            )
            nx_g.add_edge(node_to_sanitise, twin_node_id, type="SIMPLE")

            # Distributed edges across twins
            neighs = [n for n in list(nx_g.neighbors(node_to_sanitise)) if n != twin_node_id]
            degree_to_shuffle = get_node_degree(nx_g, node_to_sanitise) // 2
            random.shuffle(neighs)

            shuffle_c = 0
            for neigh in neighs:
                if shuffle_c >= degree_to_shuffle or get_node_degree(nx_g, node_to_sanitise) <= 4:
                    break
                if nx_g.has_edge(node_to_sanitise, neigh) and not nx_g.has_edge(
                    twin_node_id, neigh
                ):
                    edge_data = nx_g.get_edge_data(node_to_sanitise, neigh)
                    edge_type = edge_data.get("type", None)
                    nx_g.add_edge(twin_node_id, neigh, type=edge_type)
                    nx_g.remove_edge(node_to_sanitise, neigh)
                    shuffle_c += 1

    return nx_g


############
# BFS INIT #
############
def init_bfs(first_cube) -> tuple[deque, set, list[StandardCoord], dict, bool]:
    """Initialise key BFS management objects.

    Args:
        first_cube: Coordinates and kind for the first cube.

    Return:
        queue: The main BFS queue.
        visited: The main BFS set of visited sites.
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        edge_paths: An edge-by-edge/block-by-block summary of the space-time diagram Topologiq builds.
        run_success: Boolean flag to determine if Whether the BFS search was successful as a whole.

    """

    # Exract params
    first_id, _ = first_cube

    # Init queue & visited
    queue: deque[int] = deque([first_id])
    visited: set = {first_id}

    # Init other trackers
    taken: list[StandardCoord] = []
    edge_paths: dict = {}
    run_success = False

    return queue, visited, taken, edge_paths, run_success


##############
# BFS UPDATE #
##############
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


def update_edge_paths(
    ang: AugmentedNxGraph,
    nx_g: nx.Graph,
    edge_paths: dict,
    winner_path_standard_pass: PathBetweenNodes | None,
    winner_path_second_pass: list[StandardBlock] | None,
    taken: list[StandardCoord],
    zx_edge_type: str,
    src_id: int,
    tgt_id: int,
    second_pass: bool = False,
):
    """Write the result of a pathfinder iteration to edge_paths.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        edge_paths: An edge-by-edge summary of the 3D object Topologiq builds, updated to the last edge processsed successfully.
        winner_path_standard_pass: A winner path chosen by the value function.
        winner_path_second_pass: A list of paths each containing the 3D cubes and pipes needed to connect source and target in the 3D space.
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        zx_edge_type: The type of edge currently being processed, i.e., SIMPLE or HADAMARD.
        src_id: The ID of the current source cube.
        tgt_id: The ID of the potential target cube.
        second_pass: A boolean to flag if the edge being written as part of a "second_pass" operation.
        twin: A boolean to flag if the edge being written is the result of cloning an existing node.

    Returns:
        nx_g: An updated version of the incoming nx_graph.
        taken: An updated version of the list of taken coordinates.
        edge_paths: An updated version of the `edge_paths` object.
        edge_success: Whether the edge was succesfully written to `edge_paths` or not.

    """

    # ID edge as sorted to avoid duplicates
    edge = tuple(sorted((src_id, tgt_id)))
    edge_type_match = zx_edge_type == "SIMPLE" if ang.get_edge_type(src_id, tgt_id) == EdgeType.IDENTITY else "HADAMARD"

    # Assume failure
    edge_success = False

    # Write edge information if available
    if not second_pass and winner_path_standard_pass and edge_type_match:
        # Log as success
        edge_success = True

        # Update edge paths
        edge_paths[edge] = {
            "src_tgt_ids": (src_id, tgt_id),
            "path_coordinates": winner_path_standard_pass.coords_in_path,
            "path_nodes": winner_path_standard_pass.all_nodes_in_path,
            "edge_type": zx_edge_type,
        }

        # Update source cube info
        nx_g.nodes[src_id]["completed"] += 1

        # Update target cube info
        nx_g.nodes[tgt_id]["coords"] = winner_path_standard_pass.tgt_coords
        nx_g.nodes[tgt_id]["kind"] = winner_path_standard_pass.tgt_kind
        nx_g.nodes[tgt_id]["completed"] += 1
        nx_g.nodes[tgt_id]["beams"] = (
            []
            if nx_g.nodes[tgt_id]["completed"] >= get_node_degree(nx_g, tgt_id)
            else winner_path_standard_pass.tgt_beams
        )
        nx_g.nodes[tgt_id]["beams_short"] = (
            []
            if nx_g.nodes[tgt_id]["completed"] >= get_node_degree(nx_g, tgt_id)
            else winner_path_standard_pass.tgt_beams_short
        )

        # Add path to position to list of graphs' occupied coordinates
        all_coords_in_path = get_taken_coords(winner_path_standard_pass.all_nodes_in_path)
        taken.extend(all_coords_in_path)

        # TODO: update beams

        # Realise the target node
        target_cube = ang.realise_node(
            node = tgt_id,
            kind = CubeKind[winner_path_standard_pass.tgt_kind.upper()],
            position = winner_path_standard_pass.tgt_coords
        )

        # Prepare the proposal
        proposal = Path(
            source_cube = ang.get_cube(src_id), target_cube = target_cube,
            edge_type = ang.get_edge_type(src_id, tgt_id),
            extra_cubes = [
                (CubeKind[winner_path_standard_pass.all_nodes_in_path[idx][1].upper()],
                 winner_path_standard_pass.all_nodes_in_path[idx][0])
                for idx in range(0, len(winner_path_standard_pass.all_nodes_in_path), 2)
            ],
            proposed_pipes = [
                EdgeType.HADAMARD if 'h' in winner_path_standard_pass.all_nodes_in_path[idx][1] else EdgeType.IDENTITY
                for idx in range(1, len(winner_path_standard_pass.all_nodes_in_path), 2)
            ]
        )

        # Update the ANG with the path realising the edge
        ang.realise_edge(
            source = src_id,
            target = tgt_id,
            proposal = proposal
        )

    elif second_pass and winner_path_second_pass and edge_type_match:
        # Log as success
        edge_success = True

        # Update edge paths
        edge_paths[edge] = {
            "src_tgt_ids": (src_id, tgt_id),
            "path_coordinates": [p[0] for p in winner_path_second_pass],
            "path_nodes": winner_path_second_pass,
            "edge_type": zx_edge_type,
        }

        # Update source cube info
        nx_g.nodes[src_id]["completed"] += 1

        # Update target cube info
        nx_g.nodes[tgt_id]["completed"] += 1
        nx_g.nodes[tgt_id]["beams"] = (
            []
            if nx_g.nodes[tgt_id]["completed"] >= get_node_degree(nx_g, tgt_id)
            else nx_g.nodes[tgt_id]["beams"]
        )
        nx_g.nodes[tgt_id]["beams_short"] = (
            []
            if nx_g.nodes[tgt_id]["completed"] >= get_node_degree(nx_g, tgt_id)
            else nx_g.nodes[tgt_id]["beams_short"]
        )

        # Add path to position to list of taken coordinates
        all_coords_in_path = get_taken_coords(winner_path_second_pass)
        taken.extend(all_coords_in_path)

        # TODO: update the beams

        # Prepare the proposed path
        proposed_path = Path(
            source_cube = ang.get_cube(src_id), target_cube = ang.get_cube(tgt_id),
            edge_type = ang.get_edge_type(src_id, tgt_id),
            extra_cubes = [
                (CubeKind[winner_path_second_pass[idx][1].upper()],
                 winner_path_second_pass[idx][0])
                for idx in range(0, len(winner_path_second_pass), 2)
            ],
            proposed_pipes = [
                EdgeType.HADAMARD if 'h' in winner_path_second_pass[idx][1] else EdgeType.IDENTITY
                for idx in range(1, len(winner_path_second_pass), 2)
            ]
        )

        # Update the ANG with the path realising the edge
        ang.realise_edge(
            source = src_id,
            target = tgt_id,
            proposal = proposed_path
        )

    # Fill edge_paths with error if no paths available
    else:
        edge_paths[edge] = {
            "src_tgt_ids": "error",
            "path_coordinates": "error",
            "path_nodes": "error",
            "edge_type": "error",
        }

    # Prune beams before moving to next edge
    nx_g = prune_beams(nx_g, ang.occupied)
    return nx_g, taken, edge_paths, edge_success


def prune_beams(nx_g: nx.Graph, taken: Iterable[StandardCoord]) -> nx.Graph:
    """Remove beams that have already been broken.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        taken: A list of coordinates taken by any blocks placed as a result of previous operations.

    Returns:
        nx_g: Updated nx_graph with pruned beams.

    """

    try:
        for n_id in nx_g.nodes():
            new_beams = []
            new_beams_short = []
            if nx_g.nodes[n_id]["completed"] == []:
                pass
            elif nx_g.nodes[n_id]["completed"] >= get_node_degree(nx_g, n_id):
                nx_g.nodes[n_id]["beams"] = []
                nx_g.nodes[n_id]["beams_short"] = []
            else:
                old_beams = nx_g.nodes[n_id]["beams"]
                old_beams_short = nx_g.nodes[n_id]["beams_short"]

                if old_beams:
                    for single_beam in old_beams:
                        if not any([single_beam.contains(coord) for coord in taken]):
                            new_beams += [single_beam]
                    nx_g.nodes[n_id]["beams"] = new_beams

                if old_beams_short:
                    for single_beam_short in old_beams_short:
                        if not any([single_beam_short.contains(coord) for coord in taken]):
                            new_beams_short += [single_beam_short]
                    nx_g.nodes[n_id]["beams_short"] = new_beams_short

    except (IndexError, ValueError, LookupError, KeyError):
        pass

    return nx_g


########################
# GRAPH MANAGER OUTPUT #
########################
def reindex_path_dict(
    edge_paths: dict, fix_errors: bool = False
) -> tuple[dict[int, StandardBlock], dict[tuple[int, int], list[str]]]:
    """Distil an edge_path object into a list of nodes/blocks and edges/pipes for the space-time diagram.

    This function converts an edge_paths dictionary into a list of cubes and pipes in the final space-time diagram.
    The function can be called during the process to output a temporary snapshot of progress, or at the end
    to produce final results. If used to convert objects for visualisation, the `fix_errors` optional parameter
    should be used to avoid indexing errors. However, using the `fix_errors` parameter in final outputs
    might conceal errors one would want to see explicitly.

    Args:
        edge_paths: A dictionary containing a number of edge paths, i.e., full paths between two blocks, each path made of 3D blocks and pipes.
        fix_errors: When True, exclude any errors from edge_paths before processing.

    Returns:
        lat_nodes: The nodes/blocks of the resulting space-time diagram / lattice surgery (without redundant blocks)
        lat_edges: The edges/pipes of the resulting space-time diagram / lattice surgery (without redundant pipes)

    """

    # Exclude any errors in edge_paths
    if fix_errors is True:
        new_edge_paths = {
            key: path_data
            for key, path_data in edge_paths.items()
            if isinstance(path_data["src_tgt_ids"], tuple)
        }
        edge_paths = new_edge_paths

    max_id = 0
    idx_paths = {}
    for path in edge_paths.values():
        max_id = max(max_id, path["src_tgt_ids"][0], path["src_tgt_ids"][1])
    nxt_id = max_id + 1

    for path in edge_paths.values():
        idxd_path = {}
        key_1, key_2 = path["src_tgt_ids"]
        path_nodes = path["path_nodes"]

        idxd_path[key_1] = path_nodes[0]

        for i in range(1, len(path_nodes) - 1):
            mid_node = path_nodes[i]
            idxd_path[nxt_id] = mid_node
            nxt_id += 1

        if len(path_nodes) > 1:
            idxd_path[key_2] = path_nodes[-1]

        idx_paths[(key_1, key_2)] = idxd_path

    final_edges = {}
    for orig_key, path_id_value_map in idx_paths.items():
        n_ids = list(path_id_value_map.keys())
        b_ids = []
        e_ids = []

        for i in range(len(n_ids)):
            if i % 2 == 0:
                b_ids.append(n_ids[i])
            else:
                e_ids.append(n_ids[i])

        if len(b_ids) >= 2:
            for i in range(len(b_ids) - 1):
                n1 = b_ids[i]
                n2 = b_ids[i + 1]
                e_type = path_id_value_map[e_ids[i]][1]

                final_edges[(n1, n2)] = [e_type, orig_key]

    lat_nodes: dict[int, StandardBlock] = {}
    lat_edges: dict[tuple[int, int], list[str]] = {}
    for path in idx_paths.values():
        keys = list(path.keys())
        i = 0
        for key, info in path.items():
            if i % 2 == 0:
                lat_nodes[key] = info
            else:
                e_key = (keys[i - 1], keys[i + 1])
                e_type = info[1]
                lat_edges[e_key] = [e_type, final_edges[e_key][1]]
            i += 1

    return lat_nodes, lat_edges


############
# MISC/AUX #
############
def rm_temp_files(temp_dir_path: Path):
    """Remove any temporary files created during run."""
    try:
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path)
    except (ValueError, FileNotFoundError) as e:
        print("Unable to delete temp files or temp folder does not exist", e)
