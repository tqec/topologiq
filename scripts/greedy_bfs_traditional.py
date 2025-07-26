import random
import networkx as nx
from collections import deque
from typing import Tuple, List, Optional, Any, cast

from utils.utils_greedy_bfs import (
    check_for_exits,
    generate_tentative_target_positions,
    prune_all_beams,
)

from utils.utils_zx_graphs import (
    zx_types_validity_checks,
    get_type_family,
    get_zx_type_from_kind,
)

from utils.classes import (
    PathBetweenNodes,
    StandardBlock,
    NodeBeams,
    SimpleDictGraph,
    StandardCoord,
    Colors,
)
from scripts.pathfinder import (
    run_bfs_for_all_potential_target_nodes,
    get_coords_occupied_by_blocks,
)
from utils.grapher import visualise_3d_graph, make_graph_from_edge_paths


####################
# WORKFLOW MANAGER #
####################
def main(
    graph: SimpleDictGraph,
    circuit_name: str = "circuit",
    hide_boundaries: bool = False,
    visualise: str = "outer",
    **kwargs,
) -> Tuple[nx.Graph, dict, nx.Graph, int]:
    """Manages the generalities of the BFS process.

    Args:
        - graph: a ZX circuit as a simple dictionary of nodes and edges.
        - circuit_name: name of ZX circuit.
        - hide_boundaries:
            - true: instructs the algorithm to use boundary nodes but do not display them in visualisation,
            - false: boundary nodes are factored into the process and shown on visualisation.
        - visualise:
            - detail: visualises each iteration by outer/general BFS loop,
            - final: visualises only the final result,
            - all: visualises each iteration step by outer/general BFS loop and final result.

    Keyword arguments (**kwargs):
        - weights: weights for the value function to pick best of many paths.
        - length_of_beams: length of each of the beams coming out of open nodes.
        - max_search_space: maximum size of 3D space to generate paths for.

    Returns:
        - nx_graph: a nx_graph with the nodes and edges in the incoming ZX graph formatted to facilitate positioning of 3D blocks and pipes,
            updated regularly over the course of the process.
        - edge_paths: the raw set of 3D edges found (with redundant blocks for start and end positions of some edges),
            updated regularly over the course of the process.
        - new_nx_graph: a nx_graph containing the 3D blocks and pipes that have been placed successfully at any given point in time,
            used mainly for visualisations.
        - c: a counter for the number of top-level iterations by BFS (used to organise visualisations),
            updated regularly over the course of the process.

    """

    # KEY VARIABLES
    # Take a ZX graph and prepare a fresh 3D graph with positions set to None
    nx_graph = prepare_graph(graph)

    # Arrays/dicts to track coordinates
    occupied_coords: List[StandardCoord] = []
    all_beams: List[NodeBeams] = []
    edge_paths: dict = {}

    # VALIDITY CHECKS
    if not zx_types_validity_checks(graph):
        print(Colors.RED + "Graph validity checks failed. Aborting." + Colors.RESET)
        return (nx_graph, edge_paths, nx.Graph(), 0)

    # BFS management
    start_node: Optional[int] = _find_start_node_id(nx_graph)
    queue: deque = deque([start_node])
    visited: set = {start_node}

    # SPECIAL PROCESS FOR CENTRAL NODE
    # Terminate if there is no start node
    if start_node is None:
        print(Colors.RED + "Graph has no nodes." + Colors.RESET)
        return nx_graph, edge_paths, nx.Graph(), 0

    # Place start node at origin
    else:
        print(Colors.BLUE + "\nPlacing first node." + Colors.RESET)
        # Get kind from type family
        possible_kinds: Optional[List[str]] = nx_graph.nodes[start_node].get(
            "type_family"
        )
        randomly_chosen_kind = random.choice(possible_kinds) if possible_kinds else None

        # Write info of node
        nx_graph.nodes[start_node]["pos"] = (0, 0, 0)
        nx_graph.nodes[start_node]["kind"] = randomly_chosen_kind

        # Update occupied_coords and all_beams with node's position & beams
        occupied_coords.append((0, 0, 0))
        _, start_node_beams = check_for_exits(
            (0, 0, 0),
            randomly_chosen_kind,
            occupied_coords,
            all_beams,
            kwargs["length_of_beams"],
        )
        all_beams.append(start_node_beams)
        print(f"{start_node}: ((0, 0, 0) '{randomly_chosen_kind}').")

    # LOOP FOR ALL OTHER NODES
    c = 0  # Visualiser counter (needed to save snapshots to file)
    while queue:

        # Get current parent node
        current_parent_node: int = queue.popleft()

        # Iterate over neighbours of current parent node
        for neigh_node_id in cast(List[int], nx_graph.neighbors(current_parent_node)):

            # Queue and add to visited set if BFS just arrived at node
            if neigh_node_id not in visited:
                visited.add(neigh_node_id)
                queue.append(neigh_node_id)

                # Ensure occupied_coords has unique entries each run
                occupied_coords = list(set(occupied_coords))

                # Try to place blocks as close to one another as as possible
                step: int = 3
                while step <= 18:
                    occupied_coords, all_beams, edge_paths, successful_placement = (
                        place_next_block(
                            current_parent_node,
                            neigh_node_id,
                            nx_graph,
                            occupied_coords,
                            all_beams,
                            edge_paths,
                            step=step,
                            **kwargs,
                        )
                    )

                    # For visualisation purposes, on each step,
                    # create a new graph from edge_paths
                    if edge_paths:
                        if c < int(len(edge_paths)):
                            if list(edge_paths.values())[-1]["path_nodes"] != "error":

                                # Create graph from existing edges
                                new_nx_graph = make_graph_from_edge_paths(edge_paths)

                                # Create visualisation
                                if visualise == "detail" or visualise == "all":
                                    visualise_3d_graph(
                                        new_nx_graph, hide_boundaries=hide_boundaries
                                    )
                                visualise_3d_graph(
                                    new_nx_graph,
                                    hide_boundaries=hide_boundaries,
                                    save_to_file=True,
                                    filename=f"{circuit_name}{c:03d}",
                                )

                                c = len(edge_paths)

                    # Move to next if there is a succesful placement
                    if successful_placement:
                        break

                    # Increase distance between nodes if placement not possible
                    step += 3

    # SINCE IT WAS USED EXTENSIVELY DURING LOOP
    # ENSURE OCCUPIED COORDS ARE UNIQUE
    occupied_coords = list(set(occupied_coords))

    # RUN OVER GRAPH AGAIN IN CASE SOME EDGES WHERE NOT BUILT AS A RESULT OF MAIN LOOP
    edge_paths, c = second_pass(
        nx_graph,
        occupied_coords,
        edge_paths,
        circuit_name,
        c,
        visualise=visualise,
        **kwargs,
    )

    # CREATE A NEW GRAPH FROM FINAL EDGE PATHS RETURNS FROM ALL THE BOVE
    new_nx_graph = make_graph_from_edge_paths(edge_paths)

    # RETURN THE GRAPHS AND EDGE PATHS FOR ANY SUBSEQUENT USE
    return nx_graph, edge_paths, new_nx_graph, c


def second_pass(
    nx_graph: nx.Graph,
    occupied_coords: List[StandardCoord],
    edge_paths: dict,
    circuit_name: str,
    c: int,
    visualise: str = "outer",
    **kwargs,
) -> Tuple[dict, int]:
    """Undertakes a second pass of the graph to process any edges missed by the original BFS,
    which typically happens when there are multiple interconnected nodes.

    Args:
        - nx_graph: a nx_graph containing all nodes and edges in incoming ZX graph,
            formatted to facilitate positioning of 3D blocks and pipes,
            and updated regularly over the course of the process.
        - occupied_coords: list of coordinates occupied by any blocks/pipes placed as a result of previous operations.
        - edge_paths: the raw set of 3D edges found by the algorithm (with redundant blocks for start and end positions of some edges)
        - circuit_name: name of ZX circuit.
        - c: a counter for the number of top-level iterations by BFS (used to organise visualisations)
        - visualise:
            - detail: visualises each iteration by outer/general BFS loop,
            - final: visualises only the final result,
            - all: visualises each iteration step by outer/general BFS loop and final result.

    Keyword arguments (**kwargs):
        - weights: weights for the value function to pick best of many paths.
        - length_of_beams: length of each of the beams coming out of open nodes.
        - max_search_space: maximum size of 3D space to generate paths for.

    Returns:
        - edge_paths: updated raw set of 3D edges found by the algorithm (with redundant blocks for start and end positions of some edges)
        - c: updated counter for the number of top-level iterations by BFS (used to organise visualisations)

    """

    # BASE ALL OPERATIONS ON EDGES FROM GRAPH
    for u, v, data in nx_graph.edges(data=True):

        # Ensure occupied coords do not have duplicates
        occupied_coords = list(set(occupied_coords))

        # Get source and target node for specific edge
        u_pos = nx_graph.nodes[u].get("pos")
        v_pos = nx_graph.nodes[v].get("pos")

        # Process only if both nodes have been placed on grid already
        if u_pos is None or v_pos is None:
            # Not yet implemented. Will catch instances where some nodes are not directly connected with other nodes in circuit
            # (i.e., graphs with subgraphs).
            edge = tuple(sorted((u, v)))

        if u_pos is not None and v_pos is not None:
            # Update visualiser counter
            c += 1

            # Format adjustments to match existing operations
            u_kind = nx_graph.nodes[u].get("kind")
            v_zx_type = nx_graph.nodes[v].get("type")
            v_kind = nx_graph.nodes[v].get("kind")
            edge = tuple(sorted((u, v)))

            # Call pathfinder on any graph edge that does not have an entry in edge_paths
            if edge not in edge_paths:

                print(
                    Colors.BLUE + "\nFinding path between placed nodes." + Colors.RESET,
                    f"Node ID: {u} ({(u_pos, u_kind)}) <--> Node ID: {v} {v_pos, v_kind}",
                )

                # Check if edge is hadamard
                original_zx_edge_type = nx_graph.get_edge_data(u, v).get("type")
                hadamard_flag: bool = (
                    True if original_zx_edge_type == "HADAMARD" else False
                )
                print("Original ZX type for edge:", original_zx_edge_type)

                # Call pathfinder using optional parameters to tell the pathfinding algorithm
                # to work in pure pathfinding (rather than path creation) mode
                clean_paths = run_pathfinder(
                    (u_pos, u_kind),
                    v_zx_type,
                    3,
                    occupied_coords[:],
                    target_node_info=(v_pos, nx_graph.nodes[v].get("kind")),
                    hadamard_flag=hadamard_flag,
                    **kwargs,
                )

                # Write to edge_paths if an edge is found
                if clean_paths:

                    # Update user
                    print("\rPath found:", clean_paths[0])

                    # Update edge paths
                    coords_in_path = [
                        entry[0] for entry in clean_paths[0]
                    ]  # Take the first path
                    edge_type = data.get("type", "SIMPLE")
                    edge_paths[edge] = {
                        "src_tgt_ids": (u, v),
                        "path_coordinates": coords_in_path,
                        "path_nodes": clean_paths[0],
                        "edge_type": edge_type,
                    }

                    # Add path to position to list of graphs' occupied positions
                    full_coords_to_add = get_coords_occupied_by_blocks(clean_paths[0])
                    occupied_coords.extend(full_coords_to_add)

                    # CREATE A NEW GRAPH FROM FINAL EDGE PATHS RETURNS FROM ABOVE
                    new_nx_graph = make_graph_from_edge_paths(edge_paths)

                    # VISUALISE NEW EDGE
                    if visualise == "detail" or visualise == "all":
                        visualise_3d_graph(new_nx_graph)
                    visualise_3d_graph(
                        new_nx_graph,
                        save_to_file=True,
                        filename=f"{circuit_name}{c:03d}",
                    )

                # Write an error to edge_paths if edge not found
                else:
                    print("Path not found.")
                    edge_paths[edge] = {
                        "src_tgt_ids": "error",
                        "path_coordinates": "error",
                        "path_nodes": "error",
                        "edge_type": "error",
                    }

    # RETURN EDGE PATHS FOR FINAL CONSUMPTION
    return edge_paths, c


#######################
# CORE BFS OPERATIONS #
#######################
def prepare_graph(graph: SimpleDictGraph) -> nx.Graph:
    """Takes a simple dictionary of nodes and edges representing a ZX graph and formats all elements
    in a way that facilitates subsequent positioning of 3D blocks and pipes, without, in doing so, adding any
    information to the outgoing graph.

    Args:
        - graph: a ZX circuit as a simple dictionary of nodes and edges.

    Returns:
        - nx_graph: a nx_graph containing all nodes and edges in incoming ZX graph,
            formatted to facilitate positioning of 3D blocks and pipes.

    """

    # PREPARE EMPTY NETWORKX GRAPH
    nx_graph = nx.Graph()

    # GET NODES AND EDGES FROM INCOMING ZX GRAPH
    nodes_data: List[Tuple[int, str]] = graph.get("nodes", [])
    edges_data: List[Tuple[Tuple[int, int], str]] = graph.get("edges", [])

    # ADD NODES TO NETWORKX GRAPH
    for node_id, node_type in nodes_data:
        nx_graph.add_node(
            node_id,
            type=node_type,
            type_family=get_type_family(node_type),
            kind=None,
            pos=None,
        )

    # ADD EDGES TO NETWORKX GRAPH
    for (u, v), edge_type in edges_data:
        nx_graph.add_edge(u, v, type=edge_type)

    # IDENTIFY THE NODES WITH MORE THAN 4 CONNECTIONS
    all_nodes = list(nx_graph.nodes())
    high_degree_nodes = [
        node for node in all_nodes if _get_node_degree(nx_graph, node) > 4
    ]

    # BREAK ANY NODES WITH MORE THAN 4 CONNECTIONS
    if high_degree_nodes:

        # Determine max degree
        max_node_id = max(nx_graph.nodes) if nx_graph.nodes else 0

        # Loop over max nodes and break as appropriate
        i = 0
        while i < 100:

            # List of high degree nodes
            all_nodes_loop = list(nx_graph.nodes())
            high_degree_nodes = [
                node for node in all_nodes_loop if _get_node_degree(nx_graph, node) > 4
            ]

            # Exit loop when no nodes with more than 4 edges
            if not high_degree_nodes:
                break

            # Pick a high degree node
            node_to_sanitise = random.choice(high_degree_nodes)
            original_node_type = nx_graph.nodes[node_to_sanitise]["type"]

            # Add a twin
            max_node_id += 1
            twin_node_id = max_node_id
            nx_graph.add_node(
                twin_node_id,
                type=original_node_type,
                type_family=get_type_family(original_node_type),
                kind=None,
                pos=None,
            )
            nx_graph.add_edge(node_to_sanitise, twin_node_id, type="SIMPLE")

            # Distributed edges across twins
            neighbors = list(nx_graph.neighbors(node_to_sanitise))
            neighbors = [n for n in neighbors if n != twin_node_id]

            degree_to_move = _get_node_degree(nx_graph, node_to_sanitise) // 2

            moved_count = 0
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if (
                    moved_count >= degree_to_move
                    or _get_node_degree(nx_graph, node_to_sanitise) <= 4
                ):
                    break
                if nx_graph.has_edge(
                    node_to_sanitise, neighbor
                ) and not nx_graph.has_edge(twin_node_id, neighbor):
                    edge_data = nx_graph.get_edge_data(node_to_sanitise, neighbor)
                    edge_type = edge_data.get("type", None)
                    nx_graph.add_edge(twin_node_id, neighbor, type=edge_type)
                    nx_graph.remove_edge(node_to_sanitise, neighbor)
                    moved_count += 1

    # RETURN THE NETWORKX GRAPH
    return nx_graph


def place_next_block(
    source_node_id: int,
    neigh_node_id: int,
    nx_graph: nx.Graph,
    occupied_coords: List[StandardCoord],
    all_beams: List[NodeBeams],
    edge_paths: dict,
    step: int = 3,
    stage: float = 0.5,
    **kwargs,
) -> Tuple[List[StandardCoord], List[NodeBeams], dict, bool]:
    """Takes care of positioning nodes in the 3D space as part of the main BFS flow. The function does not explicitly create the paths,
    this is the responsibility of the inner *pathfinder* algorithm. However, the function generates a number of tentative positions
    and calls the pathfinder for each of these positions, to be able to return a best path from many.

    Args:
        - source_node_id: the ID of the source node, i.e., the one that has already been placed in the 3D space as part of previous operations.
        - neigh_node_id: the ID of the neighbouring or next node, i.e., the one that needs to be placed in the 3D space.
        - nx_graph: a nx_graph containing the nodes and edges in the incoming ZX graph formatted to facilitate placements in the 3D space,
            updated regularly over the course of the process.
        - occupied_coords: list of coordinates occupied by any blocks/pipes placed as a result of previous operations.
        - all_beams: list of coordinates occupied by the beams of all blocks in original ZX-graph.
        - edge_paths: the raw set of 3D edges found by the algorithm (with redundant blocks for start and end positions of some edges).
        - step: intended (Manhattan) distance between origin and target blocks.
        - stage (not in use): may eventually be used to determine if algorithm is at the start, middle, or end of a given circuit.

    Keyword arguments (**kwargs):
        - weights: weights for the value function to pick best of many paths.
        - length_of_beams: length of each of the beams coming out of open nodes.
        - max_search_space: maximum size of 3D space to generate paths for.

    Returns:
        - occupied_coords: updated list of coordinates occupied by any blocks/pipes placed as a result of previous operations.
        - all_beams: updated list of coordinates occupied by the beams of all blocks in original ZX-graph.
        - edge_paths: updated raw set of 3D edges found by the algorithm (with redundant blocks for start and end positions of some edges).
        - bool:
            - True: placement succesful
            - False: placement not succesful

    """

    # PRUNE BEAMS TO CONSIDER RECENT NODE PLACEMENTS
    all_beams = prune_all_beams(all_beams, occupied_coords)

    # EXTRACT STANDARD INFO APPLICABLE TO ALL NODES
    # Previous node data
    source_pos: Optional[StandardCoord] = nx_graph.nodes[source_node_id].get("pos")
    source_kind: Optional[str] = nx_graph.nodes[source_node_id].get("kind")

    if source_pos is None or source_kind is None:
        return occupied_coords, all_beams, edge_paths, False
    source_node: StandardBlock = (source_pos, source_kind)

    # Current node data
    next_neigh_node_data = nx_graph.nodes[neigh_node_id]
    next_neigh_zx_type: str = next_neigh_node_data.get("type")
    next_neigh_edge_n = int(_get_node_degree(nx_graph, neigh_node_id))
    next_neigh_pos: Optional[StandardCoord] = nx_graph.nodes[neigh_node_id].get("pos")

    # DEAL WITH CASES WHERE NEW NODE NEEDS TO BE ADDED TO GRID
    if next_neigh_pos is None:
        print(
            Colors.BLUE + "\nCreating path." + Colors.RESET,
            f"Node ID (source): {source_node_id} {source_node} --> Node ID (target): {neigh_node_id} ((?, ?, ?), ???) (ZX type: {next_neigh_zx_type})",
        )

        # Current edge data
        original_zx_edge_type = nx_graph.get_edge_data(
            source_node_id, neigh_node_id
        ).get("type")
        hadamard_flag: bool = True if original_zx_edge_type == "HADAMARD" else False
        print("Original ZX type for edge:", original_zx_edge_type)

        # Remove source coordinate from occupied coords
        occupied_coords_redux = occupied_coords[:]
        if source_pos in occupied_coords_redux:
            occupied_coords_redux.remove(source_pos)

        # Get clean candidate paths
        clean_paths = run_pathfinder(
            source_node,
            next_neigh_zx_type,
            step,
            occupied_coords_redux if occupied_coords else [],
            hadamard_flag=hadamard_flag,
            **kwargs,
        )

        # Assemble a preliminary dictionary of viable paths
        viable_paths = []
        for clean_path in clean_paths:

            target_coords, target_kind = clean_path[-1]
            target_unobstructed_exits_n, target_node_beams = check_for_exits(
                target_coords,
                target_kind,
                occupied_coords_redux,
                all_beams,
                length_of_beams=kwargs["length_of_beams"],
            )

            # Reset # of unobstructed exits and node beams if node is a boundary
            if next_neigh_zx_type == "O":
                target_unobstructed_exits_n, target_node_beams = (6, [])

            if target_unobstructed_exits_n >= next_neigh_edge_n:
                coords_in_path = [entry[0] for entry in clean_path]
                beams_broken_by_path = 0
                for beam in all_beams:
                    for coord in beam:
                        if coord in coords_in_path:
                            beams_broken_by_path += 1

                all_nodes_in_path = [entry for entry in clean_path]
                if next_neigh_zx_type == "O":
                    target_kind = "ooo"

                    all_nodes_in_path[-1] = (all_nodes_in_path[-1][0], target_kind)

                path_data = {
                    "target_pos": target_coords,
                    "target_kind": target_kind,
                    "target_beams": target_node_beams,
                    "coords_in_path": coords_in_path,
                    "all_nodes_in_path": all_nodes_in_path,
                    "beams_broken_by_path": beams_broken_by_path,
                    "len_of_path": len(clean_path),
                    "target_unobstructed_exits_n": target_unobstructed_exits_n,
                }

                viable_paths.append(PathBetweenNodes(**path_data))

        print(
            f"\r{len(viable_paths)} viable paths created. {'Algorithm picks best' if len(clean_paths)>1 else ''}."
        )

        winner_path: Optional[PathBetweenNodes] = None
        if viable_paths:
            winner_path = max(
                viable_paths, key=lambda path: path.weighed_value(stage, **kwargs)
            )

        # Rewrite current node with data of winner candidate
        if winner_path:

            # Update user
            readable_chosen_path = [
                (block[0], get_zx_type_from_kind(block[1]))
                for block in winner_path.all_nodes_in_path
            ]
            readable_chosen_path = [
                (
                    block
                    if len(block[1]) == 1 or block[1] == "BOUNDARY"
                    else (f"{block[1]} EDGE")
                )
                for block in readable_chosen_path
            ]
            print("Chosen path:", readable_chosen_path)

            # Update node information
            nx_graph.nodes[neigh_node_id]["pos"] = winner_path.target_pos
            nx_graph.nodes[neigh_node_id]["kind"] = winner_path.target_kind

            # Update edge_path dictionary
            edge = tuple(sorted((source_node_id, neigh_node_id)))
            edge_type = nx_graph.get_edge_data(source_node_id, neigh_node_id).get(
                "type", "SIMPLE"
            )  # Default to "SIMPLE" if type is not found
            edge_paths[edge] = {
                "src_tgt_ids": (source_node_id, neigh_node_id),
                "path_coordinates": winner_path.coords_in_path,
                "path_nodes": winner_path.all_nodes_in_path,
                "edge_type": edge_type,
            }

            # Add path to position to list of graphs' occupied positions
            full_coords_to_add = get_coords_occupied_by_blocks(
                winner_path.all_nodes_in_path
            )
            occupied_coords.extend(full_coords_to_add)

            # Add beams of winner's target node to list of graphs' all_beams
            all_beams.append(winner_path.target_beams)

            # Return updated occupied_coords and all_beams, with success code
            return occupied_coords, all_beams, edge_paths, True

        # Handle cases where no winner is found
        if not winner_path:

            # Explicit warning
            print(Colors.RED + "Could not find path." + Colors.RESET)

            # Fill edge_path with error (allows process to move on but error is easy to spot)
            edge = tuple(sorted((source_node_id, neigh_node_id)))
            edge_paths[edge] = {
                "src_tgt_ids": "error",
                "path_coordinates": "error",
                "path_nodes": "error",
                "edge_type": "error",
            }

            # Return unchanged occupied_coords and all_beams, with failure boolean
            return occupied_coords, all_beams, edge_paths, False

    # FAIL SAFE RETURN TO AVOID TYPE ERRORS
    return occupied_coords, all_beams, edge_paths, False


def run_pathfinder(
    previous_node_info: StandardBlock,
    next_neigh_zx_type: str,
    initial_step: int,
    occupied_coords: List[StandardCoord],
    target_node_info: Optional[StandardBlock] = None,
    hadamard_flag: bool = False,
    **kwargs,
) -> List[Any]:
    """Calls the inner pathfinder algorithm for a combination of source node and potential target position,
    with optional parameters to send the information of a target node that was already placed as part of previous operations.

    Args:
        - previous_node_info: the information of the source node including its position in the 3D space and its kind,
        - next_neigh_zx_type: the ZX type of the block that needs to be connected to the node already in the 3D space,
            which can be overriden by the optional parameter *target_node_info*.
        - initial_step: intended (Manhattan) distance between source and target blocks.
        - occupied_coords: list of coordinates occupied by any blocks/pipes placed as a result of previous operations.
        - target_node_info: optional parameter to send the information of a node that has already been placed in the 3D space,
            which overrides *next_neigh_zx_type* and tells the inner pathfinder algorithm that it is finding a path between existing blocks
            as opposed to creating a path between an existing block a new one to be placed at a tentative position.
        - hadamard_flag: a flag to tell the inner pathfinding algorithm that this edge is a Hadamard edge,
            which gets handled differently depending on the characteristics of the edge.

    Keyword arguments (**kwargs):
        - weights: weights for the value function to pick best of many paths.
        - length_of_beams: length of each of the beams coming out of open nodes.
        - max_search_space: maximum size of 3D space to generate paths for.

    Returns:
        - clean_paths: a list of 3D blocks and pipes needed to connect source and target node in the 3D space in a topologically-correct manner

    """

    # ARRAYS TO HOLD TEMPORARY PATHS
    path = []
    valid_paths = []
    clean_paths = []

    # STEP, START, & TARGET COORDS
    step = initial_step
    start_coords, _ = previous_node_info

    target_coords = None
    target_type = None
    if target_node_info:
        target_coords, target_type = target_node_info

    # COPY OCCUPIED COORDS TO AVOID OVERWRITES BY EXTERNAL FUNCTIONS
    occupied_coords_copy = occupied_coords[:]
    if start_coords in occupied_coords_copy:
        occupied_coords_copy.remove(start_coords)
    if target_coords:
        occupied_coords_copy.remove(target_coords)

    # FIND VIABLE PATHS
    # One step at a time, call separate path finding (in a 4D space) BFS algorithm
    while step <= 18:

        # Generate tentative positions for current step or use target node
        if target_node_info:
            tentative_positions = [target_coords]
        else:
            tentative_positions = generate_tentative_target_positions(
                start_coords,
                step,
                occupied_coords,  # Real occupied coords: position cannot overlap start node
            )

        # Try finding path to each tentative positions
        for position in tentative_positions:
            path_found, _, path, all_paths_from_round = (
                run_bfs_for_all_potential_target_nodes(
                    previous_node_info,
                    next_neigh_zx_type,
                    step,
                    attempts_per_distance=1,
                    occupied_coords=occupied_coords_copy,
                    overwrite_target_node=(position, target_type),
                    hadamard_flag=hadamard_flag,
                )
            )

            # Append any found paths to valid_paths
            if path_found:
                valid_paths.append(all_paths_from_round)

        # Break if valid paths generated at step

        if valid_paths and step >= kwargs["max_search_space"]:
            break

        # Increase distance if no valid paths found at current step
        step += 3

    # REMOVE PATHS THAT INTERSECT WITH EXISTING CUBES/PIPES
    if target_node_info:
        if valid_paths and valid_paths[0]:
            clean_path = valid_paths[0][0]
            if occupied_coords_copy:
                for node in clean_path:
                    if (
                        node[0] in occupied_coords_copy
                    ):  # Needs the full occupied coords: path *will* contain source
                        return []
            return [clean_path]
        else:
            return []

    else:
        if valid_paths:
            for all_paths in valid_paths:
                remove_flag = False
                for path in all_paths:
                    if occupied_coords_copy:
                        for node in path:
                            if (
                                node[0] in occupied_coords_copy
                            ):  # Copy occupied coords: path *will* contain source
                                remove_flag = True
                    if remove_flag == False:
                        clean_paths.append(path)

    # RETURN CLEAN PATHS OR EMPTY LIST IF NO VIABLE PATHS FOUND
    return clean_paths


##################
# AUX OPERATIONS #
##################
def _find_start_node_id(nx_graph: nx.Graph) -> Optional[int]:
    """Picks a node from an nx graph based on its centrality, in the context of this algorithm, for use as starting node for a BFS.

    Args:
        - nx_graph: an nx Graph.

    Returns:
        - start_node: the node with highest closeness centrality or a random selection from a list of highest degree nodes.

    """

    # TERMINATE IF THERE ARE NO NODES
    if not nx_graph.nodes:
        return None

    # LOOP OVER NODES FINDING NODES WITH HIGHEST DEGREE
    max_degree = -1
    central_nodes: List[int] = []

    node_degrees = nx_graph.degree
    if isinstance(node_degrees, int):
        print(
            "Warning: nx_graph.degree() returned an integer. Cannot determine start node."
        )
        return None  # Cannot iterate, return None
    else:
        for node, degree in node_degrees:
            if degree > max_degree:
                max_degree = degree
                central_nodes = [node]
            elif degree == max_degree:
                central_nodes.append(node)

    # PICK A HIGHEST DEGREE NODE, RANDOMLY BUT FAVOURING CENTRALITY
    if central_nodes:

        # Bias selection to slightly favour centrality when several high-degree nodes exist
        c_centrality = nx.closeness_centrality(nx_graph)
        max_centrality = max([c_centrality.get(node, -1) for node in central_nodes])
        max_nodes = sum(
            [
                1
                for node in central_nodes
                if c_centrality.get(node, -1) == max_centrality
            ]
        )
        if max_nodes != len(central_nodes):
            central_node = max(
                central_nodes, key=lambda node: c_centrality.get(node, -1)
            )
            central_nodes.append(central_node)

        # Include all high-degree nodes in potential list of start nodes
        start_node: Optional[int] = random.choice(central_nodes)

    else:
        start_node: Optional[int] = None

    # RETURN START NODE
    return start_node


def _get_node_degree(graph: nx.Graph, node: int) -> int:
    """Gets the degree (# of edges) of a given node.

    Args:
        - graph: an nx Graph.
        - node: the node of interest.

    Returns:
        - int: the degree for the node of interest, or 0 if graph has no edges.

    """

    # GET DEGREES FOR THE ENTIRE GRAPH
    degrees = graph.degree

    # GET DEGREE FOR NODE OF INTEREST
    if not isinstance(degrees, int) and hasattr(degrees, "__getitem__"):
        return degrees[node]

    # IF DEGREES NOT A LIST, RETURN 0 (SINGLE NODE WON'T HAVE EDGES)
    return 0
