import random
import networkx as nx

from collections import deque
from datetime import datetime
from typing import Tuple, List, Optional, Any, Union, cast

from utils.utils_greedy_bfs import (
    check_for_exits,
    gen_tent_tgt_coords,
    prune_beams,
    reindex_pth_dict,
)
from utils.utils_zx_graphs import validate_zx_types, get_zx_type_fam, kind_to_zx_type
from utils.classes import (
    PathBetweenNodes,
    StandardBlock,
    NodeBeams,
    SimpleDictGraph,
    StandardCoord,
    Colors,
)
from scripts.pathfinder import pthfinder, get_taken_coords
from utils.grapher import vis_3d_g, edge_pths_to_g
from utils.utils_misc import log_stats_to_file


####################
# WORKFLOW MANAGER #
####################
def graph_manager_bfs(
    g: SimpleDictGraph,
    c_name: str = "circuit",
    hide_ports: bool = False,
    visualise: Tuple[Union[None, str], Union[None, str]] = (None, None),
    log_stats: bool = False,
    **kwargs,
) -> Tuple[
    nx.Graph,
    dict,
    nx.Graph,
    int,
    Union[None, dict[int, StandardBlock]],
    Union[None, dict[Tuple[int, int], List[str]]],
]:
    """Manages the generalities of the BFS process.

    Args:
        - g: a ZX circuit as a simple dictionary of nodes and edges.
        - c_name: name of ZX circuit.
        - hide_ports:
            - true: instructs the algorithm to use boundary nodes but do not display them in visualisation,
            - false: boundary nodes are factored into the process and shown on visualisation.
        - visualise: a tuple with visualisation settings:
            - visualise[0]:
                - None: no visualisation whatsoever,
                - "final" (str): triggers a single on-screen visualisation of the final result (small performance trade-off),
                - "detail" (str): triggers on-screen visualisation for each edge in the original ZX graph (medium performance trade-off).
            - visualise[1]:
                - None: no animation whatsoever,
                - "GIF": saves step-by-step visualisation of the process in GIF format (huge performance trade-off),
                - "MP4": saves a PNG of each step/edge in the visualisation process and joins them into a GIF at the end (huge performance trade-off).
        - log_stats: boolean to determine if to log stats to CSV files in `.assets/stats/`.
            - True: log stats to file
            - False: do NOT log stats to file

    Keyword arguments (**kwargs):
        - weights: weights for the value function to pick best of many paths.
        - length_of_beams: length of each of the beams coming out of open nodes.
        - max_search_space: maximum size of 3D space to generate paths for.

    Returns:
        - nx_g: a nx_graph with the nodes and edges in the incoming ZX graph formatted to facilitate positioning of 3D blocks and pipes,
            updated regularly over the course of the process.
        - edge_pths: the raw set of 3D edges found (with redundant blocks for start and end positions of some edges),
            updated regularly over the course of the process.
        - new_nx_g: a nx_graph containing the 3D blocks and pipes that have been placed successfully at any given point in time,
            used mainly for visualisations.
        - c: a counter for the number of top-level iterations by BFS (used to organise visualisations),
            updated regularly over the course of the process.

    """

    # PRELIMS
    # Turn on logging of stats if needed
    t1 = None
    t2 = None
    unique_run_id = None
    if log_stats:
        t1 = datetime.now()
        unique_run_id = t1.strftime("%Y%m%d_%H%M%S_%f") if log_stats else None

    # Variables to hold results
    nx_g = prep_3d_g(g)
    lat_nodes: Union[None, dict[int, StandardBlock]] = None
    lat_edges: Union[None, dict[Tuple[int, int], List[str]]] = None

    # BFS management
    src: Optional[int] = _find_src_id(nx_g)
    taken: List[StandardCoord] = []
    all_beams: List[NodeBeams] = []
    edge_pths: dict = {}

    queue: deque = deque([src])
    visited: set = {src}

    # VALIDITY CHECKS
    if not validate_zx_types(g):
        print(Colors.RED + "Graph validity checks failed. Aborting." + Colors.RESET)
        return (nx_g, edge_pths, nx.Graph(), 0, lat_nodes, lat_edges)

    # SPECIAL PROCESS FOR CENTRAL NODE
    # Terminate if there is no start node
    if src is None:
        print(Colors.RED + "Graph has no nodes." + Colors.RESET)
        return nx_g, edge_pths, nx.Graph(), 0, lat_nodes, lat_edges

    # Place start node at origin
    else:

        # Get kind from type family
        tent_kinds: Optional[List[str]] = nx_g.nodes[src].get("type_fam")
        random_kind = random.choice(tent_kinds) if tent_kinds else None

        # Write info of node
        nx_g.nodes[src]["pos"] = (0, 0, 0)
        nx_g.nodes[src]["kind"] = random_kind

        # Update list of taken coords and all_beams with node's position & beams
        taken.append((0, 0, 0))
        _, src_beams = check_for_exits(
            (0, 0, 0),
            random_kind,
            taken,
            all_beams,
            kwargs["length_of_beams"],
        )
        all_beams.append(src_beams)

    # LOOP FOR ALL OTHER NODES
    c = 0  # Visualiser counter (needed to save snapshots to file)
    while queue:

        # Get current parent node
        curr_parent: int = queue.popleft()

        # Iterate over neighbours of current parent node
        for neigh_id in cast(List[int], nx_g.neighbors(curr_parent)):

            # Queue and add to visited set if BFS just arrived at node
            if neigh_id not in visited:
                visited.add(neigh_id)
                queue.append(neigh_id)

                # Ensure the list of taken coords has unique entries on each run
                taken = list(set(taken))

                # Try to place blocks as close to one another as as possible
                step: int = 3
                while step <= 18:
                    taken, all_beams, edge_pths, edge_success = place_nxt_block(
                        curr_parent,
                        neigh_id,
                        nx_g,
                        taken,
                        all_beams,
                        edge_pths,
                        init_step=step,
                        log_stats_id=unique_run_id,
                        **kwargs,
                    )

                    # For visualisation purposes, on each step,
                    # create a new graph from edge_pths
                    if edge_pths:
                        if c < int(len(edge_pths)):
                            if list(edge_pths.values())[-1]["pth_nodes"] != "error":

                                # Create graph from existing edges
                                new_nx_g = edge_pths_to_g(edge_pths)

                                # Create visualisation
                                if visualise[0]:
                                    if visualise[0].lower() == "detail":
                                        vis_3d_g(new_nx_g, hide_ports=hide_ports)

                                # Save visualisation for later animation
                                if visualise[1]:
                                    if visualise[1] == "GIF" or visualise[1] == "MP4":
                                        vis_3d_g(
                                            new_nx_g,
                                            hide_ports=hide_ports,
                                            save_to_file=True,
                                            filename=f"{c_name}{c:03d}",
                                        )

                                c = len(edge_pths)

                    # Move to next if there is a succesful placement
                    if edge_success:
                        break

                    # Increase distance between nodes if placement not possible
                    step += 3

    # SINCE IT WAS USED EXTENSIVELY DURING LOOP
    # ENSURE OCCUPIED COORDS ARE UNIQUE
    taken = list(set(taken))

    # RUN OVER GRAPH AGAIN IN CASE SOME EDGES WHERE NOT BUILT AS A RESULT OF MAIN LOOP
    if log_stats:
        t2 = datetime.now()
    edge_pths, c, num_2n_pass_edges = second_pass(
        nx_g,
        taken,
        edge_pths,
        c_name,
        c,
        hide_ports=hide_ports,
        visualise=visualise,
        log_stats_id=unique_run_id,
    )

    # ASSEMBLE FINAL LATTICE SURGERY IF NO ERRORS
    errors_in_result = False
    for _, edge_pth in edge_pths.items():
        if edge_pth["edge_type"] == "error":
            errors_in_result = True
            break

    if errors_in_result is False:
        lat_nodes, lat_edges = reindex_pth_dict(edge_pths)
        new_nx_g = edge_pths_to_g(edge_pths)
    else:
        new_nx_g = nx.Graph()

    # LOG STATS TO FILE IF NEEDED
    if log_stats:
        if t1 and t2:
            t_end = datetime.now()
            duration_first_pass = (t2 - t1).total_seconds()
            duration_second_pass = (t_end - t1).total_seconds()
            duration_total = (t_end - t1).total_seconds()

            nodes_in_input = len(nx_g.nodes)
            edges_in_input = len(nx_g.edges)
            blocks_in_output = len(lat_nodes.keys()) if lat_nodes else 0
            edges_in_output = len(lat_edges.keys()) if lat_edges else 0
            num_normal_edges = edges_in_input - num_2n_pass_edges

            bfs_manager_stats = [
                unique_run_id,
                c_name,
                lat_nodes is not None and lat_edges is not None,
                nodes_in_input,
                edges_in_input,
                num_normal_edges,
                num_2n_pass_edges,
                blocks_in_output,
                edges_in_output,
                duration_first_pass,
                duration_second_pass,
                duration_total,
            ]

            log_stats_to_file(bfs_manager_stats, f"bfs_manager")

    # RETURN THE GRAPHS AND EDGE PATHS FOR ANY SUBSEQUENT USE
    return nx_g, edge_pths, new_nx_g, c, lat_nodes, lat_edges


def second_pass(
    nx_g: nx.Graph,
    taken: List[StandardCoord],
    edge_pths: dict,
    c_name: str,
    c: int,
    hide_ports: bool = False,
    visualise: Tuple[Union[None, str], Union[None, str]] = (None, None),
    log_stats_id: Union[str, None] = None,
) -> Tuple[dict, int, int]:
    """Undertakes a second pass of the graph to process any edges missed by the original BFS,
    which typically happens when there are multiple interconnected nodes.

    Args:
        - nx_g: a nx_graph containing all nodes and edges in incoming ZX graph,
            formatted to facilitate positioning of 3D blocks and pipes,
            and updated regularly over the course of the process.
        - taken: list of coordinates occupied by any blocks/pipes placed as a result of previous operations.
        - edge_pths: the raw set of 3D edges found by the algorithm (with redundant blocks for start and end positions of some edges)
        - c_name: name of ZX circuit.
        - c: a counter for the number of top-level iterations by BFS (used to organise visualisations)
        - visualise: a tuple with visualisation settings:
            - visualise[0]:
                - None: no visualisation whatsoever,
                - "final" (str): triggers a single on-screen visualisation of the final result (small performance trade-off),
                - "detail" (str): triggers on-screen visualisation for each edge in the original ZX graph (medium performance trade-off).
            - visualise[1]:
                - None: no animation whatsoever,
                - "GIF": saves step-by-step visualisation of the process in GIF format (huge performance trade-off),
                - "MP4": saves a PNG of each step/edge in the visualisation process and joins them into a GIF at the end (huge performance trade-off).
        - log_stats_id: unique identifier for logging stats to CSV files in `.assets/stats/` (`None` keeps logging is off).

    Keyword arguments (**kwargs):
        - weights: weights for the value function to pick best of many paths.
        - length_of_beams: length of each of the beams coming out of open nodes.
        - max_search_space: maximum size of 3D space to generate paths for.

    Returns:
        - edge_pths: updated raw set of 3D edges found by the algorithm (with redundant blocks for start and end positions of some edges)
        - c: updated counter for the number of top-level iterations by BFS (used to organise visualisations)

    """

    # BASE ALL OPERATIONS ON EDGES FROM GRAPH
    num_2n_pass_edges = 0
    for u, v, data in nx_g.edges(data=True):

        # Ensure occupied coords do not have duplicates
        taken = list(set(taken))

        # Get source and target node for specific edge
        u_coords = nx_g.nodes[u].get("pos")
        v_coords = nx_g.nodes[v].get("pos")

        if u_coords is not None and v_coords is not None:

            # Update visualiser counter
            c += 1

            # Format adjustments to match existing operations
            u_kind = nx_g.nodes[u].get("kind")
            v_zx_type = nx_g.nodes[v].get("type")
            edge = tuple(sorted((u, v)))

            # Call pathfinder on any graph edge that does not have an entry in edge_pths
            if edge not in edge_pths:

                # Update edge counter
                num_2n_pass_edges += 1

                # Check if edge is hadamard
                zx_edge_type = nx_g.get_edge_data(u, v).get("type")
                hdm: bool = True if zx_edge_type == "HADAMARD" else False

                # Call pathfinder using optional parameters to tell the pathfinding algorithm
                # to work in pure pathfinding (rather than path creation) mode
                clean_pths = run_pthfinder(
                    (u_coords, u_kind),
                    v_zx_type,
                    3,
                    taken[:],
                    tgt=(v_coords, nx_g.nodes[v].get("kind")),
                    hdm=hdm,
                )

                # Write to edge_pths if an edge is found
                if clean_pths:

                    # Update edge paths
                    coords_in_pth = [p[0] for p in clean_pths[0]]  # Take the first path
                    edge_type = data.get("type", "SIMPLE")
                    edge_pths[edge] = {
                        "src_tgt_ids": (u, v),
                        "pth_coordinates": coords_in_pth,
                        "pth_nodes": clean_pths[0],
                        "edge_type": edge_type,
                    }

                    # Add path to position to list of graphs' occupied positions
                    all_coords_in_pth = get_taken_coords(clean_pths[0])
                    taken.extend(all_coords_in_pth)

                    # Create graph from existing edges
                    new_nx_g = edge_pths_to_g(edge_pths)

                    if log_stats_id:
                        print(f"Path discovery: {u} -> {v}. SUCCESS.")

                    # Create visualisation
                    if visualise[0]:
                        if visualise[0].lower() == "detail":
                            vis_3d_g(new_nx_g, hide_ports=hide_ports)

                    # Save visualisation for later animation
                    if visualise[1]:
                        if visualise[1] == "GIF" or visualise[1] == "MP4":
                            vis_3d_g(
                                new_nx_g,
                                hide_ports=hide_ports,
                                save_to_file=True,
                                filename=f"{c_name}{c:03d}",
                            )

                # Write an error to edge_pths if edge not found
                else:
                    edge_pths[edge] = {
                        "src_tgt_ids": "error",
                        "pth_coordinates": "error",
                        "pth_nodes": "error",
                        "edge_type": "error",
                    }

                    if log_stats_id:
                        print(f"Path discovery: {u} -> {v}. FAIL.")

    # RETURN EDGE PATHS FOR FINAL CONSUMPTION
    return edge_pths, c, num_2n_pass_edges


#######################
# CORE BFS OPERATIONS #
#######################
def prep_3d_g(g: SimpleDictGraph) -> nx.Graph:
    """Takes a simple dictionary of nodes and edges representing a ZX graph and formats all elements
    in a way that facilitates subsequent positioning of 3D blocks and pipes, without, in doing so, adding any
    information to the outgoing graph.

    Args:
        - g: a ZX circuit as a simple dictionary of nodes and edges.

    Returns:
        - nx_g: a nx_graph containing all nodes and edges in incoming ZX graph,
            formatted to facilitate positioning of 3D blocks and pipes.

    """

    # PREPARE EMPTY NETWORKX GRAPH
    nx_g = nx.Graph()

    # GET NODES AND EDGES FROM INCOMING ZX GRAPH
    nodes: List[Tuple[int, str]] = g.get("nodes", [])
    edges: List[Tuple[Tuple[int, int], str]] = g.get("edges", [])

    # ADD NODES TO NETWORKX GRAPH
    for n_id, n_type in nodes:
        nx_g.add_node(
            n_id,
            type=n_type,
            type_fam=get_zx_type_fam(n_type),
            kind=None,
            pos=None,
        )

    # ADD EDGES TO NETWORKX GRAPH
    for (u, v), e_type in edges:
        nx_g.add_edge(u, v, type=e_type)

    # IDENTIFY THE NODES WITH MORE THAN 4 CONNECTIONS
    all_nodes = list(nx_g.nodes())
    centr_nodes = [n for n in all_nodes if _get_node_degree(nx_g, n) > 4]

    # BREAK ANY NODES WITH MORE THAN 4 CONNECTIONS
    if centr_nodes:

        # Determine max degree
        centr_node = max(nx_g.nodes) if nx_g.nodes else 0

        # Loop over max nodes and break as appropriate
        i = 0
        while i < 100:

            # List of high degree nodes
            all_nodes_loop = list(nx_g.nodes())
            centr_nodes = [n for n in all_nodes_loop if _get_node_degree(nx_g, n) > 4]

            # Exit loop when no nodes with more than 4 edges
            if not centr_nodes:
                break

            # Pick a high degree node
            node_to_sanitise = random.choice(centr_nodes)
            orig_node_type = nx_g.nodes[node_to_sanitise]["type"]

            # Add a twin
            centr_node += 1
            twin_node_id = centr_node
            nx_g.add_node(
                twin_node_id,
                type=orig_node_type,
                type_fam=get_zx_type_fam(orig_node_type),
                kind=None,
                pos=None,
            )
            nx_g.add_edge(node_to_sanitise, twin_node_id, type="SIMPLE")

            # Distributed edges across twins
            neighs = list(nx_g.neighbors(node_to_sanitise))
            neighs = [n for n in neighs if n != twin_node_id]

            degree_to_shuffle = _get_node_degree(nx_g, node_to_sanitise) // 2

            shuffle_c = 0
            random.shuffle(neighs)

            for neigh in neighs:
                if (
                    shuffle_c >= degree_to_shuffle
                    or _get_node_degree(nx_g, node_to_sanitise) <= 4
                ):
                    break
                if nx_g.has_edge(node_to_sanitise, neigh) and not nx_g.has_edge(
                    twin_node_id, neigh
                ):
                    edge_data = nx_g.get_edge_data(node_to_sanitise, neigh)
                    edge_type = edge_data.get("type", None)
                    nx_g.add_edge(twin_node_id, neigh, type=edge_type)
                    nx_g.remove_edge(node_to_sanitise, neigh)
                    shuffle_c += 1

    # RETURN THE NETWORKX GRAPH
    return nx_g


def place_nxt_block(
    src_id: int,
    neigh_id: int,
    nx_g: nx.Graph,
    taken: List[StandardCoord],
    all_beams: List[NodeBeams],
    edge_pths: dict,
    init_step: int = 3,
    log_stats_id: Union[str, None] = None,
    **kwargs,
) -> Tuple[List[StandardCoord], List[NodeBeams], dict, bool]:
    """Takes care of positioning nodes in the 3D space as part of the outer (graph manager) BFS flow. The function does not explicitly create the paths,
    this is the responsibility of the inner *pathfinder* algorithm. However, the function generates a number of tentative positions
    and calls the pathfinder for each of these positions, to be able to return a best path from many.

    Args:
        - src_id: the ID of the source node, i.e., the one that has already been placed in the 3D space as part of previous operations.
        - neigh_id: the ID of the neighbouring or next node, i.e., the one that needs to be placed in the 3D space.
        - nx_g: a nx_graph containing the nodes and edges in the incoming ZX graph formatted to facilitate placements in the 3D space,
            updated regularly over the course of the process.
        - taken: list of coordinates occupied by any blocks/pipes placed as a result of previous operations.
        - all_beams: list of coordinates occupied by the beams of all blocks in original ZX graph.
        - edge_pths: the raw set of 3D edges found by the algorithm (with redundant blocks for start and end positions of some edges).
        - init_step: intended (Manhattan) distance between origin and target blocks.
        - log_stats_id: unique identifier for logging stats to CSV files in `.assets/stats/` (`None` keeps logging is off).

    Keyword arguments (**kwargs):
        - weights: weights for the value function to pick best of many paths.
        - length_of_beams: length of each of the beams coming out of open nodes.
        - max_search_space: maximum size of 3D space to generate paths for.

    Returns:
        - taken: updated list of coordinates occupied by any blocks/pipes placed as a result of previous operations.
        - all_beams: updated list of coordinates occupied by the beams of all blocks in original ZX graph.
        - edge_pths: updated raw set of 3D edges found by the algorithm (with redundant blocks for start and end positions of some edges).
        - bool:
            - True: placement succesful
            - False: placement not succesful

    """

    # PRUNE BEAMS TO CONSIDER RECENT NODE PLACEMENTS
    all_beams = prune_beams(all_beams, taken)

    # EXTRACT STANDARD INFO APPLICABLE TO ALL NODES
    # Previous node data
    src_coords: Optional[StandardCoord] = nx_g.nodes[src_id].get("pos")
    src_kind: Optional[str] = nx_g.nodes[src_id].get("kind")

    if src_coords is None or src_kind is None:
        return taken, all_beams, edge_pths, False
    src: StandardBlock = (src_coords, src_kind)

    # Current node data
    nxt_neigh_node_data = nx_g.nodes[neigh_id]
    nxt_neigh_zx_type: str = nxt_neigh_node_data.get("type")
    nxt_neigh_neigh_n = int(_get_node_degree(nx_g, neigh_id))
    nxt_neigh_coords: Optional[StandardCoord] = nx_g.nodes[neigh_id].get("pos")

    # DEAL WITH CASES WHERE NEW NODE NEEDS TO BE ADDED TO GRID
    if nxt_neigh_coords is None:

        # Current edge data
        zx_edge_type = nx_g.get_edge_data(src_id, neigh_id).get("type")
        hdm: bool = True if zx_edge_type == "HADAMARD" else False

        # Remove source coordinate from occupied coords
        taken_coords_c = taken[:]
        if src_coords in taken_coords_c:
            taken_coords_c.remove(src_coords)

        # Get clean candidate paths
        clean_pths = run_pthfinder(
            src,
            nxt_neigh_zx_type,
            init_step,
            taken_coords_c if taken else [],
            hdm=hdm,
            log_stats_id=log_stats_id,
        )

        # Assemble a preliminary dictionary of viable paths
        viable_pths = []
        for clean_pth in clean_pths:
            tgt_coords, tgt_kind = clean_pth[-1]
            tgt_unobstr_exit_n, tgt_beams = check_for_exits(
                tgt_coords,
                tgt_kind,
                taken_coords_c,
                all_beams,
                beams_len=kwargs["length_of_beams"],
            )

            # Reset # of unobstructed exits and node beams if node is a boundary
            if nxt_neigh_zx_type == "O":
                tgt_unobstr_exit_n, tgt_beams = (6, [])

            if tgt_unobstr_exit_n >= nxt_neigh_neigh_n:
                coords_in_pth = [entry[0] for entry in clean_pth]
                beams_broken_by_pth = 0
                for beam in all_beams:
                    for coord in beam:
                        if coord in coords_in_pth:
                            beams_broken_by_pth += 1

                all_nodes_in_pth = [entry for entry in clean_pth]
                if nxt_neigh_zx_type == "O":
                    tgt_kind = "ooo"

                    all_nodes_in_pth[-1] = (all_nodes_in_pth[-1][0], tgt_kind)

                pth_data = {
                    "tgt_pos": tgt_coords,
                    "tgt_kind": tgt_kind,
                    "tgt_beams": tgt_beams,
                    "coords_in_pth": coords_in_pth,
                    "all_nodes_in_pth": all_nodes_in_pth,
                    "beams_broken_by_pth": beams_broken_by_pth,
                    "len_of_pth": len(clean_pth),
                    "tgt_unobstr_exit_n": tgt_unobstr_exit_n,
                }

                viable_pths.append(PathBetweenNodes(**pth_data))

        winner_pth: Optional[PathBetweenNodes] = None
        if viable_pths:
            winner_pth = max(viable_pths, key=lambda pth: pth.weighed_value(**kwargs))

        # Rewrite current node with data of winner candidate
        if winner_pth:

            # Update user
            pretty_winner_pth = [
                (block[0], kind_to_zx_type(block[1]))
                for block in winner_pth.all_nodes_in_pth
            ]
            pretty_winner_pth = [
                (
                    block
                    if len(block[1]) == 1 or block[1] == "BOUNDARY"
                    else (f"{block[1]} EDGE")
                )
                for block in pretty_winner_pth
            ]

            # Update node information
            nx_g.nodes[neigh_id]["pos"] = winner_pth.tgt_pos
            nx_g.nodes[neigh_id]["kind"] = winner_pth.tgt_kind

            # Update edge_pth dictionary
            edge = tuple(sorted((src_id, neigh_id)))
            edge_type = nx_g.get_edge_data(src_id, neigh_id).get(
                "type", "SIMPLE"
            )  # Default to "SIMPLE" if type is not found
            edge_pths[edge] = {
                "src_tgt_ids": (src_id, neigh_id),
                "pth_coordinates": winner_pth.coords_in_pth,
                "pth_nodes": winner_pth.all_nodes_in_pth,
                "edge_type": edge_type,
            }

            # Add path to position to list of graphs' occupied positions
            all_coords_in_pth = get_taken_coords(winner_pth.all_nodes_in_pth)
            taken.extend(all_coords_in_pth)

            # Add beams of winner's target node to list of graphs' all_beams
            all_beams.append(winner_pth.tgt_beams)

            if log_stats_id:
                print(f"Path creation: {src_id} -> {neigh_id}. SUCCESS.")

            # Return updated list of taken coords and all_beams, with success code
            return taken, all_beams, edge_pths, True

        # Handle cases where no winner is found
        if not winner_pth:

            # Fill edge_pth with error (allows process to move on but error is easy to spot)
            edge = tuple(sorted((src_id, neigh_id)))
            edge_pths[edge] = {
                "src_tgt_ids": "error",
                "pth_coordinates": "error",
                "pth_nodes": "error",
                "edge_type": "error",
            }

            # Explicit warning
            if log_stats_id:
                print(f"Path creation: {src_id} -> {neigh_id}. FAIL.")

            # Return unchanged list of taken coords and all_beams, with failure boolean
            return taken, all_beams, edge_pths, False

    # FAIL SAFE RETURN TO AVOID TYPE ERRORS
    return taken, all_beams, edge_pths, False


def run_pthfinder(
    src: StandardBlock,
    nxt_zx_type: str,
    init_step: int,
    taken: List[StandardCoord],
    tgt: Optional[StandardBlock] = None,
    hdm: bool = False,
    log_stats_id: Union[str, None] = None,
) -> List[Any]:
    """Calls the inner pathfinder algorithm for a combination of source node and potential target position,
    with optional parameters to send the information of a target node that was already placed as part of previous operations.

    Args:
        - src: the information of the source node including its position in the 3D space and its kind,
        - nxt_zx_type: the ZX type of the block that needs to be connected to the node already in the 3D space,
            which can be overriden by the optional parameter *tgt*.
        - init_step: intended (Manhattan) distance between source and target blocks.
        - taken: list of coordinates occupied by any blocks/pipes placed as a result of previous operations.
        - tgt: optional parameter to send the information of a node that has already been placed in the 3D space,
            which overrides *nxt_zx_type* and tells the inner pathfinder algorithm that it is finding a path between existing blocks
            as opposed to creating a path between an existing block a new one to be placed at a tentative position.
        - hdm: a flag to tell the inner pathfinding algorithm that this edge is a Hadamard edge,
            which gets handled differently depending on the characteristics of the edge.
        - log_stats_id: unique identifier for logging stats to CSV files in `.assets/stats/` (`None` keeps logging is off).

    Returns:
        - clean_pths: a list of 3D blocks and pipes needed to connect source and target node in the 3D space in a topologically-correct manner

    """

    # ARRAYS TO HOLD TEMPORARY PATHS
    valid_pths: Union[dict[StandardBlock, List[StandardBlock]], None] = None
    clean_pths = []

    # STEP, START, & TARGET COORDS
    step = init_step
    src_coords, _ = src
    tgt_coords, tgt_type = tgt if tgt else (None, None)

    # COPY OCCUPIED COORDS TO AVOID OVERWRITES BY EXTERNAL FUNCTIONS
    taken_cc = taken[:]
    if src_coords in taken_cc:
        taken_cc.remove(src_coords)
    if tgt_coords:
        taken_cc.remove(tgt_coords)

    # FIND VIABLE PATHS
    # Pathfinder BFS loop
    max_step = 2 * init_step if tgt else 9
    while step <= max_step:

        # Generate tentative positions for current step or use target node
        if tgt_coords:
            tent_coords = [tgt_coords]
        else:
            tent_coords = gen_tent_tgt_coords(
                src_coords,
                step,
                taken,  # Real occupied coords: position cannot overlap start node
            )

        # Try finding path to each tentative positions
        valid_pths = pthfinder(
            src,
            tent_coords,
            nxt_zx_type,
            taken=taken_cc,
            tgt=(tent_coords[0], tgt_type),
            hdm=hdm,
            log_stats_id=log_stats_id,
        )

        # Append usable paths to clean paths
        if valid_pths:
            for path in valid_pths.values():
                pth_checks = True
                for node in path:
                    if node[0] in taken_cc:
                        pth_checks = False
                if pth_checks:
                    clean_pths.append(path)

        # Break if valid paths generated at step
        if clean_pths:
            break

        # Increase distance if no valid paths found at current step
        step += 3

    # RETURN CLEAN PATHS OR EMPTY LIST IF NO VIABLE PATHS FOUND
    return clean_pths


##################
# AUX OPERATIONS #
##################
def _find_src_id(nx_g: nx.Graph) -> Optional[int]:
    """Picks a node from an nx graph based on its centrality, in the context of this algorithm, for use as starting node for a BFS.

    Args:
        - nx_g: an nx_graph.

    Returns:
        - src_id: the ID of the node with highest closeness centrality or a random selection from a list of highest degree nodes.

    """

    # TERMINATE IF THERE ARE NO NODES
    if not nx_g.nodes:
        return None

    # LOOP OVER NODES FINDING NODES WITH HIGHEST DEGREE
    max_d = -1
    centr_nodes: List[int] = []

    nodes_ds = nx_g.degree
    if isinstance(nodes_ds, int):
        print(
            "Warning: nx_g.degree() returned an integer. Cannot determine start node."
        )
        return None  # Cannot iterate, return None
    else:
        for n, d in nodes_ds:
            if d > max_d:
                max_d = d
                centr_nodes = [n]
            elif d == max_d:
                centr_nodes.append(n)

    # PICK A HIGHEST DEGREE NODE, RANDOMLY BUT FAVOURING CENTRALITY
    if centr_nodes:

        # Bias selection to slightly favour centrality when several high-degree nodes exist
        c_centrality = nx.closeness_centrality(nx_g)
        max_centrality = max([c_centrality.get(n, -1) for n in centr_nodes])
        max_nodes = sum(
            [1 for n in centr_nodes if c_centrality.get(n, -1) == max_centrality]
        )
        if max_nodes != len(centr_nodes):
            centr_node = max(centr_nodes, key=lambda n: c_centrality.get(n, -1))
            centr_nodes.append(centr_node)

        # Include all high-degree nodes in potential list of start nodes
        src_id: Optional[int] = random.choice(centr_nodes)

    else:
        src_id: Optional[int] = None

    # RETURN START NODE
    return src_id


def _get_node_degree(g: nx.Graph, node: int) -> int:
    """Gets the degree (# of edges) of a given node.

    Args:
        - g: an nx Graph.
        - node: the node of interest.

    Returns:
        - int: the degree for the node of interest, or 0 if graph has no edges.

    """

    # GET DEGREES FOR THE ENTIRE GRAPH
    degrees = g.degree

    # GET DEGREE FOR NODE OF INTEREST
    if not isinstance(degrees, int) and hasattr(degrees, "__getitem__"):
        return degrees[node]

    # IF DEGREES NOT A LIST, RETURN 0 (SINGLE NODE WON'T HAVE EDGES)
    return 0
