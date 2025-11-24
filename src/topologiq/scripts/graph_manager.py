"""
Manages the main/outer graph manager BFS process. 

This file contains functions that altogether determine the first spider to process,
the order in which subsequent spiders get processed, and all operations needed to 
determine potential/tentative placements, call the inner pathfinder algorithm to get
paths to the tentative placements, and select a winner path from amongst all paths.

Usage:
    Call `graph_manager_bfs()` programmatically from a separate script. 

Notes:
    For now, none of the functions in this file are to be called individually.
    In the future, some of the functions could be called by variant algorithms that 
        do not necessarily need or want to implement all separate features.
"""

import random
import networkx as nx
import matplotlib.figure

from datetime import datetime
from collections import deque
from typing import Tuple, List, Optional, Any, Union, cast

from topologiq.scripts.pathfinder import pathfinder, get_taken_coords
from topologiq.utils.animation import create_animation
from topologiq.utils.utils_greedy_bfs import (
    find_first_id,
    get_node_degree,
    gen_tent_tgt_coords,
    prune_beams,
    reindex_path_dict,
)
from topologiq.utils.utils_pathfinder import check_exits
from topologiq.utils.utils_zx_graphs import check_zx_types, get_zx_type_fam, kind_to_zx_type
from topologiq.utils.grapher_common import lattice_to_g
from topologiq.utils.grapher import vis_3d
from topologiq.utils.utils_misc import prep_stats_n_log
from topologiq.utils.classes import (
    PathBetweenNodes,
    StandardBlock,
    NodeBeams,
    SimpleDictGraph,
    StandardCoord,
    Colors,
)


###############################
# MAIN GRAPH MANAGER WORKFLOW #
###############################
def graph_manager_bfs(
    simple_graph: SimpleDictGraph,
    circuit_name: str = "circuit",
    min_succ_rate: int = 50,
    hide_ports: bool = False,
    vis_options: Tuple[Union[None, str], Union[None, str]] = (None, None),
    log_stats_id: Union[str, None] = None,
    debug: int = 0,
    fig_data: Optional[matplotlib.figure.Figure] = None,
    first_cube: Tuple[Union[int, None], Union[str, None]] = (None, None),
    **kwargs,
) -> Tuple[
    nx.Graph,
    dict,
    int,
    Union[None, dict[int, StandardBlock]],
    Union[None, dict[Tuple[int, int], List[str]]],
]:
    """Process all nodes/edges in the input ZX graph and select best paths.

    This function manages a greedy Breadth-First-Search (BFS) process that takes care of calling 
    a number of operations that altogether enable the conversion of all spiders and edges in the 
    input ZX graph into corresponding 3D cubes and pipes. It chooses a first spider and places
    its corresponding 3D cube at origin, then manages calls to other functions that place all 
    subsequent spiders in the order given by the nature of the BFS process.

    Args:
        simple_graph: The `simple_graph` form of an arbitrary ZX circuit.
        circuit_name: The name of the ZX circuit.
        min_succ_rate (optional): Minimum % of tentative coordinates that must be filled for each edge.
        hide_ports (optional): If True, boundary spiders are considered by Topologiq but not displayed in visualisations.
        vis_options (optional): Visualisation settings provided as a Tuple.
            vis_options[0]: If enabled, triggers "final" or "detail" visualisations.
                (None): No visualisation.
                (str) "final" | "detail": A single visualisation of the final result or one visualisation per completed edge.
            vis_options[1]: If enabled, triggers creation of an animated summary for the entire process.
                (None): No animation.
                (str) "GIF" | "MP4": A step-by-step visualisation of the process in GIF or MP4 format.
        log_stats_id (optional): A unique datetime-based identifier for the purposes of logging stats for an specific run.
        debug (optional): Debug mode (0: off, 1: graph manager, 2: pathfinder, 3: pathfinder w. discarded paths).
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).
        first_cube (optional): the ID and kind of the first cube to place in 3D space (used to replicate specific cases).

    Keyword arguments (**kwargs):
        weights: A tuple (int, int) of weights used to pick the best of several paths when there are several valid alternatives.
        length_of_beams: The length of each of the beams coming out of cubes still needing connections at any given point in time.

    Returns:
        nx_g: A nx_graph with the same spiders/edges as incoming ZX graph but in 3D-amicable format/structure.
        edge_paths: An edge-by-edge/block-by-block summary of the space-time diagram Topologiq builds.
        c: a counter for the number of completed (top-level) iterations by the main loop in this function (used to organise visualisations).

    """

    # Preliminaries
    # Set all timers to None in case there are errors
    t1 = None
    t2 = None
    t_end = None

    # Start outer timer if log_stats is on
    if log_stats_id is not None:
        t1 = datetime.now()

    # Key parameters - Graph & outputs
    nx_g = prep_3d_g(simple_graph)
    num_nodes_input: int = 0
    num_1st_pass_edges: int = 0

    lat_nodes: Union[None, dict[int, StandardBlock]] = None
    lat_edges: Union[None, dict[Tuple[int, int], List[str]]] = None
    
    # Key parameters - BFS management
    first_id: Optional[int] = find_first_id(nx_g) if first_cube[0] is None else first_cube[0]
    taken: List[StandardCoord] = []
    all_beams: List[NodeBeams] = []
    edge_paths: dict = {}
    queue: deque[int] = deque([first_id])
    visited: set = {first_id}

    # Validity checks
    if not check_zx_types(simple_graph):
        print(Colors.RED + "Graph validity checks failed. Aborting." + Colors.RESET)
        return (nx_g, edge_paths, 0, lat_nodes, lat_edges)

    if first_id is None:
        print(Colors.RED + "Graph has no nodes." + Colors.RESET)
        return nx_g, edge_paths, 0, lat_nodes, lat_edges

    # Place first spider at origin
    else:

        # Get kind from type family
        tentative_kinds: Optional[List[str]] = nx_g.nodes[first_id].get("type_fam") if first_cube[1] is None else [first_cube[1]]
        first_kind = random.choice(tentative_kinds) if tentative_kinds else None

        # Update list of taken coords and all_beams with node's position & beams
        taken.append((0, 0, 0))
        _, src_beams = check_exits(
            (0, 0, 0),
            first_kind,
            taken,
            all_beams,
            kwargs["length_of_beams"],
        )

        # Write info of node
        nx_g.nodes[first_id]["coords"] = (0, 0, 0)
        nx_g.nodes[first_id]["kind"] = first_kind
        nx_g.nodes[first_id]["beams"] = src_beams

        # Update global beams array
        all_beams.append(src_beams)

        # Update node counter
        num_nodes_input += 1

    # Loop over all other spiders
    c = 0  # Visualiser counter (needed to save snapshots to file)
    while queue:

        # Get current parent node
        src_id: int = queue.popleft()

        # Iterate over neighbours of current parent node
        for tgt_id in cast(List[int], nx_g.neighbors(src_id)):

            # Queue and add to visited set if BFS just arrived at node
            if tgt_id not in visited:
                visited.add(tgt_id)
                queue.append(tgt_id)

                # Ensure the list of taken coords has unique entries on each run
                taken = list(set(taken))

                # Try to place blocks as close to one another as as possible
                step: int = 3
                while step <= 9:

                    taken, all_beams, edge_paths, edge_success = place_nxt_block(
                        src_id,
                        tgt_id,
                        nx_g,
                        taken,
                        all_beams,
                        edge_paths,
                        circuit_name=circuit_name,
                        init_step=step,
                        min_succ_rate=min_succ_rate,
                        hide_ports=hide_ports,
                        vis_options=vis_options,
                        fig_data=fig_data,
                        log_stats_id=log_stats_id,
                        debug=debug,
                        **kwargs,
                    )
                    c = len(edge_paths)

                    # Move to next if there is a succesful placement
                    if edge_success:
                        num_nodes_input += 1
                        num_1st_pass_edges += 1
                        break

                    else:
                        if step == 9:  # If we get here, edge did not complete and algorithm will fail
                            # Create animation of failed attempt
                            if vis_options[1]:
                                create_animation(
                                    filename_prefix=f"FAIL_{circuit_name}",
                                    restart_delay=5000,
                                    duration=2500,
                                    video=True if vis_options[1] == "MP4" else False,
                                )

                            # Log stats of failed attempt
                            if log_stats_id is not None:
                                t_end = datetime.now()
                                times = {"t1": t1, "t2": t2, "t_end": t_end}
                                run_success = False
                                counts = {
                                    "num_input_nodes_processed": num_nodes_input,
                                    "num_input_edges_processed": num_1st_pass_edges,
                                    "num_1st_pass_edges_processed": num_1st_pass_edges,
                                    "num_2n_pass_edges_processed": 0,
                                }

                                prep_stats_n_log(
                                    "graph_manager",
                                    log_stats_id,
                                    run_success,
                                    counts,
                                    times,
                                    circuit_name=circuit_name,
                                    edge_paths=edge_paths,
                                    lat_nodes=lat_nodes,
                                    lat_edges=lat_edges,
                                    run_params={"min_succ_rate": min_succ_rate, **kwargs},
                                )

                            raise ValueError(
                                f"Path creation. Error with edge: {src_id} -> {tgt_id}."
                            )

                    # Increase distance between nodes if placement not possible
                    step += 3

    # Since it is used extensively in loop, remove any redundancies from `taken`
    taken = list(set(taken))

    # Run over graph again in case some edges were not considered in main loop
    num_2n_pass_edges = 0
    try:

        # Start second pass timer
        if log_stats_id is not None:
            t2 = datetime.now()

        # Call second pass on graph
        edge_paths, c, num_2n_pass_edges = second_pass(
            nx_g,
            taken,
            edge_paths,
            c,
            all_beams,
            circuit_name=circuit_name,
            min_succ_rate=min_succ_rate,
            hide_ports=hide_ports,
            vis_options=vis_options,
            log_stats_id=log_stats_id,
            debug=debug,
            fig_data=fig_data,
        )
    except ValueError as e:  # If we get here, algorithm will fail
        # Create animation of failed attempt
        if vis_options[1]:
            create_animation(
                filename_prefix=f"FAIL_{circuit_name}",
                restart_delay=5000,
                duration=2500,
                video=True if vis_options[1] == "MP4" else False,
            )

        # Log stats for failed attempt
        if log_stats_id is not None:
            t_end = datetime.now()
            times = {"t1": t1, "t2": t2, "t_end": t_end}
            run_success = False
            counts = {
                "num_input_nodes_processed": num_nodes_input,
                "num_input_edges_processed": num_1st_pass_edges + num_2n_pass_edges,
                "num_1st_pass_edges_processed": num_1st_pass_edges,
                "num_2n_pass_edges_processed": num_2n_pass_edges,
            }

            prep_stats_n_log(
                "graph_manager",
                log_stats_id,
                run_success,
                counts,
                times,
                circuit_name=circuit_name,
                edge_paths=edge_paths,
                lat_nodes=lat_nodes,
                lat_edges=lat_edges,
                run_params={"min_succ_rate": min_succ_rate, **kwargs},
            )

        # Raise
        raise ValueError(e)

    # If we make it here, all edges completed
    # Assemble final lattice objects
    lat_nodes, lat_edges = reindex_path_dict(edge_paths)

    # Log stats
    if log_stats_id is not None:
        t_end = datetime.now()
        times = {"t1": t1, "t2": t2, "t_end": t_end}
        run_success = lat_nodes is not None and lat_edges is not None
        counts = {
            "num_input_nodes_processed": num_nodes_input,
            "num_input_edges_processed": num_1st_pass_edges + num_2n_pass_edges,
            "num_1st_pass_edges_processed": num_1st_pass_edges,
            "num_2n_pass_edges_processed": num_2n_pass_edges,
        }

        prep_stats_n_log(
            "graph_manager",
            log_stats_id,
            run_success,
            counts,
            times,
            circuit_name=circuit_name,
            edge_paths=edge_paths,
            lat_nodes=lat_nodes,
            lat_edges=lat_edges,
            run_params={"min_succ_rate": min_succ_rate, **kwargs},
        )

    return nx_g, edge_paths, c, lat_nodes, lat_edges


##################
# EDGE RENDERERS #
##################
def place_nxt_block(
    src_id: int,
    tgt_id: int,
    nx_g: nx.Graph,
    taken: List[StandardCoord],
    all_beams: List[NodeBeams],
    edge_paths: dict,
    circuit_name: str = "circuit",
    init_step: int = 3,
    min_succ_rate: int = 60,
    hide_ports: bool = False,
    vis_options: Tuple[Union[None, str], Union[None, str]] = (None, None),
    fig_data: Optional[matplotlib.figure.Figure] = None,
    log_stats_id: Union[str, None] = None,
    debug: int = 0,
    **kwargs,
) -> Tuple[List[StandardCoord], List[NodeBeams], dict, bool]:
    """Position target cube in the 3D space as part of the primary BFS flow.
    
    This function calls the inner pathfinder algorithm on any arbitrary combination of an already-placed
    `src_id` and a yet-to-be-placed `tgt_id`. The inner pathfinder algorithm returns a list of viable 
    paths to a number of valid placements for `tgt_id`, and chooses a best path from this list 
    using hyperparameters passed as `kwargs` and a value function.

    Args:
        src_id: The ID of the source node, i.e., the one that has already been placed in the 3D space as part of previous operations.
        tgt_id: The ID of the neighbouring or next node, i.e., the one that needs to be placed in the 3D space.
        nx_g: A nx_graph with the same spiders/edges as incoming ZX graph but in 3D-amicable format/structure.
            updated regularly over the course of the process.
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        all_beams: A list of coordinates occupied by the beams of already-placed cubes that still require connections.
        edge_paths: An edge-by-edge summary of the 3D object Topologiq builds, updated to the last edge processsed successfully.
        circuit_name: The name of the ZX circuit.
        init_step: The ideal/intended (Manhattan) distance between source and target blocks.
        min_succ_rate (optional): Minimum % of tentative coordinates that must be filled for each edge.
        hide_ports (optional): If True, boundary spiders are considered by Topologiq but not displayed in visualisations.
        vis_options (optional): Visualisation settings provided as a Tuple.
            vis_options[0]: If enabled, triggers "final" or "detail" visualisations.
                (None): No visualisation.
                (str) "final" | "detail": A single visualisation of the final result or one visualisation per completed edge.
            vis_options[1]: If enabled, triggers creation of an animated summary for the entire process.
                (None): No animation.
                (str) "GIF" | "MP4": A step-by-step visualisation of the process in GIF or MP4 format.
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).
        log_stats_id (optional): A unique datetime-based identifier for the purposes of logging stats for an specific run.
        debug (optional): Debug mode (0: off, 1: graph manager, 2: pathfinder, 3: pathfinder w. discarded paths).

    Keyword arguments (**kwargs):
        weights: A tuple (int, int) of weights used to pick the best of several paths when there are several valid alternatives.
        length_of_beams: The length of each of the beams coming out of cubes still needing connections at any given point in time.

    Returns:
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        all_beams: A list of coordinates occupied by the beams of already-placed cubes that still require connections.
        edge_paths: An edge-by-edge summary of the 3D object Topologiq builds, updated to the last edge processsed successfully.
        (bool): A boolean flag to signal success (True if placement was succesful).

    """

    # Always prune beams to ensure recent placements are accounted for
    nx_g, all_beams = prune_beams(nx_g, all_beams, taken)

    # Get source cube data
    src_coords: Optional[StandardCoord] = nx_g.nodes[src_id].get("coords")
    src_kind: Optional[str] = nx_g.nodes[src_id].get("kind")

    if src_coords is None or src_kind is None:
        return taken, all_beams, edge_paths, False
    src_block_info: StandardBlock = (src_coords, src_kind)

    # Check position of target cube (should be None)
    nxt_neigh_coords: Optional[StandardCoord] = nx_g.nodes[tgt_id].get("coords")

    # Process targets that have yet to be placed in the 3D space
    if nxt_neigh_coords is None:

        # Geat target information
        nxt_neigh_node_data = nx_g.nodes[tgt_id]
        nxt_neigh_zx_type: str = cast(str, nxt_neigh_node_data.get("type"))

        # Get edge information
        zx_edge_type = nx_g.get_edge_data(src_id, tgt_id).get("type")
        hdm: bool = True if zx_edge_type == "HADAMARD" else False

        # Remove source coordinates from occupied coords
        # Note. This function needs access to the source coordinates
        taken_coords_c = taken[:]
        if src_coords in taken_coords_c:
            taken_coords_c.remove(src_coords)

        # Get clean candidate paths
        # Note. Topologically correct but not necessarily smart paths
        clean_paths, pathfinder_vis_data = run_pathfinder(
            src_block_info,
            nxt_neigh_zx_type,
            init_step,
            taken_coords_c if taken else [],
            hdm=hdm,
            min_succ_rate=min_succ_rate,
            src_tgt_ids=(src_id, tgt_id),
            log_stats_id=log_stats_id,
            debug=debug,
            nx_g=nx_g,
        )

        # Assemble a preliminary dictionary of viable paths
        # Note. A smart subset of clean paths
        viable_paths = []
        nxt_neigh_neigh_n = int(get_node_degree(nx_g, tgt_id))
        for clean_path in clean_paths:
            tgt_coords, tgt_kind = clean_path[-1]
            tgt_unobstr_exit_n, tgt_beams = check_exits(
                tgt_coords,
                tgt_kind,
                taken_coords_c,
                all_beams,
                beams_len=kwargs["length_of_beams"],
            )

            # Check path doesn't obstruct an absolutely necessary exit for a pre-existing cube
            coords_in_path = get_taken_coords(clean_path)

            # Reset # of unobstructed exits and node beams if target is a boundary
            if nxt_neigh_zx_type == "O":
                tgt_unobstr_exit_n, tgt_beams = (6, [])

            # Guarantee minimum necessary number of exits
            if tgt_unobstr_exit_n >= nxt_neigh_neigh_n - 1:
                # Allow path to break some beams
                # but ensure it does not break more beams than needed
                beams_broken_by_path = 0
                critical_beams_broken = False
                for n_id in nx_g.nodes():
                    if nx_g.nodes[n_id]["beams"]:
                        broken = 0
                        for bm in nx_g.nodes[n_id]["beams"]:
                            critical_beams_broken = False
                            if any([(c in coords_in_path) for c in bm]):
                                beams_broken_by_path += 1
                                broken += 1
                        adjust_for_source_node = 1 if n_id == src_id else 0
                        if broken > 4 - (get_node_degree(nx_g, n_id) - adjust_for_source_node):
                            critical_beams_broken = True

                # Append path to viable paths if path clears all checks
                if critical_beams_broken is not True:
                    all_nodes_in_path = [p for p in clean_path]

                    # Re-write type of boundary nodes for consistency
                    if nxt_neigh_zx_type == "O":
                        tgt_kind = "ooo"
                        all_nodes_in_path[-1] = (all_nodes_in_path[-1][0], tgt_kind)

                    # Consolidate path data
                    path_data = {
                        "tgt_coords": tgt_coords,
                        "tgt_kind": tgt_kind,
                        "tgt_beams": tgt_beams,
                        "coords_in_path": coords_in_path,
                        "all_nodes_in_path": all_nodes_in_path,
                        "beams_broken_by_path": beams_broken_by_path,
                        "len_of_path": len(clean_path),
                        "tgt_unobstr_exit_n": tgt_unobstr_exit_n,
                    }

                    # Append to viable paths
                    viable_paths.append(PathBetweenNodes(**path_data))

        # Choose a winner path from all viable paths
        winner_path: Optional[PathBetweenNodes] = None
        if viable_paths:
            winner_path = max(viable_paths, key=lambda path: path.weighed_value(**kwargs))
            
            # For visualisation, create a new graph on each step
            if debug > 0:
                # Number of edges in current lattice
                c = len(edge_paths)

                # Create partial progress graph from current edges
                partial_lat_nodes, partial_lat_edges = (reindex_path_dict(edge_paths))
                partial_nx_g, _ = lattice_to_g(partial_lat_nodes, partial_lat_edges, nx_g)

                # Detailed interactive visualisation of progress
                tent_coords, tent_tgt_kinds, all_search_paths, valid_paths = pathfinder_vis_data
                vis_3d(
                    nx_g,
                    partial_nx_g,
                    edge_paths,
                    valid_paths if valid_paths else None,
                    winner_path if winner_path else None,
                    src_block_info,
                    tent_coords,
                    tent_tgt_kinds,
                    hide_ports=hide_ports,
                    all_search_paths=all_search_paths,
                    debug=debug,
                    src_tgt_ids=(src_id, tgt_id),
                    fig_data=fig_data,
                    filename_info=(circuit_name, c) if vis_options[1] or debug == 4 else None,
                )

        # Write winner path and related info
        if winner_path:
            # Beautify path
            pretty_winner_path = [
                (block[0], kind_to_zx_type(block[1]))
                for block in winner_path.all_nodes_in_path
            ]
            pretty_winner_path = [
                (
                    block
                    if len(block[1]) == 1 or block[1] == "BOUNDARY"
                    else (f"{block[1]} EDGE")
                )
                for block in pretty_winner_path
            ]

            # Update source
            nx_g.nodes[src_id]["completed"] += 1

            # Update target
            nx_g.nodes[tgt_id]["coords"] = winner_path.tgt_coords
            nx_g.nodes[tgt_id]["kind"] = winner_path.tgt_kind
            nx_g.nodes[tgt_id]["completed"] += 1
            nx_g.nodes[tgt_id]["beams"] = (
                []
                if nx_g.nodes[tgt_id]["completed"] >= get_node_degree(nx_g, tgt_id)
                else winner_path.tgt_beams
            )

            # Add winner's beams to list of all_beams
            all_beams.append(winner_path.tgt_beams)

            # Update edge_paths
            edge = tuple(sorted((src_id, tgt_id)))
            edge_type = nx_g.get_edge_data(src_id, tgt_id).get(
                "type", "SIMPLE"
            )  # Default to "SIMPLE" if type is not found

            edge_paths[edge] = {
                "src_tgt_ids": (src_id, tgt_id),
                "path_coordinates": winner_path.coords_in_path,
                "path_nodes": winner_path.all_nodes_in_path,
                "edge_type": edge_type,
            }

            # Add path to position to list of graphs' occupied coordinates
            all_coords_in_path = get_taken_coords(winner_path.all_nodes_in_path)
            taken.extend(all_coords_in_path)

            # Update user if log_stats or debug are enabled
            if log_stats_id or debug in [1, 2, 3]:
                print(f"Path creation: {src_id} -> {tgt_id}. SUCCESS.")

            # Return updated list of taken coords and all_beams, with success flag
            nx_g, all_beams = prune_beams(nx_g, all_beams, taken)
            return taken, all_beams, edge_paths, True

        # Handle cases where no winner is found
        if not winner_path:

            # Explicit warning if log_stats or debug are enabled 
            if log_stats_id or debug in [1, 2, 3]:
                print(f"Path creation: {src_id} -> {tgt_id}. FAIL.")

            # Fill edge_paths with error
            edge = tuple(sorted((src_id, tgt_id)))
            edge_paths[edge] = {
                "src_tgt_ids": "error",
                "path_coordinates": "error",
                "path_nodes": "error",
                "edge_type": "error",
            }

            # Return unchanged list of taken coords and all_beams, with failure flag
            nx_g, all_beams = prune_beams(nx_g, all_beams, taken)
            return taken, all_beams, edge_paths, False

    # Fail-safe return to avoid type errors
    return taken, all_beams, edge_paths, False


def second_pass(
    nx_g: nx.Graph,
    taken: List[StandardCoord],
    edge_paths: dict,
    c: int,
    all_beams: List[NodeBeams],
    circuit_name: str = "circuit",
    min_succ_rate: int = 50,
    hide_ports: bool = False,
    vis_options: Tuple[Union[None, str], Union[None, str]] = (None, None),
    log_stats_id: Union[str, None] = None,
    debug: int = 0,
    fig_data: Optional[matplotlib.figure.Figure] = None,
) -> Tuple[dict, int, int]:
    """Perform a second pass of the graph to process any edges missed by the primary BFS.
    
    This function is a backup facility that goes over the ZX graph after the primary BFS finishes, 
    identifying any edge that is in the ZX graph but has not been yet transformed into a corresponding
    3D edge. This typically happens when there are multiple interconnected nodes, which causes the BFS to 
    run out of spiders/cubes to place before all edges are rendered in 3D.

    Args:
        nx_g: A nx_graph with the same spiders/edges as incoming ZX graph but in 3D-amicable format/structure.
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        edge_paths: An edge-by-edge summary of the 3D object Topologiq builds, updated to the last edge processsed successfully.
        c: A counter for the number of completed edge iterations (used to organise visualisations).
        all_beams: A list of coordinates occupied by the beams of already-placed cubes that still require connections.
        circuit_name: The name of the ZX circuit.
        min_succ_rate (optional): Minimum % of tentative coordinates that must be filled for each edge.
        hide_ports (optional): If True, boundary spiders are considered by Topologiq but not displayed in visualisations.
        vis_options (optional): Visualisation settings provided as a Tuple.
            vis_options[0]: If enabled, triggers "final" or "detail" visualisations.
                (None): No visualisation.
                (str) "final" | "detail": A single visualisation of the final result or one visualisation per completed edge.
            vis_options[1]: If enabled, triggers creation of an animated summary for the entire process.
                (None): No animation.
                (str) "GIF" | "MP4": A step-by-step visualisation of the process in GIF or MP4 format.
        log_stats_id (optional): A unique datetime-based identifier for the purposes of logging stats for an specific run.
        debug (optional): Debug mode (0: off, 1: graph manager, 2: pathfinder, 3: pathfinder w. discarded paths).
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).

    Keyword arguments (**kwargs):
        weights: A tuple (int, int) of weights used to pick the best of several paths when there are several valid alternatives.
        length_of_beams: The length of each of the beams coming out of cubes still needing connections at any given point in time.

    Returns:
        edge_paths: An edge-by-edge summary of the 3D object Topologiq builds, updated to the last edge processsed successfully.
        c: A counter for the number of completed edge iterations (used to organise visualisations).

    """

    # Get graph edges
    num_2n_pass_edges = 0
    for src_id, tgt_id, data in nx_g.edges(data=True):

        # Ensure occupied coords do not have duplicates
        taken = list(set(taken))

        # Prune beams for good practice
        nx_g, all_beams = prune_beams(nx_g, all_beams, taken)

        # Get source and target data for current (src_id, tgt_id) pair
        u_coords: Optional[StandardCoord] = nx_g.nodes[src_id].get("coords")
        v_coords: Optional[StandardCoord] = nx_g.nodes[tgt_id].get("coords")

        # Process edge only if both src_id and tgt_id have already been placed in the 3D space
        # Note. Function should never run into (src_id, tgt_id) pairs not already in 3D space
        if u_coords is not None and v_coords is not None:
            # Update visualiser counter
            c += 1

            # Format adjustments to match existing operations
            u_kind = cast(str, nx_g.nodes[src_id].get("kind"))
            v_zx_type = cast(str, nx_g.nodes[tgt_id].get("type"))
            edge = tuple(sorted((src_id, tgt_id)))

            # Call pathfinder on any graph edge that does not have an entry in edge_paths
            if edge not in edge_paths:

                # Update edge counter
                num_2n_pass_edges += 1

                # Prune beams to consider recent node placements
                nx_g, all_beams = prune_beams(nx_g, all_beams, taken)

                critical_beams: dict[int, Tuple[int, NodeBeams]] = {}
                num_edges_still_to_complete = 0
                for node_id in nx_g.nodes():
                    n_beams = nx_g.nodes[node_id]["beams"]
                    if n_beams != []:
                        n_degree = get_node_degree(nx_g, node_id)
                        n_edges_completed = nx_g.nodes[node_id]["completed"]
                        num_edges_still_to_complete = n_degree - n_edges_completed
                        if num_edges_still_to_complete == 0:
                            pass
                        else:
                            critical_beams[node_id] = (
                                num_edges_still_to_complete,
                                n_beams,
                            )

                # Check if edge is hadamard
                zx_edge_type = nx_g.get_edge_data(src_id, tgt_id).get("type")
                hdm: bool = True if zx_edge_type == "HADAMARD" else False

                # Call pathfinder using optional parameters that flag second pass nature of operation
                v_kind: Optional[str] = nx_g.nodes[tgt_id].get("kind")
                if v_coords and v_kind:
                    clean_paths, pathfinder_vis_data = run_pathfinder(
                        (u_coords, u_kind),
                        v_zx_type,
                        3,
                        taken[:],
                        tgt_block_info=(v_coords, v_kind),
                        hdm=hdm,
                        min_succ_rate=min_succ_rate,
                        log_stats_id=log_stats_id,
                        critical_beams=critical_beams,
                        src_tgt_ids=(src_id, tgt_id),
                        debug=debug,
                        nx_g=nx_g,
                    )

                    # For visualisation, create a new graph on each step
                    if debug > 0:
                        # Number of edges in current lattice
                        c = len(edge_paths)

                        # Create partial progress graph from current edges
                        partial_lat_nodes, partial_lat_edges = (reindex_path_dict(edge_paths))
                        partial_nx_g, _ = lattice_to_g(partial_lat_nodes, partial_lat_edges, nx_g)

                        # Detailed interactive visualisation of progress
                        tent_coords, tent_tgt_kinds, all_search_paths, valid_paths = pathfinder_vis_data
                        vis_3d(
                            nx_g,
                            partial_nx_g,
                            edge_paths,
                            valid_paths if valid_paths else None,
                            clean_paths[0] if clean_paths else None,
                            (u_coords, u_kind),
                            tent_coords,
                            tent_tgt_kinds,
                            hide_ports=hide_ports,
                            all_search_paths=all_search_paths,
                            debug=debug,
                            src_tgt_ids=(src_id, tgt_id),
                            fig_data=fig_data,
                            filename_info=(circuit_name, c) if vis_options[1] or debug == 4 else None,
                        )

                    # Write to edge_paths if an edge is found
                    # Note. Since both (src_id, tgt_id) are in 3D space, pathfinder will return only one path
                    if clean_paths:

                        # Update edge paths
                        coords_in_path = [p[0] for p in clean_paths[0]]  # Take the first path
                        edge_type = data.get("type", "SIMPLE")
                        edge_paths[edge] = {
                            "src_tgt_ids": (src_id, tgt_id),
                            "path_coordinates": coords_in_path,
                            "path_nodes": clean_paths[0],
                            "edge_type": edge_type,
                        }

                        # Update source info
                        nx_g.nodes[src_id]["completed"] += 1

                        # Update target node information
                        nx_g.nodes[tgt_id]["completed"] += 1
                        nx_g.nodes[tgt_id]["beams"] = (
                            []
                            if nx_g.nodes[tgt_id]["completed"] >= get_node_degree(nx_g, tgt_id)
                            else nx_g.nodes[tgt_id]["beams"]
                        )

                        # Add path to position to list of taken coordinates
                        all_coords_in_path = get_taken_coords(clean_paths[0])
                        taken.extend(all_coords_in_path)
                        
                        # Prune beams before moving to next edge
                        nx_g, all_beams = prune_beams(nx_g, all_beams, taken)

                        # Update user if log_stats or debug mode are enabled
                        if log_stats_id or debug in [1, 2, 3]:
                            print(f"Path discovery: {src_id} -> {tgt_id}. SUCCESS.")

                    # Write an error to edge_paths if edge not found
                    else:
                        raise ValueError(f"Path discovery. Error with edge: {src_id} -> {tgt_id}.")

    # Return edge_paths for final consumption
    return edge_paths, c, num_2n_pass_edges


def run_pathfinder(
    src_block_info: StandardBlock,
    tgt_zx_type: str,
    init_step: int,
    taken: List[StandardCoord],
    tgt_block_info: Optional[StandardBlock] = None,
    hdm: bool = False,
    min_succ_rate: int = 60,
    critical_beams: dict[int, Tuple[int, NodeBeams]] = {},
    src_tgt_ids: Optional[Tuple[int,int]] = None,
    log_stats_id: Union[str, None] = None,
    debug: int = 0,
    nx_g: nx.Graph = [],
) -> Tuple[
    List[Any],
    Tuple[List[StandardCoord], List[str], Union[None, dict[StandardBlock, List[StandardBlock]]]],
    ]:
    """Call the pathfinder algorithm for an arbitrary combination of source and target spiders/cubes.
    
    This function calls the inner pathfinder algorith with the information using a variable combination of parameters. 
    If the function does not get information about the desired target, it assumes it is creating a path between an 
    already-placed cube and a new cube. In such case, the function generates a list of tentative target coordinates, 
    which the inner pathfinder algorithm fulfills up to `min_succ_rate` percent. Once the inner pathfinder algorithm
    returns all paths fulfilled, this function eliminates paths not meeting key heuristics and chooses the best 
    amongst all surviving paths. 

    Args:
        src_block_info: The information of the source cube including its position in the 3D space and its kind.
        tgt_zx_type: The ZX type of the target spider/cube.
        init_step: The ideal/intended (Manhattan) distance between source and target blocks.
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        tgt_block_info (optional): An optional parameter to send the information of a node that has already been placed in the 3D space.
        hdm (optional): If True, it tells the inner pathfinding algorithm that the original ZX-edge is a Hadamard edge.
        min_succ_rate (optional): Minimum % of tentative coordinates that must be filled for each edge.
        critical_beams (optional): Annotated beams object with details about minimum number of beams needed per node.
        log_stats_id (optional): A unique datetime-based identifier for the purposes of logging stats for an specific run.
        debug (optional): Debug mode (0: off, 1: graph manager, 2: pathfinder, 3: pathfinder w. discarded paths).

    Returns:
        clean_paths: A list of paths each containing the 3D cubes and pipes needed to connect source and target in the 3D space.

    """

    # Edge path management
    valid_paths: Union[dict[StandardBlock, List[StandardBlock]], None] = None
    clean_paths = []

    step = init_step
    src_coords, _ = src_block_info
    tgt_coords, tgt_type = tgt_block_info if tgt_block_info else (None, None)

    # Copy taken to avoid accidental overwrites
    taken_cc = taken[:]
    if src_coords in taken_cc:
        taken_cc.remove(src_coords)
    if tgt_coords:
        taken_cc.remove(tgt_coords)

    # Loop call the inner pathfinder in case there is a need to re-run the pathfinder
    max_step = 2 * init_step if tgt_block_info else 9
    while step <= max_step:
        # Generate tentative coordinates for current step or use target node
        if tgt_coords:
            tent_coords = [tgt_coords]
        else:
            tent_coords = gen_tent_tgt_coords(
                src_coords,
                step,
                taken,  # Real occupied coords: position cannot overlap start node
            )

        # Try finding paths to each tentative coordinates
        valid_paths, pathfinder_vis_data = pathfinder(
            src_block_info,
            tent_coords,
            tgt_zx_type,
            taken=taken_cc,
            tgt_block_info=(tent_coords[0], tgt_type),
            hdm=hdm,
            min_succ_rate=min_succ_rate,
            critical_beams=critical_beams,
            src_tgt_ids=src_tgt_ids,
            log_stats_id=log_stats_id,
        )

        # Append usable paths to clean paths
        if valid_paths:
            for path in valid_paths.values():
                path_checks = True
                for node in path:
                    if node[0] in taken_cc:
                        path_checks = False
                if path_checks:
                    clean_paths.append(path)

        # Break if valid paths generated at step
        if clean_paths:
            break

        # Increase distance if no valid paths found at current step
        step += 3

    return clean_paths, pathfinder_vis_data


#######################
# CORE AUX OPERATIONS #
#######################
def prep_3d_g(simple_graph: SimpleDictGraph) -> nx.Graph:
    """Convert a `simple_graph` into an NX graph with syntax and structure amicable to 3D transformations.

    This function takes a `simple_graph` containing the spiders and edges of a ZX graph and converts it into
    an NX graph. The resulting NX graph contains the same information as the `simple_graph` but has a number
    of placeholders that enable the algorithm to overwrite the NX graph with 3D information as the algorithm
    traverses the graph making 3D placements. 

    Args:
        simple_graph: The `simple_graph` form of an arbitrary ZX circuit.

    Returns:
        nx_g: A nx_graph with the same spiders/edges as incoming ZX graph but in 3D-amicable format/structure.

    """

    # Prepare an empty NX graph
    nx_g = nx.Graph()

    # Get the spiders and edges of incoming `simple_graph`
    nodes: List[Tuple[int, str]] = simple_graph.get("nodes", [])
    edges: List[Tuple[Tuple[int, int], str]] = simple_graph.get("edges", [])

    # Add the spiders to the NX graph
    for n_id, n_type in nodes:
        nx_g.add_node(
            n_id,
            type=n_type,
            type_fam=get_zx_type_fam(n_type),
            kind=None,
            coords=None,
            beams=None,
            completed=0,
        )

    # Add the edges to the NX graph
    for (src_id, tgt_id), e_type in edges:
        nx_g.add_edge(src_id, tgt_id, type=e_type)

    # ID any spider with more than 4 edges/neighbours
    all_nodes = list(nx_g.nodes())
    centr_nodes = [n for n in all_nodes if get_node_degree(nx_g, n) > 4]

    # Break any spiders iwth mode than 4 edges/neigbours
    # Note. This operation is a backup facility. Ideally,
    # incoming `simple_graph` will have been pre-processed
    # in a way that avoids >4-edge spiders.
    if centr_nodes:
        # Determine max degree
        centr_node = max(nx_g.nodes) if nx_g.nodes else 0

        # Loop over max nodes and break as appropriate
        i = 0
        while i < 100:
            # List of high degree nodes
            all_nodes_loop = list(nx_g.nodes())
            centr_nodes = [n for n in all_nodes_loop if get_node_degree(nx_g, n) > 4]

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
                coords=None,
                beams=None,
                completed=0,
            )
            nx_g.add_edge(node_to_sanitise, twin_node_id, type="SIMPLE")

            # Distributed edges across twins
            neighs = list(nx_g.neighbors(node_to_sanitise))
            neighs = [n for n in neighs if n != twin_node_id]

            degree_to_shuffle = get_node_degree(nx_g, node_to_sanitise) // 2

            shuffle_c = 0
            random.shuffle(neighs)

            for neigh in neighs:
                if (
                    shuffle_c >= degree_to_shuffle
                    or get_node_degree(nx_g, node_to_sanitise) <= 4
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

    return nx_g
