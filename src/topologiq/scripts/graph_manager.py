"""Manage the main/outer graph manager BFS process.

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
from collections import deque
from datetime import datetime
from typing import Any, cast

import matplotlib.figure
import networkx as nx

from topologiq.scripts.pathfinder import get_taken_coords, pathfinder
from topologiq.utils.animation import create_animation
from topologiq.utils.classes import (
    Colors,
    CubeBeams,
    PathBetweenNodes,
    SimpleDictGraph,
    StandardBlock,
    StandardCoord,
)
from topologiq.utils.grapher import vis_3d
from topologiq.utils.grapher_common import lattice_to_g
from topologiq.utils.utils_greedy_bfs import (
    find_first_id,
    gen_tent_tgt_coords,
    get_node_degree,
    prune_beams,
    reindex_path_dict,
)
from topologiq.utils.utils_misc import prep_stats_n_log
from topologiq.utils.utils_pathfinder import check_exits
from topologiq.utils.utils_zx_graphs import check_zx_types, get_zx_type_fam, kind_to_zx_type


###############################
# MAIN GRAPH MANAGER WORKFLOW #
###############################
def graph_manager_bfs(
    simple_graph: SimpleDictGraph,
    circuit_name: str = "circuit",
    min_succ_rate: int = 50,
    hide_ports: bool = False,
    vis_options: tuple[str | None, str | None] = (None, None),
    log_stats_id: str | None = None,
    debug: int = 0,
    fig_data: matplotlib.figure.Figure | None = None,
    first_cube: tuple[int | None, str | None] = (None, None),
    **kwargs,
) -> tuple[
    nx.Graph,
    dict,
    dict[int, StandardBlock] | None,
    dict[tuple[int, int], list[str]] | None,
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
        vis_options (optional): Visualisation settings provided as a tuple.
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
        **kwargs: !
            weights: A tuple (int, int) of weights used to pick the best of several paths when there are several valid alternatives.
            deterministic: A boolean flag to tell the function if choice is deterministic or random.
            random_seed: Typically `None`, but can be used to pass a specific seed across the entire algorithm.

    Returns:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        edge_paths: An edge-by-edge/block-by-block summary of the space-time diagram Topologiq builds.
        lat_nodes: The cubes of the final space-time diagram produced by Topologiq.
        lat_edges: The pipes of the final space-time diagram produced by Topologiq.

    """

    # 1. Key parameters
    # Stats trackers
    t_start = datetime.now()
    duration_1st_pass, duration_2nd_pass = (0.0, 0.0)
    run_success = False

    num_input_nodes, num_input_edges = (len(simple_graph["nodes"]), len(simple_graph["edges"]))
    num_1st_pass_edges, num_2nd_pass_edges = (0, 0)
    num_edges_processed = 0  # Usefulness not obvious, needed to save snapshots for animations

    # Graph & outputs
    nx_g = prep_3d_g(simple_graph)
    lat_nodes: dict[int, StandardBlock] | None = None
    lat_edges: dict[tuple[int, int], list[str]] | None = None

    # First spider/cube
    first_cube = get_first_cube(nx_g, first_cube=first_cube, deterministic=kwargs["deterministic"], random_seed=kwargs["seed"])
    first_id, first_kind = first_cube

    # BFS management
    taken: list[StandardCoord] = []
    edge_paths: dict = {}
    queue: deque[int] = deque([first_id])
    visited: set = {first_id}

    # 2. Validity checks
    if not validity_checks(simple_graph, first_cube):
        return nx_g, edge_paths, lat_nodes, lat_edges

    # 3. Place first spider/cube
    nx_g, taken = place_first_cube(nx_g, taken, first_cube)
    if log_stats_id or debug > 0:
        print(f"First cube ID: {first_id} ({first_kind}).")

    # 4. Graph manager BFS
    while queue:
        # Get first cube or current source cube
        src_id: int = queue.popleft()

        # Iterate over neighbours of current source
        for tgt_id in cast(list[int], nx_g.neighbors(src_id)):
            # Handle cubes that need to be placed for the first time
            if tgt_id not in visited:
                # Start iteration timer
                t1_1st_pass_iter = datetime.now()

                # Add/append ID to visited and queue
                visited.add(tgt_id)
                queue.append(tgt_id)

                # Ensure taken has unique entries on each run
                taken = list(set(taken))

                # Try to place blocks as close to one another as as possible
                step, max_step = (3, 15)
                while step <= max_step:
                    taken, edge_paths, edge_success = place_nxt_block(
                        src_id,
                        tgt_id,
                        nx_g,
                        taken,
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

                    # Stop iteration timers
                    t_end_1st_pass_iter = datetime.now()
                    duration_1st_pass += (t_end_1st_pass_iter - t1_1st_pass_iter).total_seconds()

                    # Move to next if there is a succesful placement
                    if edge_success:
                        run_success = True
                        num_1st_pass_edges += 1
                        num_edges_processed += 1
                        break

                    # Fail if max_step is reached
                    elif step >= max_step:
                        run_success = False
                        break

                    # Increase distance between nodes if placement not possible
                    step += 3

                # Exit BFS loop if a single edge fails to build
                if run_success is False:
                    break

            # Handle connections between previously placed blocks
            elif (src_id, tgt_id) not in edge_paths and (tgt_id, src_id) not in edge_paths:
                # Start iteration timer for 2st pass iteration
                t1_2nd_pass_iter = datetime.now()

                # Trigger connection for previously placed cubes
                taken, edge_paths, edge_success = connect_prev_placed_cubes(
                    src_id,
                    tgt_id,
                    nx_g,
                    taken,
                    edge_paths,
                    circuit_name=circuit_name,
                    min_succ_rate=min_succ_rate,
                    hide_ports=hide_ports,
                    vis_options=vis_options,
                    fig_data=fig_data,
                    log_stats_id=log_stats_id,
                    debug=debug,
                )

                # Stop & log times
                t_end_2nd_pass_iter = datetime.now()
                duration_2nd_pass += (t_end_2nd_pass_iter - t1_2nd_pass_iter).total_seconds()

                if edge_success:
                    run_success = True
                    num_2nd_pass_edges += 1
                    num_edges_processed += 1

                else:
                    run_success = False
                    break

        if run_success is False:
            break

    # Taken is used extensively, so prune it again
    taken = list(set(taken))

    # Assemble final lattice if run is successfull
    if run_success:
        lat_nodes, lat_edges = reindex_path_dict(edge_paths)

    # Log stats
    if log_stats_id is not None:
        duration_total = (datetime.now() - t_start).total_seconds()

        call_logger(
            [circuit_name, log_stats_id, run_success],
            [edge_paths, lat_nodes, lat_edges],
            [duration_1st_pass, duration_2nd_pass, duration_total],
            [
                num_input_nodes,
                num_input_edges,
                num_1st_pass_edges,
                num_2nd_pass_edges,
            ],
            [min_succ_rate, vis_options[1], kwargs],
        )

    # Raise
    if not run_success:
        raise ValueError(f"ERROR. Run aborted. Failed to complete edge: {src_id} -> {tgt_id}.")

    return nx_g, edge_paths, lat_nodes, lat_edges


##################
# EDGE BUILDERS #
##################
def place_nxt_block(
    src_id: int,
    tgt_id: int,
    nx_g: nx.Graph,
    taken: list[StandardCoord],
    edge_paths: dict,
    circuit_name: str = "circuit",
    init_step: int = 3,
    min_succ_rate: int = 60,
    hide_ports: bool = False,
    vis_options: tuple[str | None, str | None] = (None, None),
    fig_data: matplotlib.figure.Figure | None = None,
    log_stats_id: str | None = None,
    debug: int = 0,
    **kwargs,
) -> tuple[list[StandardCoord], list[CubeBeams], dict, bool]:
    """Position target cube in the 3D space as part of the primary BFS flow.

    This function calls the inner pathfinder algorithm on any arbitrary combination of an already-placed
    `src_id` and a yet-to-be-placed `tgt_id`. The inner pathfinder algorithm returns a list of viable
    paths to a number of valid placements for `tgt_id`, and chooses a best path from this list
    using hyperparameters passed as `kwargs` and a value function.

    Args:
        src_id: The ID of the source node, i.e., the one that has already been placed in the 3D space as part of previous operations.
        tgt_id: The ID of the neighbouring or next node, i.e., the one that needs to be placed in the 3D space.
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        edge_paths: An edge-by-edge summary of the 3D object Topologiq builds, updated to the last edge processsed successfully.
        circuit_name: The name of the ZX circuit.
        init_step: The ideal/intended (Manhattan) distance between source and target blocks.
        min_succ_rate (optional): Minimum % of tentative coordinates that must be filled for each edge.
        hide_ports (optional): If True, boundary spiders are considered by Topologiq but not displayed in visualisations.
        vis_options (optional): Visualisation settings provided as a tuple.
            vis_options[0]: If enabled, triggers "final" or "detail" visualisations.
                (None): No visualisation.
                (str) "final" | "detail": A single visualisation of the final result or one visualisation per completed edge.
            vis_options[1]: If enabled, triggers creation of an animated summary for the entire process.
                (None): No animation.
                (str) "GIF" | "MP4": A step-by-step visualisation of the process in GIF or MP4 format.
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).
        log_stats_id (optional): A unique datetime-based identifier for the purposes of logging stats for an specific run.
        debug (optional): Debug mode (0: off, 1: graph manager, 2: pathfinder, 3: pathfinder w. discarded paths).
        **kwargs: !
            weights: A tuple (int, int) of weights used to pick the best of several paths when there are several valid alternatives.
            deterministic: A boolean flag to tell the function if choice is deterministic or random.
            random_seed: Typically `None`, but can be used to pass a specific seed across the entire algorithm.

    Returns:
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        edge_paths: An edge-by-edge summary of the 3D object Topologiq builds, updated to the last edge processsed successfully.
        (bool): A boolean flag to signal success (True if placement was succesful).

    """

    # Start timer
    t1 = datetime.now()

    # Always prune beams to ensure recent placements are accounted for
    nx_g = prune_beams(nx_g, taken)

    # Get source cube data
    src_coords: StandardCoord | None = nx_g.nodes[src_id].get("coords")
    src_kind: str | None = nx_g.nodes[src_id].get("kind")

    if src_coords is None or src_kind is None:
        return taken, edge_paths, False
    src_block_info: StandardBlock = (src_coords, src_kind)

    # Check position of target cube (should be None)
    nxt_neigh_coords: StandardCoord | None = nx_g.nodes[tgt_id].get("coords")

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
        )

        # Assemble a preliminary dictionary of viable paths
        # Note. A smart subset of clean paths
        viable_paths = []
        nxt_neigh_neigh_n = int(get_node_degree(nx_g, tgt_id))
        for clean_path in clean_paths:
            # Extract key path information
            tgt_coords, tgt_kind = clean_path[-1]
            coords_in_path = get_taken_coords(clean_path)

            # Check if exits are unobstructed
            tgt_unobstr_exit_n, tgt_beams = check_exits(
                tgt_coords,
                tgt_kind,
                taken_coords_c,
                coords_in_path
            )

            # Check path doesn't obstruct an absolutely necessary exit for a pre-existing cube
            # Reset # of unobstructed exits and node beams if target is a boundary
            if nxt_neigh_zx_type == "O":
                tgt_unobstr_exit_n, tgt_beams = (6, [])

            if tgt_unobstr_exit_n >= nxt_neigh_neigh_n - 1:
                # Allow path to break some beams
                # but ensure it does not break more beams than needed
                beams_broken_by_path = 0
                for n_id in nx_g.nodes():
                    critical_beams_broken = False
                    broken = 0
                    if nx_g.nodes[n_id]["beams"]:
                        for single_beam in nx_g.nodes[n_id]["beams"]:
                            if any([single_beam.contains(c) for c in coords_in_path]):
                                beams_broken_by_path += 1
                                broken += 1
                        adjust_for_source_node = 1 if n_id == src_id else 0
                        n_degree = get_node_degree(nx_g, n_id)
                        n_edges_completed = nx_g.nodes[n_id]["completed"]
                        num_edges_still_to_complete = n_degree - n_edges_completed
                        if (
                            len(nx_g.nodes[n_id]["beams"]) - broken + adjust_for_source_node
                            < num_edges_still_to_complete
                        ):
                            critical_beams_broken = True
                            break

                # Watch out for critical beam clashes
                # It is hard to manage situations where the beams of a beam clash with the beams of other beams
                # Most such clashes are harmless, so forbidding all such situations would harm volume.
                # Some clashes create a snowball effect that guarantees failures down the line.
                for n_id in nx_g.nodes():
                    critical_beams_clash = False
                    beam_clash_count = 0

                    if n_id not in (src_id, tgt_id) and nx_g.nodes[n_id]["beams"] and nx_g.nodes[n_id]["coords"]:
                        n_degree = get_node_degree(nx_g, n_id)
                        n_edges_completed = nx_g.nodes[n_id]["completed"]
                        num_edges_still_to_complete = n_degree - n_edges_completed

                        if tgt_beams:
                            for single_beam in nx_g.nodes[n_id]["beams"]:
                                beam_clash_count = sum(
                                    [
                                        single_beam.intersects(single_beam_of_tgt_cube)
                                        for single_beam_of_tgt_cube in tgt_beams
                                    ]
                                )

                                if (
                                    len(nx_g.nodes[n_id]["beams"]) - beam_clash_count
                                    < num_edges_still_to_complete
                                ):
                                    critical_beams_clash = True
                                    break

                # Append path to viable paths if path clears all checks
                if critical_beams_broken is not True and critical_beams_clash is not True:
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
        winner_path: PathBetweenNodes | None = None
        if viable_paths:
            winner_path = max(viable_paths, key=lambda path: path.weighed_value(**kwargs))

        # Finish timer before popping up visualisation
        duration_iter = (datetime.now() - t1).total_seconds()

        # For visualisation, create a new graph on each step
        debug = debug if debug >= 1 else 1 if vis_options[0] == "detail" or vis_options[1] else 0
        if debug > 0:
            # Create partial progress graph from current edges
            partial_lat_nodes, partial_lat_edges = reindex_path_dict(edge_paths, fix_errors=True)
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
                vis_options=vis_options,
                src_tgt_ids=(src_id, tgt_id),
                fig_data=fig_data,
                filename_info=(circuit_name, len(edge_paths) + 1)
                if vis_options[1] or debug == 4
                else None,
            )

        # Write winner path and related info
        if winner_path:
            # Beautify path
            pretty_winner_path = [
                (block[0], kind_to_zx_type(block[1])) for block in winner_path.all_nodes_in_path
            ]
            pretty_winner_path = [
                (block if len(block[1]) == 1 or block[1] == "BOUNDARY" else (f"{block[1]} EDGE"))
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
            if log_stats_id or debug > 0:
                volume = len(
                    [block for block in winner_path.all_nodes_in_path if "o" not in block[1]]
                )
                print(
                    f"New path created: {src_id} -> {tgt_id} (vol += {volume - 1}) (runtime ~ {int(duration_iter * 1000)}ms)."
                )

            # Return updated list of taken coords, edge_paths, and a fail/success flag
            nx_g = prune_beams(nx_g, taken)
            return taken, edge_paths, True

        # Handle cases where no winner is found
        if not winner_path:
            # Explicit warning if log_stats or debug are enabled
            if log_stats_id or debug > 0:
                print(
                    f"{'ERROR' if init_step == 15 else 'Partial error'}. New path creation: {src_id} -> {tgt_id} (runtime ~ {int(duration_iter * 1000)}ms). -> {'Increasing search distance' if init_step < 15 else 'Shut down unavoidable.'}"
                )

            # Fill edge_paths with error
            edge = tuple(sorted((src_id, tgt_id)))
            edge_paths[edge] = {
                "src_tgt_ids": "error",
                "path_coordinates": "error",
                "path_nodes": "error",
                "edge_type": "error",
            }

            # Return updated list of taken coords, edge_paths, and a fail/success flag
            nx_g = prune_beams(nx_g, taken)
            return taken, edge_paths, False

    # Fail-safe return to avoid type errors
    return taken, edge_paths, False


def connect_prev_placed_cubes(
    src_id: int,
    tgt_id: int,
    nx_g: nx.Graph,
    taken: list[StandardCoord],
    edge_paths: dict,
    circuit_name: str = "circuit",
    min_succ_rate: int = 60,
    hide_ports: bool = False,
    vis_options: tuple[str | None, str | None] = (None, None),
    fig_data: matplotlib.figure.Figure | None = None,
    log_stats_id: str | None = None,
    debug: int = 0,
) -> tuple[list[StandardCoord], list[CubeBeams], dict, bool]:
    """Search for a path between two cubes that have already been placed in 3D space.

    This function calls the inner pathfinder algorithm to search for paths between cubes that have
    been already-placed in the 3D space. The inner pathfinder algorithm returns a single paths, indeed
    the shortest path respecting all restrictoins.

    Args:
        src_id: The ID of the source node, i.e., the one that has already been placed in the 3D space as part of previous operations.
        tgt_id: The ID of the neighbouring or next node, i.e., the one that needs to be placed in the 3D space.
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        edge_paths: An edge-by-edge summary of the 3D object Topologiq builds, updated to the last edge processsed successfully.
        circuit_name: The name of the ZX circuit.
        min_succ_rate (optional): Minimum % of tentative coordinates that must be filled for each edge.
        hide_ports (optional): If True, boundary spiders are considered by Topologiq but not displayed in visualisations.
        vis_options (optional): Visualisation settings provided as a tuple.
            vis_options[0]: If enabled, triggers "final" or "detail" visualisations.
                (None): No visualisation.
                (str) "final" | "detail": A single visualisation of the final result or one visualisation per completed edge.
            vis_options[1]: If enabled, triggers creation of an animated summary for the entire process.
                (None): No animation.
                (str) "GIF" | "MP4": A step-by-step visualisation of the process in GIF or MP4 format.
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).
        log_stats_id (optional): A unique datetime-based identifier for the purposes of logging stats for an specific run.
        debug (optional): Debug mode (0: off, 1: graph manager, 2: pathfinder, 3: pathfinder w. discarded paths).
        **kwargs: !
            weights: A tuple (int, int) of weights used to pick the best of several paths when there are several valid alternatives.
            deterministic: A boolean flag to tell the function if choice is deterministic or random.
            random_seed: Typically `None`, but can be used to pass a specific seed across the entire algorithm.

    Returns:
        taken: Updated list of all coordinates occupied by any blocks/pipes, including any placed by this function iteration.
        edge_paths: An edge-by-edge summary of the 3D object Topologiq builds, updated to the last edge processsed successfully.
        edge_success: A boolean flag declaring whether the edge was built successfully or not.

    """
    # Start timer
    t1 = datetime.now()

    # Prune taken and beams
    edge_success = False
    taken = list(set(taken))
    nx_g = prune_beams(nx_g, taken)

    # Get source and target data for current (src_id, tgt_id) pair
    u_coords: StandardCoord | None = nx_g.nodes[src_id].get("coords")
    v_coords: StandardCoord | None = nx_g.nodes[tgt_id].get("coords")

    # Process edge only if both src_id and tgt_id have already been placed in the 3D space
    # Note. Function should never run into (src_id, tgt_id) pairs not already in 3D space
    if u_coords is not None and v_coords is not None:
        # Format adjustments to match existing operations
        u_kind = cast(str, nx_g.nodes[src_id].get("kind"))
        v_zx_type = cast(str, nx_g.nodes[tgt_id].get("type"))
        edge = tuple(sorted((src_id, tgt_id)))

        # Call pathfinder on any graph edge that does not have an entry in edge_paths
        if edge not in edge_paths:
            critical_beams: dict[int, tuple[int, CubeBeams]] = {}
            num_edges_still_to_complete = 0
            for node_id in nx_g.nodes():
                node_coords = nx_g.nodes[node_id]["coords"]
                all_beams_for_node = nx_g.nodes[node_id]["beams"]
                if all_beams_for_node != [] and all_beams_for_node is not None:
                    n_degree = get_node_degree(nx_g, node_id)
                    n_edges_completed = nx_g.nodes[node_id]["completed"]
                    num_edges_still_to_complete = n_degree - n_edges_completed
                    if num_edges_still_to_complete == 0:
                        pass
                    else:
                        critical_beams[node_id] = (
                            node_coords,
                            num_edges_still_to_complete,
                            all_beams_for_node,
                        )

            # Check if edge is hadamard
            zx_edge_type = nx_g.get_edge_data(src_id, tgt_id).get("type")
            hdm: bool = True if zx_edge_type == "HADAMARD" else False

            # Call pathfinder using optional parameters that flag second pass nature of operation
            v_kind: str | None = nx_g.nodes[tgt_id].get("kind")
            if v_coords and v_kind:
                clean_paths, pathfinder_vis_data = run_pathfinder(
                    (u_coords, u_kind),
                    v_zx_type,
                    3,
                    taken[:],
                    tgt_block_info=(v_coords, v_kind),
                    hdm=hdm,
                    min_succ_rate=min_succ_rate,
                    critical_beams=critical_beams,
                    log_stats_id=log_stats_id,
                    src_tgt_ids=(src_id, tgt_id),
                )

                # Finish timer before popping up visualisation
                duration_iter = (datetime.now() - t1).total_seconds()

                # For visualisation, create a new graph on each step irrespective of outcome
                debug = (
                    debug
                    if debug >= 1
                    else 1
                    if vis_options[0] == "detail" or vis_options[1]
                    else 0
                )
                if debug > 0:
                    # Create partial progress graph from current edges
                    partial_lat_nodes, partial_lat_edges = reindex_path_dict(edge_paths)
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
                        vis_options=vis_options,
                        src_tgt_ids=(src_id, tgt_id),
                        fig_data=fig_data,
                        filename_info=(circuit_name, len(edge_paths) + 1)
                        if vis_options[1] or debug == 4
                        else None,
                    )

                # Write to edge_paths if an edge is found
                # NB! As both (src_id, tgt_id) were already placed, pathfinder
                # will return a list with ONE single path if successful
                # or an empty clean_paths otherwise
                if clean_paths:
                    # Log run as success
                    edge_success = True

                    # Update edge paths
                    coords_in_path = [p[0] for p in clean_paths[0]]  # Take the first path
                    edge_type = zx_edge_type
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
                    nx_g = prune_beams(nx_g, taken)

                    # Update user if log_stats or debug mode are enabled
                    if log_stats_id or debug > 0:
                        volume = len([block for block in clean_paths[0] if "o" not in block[1]])
                        print(
                            f"Path between fixed cubes found: {src_id} -> {tgt_id} (vol += {volume - 2}) (runtime ~ {int(duration_iter * 1000)}ms)."
                        )
                else:
                    # Fill edge_paths with error
                    edge_success = False
                    edge = tuple(sorted((src_id, tgt_id)))
                    edge_paths[edge] = {
                        "src_tgt_ids": "error",
                        "path_coordinates": "error",
                        "path_nodes": "error",
                        "edge_type": "error",
                    }

                    # Return updated list of taken coords, edge_paths, and a fail/success flag
                    nx_g = prune_beams(nx_g, taken)

                    # Explicit warning if log_stats or debug are enabled
                    if log_stats_id or debug > 0:
                        print(
                            f"ERROR. Path between fixed cubes: {src_id} -> {tgt_id} (runtime ~ {int(duration_iter * 1000)}ms). Shut down unavoidable."
                        )

    return taken, edge_paths, edge_success


##############################
# LIASION W INNER PATHFINDER #
##############################
def run_pathfinder(
    src_block_info: StandardBlock,
    tgt_zx_type: str,
    init_step: int,
    taken: list[StandardCoord],
    tgt_block_info: StandardCoord | None = None,
    hdm: bool = False,
    min_succ_rate: int = 60,
    critical_beams: dict[int, tuple[int, CubeBeams]] = {},
    src_tgt_ids: tuple[int, int] | None = None,
    log_stats_id: str | None = None,
) -> tuple[
    list[Any],
    tuple[list[StandardCoord], list[str], dict[StandardBlock, list[StandardBlock]] | None],
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
        src_tgt_ids (optional): The exact IDs of the source and target cubes.
        log_stats_id (optional): A unique datetime-based identifier for the purposes of logging stats for an specific run.

    Returns:
        clean_paths: A list of paths each containing the 3D cubes and pipes needed to connect source and target in the 3D space.

    """

    # Edge path management
    pathfinder_vis_data = [None, None, None, None]
    valid_paths: dict[StandardBlock, list[StandardBlock]] | None = None
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
    max_step = init_step if tgt_block_info else 15
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
        if tent_coords:
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
        else:
            raise ValueError(f"tent_coords: {tent_coords}")

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


def validity_checks(simple_graph: SimpleDictGraph, first_cube: StandardBlock) -> bool:
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


def get_first_cube(
    nx_g: nx.Graph,
    first_cube: tuple[int | None, str | None] = (None, None),
    deterministic: bool = False,
    random_seed: int | None = None,
) -> tuple[int, str]:
    """Determine the iID and kind of the first block to place in 3D space.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        first_cube (optional): Override ID and kind (used to replicate specific cases).
        deterministic: A boolean flag to tell the function if choice is deterministic or random.
        random_seed: Typically `None`, but can be used to pass a specific seed across the entire algorithm.

    Returns:
        first_id: ID of the first block to place in 3D space
        first_kind: Kind of the first block to place in 3D space

    """

    first_id, first_kind = first_cube

    if (not first_id or not first_kind) and not deterministic and random_seed:
        random.seed(random_seed)

    if not first_id:
        first_id = find_first_id(nx_g, deterministic=deterministic)

    if not first_kind:
        tentative_kinds = nx_g.nodes[first_id].get("type_fam")
        first_kind = tentative_kinds[0] if deterministic else random.choice(tentative_kinds)

    return first_id, first_kind


def place_first_cube(
    nx_g: nx.Graph, taken: list[StandardCoord], first_cube: StandardBlock
) -> tuple[list[StandardCoord], nx.Graph]:
    """Place the first cube in the 3D space.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        first_cube: ID and kind for the very first spider/cube to place in 3D space.

    Returns:
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.

    """

    # Update taken
    taken.append((0, 0, 0))

    # Get beams
    first_id, first_kind = first_cube
    _, src_beams = check_exits((0, 0, 0), first_kind, taken, [(0, 0, 0)])

    # Write info to nx_g
    nx_g.nodes[first_id]["coords"] = (0, 0, 0)
    nx_g.nodes[first_id]["kind"] = first_kind
    nx_g.nodes[first_id]["beams"] = src_beams

    return nx_g, taken


def call_logger(
    circuit_info: list[str],
    outputs: list[Any],
    runtimes: list[float],
    metrics: list[int],
    misc_params: list[Any],
):
    """Call animation for iteration and log key stats.

    Args:
        circuit_info: A list containing the name of the ZX circuit and the log_stats_id to use for logging.
        outputs: Key outputs for graph manager including edge_paths, lat_nodes, and lat_edges.
        runtimes: Key runtime metrics for graph manager BFS.
        metrics: Key performance metrics for graph manager BFS.
        misc_params: Key settings needed for potential replicability.

    """

    # Extract key metrics
    circuit_name, log_stats_id, run_success = circuit_info
    edge_paths, lat_nodes, lat_edges = outputs
    duration_1st_pass, duration_2nd_pass, duration_total = runtimes
    num_input_nodes, num_input_edges, num_1st_pass_edges, num_2nd_pass_edges = metrics
    min_succ_rate, animate, kwargs = misc_params

    # Assemble objects for logger
    times = {
        "t_1st_pass": duration_1st_pass,
        "t_2nd_pass": duration_2nd_pass,
        "t_total": duration_total,
    }

    counts = {
        "num_input_nodes": num_input_nodes,
        "num_input_edges": num_input_edges,
        "num_input_nodes_processed": num_1st_pass_edges + 1,
        "num_input_edges_processed": num_1st_pass_edges + num_2nd_pass_edges,
        "num_1st_pass_edges_processed": num_1st_pass_edges,
        "num_2n_pass_edges_processed": num_2nd_pass_edges,
    }

    # Animate & log as appropriate
    if animate:
        create_animation(
            filename_prefix=f"{'' if run_success else 'FAIL_'}{circuit_name}",
            restart_delay=5000,
            duration=2500,
            video=True if animate == "MP4" else False,
        )

    # Log stats of failed attempt
    if log_stats_id is not None:
        try:
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
        except Exception as e:
            print(f"Unable to log stats for failed graph manager run: {e}")
