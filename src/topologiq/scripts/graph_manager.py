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
from typing import Any, cast

import matplotlib.figure
import networkx as nx
import numpy as np

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
from topologiq.utils.utils_misc import datetime_manager, init_bfs, prep_stats_n_log
from topologiq.utils.utils_pathfinder import check_exits, get_manhattan
from topologiq.utils.utils_zx_graphs import check_zx_types, get_zx_type_fam


###############################
# MAIN GRAPH MANAGER WORKFLOW #
###############################
def graph_manager_bfs(
    simple_graph: SimpleDictGraph,
    circuit_name: str = "circuit",
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
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).
        first_cube (optional): the ID and kind of the first cube to place in 3D space (used to replicate specific cases).
        **kwargs: !

    Returns:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        edge_paths: An edge-by-edge/block-by-block summary of the space-time diagram Topologiq builds.
        lat_nodes: The cubes of the final space-time diagram produced by Topologiq.
        lat_edges: The pipes of the final space-time diagram produced by Topologiq.

    """

    # 1. Initialise trackers & BFS variables
    t_1, _ = datetime_manager()
    t_total, t_std_edges, t_cross_edges = (0.0, 0.0, 0.0)
    zx_spiders_num, zx_edges_num = (len(simple_graph["nodes"]), len(simple_graph["edges"]))
    std_edges_processed, cross_edges_processed, num_edges_processed = (0, 0, 0)

    # First spider/cube
    nx_g = prep_3d_g(simple_graph)
    first_cube = get_first_cube(
        nx_g,
        first_cube=first_cube,
        deterministic=kwargs["deterministic"],
        random_seed=kwargs["seed"],
    )

    # BFS management
    queue, visited, taken, edge_paths, run_success = init_bfs(first_cube)

    # Outputs
    lat_nodes: dict[int, StandardBlock] | None = None
    lat_edges: dict[tuple[int, int], list[str]] | None = None

    # 2. Validity checks
    # Health check depating point
    if not validity_checks(simple_graph, first_cube):
        return nx_g, edge_paths, lat_nodes, lat_edges

    # 3. Place first spider/cube
    nx_g, taken = place_first_cube(nx_g, taken, first_cube)

    # 4. Graph manager BFS
    # Group parameters for readability
    duration_trackers = t_std_edges, t_cross_edges
    edge_trackers = std_edges_processed, num_edges_processed, cross_edges_processed
    trackers = duration_trackers, edge_trackers

    try:
        edge_paths, taken, run_success, trackers, _ = do_bfs(
            nx_g,
            queue,
            visited,
            taken,
            circuit_name,
            edge_paths,
            trackers,
            fig_data=fig_data,
            **kwargs,
        )

        # Reassemble trackers
        duration_trackers, edge_trackers = trackers
        t_std_edges, t_cross_edges = duration_trackers
        std_edges_processed, num_edges_processed, cross_edges_processed = edge_trackers

        # Taken is used extensively, so prune it again
        taken = list(set(taken))

        # Assemble final lattice if run is successfull
        if run_success:
            lat_nodes, lat_edges = reindex_path_dict(edge_paths)

        # Log stats
        if kwargs["log_stats_id"] is not None:
            _, t_total = datetime_manager(t_1=t_1)
            call_logger(
                [circuit_name, run_success, edge_paths, lat_nodes, lat_edges],
                [t_std_edges, t_cross_edges, t_total],
                [zx_spiders_num, zx_edges_num, std_edges_processed, cross_edges_processed],
                **kwargs,
            )

        if not run_success:
                raise ValueError("ERROR. Graph manager ran but reported an successful outcome.")

    except Exception as e:
        raise ValueError("ERROR. The graph_manager BFS crashed.", e)

    return nx_g, edge_paths, lat_nodes, lat_edges


def do_bfs(
    nx_g: nx.Graph,
    queue: deque,
    visited: set,
    taken: list[StandardCoord],
    circuit_name: str,
    edge_paths: dict,
    trackers: list[any],
    fig_data: matplotlib.figure.Figure | None = None,
    **kwargs,
):
    """Undertake a BFS search of a ZX graph.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        queue: The main BFS queue.
        visited: The main BFS set of visited sites.
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        circuit_name: The name of the ZX circuit.
        edge_paths: An edge-by-edge/block-by-block summary of the space-time diagram Topologiq builds.
        trackers: A list containing several trackers used to track Topologiq statistics.
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).
        **kwargs: !

    Returns:
        edge_paths: Updated edge-by-edge/block-by-block summary of the space-time diagram Topologiq builds.
        taken: Updated list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        run_success: Boolean flag to determine if Whether the BFS search was successful as a whole.
        trackers: A list containing several trackers used to track Topologiq statistics.

    """

    # Extract various params
    duration_trackers, edge_trackers = trackers
    t_std_edges, t_cross_edges = duration_trackers
    std_edges_processed, num_edges_processed, cross_edges_processed = edge_trackers

    # Start the queue
    twins = {}
    run_success = False
    while queue:
        _, _, priority_ids = check_path_to_beam_clashes(nx_g, 0, 0, taken, priority_ids=[])
        priority_ids = check_multi_beam_clashes(nx_g, priority_ids=priority_ids)
        priority_ids = list(set(priority_ids))

        if priority_ids:
            if kwargs["log_stats_id"] or kwargs["debug"] > 0:
                print(Colors.BLUE, "==> Adding twin nodes for IDs:" + Colors.RESET, priority_ids)

            for priority_id in priority_ids:
                twin_id = max(nx_g.nodes) + 1
                twins[priority_id] = twin_id

                nx_g.add_node(
                    twin_id,
                    type=nx_g.nodes[priority_id]["type"],
                    type_fam=nx_g.nodes[priority_id]["type_fam"],
                    kind=None,
                    coords=None,
                    beams=None,
                    beams_short=None,
                    completed=0,
                )

                twin_pending_neighs = [
                    n
                    for n in nx_g.neighbors(priority_id)
                    if tuple(sorted((n, priority_id))) not in list(edge_paths.keys())
                ]

                nx_g.add_edge(priority_id, twin_id, type="SIMPLE")
                for twin_neigh_id in twin_pending_neighs:
                    edge_type = nx_g.get_edge_data(priority_id, twin_neigh_id)
                    nx_g.remove_edge(priority_id, twin_neigh_id)
                    nx_g.add_edge(twin_id, twin_neigh_id, type=edge_type)

                # Try to place twin slightly away from current blockgraph
                taken = list(set(taken))
                step, max_step = (6, 15)
                while step <= max_step:
                    nx_g, taken, edge_paths, edge_success = place_nxt_block(
                        priority_id,
                        twin_id,
                        nx_g,
                        taken,
                        edge_paths,
                        circuit_name=circuit_name,
                        init_step=step,
                        fig_data=fig_data,
                        **kwargs,
                    )

                    # Move to next if there is a succesful placement
                    if edge_success:
                        nx_g = prune_beams(nx_g, taken)
                        break

                    if step >= max_step:
                        print(
                            Colors.RED,
                            "==> Failed to add twin nodes:" + Colors.RESET,
                            priority_ids,
                        )

                    # Increase distance between nodes if placement not possible
                    step += 3

            # Re-write queue to exchange priority IDs with new twin IDs
            new_queue = []
            while queue:
                next_in_queue = queue.popleft()
                if next_in_queue in priority_ids:
                    new_queue.append(twins[next_in_queue])
                    visited.add(twins[next_in_queue])
                else:
                    new_queue.append(next_in_queue)
            queue.extend(new_queue)
            priority_ids = []

        # Get first cube from queue
        src_id: int = queue.popleft()

        # Iterate over neighbours of current source
        fixed_list_of_neighs = cast(list[int], nx_g.neighbors(src_id))
        for unsanitised_tgt_id in fixed_list_of_neighs:
            if unsanitised_tgt_id in twins:
                tgt_id = (
                    unsanitised_tgt_id
                    if src_id == twins[unsanitised_tgt_id]
                    else twins[unsanitised_tgt_id]
                )
            else:
                tgt_id = unsanitised_tgt_id

            # Handle cubes that need to be placed for the first time
            if tgt_id not in visited:
                # Start iteration timer
                t_1_std_edge_iter, _ = datetime_manager()

                # Add/append ID to visited and queue
                queue.append(tgt_id)
                visited.add(tgt_id)

                # Ensure taken has unique entries on each run
                taken = list(set(taken))

                # Try to place blocks as close to one another as as possible
                step, max_step = (3, 15)
                while step <= max_step:
                    nx_g, taken, edge_paths, edge_success = place_nxt_block(
                        src_id,
                        tgt_id,
                        nx_g,
                        taken,
                        edge_paths,
                        circuit_name=circuit_name,
                        init_step=step,
                        fig_data=fig_data,
                        **kwargs,
                    )

                    # Stop iteration timers
                    _, duration_std_edge_iter = datetime_manager(t_1=t_1_std_edge_iter)
                    t_std_edges += duration_std_edge_iter

                    # Move to next if there is a succesful placement
                    if edge_success:
                        run_success = True
                        std_edges_processed += 1
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

            elif (src_id, tgt_id) not in edge_paths and (tgt_id, src_id) not in edge_paths:

                # Start iteration timer for 2st pass iteration
                t_1_cross_edge_iter, _ = datetime_manager()

                # Trigger connection for previously placed cubes
                nx_g, taken, edge_paths, edge_success = connect_prev_placed_cubes(
                    src_id,
                    tgt_id,
                    nx_g,
                    taken,
                    edge_paths,
                    circuit_name=circuit_name,
                    fig_data=fig_data,
                    **kwargs,
                )

                # Stop & log times
                _, duration_cross_edge_iter = datetime_manager(t_1=t_1_cross_edge_iter)
                t_cross_edges += duration_cross_edge_iter

                if edge_success:
                    run_success = True
                    cross_edges_processed += 1
                    num_edges_processed += 1

                else:
                    run_success = False
                    break

        if run_success is False:
            break

    duration_trackers = t_std_edges, t_cross_edges
    edge_trackers = std_edges_processed, num_edges_processed, cross_edges_processed
    trackers = duration_trackers, edge_trackers

    return edge_paths, taken, run_success, trackers, visited


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
    fig_data: matplotlib.figure.Figure | None = None,
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
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).
        **kwargs: !

    Returns:
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        edge_paths: An edge-by-edge summary of the 3D object Topologiq builds, updated to the last edge processsed successfully.
        (bool): A boolean flag to signal success (True if placement was succesful).

    """

    # Start timer
    t_1, _ = datetime_manager()

    # Always prune beams to ensure recent placements are accounted for
    nx_g = prune_beams(nx_g, taken)

    # Get source cube data
    src_coords: StandardCoord | None = nx_g.nodes[src_id].get("coords")
    src_kind: str | None = nx_g.nodes[src_id].get("kind")
    if src_coords is None or src_kind is None:
        return nx_g, taken, edge_paths, False
    src_block_info: StandardBlock = (src_coords, src_kind)

    # Check position of target cube (should be None)
    nxt_neigh_coords: StandardCoord | None = nx_g.nodes[tgt_id].get("coords")

    # Process targets that have yet to be placed in the 3D space
    edge_success = False
    if nxt_neigh_coords is None:
        # Geat target information
        nxt_neigh_node_data = nx_g.nodes[tgt_id]
        nxt_neigh_zx_type: str = cast(str, nxt_neigh_node_data.get("type"))

        # Get edge information
        zx_edge_type = nx_g.get_edge_data(src_id, tgt_id).get("type")
        hdm: bool = True if zx_edge_type == "HADAMARD" else False

        # Remove source coordinates from occupied coords
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
            src_tgt_ids=(src_id, tgt_id),
            **kwargs,
        )

        # Assemble a preliminary dictionary of viable paths
        # Note. A smart subset of clean paths
        viable_paths = []
        tgt_degree = int(get_node_degree(nx_g, tgt_id))

        for clean_path in clean_paths:
            # Extract key path information
            tgt_coords, tgt_kind = clean_path[-1]
            coords_in_path = get_taken_coords(clean_path)

            # Check if exits are unobstructed
            tgt_unobstr_exit_n, tgt_beams, tgt_beams_short = check_exits(
                tgt_coords, tgt_kind, taken_coords_c, coords_in_path
            )

            # Check path doesn't obstruct an absolutely necessary exit for a pre-existing cube
            # Reset # of unobstructed exits and node beams if target is a boundary
            if nxt_neigh_zx_type == "O":
                tgt_unobstr_exit_n, tgt_beams = (6, [])

            if tgt_unobstr_exit_n >= tgt_degree - 1:
                # Check if path breaks more beams than tolerable
                path_to_beam_clashes, beams_broken_by_path, _ = check_path_to_beam_clashes(
                    nx_g, src_id, tgt_id, coords_in_path, strict=False
                )

                # Check if there are more beam-to-beam clashes than tolerable
                tgt_beam_clashes, beams_broken_by_path = check_tgt_beam_clashes(
                    nx_g,
                    src_id,
                    tgt_id,
                    tgt_beams,
                    tgt_beams_short,
                    tgt_degree,
                    beams_broken_by_path,
                    **kwargs,
                )

                # Append path to viable paths if path clears all checks
                if path_to_beam_clashes is not True and tgt_beam_clashes is not True:
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
                        "tgt_beams_short": tgt_beams_short,
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
        _, t_total_iter = datetime_manager(t_1=t_1)

        # For visualisation, create a new graph on each step
        if kwargs["debug"] > 0 or kwargs["vis_options"][0] == "detail" or kwargs["vis_options"][1]:

            # Call visualisation
            call_debug_vis(
                circuit_name,
                nx_g,
                edge_paths,
                winner_path,
                None,
                (src_id, tgt_id),
                src_block_info,
                pathfinder_vis_data,
                fig_data=fig_data,
                **kwargs,
            )

        # Write to edge_paths if winner is found
        if winner_path:
            nx_g, taken, edge_paths, edge_success = update_edge_paths(
                nx_g, edge_paths, winner_path, clean_paths, taken, zx_edge_type, src_id, tgt_id
            )

        # Update user
        if kwargs["log_stats_id"] or kwargs["debug"] > 0:
            volume = (
                len([i for i in winner_path.all_nodes_in_path if "o" not in i[1]])
                if winner_path
                else 0
            )
            print(
                f"ADD CUBE: {src_id} -> {tgt_id}.",
                (Colors.GREEN + "Success." + Colors.RESET)
                if edge_success
                else f"{(Colors.YELLOW + 'Increasing search distance.' + Colors.RESET) if init_step < 15 else (Colors.RED + 'FAIL.' + Colors.RESET)}",
                f"Vol: {volume - 1}." if edge_success else "",
                f"Runtime: ~{int(t_total_iter * 1000)}ms.",
            )

    return nx_g, taken, edge_paths, edge_success


def connect_prev_placed_cubes(
    src_id: int,
    tgt_id: int,
    nx_g: nx.Graph,
    taken: list[StandardCoord],
    edge_paths: dict,
    circuit_name: str = "circuit",
    fig_data: matplotlib.figure.Figure | None = None,
    **kwargs,
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
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).
        **kwargs: !

    Returns:
        taken: Updated list of all coordinates occupied by any blocks/pipes, including any placed by this function iteration.
        edge_paths: An edge-by-edge summary of the 3D object Topologiq builds, updated to the last edge processsed successfully.
        edge_success: A boolean flag declaring whether the edge was built successfully or not.

    """
    # Start timer
    t_1, _ = datetime_manager()

    # Prune taken and beams
    edge_success = False
    taken = list(set(taken))
    nx_g = prune_beams(nx_g, taken)

    # Get source and target data for current (src_id, tgt_id) pair
    u_coords, v_coords = (nx_g.nodes[src_id].get("coords"), nx_g.nodes[tgt_id].get("coords"))

    # Process edge only if both src_id and tgt_id have already been placed in the 3D space
    # Note. Function should never run into (src_id, tgt_id) pairs not already in 3D space
    if u_coords is not None and v_coords is not None:
        # Format adjustments to match existing operations
        u_kind = cast(str, nx_g.nodes[src_id].get("kind"))
        v_zx_type = cast(str, nx_g.nodes[tgt_id].get("type"))
        edge = tuple(sorted((src_id, tgt_id)))

        # Call pathfinder on any graph edge that does not have an entry in edge_paths
        if edge not in edge_paths:
            critical_beams = assemble_critical_beams(nx_g)

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
                    critical_beams=critical_beams,
                    src_tgt_ids=(src_id, tgt_id),
                    **kwargs,
                )
                clean_paths = []  # xyz
                # Finish timer before popping up visualisation
                _, t_total_iter = datetime_manager(t_1=t_1)

                # For visualisation, create a new graph on each step irrespective of outcome
                if (
                    kwargs["debug"] > 1
                    or kwargs["vis_options"][0] == "detail"
                    or kwargs["vis_options"][1]
                ):
                    # Call visualisation
                    call_debug_vis(
                        circuit_name,
                        nx_g,
                        edge_paths,
                        None,
                        clean_paths[0] if clean_paths else None,
                        (src_id, tgt_id),
                        (u_coords, u_kind),
                        pathfinder_vis_data,
                        fig_data=fig_data,
                        **kwargs,
                    )

                # Write to edge_paths if an edge is found
                nx_g, taken, edge_paths, edge_success = update_edge_paths(
                    nx_g,
                    edge_paths,
                    None,
                    clean_paths[0] if clean_paths else None,
                    taken,
                    zx_edge_type,
                    src_id,
                    tgt_id,
                    second_pass=True,
                )

                # Update user
                if kwargs["log_stats_id"] or kwargs["debug"] > 0:
                    volume = (
                        len([i for i in clean_paths[0] if "o" not in i[1]]) if edge_success else 0
                    )
                    print(
                        f"CONNECT PRE-EXISTING CUBES: {src_id} -> {tgt_id}.",
                        (Colors.GREEN + "Success." + Colors.RESET)
                        if edge_success
                        else Colors.RED + "FAIL." + Colors.RESET,
                        f"Vol: {volume - 2}." if edge_success else "",
                        f"Runtime: ~{int(t_total_iter * 1000)}ms.",
                    )

    nx_g = prune_beams(nx_g, taken)
    return nx_g, taken, edge_paths, edge_success


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
    critical_beams: dict[StandardCoord, int, tuple[int, CubeBeams], tuple[int, CubeBeams]] = {},
    src_tgt_ids: tuple[int, int] | None = None,
    **kwargs,
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
        critical_beams (optional): Annotated beams object with details about minimum number of beams needed per node.
        src_tgt_ids (optional): The exact IDs of the source and target cubes.
        **kwargs: !

    Returns:
        clean_paths: A list of paths each containing the 3D cubes and pipes needed to connect source and target in the 3D space.
        pathfinder_vis_data: A list containing data for visualisation of a given pathfinder run.

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
                critical_beams=critical_beams,
                src_tgt_ids=src_tgt_ids,
                **kwargs,
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
    nx_g: nx.Graph,
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


def call_logger(
    circuit_info: list[str],
    runtimes: list[float],
    metrics: list[int],
    **kwargs,
):
    """Call animation for iteration and log key stats.

    Args:
        circuit_info: A list containing name of circuit, log_stats_id, edge_paths, lat_nodes, and lat_edges.
        runtimes: Key runtime metrics for graph manager BFS.
        metrics: Key performance metrics for graph manager BFS.
        **kwargs: !

    """

    # Extract key metrics
    circuit_name, run_success, edge_paths, lat_nodes, lat_edges = circuit_info
    t_std_edges, t_cross_edges, t_total = runtimes
    zx_spiders_num, zx_edges_num, std_edges_processed, cross_edges_processed = metrics

    # Assemble objects for logger
    times = {
        "t_std_edges": t_std_edges,
        "t_cross_edges": t_cross_edges,
        "t_total": t_total,
    }

    counts = {
        "zx_spiders_num": zx_spiders_num,
        "zx_edges_num": zx_edges_num,
        "std_edges_processed": std_edges_processed,
        "cross_edges_processed": cross_edges_processed,
    }

    # Animate & log as appropriate
    if kwargs["vis_options"][1]:
        create_animation(
            filename_prefix=f"{'' if run_success else 'FAIL_'}{circuit_name}",
            restart_delay=5000,
            duration=2500,
            video=True if kwargs["vis_options"][1] == "MP4" else False,
        )

    # Log stats of failed attempt
    if kwargs["log_stats_id"] is not None:
        try:
            prep_stats_n_log(
                "graph_manager",
                run_success,
                counts,
                times,
                circuit_name=circuit_name,
                edge_paths=edge_paths,
                lat_nodes=lat_nodes,
                lat_edges=lat_edges,
                **kwargs,
            )
        except Exception as e:
            print(f"Unable to log stats for failed graph manager run: {e}")


def call_debug_vis(
    circuit_name: str,
    nx_g: nx.Graph,
    edge_paths: dict,
    winner_path_standard_pass: PathBetweenNodes | None,
    winner_path_second_pass: list[StandardBlock] | None,
    src_tgt_ids: tuple[int, int] | None,
    src_block_info: StandardBlock,
    pathfinder_vis_data: tuple[Any],
    fig_data: matplotlib.figure.Figure | None = None,
    **kwargs,
):
    """Assemble objects for and call intermediate 'debug' visualisation.

    Args:
        circuit_name: The name of the ZX circuit.
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        edge_paths: An edge-by-edge summary of the 3D object Topologiq builds, updated to the last edge processsed successfully.
        winner_path_standard_pass: A winner path chosen by the value function.
        winner_path_second_pass: A list of paths returned by the pathfinder algorithm.
        src_tgt_ids: tuple[int, int] | None = None,
        src_block_info: The information of the source cube including its position in the 3D space and its kind.
        pathfinder_vis_data: A list containing data for visualisation of a given pathfinder run.
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).
        **kwargs: !

    """

    # Unpack vis data
    tent_coords, tent_tgt_kinds, all_search_paths, valid_paths = pathfinder_vis_data
    winner_path = (
        winner_path_standard_pass
        if winner_path_standard_pass
        else winner_path_second_pass
        if winner_path_second_pass
        else None
    )

    # Turn debug mode ON if OFF (happens if function is called via `vis_options`).
    kwargs["debug"] = kwargs["debug"] if kwargs["debug"] > 0 else 1

    # Create partial progress graph from current edges
    partial_lat_nodes, partial_lat_edges = reindex_path_dict(edge_paths, fix_errors=True)
    partial_nx_g, _ = lattice_to_g(partial_lat_nodes, partial_lat_edges, nx_g)

    # Call visualisation
    vis_3d(
        nx_g,
        partial_nx_g,
        edge_paths,
        valid_paths if valid_paths else None,
        winner_path,
        src_block_info,
        tent_coords,
        tent_tgt_kinds,
        all_search_paths=all_search_paths,
        src_tgt_ids=src_tgt_ids,
        fig_data=fig_data,
        filename_info=(circuit_name, len(edge_paths) + 1)
        if kwargs["vis_options"][1] or kwargs["debug"] == 4
        else None,
        **kwargs,
    )


def check_path_to_beam_clashes(
    nx_g: nx.Graph,
    src_id: int,
    tgt_id: int,
    coords_in_path: list[StandardCoord],
    beams_broken_by_path: int | None = None,
    priority_ids: list[int | None] | None = None,
    strict: bool = True,
) -> tuple[bool, int, list[int | None] | None]:
    """Determine if placement triggers critical multi-beam clashes.

    This function checks if a given placement blocks more beams that tolerable.
    A single beam being broken is not necessarily a problem, as some cubes can lose
    some beams. However, if a new placement breaks more beams than what any one cube
    can lose, it will become impossible to make all connections for the said cube.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        src_id: The ID of the current source cube.*
            To use this function to test all of `taken`, feed any int and repeat the same int in tgt_id.
        tgt_id: The ID of the potential target cube.*
            To use this function to test all of `taken`, feed src_id again.
        coords_in_path: All coords in the current path.*
            To use this function to test all of `taken`, feed `taken` as `coords_in_path`.
        beams_broken_by_path (optional): A pre-existent count of broken beams.
        priority_ids (optional): A list of spider IDs with one or another kind of conflict.
        strict (optional): Whether to perform a strict or loose check.

    Returns:
        clash: False if no critical beam clashed found, else True.
        beams_broken_by_path: Accumulated total number of beams for which path creates some kind of problem.
        priority_ids: Cube IDs flagged as potentially problematic.

    """

    #####

    # Initialise beams broken by path if it hasn't been initialised
    beams_broken_by_path = 0 if beams_broken_by_path is None else beams_broken_by_path

    # Default to False in case no cubes with beams exist
    clash = False
    priority_ids = priority_ids if priority_ids else []

    # Loop over all cubes in 3D space
    for cube_id in nx_g.nodes():
        if nx_g.nodes[cube_id]["beams"] if strict else nx_g.nodes[cube_id]["beams_short"]:
            # Use infinite or short beams according to `strict`
            beams_to_check = (
                nx_g.nodes[cube_id]["beams"] if strict else nx_g.nodes[cube_id]["beams_short"]
            )
            cube_broken_count = 0
            cube_degree = get_node_degree(nx_g, cube_id)
            cube_pending_edges = cube_degree - nx_g.nodes[cube_id]["completed"]
            for beam in beams_to_check:
                if any([beam.contains(coord) for coord in coords_in_path]):
                    beams_broken_by_path += 1
                    cube_broken_count += 1

            # Append to priority IDs for all cubes with problems
            # Flip check if even ONE cube has problems
            src_tgt_adjust = 1 if (cube_id in [src_id, tgt_id] and src_id != tgt_id) else 0
            if len(beams_to_check) - cube_broken_count + src_tgt_adjust < cube_pending_edges:
                priority_ids.append(cube_id)
                clash = True

    return clash, beams_broken_by_path, priority_ids


def check_path_to_beam_clashes_old(
    nx_g: nx.Graph,
    src_id: int,
    coords_in_path: list[StandardCoord],
    beams_broken_by_path: int | None = None,
    strict: bool = True,
) -> bool:
    """Determine if placement triggers critical multi-beam clashes.

    This function checks if a given placement blocks more beams that tolerable.
    A single beam being broken is not necessarily a problem, as some cubes can lose
    some beams. However, if a new placement breaks more beams than what any one cube
    can lose, it will become impossible to make all connections for the said cube.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        src_id: The ID of the current source cube.
        coords_in_path: All coords in the current path.
        beams_broken_by_path (optional): A pre-existent count of broken beams.
        strict (optional): Whether to perform a strict or loose check.

    Returns:
        clash: False if no critical beam clashed found, else True.
        beams_broken_by_path: Accumulated total number of beams for which path creates some kind of problem.
        priority_ids: Cube IDs flagged as potentially problematic.

    """

    # Initialise beams broken by path if it hasn't been initialised
    beams_broken_by_path = 0 if beams_broken_by_path is None else beams_broken_by_path

    # Check for clashes
    clash = False
    priority_ids = []
    for n_id in nx_g.nodes():
        broken = 0  # Tracks current cube

        # Only check if beams exist
        if nx_g.nodes[n_id]["beams"]:
            # Establish minimum needs
            num_beams = len(nx_g.nodes[n_id]["beams"])
            n_degree = get_node_degree(nx_g, n_id)
            n_edges_completed = nx_g.nodes[n_id]["completed"]
            num_edges_still_to_complete = n_degree - n_edges_completed

            # Count clashes
            if strict:
                for single_beam in nx_g.nodes[n_id]["beams"]:
                    if any([single_beam.contains(c) for c in coords_in_path]):
                        beams_broken_by_path += 1
                        broken += 1
            else:
                for single_beam in nx_g.nodes[n_id]["beams_short"]:
                    if any([single_beam.contains(c) for c in coords_in_path]):
                        beams_broken_by_path += 1
                        broken += 1

            # Evaluate tolerances
            adjust_for_source_node = 1 if n_id == src_id else 0

            if strict:
                if num_beams - broken + adjust_for_source_node < num_edges_still_to_complete:
                    clash = True
                    if strict:
                        priority_ids.append(n_id)

    return clash, beams_broken_by_path, priority_ids


def check_tgt_beam_clashes(
    nx_g: nx.Graph,
    src_id: int,
    tgt_id: int,
    tgt_beams: CubeBeams,
    tgt_beams_short: CubeBeams,
    tgt_degree: int,
    beams_broken_by_path: int = 0,
    strict: bool = True,
    **kwargs,
) -> bool:
    """Determine if placement triggers critical multi-beam clashes.

    This function checks if a given placement blocks more beams that tolerable.
    A single beam being broken is not necessarily a problem, as some cubes can lose
    some beams. However, if a new placement breaks more beams than what any one cube
    can lose, it will become impossible to make all connections for the said cube.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        src_id: The ID of the current source cube.
        tgt_id: The ID of the potential target cube.
        tgt_beams: The beams of the potential target cube.
        tgt_beams_short: The short beams of the potential target cube.
        tgt_degree: The number of neighbours of the potential target cube.
        beams_broken_by_path (optional): A pre-existent count of broken beams.
        strict (optional): Whether to perform a strict or loose check.
        **kwargs: !

    Returns:
        clash: False if no critical beam clashed found, else True.
        beams_broken_by_path: Accumulated total number of beams for which path creates some kind of problem.

    """

    # Aux params
    clash = False
    tgt_beams_to_check = tgt_beams if strict else tgt_beams_short

    # Check tgt against beams of each other cube in 3D space
    if tgt_beams_to_check:
        for cube_id in nx_g.nodes():
            # Reset trackers on every cube irrespectively
            tgt_clash_tracker = np.array([False for _ in tgt_beams_short])
            clash = False
            cube_clash_count = 0

            # Count clashes for cubes with beams
            if cube_id not in (src_id, tgt_id) and (
                nx_g.nodes[cube_id]["beams"] if strict else nx_g.nodes[cube_id]["beams_short"]
            ):
                cube_degree = get_node_degree(nx_g, cube_id)
                cube_pending_edges = cube_degree - nx_g.nodes[cube_id]["completed"]
                for cube_beam in nx_g.nodes[cube_id]["beams_short"]:
                    if nx_g.nodes[cube_id]["beams_short"]:
                        intersections = [
                            tgt_beam.intersects(cube_beam, kwargs["beams_len_short"])
                            for tgt_beam in tgt_beams_short
                        ]
                        tgt_clash_tracker = tgt_clash_tracker + np.array(intersections)
                        cube_clash_count += 1 if any(intersections) else 0

                src_tgt_adjust = 1 if cube_id in [src_id, tgt_id] else 0
                if (
                    len(nx_g.nodes[cube_id]["beams"]) - cube_clash_count + src_tgt_adjust
                    < cube_pending_edges
                ):
                    beams_broken_by_path += 1
                    clash = True

        if len(tgt_beams) - sum(tgt_clash_tracker) < tgt_degree - 1:
            beams_broken_by_path += 1
            clash = True

    return clash, beams_broken_by_path


def check_tgt_beam_clashes_old(
    nx_g: nx.Graph,
    src_id: int,
    tgt_id: int,
    tgt_coords: StandardCoord,
    tgt_beams: CubeBeams,
    tgt_degree: int,
    beams_broken_by_path: int = 0,
    strict: bool = True,
) -> bool:
    """Determine if placement triggers critical multi-beam clashes.

    This function checks if a given placement blocks more beams that tolerable.
    A single beam being broken is not necessarily a problem, as some cubes can lose
    some beams. However, if a new placement breaks more beams than what any one cube
    can lose, it will become impossible to make all connections for the said cube.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        src_id: The ID of the current source cube.
        tgt_id: The ID of the potential target cube.
        tgt_coords: The coords of the potential target cube.
        tgt_beams: The beams of the potential target cube.
        tgt_degree: The number of neighbours of the potential target cube.
        beams_broken_by_path (optional): A pre-existent count of broken beams.
        strict (optional): Whether to perform a strict or loose check.

    Returns:
        clash: False if no critical beam clashed found, else True.
        beams_broken_by_path: Accumulated total number of beams for which path creates some kind of problem.

    """

    # Assume no clashes
    clash = False

    # Check beams of all cubes against target beams
    tgt_num_beams = len(tgt_beams)
    beam_tracker = np.array([False for _ in tgt_beams])  # Tracks target
    for n_id in nx_g.nodes():
        if (
            n_id not in (src_id, tgt_id)
            and (nx_g.nodes[n_id]["beams"] if strict else nx_g.nodes[n_id]["beams_short"])
            and nx_g.nodes[n_id]["coords"]
        ):
            cube_beams = nx_g.nodes[n_id]["beams"] if strict else nx_g.nodes[n_id]["beams_short"]
            manhattan_between = get_manhattan(tgt_coords, nx_g.nodes[n_id]["coords"])
            if tgt_beams:
                for beam in cube_beams:
                    broken_beams = [
                        beam.intersects(single_tgt_beam, manhattan_between)
                        for single_tgt_beam in tgt_beams
                    ]
                    beam_tracker = beam_tracker + np.array(broken_beams)
                    beams_broken_by_path += sum(broken_beams)

    if tgt_num_beams - sum(beam_tracker) < tgt_degree - 1:
        clash = True

    return clash, beams_broken_by_path


def check_multi_beam_clashes(
    nx_g: nx.Graph,
    priority_ids: list[int | None] | None = None,
    strict: bool = True,
) -> bool:
    """Determine if there are critical beam clashes for any given node.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        priority_ids (optional): A list of spider IDs with one or another kind of conflict.
        strict (optional): Whether to perform a strict or loose check.

    Returns:
        priority_ids: Cube IDs flagged as potentially problematic.

    """

    # Check beams of all cubes against target beams
    priority_ids = priority_ids if priority_ids else []
    for out_id in nx_g.nodes():
        if (
            nx_g.nodes[out_id]["beams"] if strict else nx_g.nodes[out_id]["beams_short"]
        ) and nx_g.nodes[out_id]["coords"]:
            out_coords = nx_g.nodes[out_id]["coords"]
            out_beams = nx_g.nodes[out_id]["beams"] if strict else nx_g.nodes[out_id]["beams_short"]
            out_beams_num = len(out_beams)
            out_degree = get_node_degree(nx_g, out_id)
            out_pending = out_degree - nx_g.nodes[out_id]["completed"]

            out_tracker = np.array([False for beam in out_beams])
            for in_id in nx_g.nodes():
                inner_count = 0  # Tracks each "in" cube
                if (
                    nx_g.nodes[out_id]["beams"] if strict else nx_g.nodes[in_id]["beams_short"]
                ) and nx_g.nodes[in_id]["coords"]:
                    in_coords = nx_g.nodes[in_id]["coords"]
                    in_beams = (
                        nx_g.nodes[in_id]["beams"] if strict else nx_g.nodes[in_id]["beams_short"]
                    )
                    in_beams_num = len(in_beams)
                    in_degree = get_node_degree(nx_g, in_id)
                    in_pending = in_degree - nx_g.nodes[in_id]["completed"]
                    manhattan_between = get_manhattan(out_coords, in_coords)

                    for beam in in_beams:
                        broken_beams = [
                            beam.intersects(out_beam, manhattan_between) for out_beam in out_beams
                        ]
                        out_tracker = out_tracker + np.array(broken_beams)

                        inner_count += sum(broken_beams)
                        if in_beams_num - inner_count < in_pending:
                            priority_ids.append(in_id)

            if out_beams_num - sum(out_tracker) < out_pending:
                priority_ids.append(out_id)

    return priority_ids


def assemble_critical_beams(
    nx_g: nx.Graph,
) -> dict[StandardCoord, int, tuple[int, CubeBeams], tuple[int, CubeBeams]]:
    """Assemble a dictionary of beams and related information.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.

    Returns:
        critical_beams: A dictionary containing beams to and key related information, to be used by the pathfinder.
            Keys:
                cube_id: The ID of an arbitrary cube.
            Values:
                cube_coords: The coordinates of the cube.
                cube_pending_edges: The number of edges still needed by the particular cube.
                cube_beams: The infinite beams for the specific node.
                cube_beams_short: The short beams for the specific node.

    """

    critical_beams = {}
    for cube_id in nx_g.nodes():
        cube_coords = nx_g.nodes[cube_id]["coords"]
        cube_beams = nx_g.nodes[cube_id]["beams"]
        cube_beams_short = nx_g.nodes[cube_id]["beams_short"]
        if cube_beams or cube_beams_short:
            cube_degree = get_node_degree(nx_g, cube_id)
            cube_pending_edges = cube_degree - nx_g.nodes[cube_id]["completed"]
            if cube_pending_edges != 0:
                critical_beams[cube_id] = (
                    cube_coords,
                    cube_pending_edges,
                    cube_beams,
                    cube_beams_short,
                )

    return critical_beams


def update_edge_paths(
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
    edge_type_match = zx_edge_type == nx_g.get_edge_data(src_id, tgt_id).get("type")

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

    # Fill edge_paths with error if no paths available
    else:
        edge_paths[edge] = {
            "src_tgt_ids": "error",
            "path_coordinates": "error",
            "path_nodes": "error",
            "edge_type": "error",
        }

    # Prune beams before moving to next edge
    nx_g = prune_beams(nx_g, taken)
    return nx_g, taken, edge_paths, edge_success
