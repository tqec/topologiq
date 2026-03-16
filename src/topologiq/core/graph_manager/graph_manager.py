"""Primary operations for the outer graph manager BFS.

This file contains functions that altogether determine the first spider to process,
the order in which subsequent spiders get processed, calling the inner pathfinder
in the appropriate mode for any given edge, and selecting a winner path from any
paths returned by the pathfinder.

Usage:
    Call `runner()` programmatically from a separate script to trigger the BFS flow.

Notes:
    For now, none of the functions in this file are to be called individually.
    In the future, some of the functions could be called by variant algorithms that
        do not necessarily need or want to implement all separate features.

"""

import traceback
from collections import deque
from pathlib import Path
from typing import cast

import matplotlib.figure
import networkx as nx

from topologiq.core.graph_manager.beams import check_need_for_twins
from topologiq.core.graph_manager.callers import call_logger
from topologiq.core.graph_manager.edge_handlers import add_twin, handle_cross_edge, handle_std_edge
from topologiq.core.graph_manager.first_cube import get_first_cube, place_first_cube
from topologiq.core.graph_manager.kwargs import check_assemble_kwargs
from topologiq.core.graph_manager.utils import (
    init_bfs,
    prep_3d_g,
    prune_beams,
    reindex_path_dict,
    rm_temp_files,
    validity_checks,
)
from topologiq.input.simple_graphs import break_single_spider_graph, strip_boundaries
from topologiq.utils.classes import Colors, SimpleDictGraph, StandardBlock, StandardCoord
from topologiq.utils.core import datetime_manager
from topologiq.utils.read_write import write_bgraph
from topologiq.vis.animation import create_animation
from topologiq.vis.blockgraph import vis_3d
from topologiq.vis.common import lattice_to_g

#########
# PATHS #
#########
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent.parent
OUTPUT_DIR_PATH = REPO_ROOT / "output/txt"
TEMP_DIR_PATH = REPO_ROOT / "output/temp"


##########
# RUNNER #
##########
def runner(
    simple_graph: SimpleDictGraph,
    circuit_name: str,
    fig_data: matplotlib.figure.Figure | None = None,
    first_cube: tuple[int | None, str | None] = (None, None),
    **kwargs,
) -> tuple[
    SimpleDictGraph,
    None | dict,
    None | dict[int, StandardBlock],
    None | dict[tuple[int, int], list[str]],
]:
    """Run Topologiq on an arbitrary circuit provided as `simple_graph`.

    This function calls any available and applicable pre-processing on the input `simple_graph`,
    then sends the graph for processing by Topologiq. If the algorith succeeds, the function
    gathers and returns the main outputs produced by Topologiq.

    Args:
        simple_graph: The `simple_graph` form of an arbitrary ZX circuit.
        circuit_name: The name of the ZX circuit.
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).
        first_cube (optional): The ID and kind of the first cube to place in 3D space (used to replicate specific cases).
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

    Returns:
        simple_graph: The original `simple_graph` given to function (returned for ease of use and traceability).
        edge_paths: An edge-by-edge summary of the 3D object Topologiq builds, updated to the last edge processsed successfully.
        lat_nodes: The cubes of the final space-time diagram produced by Topologiq.
        lat_edges: The pipes of the final space-time diagram produced by Topologiq.

    """

    # Preliminaries
    t_1, _ = datetime_manager()
    Path(OUTPUT_DIR_PATH).mkdir(parents=True, exist_ok=True)

    # Assemble kwargs if not detected
    kwargs = check_assemble_kwargs(**kwargs)

    # Optimise incoming graph if applicable (or leave unchanged if no optimisation is available)
    simple_graph_optimised = break_single_spider_graph(simple_graph)
    simple_graph = (
        simple_graph_optimised if simple_graph != simple_graph_optimised else simple_graph
    )
    fig_data = None if simple_graph != simple_graph_optimised else fig_data

    # Optional graph transformations
    if kwargs["strip_ports"]:
        simple_graph = strip_boundaries(simple_graph)

    # Call algorithm on a loop up to `max_attempts` tries
    i = 0
    while i < kwargs["max_attempts"]:
        # Update loop counter & start clock
        i += 1
        t_1_inner, _ = datetime_manager()

        # Update user
        print(f"\nCircuit name: {circuit_name}. Attempt #{i}.")

        # Call algorithm
        edge_paths, lat_nodes, lat_edges = (None, None, None)
        try:
            nx_g, edge_paths, lat_nodes, lat_edges = graph_manager_bfs(
                simple_graph,
                circuit_name=circuit_name,
                fig_data=fig_data,
                first_cube=first_cube,
                **kwargs,
            )
        except ValueError as e:
            print("ERROR. Graph manager failed", e)

        _, duration_iter = datetime_manager(t_1=t_1_inner)
        _, duration_all = datetime_manager(t_1=t_1)

        # Return result if any
        if lat_nodes and lat_edges:
            lat_volume = sum([1 for node in lat_nodes.values() if node[1] != "ooo"])
            print(
                Colors.GREEN + "SUCCESS!!!" + Colors.RESET,
                f"Volume: {lat_volume}. Duration: {duration_iter:.2f}s (attempt), {duration_all:.2f}s (total).",
            )

            # Write outputs
            write_bgraph(OUTPUT_DIR_PATH, circuit_name, lat_nodes, lat_edges)

            # vis_options result
            if kwargs["vis_options"][0] or kwargs["vis_options"][1] or kwargs["debug"] > 1:
                final_nx_g, _ = lattice_to_g(lat_nodes, lat_edges, nx_g)
                vis_3d(
                    nx_g,
                    final_nx_g,
                    edge_paths,
                    None,
                    None,
                    None,
                    None,
                    None,
                    fig_data=fig_data,
                    filename_info=(circuit_name, len(edge_paths) + 1)
                    if (kwargs["vis_options"][1] or kwargs["debug"] == 4)
                    else None,
                    **kwargs,
                )

                # Animation
                if kwargs["vis_options"][1] or kwargs["debug"] == 4:
                    create_animation(
                        filename_prefix=circuit_name,
                        restart_delay=5000,
                        duration=2500,
                        video=True if kwargs["vis_options"][1] == "MP4" else False,
                    )

            # End loop
            if kwargs["stop_on_first_success"]:
                break

        # Delete temporary files
        rm_temp_files(TEMP_DIR_PATH)

    return simple_graph, edge_paths, lat_nodes, lat_edges


####################
# WORKFLOW MANAGER #
####################
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
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

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
        first_id_strategy=kwargs["first_id_strategy"],
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
        traceback.print_exc()
        raise ValueError("ERROR. The graph_manager BFS crashed.", e, "\n")

    return nx_g, edge_paths, lat_nodes, lat_edges


#######
# BFS #
#######
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
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

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
    hold_for_edge_removal = []

    run_success = False
    while queue:
        repeat_current_src = False
        if hold_for_edge_removal:
            nx_g.remove_edges_from(hold_for_edge_removal)
            nx_g = prune_beams(nx_g, taken)

            hold_for_edge_removal = []

        # Get first cube from queue
        src_id: int = queue.popleft()

        # Iterate over neighbours of current source
        for tgt_id in cast(list[int], nx_g.neighbors(src_id)):
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
                    nx_g, taken, edge_paths, edge_success = handle_std_edge(
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

                        # Check move didn't cause problems elsewhere
                        priority_ids = check_need_for_twins(
                            nx_g, src_id, tgt_id, taken, priority_ids=[], strict=True
                        )

                        # Add twin cubes if problems detected
                        if priority_ids:
                            if kwargs["log_stats_id"] or kwargs["debug"] > 0:
                                print(
                                    Colors.BLUE,
                                    "==> Adding twin nodes for IDs:" + Colors.RESET,
                                    priority_ids,
                                )

                            (
                                nx_g,
                                queue,
                                visited,
                                twins,
                                taken,
                                priority_ids,
                                hold_for_edge_removal,
                            ) = add_twin(
                                circuit_name,
                                nx_g,
                                queue,
                                visited,
                                edge_paths,
                                taken,
                                fig_data,
                                twins,
                                priority_ids,
                                src_id,
                                **kwargs,
                            )
                            repeat_current_src = True

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
                nx_g, taken, edge_paths, edge_success = handle_cross_edge(
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

                    # Check move didn't cause problems elsewhere
                    priority_ids = check_need_for_twins(
                        nx_g, src_id, tgt_id, taken, priority_ids=[], strict=True
                    )

                    # Add twin cubes if problems detected
                    if priority_ids:
                        if kwargs["log_stats_id"] or kwargs["debug"] > 0:
                            print(
                                Colors.BLUE,
                                "==> Adding twin nodes for IDs:" + Colors.RESET,
                                priority_ids,
                            )

                        nx_g, queue, visited, twins, taken, priority_ids, hold_for_edge_removal = (
                            add_twin(
                                circuit_name,
                                nx_g,
                                queue,
                                visited,
                                edge_paths,
                                taken,
                                fig_data,
                                twins,
                                priority_ids,
                                src_id,
                                **kwargs,
                            )
                        )
                        repeat_current_src = True

                else:
                    run_success = False
                    break

            if repeat_current_src:
                break

        if run_success is False:
            break

    duration_trackers = t_std_edges, t_cross_edges
    edge_trackers = std_edges_processed, num_edges_processed, cross_edges_processed
    trackers = duration_trackers, edge_trackers

    return edge_paths, taken, run_success, trackers, visited
