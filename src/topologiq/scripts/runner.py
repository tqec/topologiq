"""
Run Topologiq programmatically using an arbitrary circuit provided as a `simple_graph`.

Usage:
    Call `runner()` programmatically from a separate script. 

Notes:
    Examples of how to run this file using combined options are available in `./docs`.
    MP4 animations require FFmpeg (the actual thing, not just the Python wrapper).
"""

import shutil
import matplotlib.figure

from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Union

from topologiq.scripts.graph_manager import graph_manager_bfs
from topologiq.utils.simple_grapher import simple_graph_vis
from topologiq.utils.utils_misc import write_outputs
from topologiq.utils.utils_zx_graphs import strip_boundaries
from topologiq.utils.grapher_common import lattice_to_g
from topologiq.utils.grapher import vis_3d
from topologiq.utils.animation import create_animation
from topologiq.utils.utils_zx_graphs import break_single_spider_graph
from topologiq.utils.classes import Colors, SimpleDictGraph, StandardBlock
from topologiq.run_hyperparams import VALUE_FUNCTION_HYPERPARAMS, LENGTH_OF_BEAMS


####################
# MAIN RUN MANAGER #
####################
def runner(
    simple_graph: SimpleDictGraph,
    circuit_name: str,
    min_succ_rate: int = 60,
    strip_ports: bool = False,
    hide_ports: bool = False,
    max_attempts: int = 10,
    stop_on_first_success: bool = True,
    vis_options: Tuple[Union[None, str], Union[None, str]] = (None, None),
    log_stats: bool = False,
    debug: int = 0,
    fig_data: matplotlib.figure.Figure | None = None,
    first_cube: Tuple[Union[int, None], Union[str, None]] = (None, None),
    **kwargs,
) -> Tuple[
    SimpleDictGraph,
    Union[None, dict],
    Union[None, dict[int, StandardBlock]],
    Union[None, dict[Tuple[int, int], List[str]]],
]:
    """Run Topologiq on an arbitrary circuit provided as `simple_graph`.

    This function calls any available and applicable pre-processing on the input `simple_graph`,
    then sends the graph for processing by Topologiq. If the algorith succeeds, the function 
    gathers and returns the main outputs produced by Topologiq.

    Args:
        simple_graph: The `simple_graph` form of an arbitrary ZX circuit.
        circuit_name: The name of the ZX circuit.
        min_succ_rate (optional): Minimum % of tentative coordinates that must be filled for each edge.
        strip_ports (optional): If True, boundary spiders are removed from the `simple_graph` prior to calling Topologiq.
        hide_ports (optional): If True, boundary spiders are considered by Topologiq but not displayed in visualisations.
        max_attempts (optional): The maximum number of times to repeat-call the algorithm on a given circuit.
        stop_on_first_success (boolean): If True, forces exit on first successful outcome irrespective of `max_attempts`.
        vis_options (optional): Visualisation settings provided as a Tuple.
            vis_options[0]: If enabled, triggers "final" or "detail" visualisations.
                (None): No visualisation.
                (str) "final" | "detail": A single visualisation of the final result or one visualisation per completed edge.
            vis_options[1]: If enabled, triggers creation of an animated summary for the entire process.
                (None): No animation.
                (str) "GIF" | "MP4": A step-by-step visualisation of the process in GIF or MP4 format.
        log_stats (optional): If True, triggers automated stats logging to CSV files in `.assets/stats/`.
        debug (optional): Debug mode (0: off, 1: graph manager, 2: pathfinder, 3: pathfinder w. discarded paths).
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).
        first_cube (optional): The ID and kind of the first cube to place in 3D space (used to replicate specific cases).

    Keyword arguments (**kwargs):
        weights: A tuple (int, int) of weights used to pick the best of several paths when there are several valid alternatives.
        length_of_beams: The length of each of the beams coming out of cubes still needing connections at any given point in time.

    Returns:
        simple_graph: The original `simple_graph` given to function (returned for ease of use and traceability).
        edge_paths: An edge-by-edge summary of the 3D object Topologiq builds, updated to the last edge processsed successfully.
        lat_nodes: The cubes of the final space-time diagram produced by Topologiq.
        lat_edges: The pipes of the final space-time diagram produced by Topologiq.

    """

    # Preliminaries
    t1 = datetime.now()
    repo_root: Path = Path(__file__).resolve().parent.parent
    output_dir_path = repo_root / "output/txt"
    temp_dir_path = repo_root / "output/temp"
    Path(output_dir_path).mkdir(parents=True, exist_ok=True)
    unique_run_id = None

    # Assemble kwargs if not detected
    if len(kwargs) == 0:
        kwargs: dict[str, Tuple[int, int] | int] = {
            "weights": VALUE_FUNCTION_HYPERPARAMS,
            "length_of_beams": LENGTH_OF_BEAMS,
        }

    # Optimise incoming graph if applicable
    # The following operations leave the simple_graph unchanged if no optimisation is available
    # Eventually, there should be a manager that calls only applicable optimisations
    simple_graph_optimised = break_single_spider_graph(simple_graph)

    # Update user if graph was auto-optimised
    if simple_graph != simple_graph_optimised:  

        # Override simple graph
        simple_graph = simple_graph_optimised
        
        # Nullify any pre-existing fig_data as overlay would no longer correspond
        fig_data = None
        print("Note! Graph auto-optimised to reduce final volume.")
        simple_graph_vis(simple_graph, layout_method="planar")

    # Optional graph transformations
    if strip_ports:
        simple_graph = strip_boundaries(simple_graph)

    # Call algorithm on a loop up to `max_attempts` tries
    i: int = 0
    while i < max_attempts:
        # Update counters
        t1_inner = datetime.now()
        i += 1

        # Verbose updates if log_stats or debug mode is on
        if log_stats or debug in [1, 2, 3]:
            print(f"\nAttempt {i} of {max_attempts}:")
        
        # Create unique run ID if stats logging is on
        if log_stats:
            unique_run_id = t1_inner.strftime("%Y%m%d_%H%M%S_%f")
        else: 
            print(".")

        # Call algorithm
        edge_paths = None
        lat_nodes = None
        lat_edges = None
        try:
            nx_g, edge_paths, c, lat_nodes, lat_edges = graph_manager_bfs(
                simple_graph,
                circuit_name=circuit_name,
                min_succ_rate=min_succ_rate,
                hide_ports=hide_ports,
                vis_options=vis_options,
                log_stats_id=unique_run_id,
                debug=debug,
                fig_data=fig_data,
                first_cube=first_cube,
                **kwargs,
            )
            lat_volume = sum([1 for node in lat_nodes.values() if node[1] != "ooo"])

            # Return result if any
            if lat_nodes is not None and lat_edges is not None:
                # Stop timer
                duration_iter = (datetime.now() - t1_inner).total_seconds()
                duration_all = (datetime.now() - t1).total_seconds()

                # Update user
                print(
                    Colors.GREEN + "SUCCESS!!!" + Colors.RESET,
                    f"Volume: {lat_volume}.",
                    f"Duration: {duration_iter:.2f}s (attempt), {duration_all:.2f}s (total).",
                )

                if vis_options[0] is not None or vis_options[1] is not None:
                    print(
                        "Visualisations enabled. For faster runtimes, disable visualisations."
                    )

                # Write outputs
                write_outputs(
                    simple_graph, circuit_name, edge_paths, lat_nodes, lat_edges, output_dir_path
                )

                # vis_options result
                if vis_options[0] or vis_options[1]:
                    final_nx_g, _ = lattice_to_g(lat_nodes, lat_edges, nx_g)

                    # 3D interactive visualisation
                    if vis_options[0]:
                        if vis_options[0].lower() in ["final", "details"]:
                            vis_3d(
                                nx_g,
                                final_nx_g,
                                edge_paths,
                                None,
                                None,
                                None,
                                None,
                                None,
                                hide_ports=hide_ports,
                                debug=0,
                                fig_data=fig_data,
                                filename=f"{circuit_name}{c:03d}" if vis_options[1] else None,
                            )

                    # Animation
                    if vis_options[1]:
                            create_animation(
                                filename_prefix=circuit_name,
                                restart_delay=5000,
                                duration=2500,
                                video=True if vis_options[1] == "MP4" else False,
                            )

                # End loop
                if stop_on_first_success:
                    break

        except ValueError as e:
            # Stop timer
            duration_iter = (datetime.now() - t1_inner).total_seconds()
            duration_all = (datetime.now() - t1).total_seconds()

            # Update user
            if log_stats or debug in [1, 2, 3]:
                print(
                    Colors.RED + f"ATTEMPT FAILED.\n{e}" + Colors.RESET,
                    f"Duration: {duration_iter:.2f}s. (attempt), {duration_all:.2f}s (total).",
                )
                if vis_options[0] is not None or vis_options[1] is not None:
                    print(
                        "Visualisations enabled. For faster runtimes, disable visualisations."
                    )

        # Delete temporary files
        try:
            if temp_dir_path.exists():
                shutil.rmtree(temp_dir_path)
        except (ValueError, FileNotFoundError) as e:
            print("Unable to delete temp files or temp folder does not exist", e)

    return simple_graph, edge_paths, lat_nodes, lat_edges
