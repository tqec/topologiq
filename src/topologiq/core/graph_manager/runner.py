"""Run Topologiq programmatically using an arbitrary circuit provided as a `simple_graph`.

Usage:
    Call `runner()` programmatically from a separate script.

Notes:
    Examples of how to run this file using combined options are available in `./docs`.
    MP4 animations require FFmpeg (the actual thing, not just the Python wrapper).

"""

import shutil
from pathlib import Path

import matplotlib.figure

from topologiq.core.graph_manager.graph_manager import graph_manager_bfs
from topologiq.input.simple_graphs import break_single_spider_graph, strip_boundaries
from topologiq.kwargs import (
    BEAMS_SHORT_LEN,
    DEBUG,
    FIRST_ID_STRATEGY,
    HIDE_PORTS,
    LOG_STATS,
    MAX_ATTEMPTS,
    MIN_SUCC_RATE,
    SEED,
    STOP_ON_FIRST_SUCCESS,
    STRIP_PORTS,
    VALUE_FUNCTION_HYPERPARAMS,
)
from topologiq.utils.classes import Colors, SimpleDictGraph, StandardBlock
from topologiq.utils.utils_misc import datetime_manager, write_outputs
from topologiq.vis.animation import create_animation
from topologiq.vis.grapher import vis_3d
from topologiq.vis.grapher_common import lattice_to_g

#########
# PATHS #
#########
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR_PATH = REPO_ROOT / "output/txt"
TEMP_DIR_PATH = REPO_ROOT / "output/temp"


####################
# MAIN RUN MANAGER #
####################
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
        **kwargs: !

            ! If a given kwarg is not given explicitly, this function will create it based on `./src/topologiq/run_hyperparams.py`.
            By extension, it only makes sense to give kwargs initially to deviate from defaults.

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
            write_outputs(
                simple_graph, circuit_name, edge_paths, lat_nodes, lat_edges, OUTPUT_DIR_PATH
            )

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
        _rm_temp_files()

    return simple_graph, edge_paths, lat_nodes, lat_edges


#######
# AUX #
#######
def check_assemble_kwargs(**kwargs) -> dict[str, any]:
    """Check if all kwargs are present and add any missing."""

    if len(kwargs) == 0:
        kwargs = {
            "weights": VALUE_FUNCTION_HYPERPARAMS,
            "first_id_strategy": FIRST_ID_STRATEGY,
            "beams_len_short": BEAMS_SHORT_LEN,
            "seed": SEED,
            "vis_options": (None, None),
            "max_attempts": MAX_ATTEMPTS,
            "stop_on_first_success": STOP_ON_FIRST_SUCCESS,
            "min_succ_rate": MIN_SUCC_RATE,
            "strip_ports": STRIP_PORTS,
            "hide_ports": HIDE_PORTS,
            "log_stats": LOG_STATS,
            "log_stats_id": None,
            "debug": DEBUG,
        }


    if "weights" not in kwargs:
        kwargs["weights"] = VALUE_FUNCTION_HYPERPARAMS
    if "first_id_strategy" not in kwargs:
        kwargs["first_id_strategy"] = FIRST_ID_STRATEGY
    if "beams_len_short" not in kwargs:
        kwargs["beams_len_short"] = BEAMS_SHORT_LEN
    if "seed" not in kwargs:
        kwargs["seed"] = SEED
    if "vis_options" not in kwargs:
        kwargs["vis_options"] = (None, None)
    if "max_attempts" not in kwargs:
        kwargs["max_attempts"] = MAX_ATTEMPTS
    if "stop_on_first_success" not in kwargs:
        kwargs["stop_on_first_success"] = STOP_ON_FIRST_SUCCESS
    if "min_succ_rate" not in kwargs:
        kwargs["min_succ_rate"] = MIN_SUCC_RATE
    if "strip_ports" not in kwargs:
        kwargs["strip_ports"] = STRIP_PORTS
    if "hide_ports" not in kwargs:
        kwargs["hide_ports"] = HIDE_PORTS
    if "log_stats" not in kwargs:
        kwargs["log_stats"] = LOG_STATS
    if "log_stats_id" not in kwargs:
        kwargs["log_stats_id"] = None
    if "debug" not in kwargs:
        kwargs["debug"] = DEBUG

    # Create unique run ID if stats logging is on
    if kwargs["log_stats"]:
        timestamp, _ = datetime_manager()
        kwargs["log_stats_id"] = timestamp.strftime("%Y%m%d_%H%M%S_%f")

    return kwargs


def _rm_temp_files():
    """Remove any temporary files created during run."""
    try:
        if TEMP_DIR_PATH.exists():
            shutil.rmtree(TEMP_DIR_PATH)
    except (ValueError, FileNotFoundError) as e:
        print("Unable to delete temp files or temp folder does not exist", e)
