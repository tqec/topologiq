import shutil

from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Union
import matplotlib.figure

from topologiq.scripts.graph_manager import graph_manager_bfs
from topologiq.utils.utils_misc import write_outputs
from topologiq.utils.utils_zx_graphs import strip_boundaries
from topologiq.utils.grapher import vis_3d_g, lattice_to_g
from topologiq.utils.animation import create_animation
from topologiq.utils.classes import Colors, SimpleDictGraph, StandardBlock


####################
# MAIN RUN MANAGER #
####################
def runner(
    c_g_dict: SimpleDictGraph,
    c_name: str,
    min_succ_rate: int = 50,
    strip_ports: bool = False,
    hide_ports: bool = False,
    max_attempts: int = 10,
    stop_on_first_success: bool = True,
    visualise: Tuple[Union[None, str], Union[None, str]] = (None, None),
    log_stats: bool = False,
    debug: bool = False,
    fig_data: Optional[matplotlib.figure.Figure] = None,
    **kwargs,
) -> Tuple[
    SimpleDictGraph,
    Union[None, dict],
    Union[None, dict[int, StandardBlock]],
    Union[None, dict[Tuple[int, int], List[str]]],
]:
    """Runs the algorithm on any circuit given to it

    Args:
        - c_g_dict: a ZX circuit as a simple dictionary of nodes and edges.
        - c_name: name of ZX circuit.
        - min_succ_rate: min % of tent_coords that need to be filled on each run of the pathfinder, used as exit condition.
        - strip_ports:
            - true: instructs the algorithm to eliminate any boundary nodes and their corresponding edges,
            - false: nodes are factored into the process and shown on visualisation.
        - hide_ports:
            - true: instructs the algorithm to use boundary nodes but do not display them in visualisation,
            - false: boundary nodes are factored into the process and shown on visualisation.
        - visualise: a tuple with visualisation settings:
            - visualise[0]:
                - None: no visualisation whatsoever,
                - "final" (str): triggers a single on-screen visualisation of the final result (small performance trade-off),
                - "detail" (str): triggers on-screen visualisation for each edge in the original ZX-graph (medium performance trade-off).
            - visualise[1]:
                - None: no animation whatsoever,
                - "GIF": saves step-by-step visualisation of the process in GIF format (huge performance trade-off),
                - "MP4": saves a PNG of each step/edge in the visualisation process and joins them into a GIF at the end (huge performance trade-off).
        - log_stats: boolean to determine if to log stats to CSV files in `.assets/stats/`.
            - True: log stats to file
            - False: do NOT log stats to file
        - debug: optional parameter to turn debugging mode on (added details will be visualised on each step).
            - True: debugging mode on,
            - False: debugging mode off.
        - fig_data: optional parameter to pass the original visualisation for input graph (currently only available for PyZX graphs).

    Keyword arguments (**kwargs):
        - weights: weights for the value function to pick best of many paths.
        - length_of_beams: length of each of the beams coming out of open nodes.

    Returns:
        - c_g_dict: original circuit given to function returns for easy traceability.
        - edge_pths: the raw set of 3D edges found by the algorithm (with redundant blocks for start and end positions of some edges).
        - lat_nodes: the nodes/blocks of the resulting space-time diagram (without redundant blocks).
        - lat_edges: the edges/pipes of the resulting space-time diagram (without redundant pipes).

    """

    # PRELIMINARIES
    unique_run_id = None
    t1 = datetime.now()
    if log_stats:
        unique_run_id = t1.strftime("%Y%m%d_%H%M%S_%f") if log_stats else None

    repo_root: Path = Path(__file__).resolve().parent.parent
    out_dir_pth = repo_root / "outputs/txt"
    temp_dir_pth = repo_root / "outputs/temp"
    Path(out_dir_pth).mkdir(parents=True, exist_ok=True)

    # APPLICABLE GRAPH TRANSFORMATIONS
    if strip_ports:
        c_g_dict = strip_boundaries(c_g_dict)

    # VARS TO HOLD RESULTS
    edge_pths: Union[None, dict] = None
    lat_nodes: Union[None, dict[int, StandardBlock]] = None
    lat_edges: Union[None, dict[Tuple[int, int], List[str]]] = None

    # LOOP UNTIL SUCCESS OR LIMIT
    i: int = 0
    while i < max_attempts:

        # Update counters
        t1_inner = datetime.now()
        i += 1

        # Update user
        if log_stats:
            print(f"\nAttempt {i} of {max_attempts}:")
        else: 
            print(".")

        # Call algorithm
        try:
            nx_g, edge_pths, c, lat_nodes, lat_edges = graph_manager_bfs(
                c_g_dict,
                c_name=c_name,
                min_succ_rate=min_succ_rate,
                hide_ports=hide_ports,
                visualise=visualise,
                log_stats_id=unique_run_id,
                debug=debug,
                fig_data=fig_data,
                **kwargs,
            )

            # Return result if any
            if lat_nodes is not None and lat_edges is not None:

                # Stop timer
                duration_iter = (datetime.now() - t1_inner).total_seconds()
                duration_all = (datetime.now() - t1).total_seconds()

                # Update user
                print(
                    Colors.GREEN + "SUCCESS!!!" + Colors.RESET,
                    f"Duration: {duration_iter:.2f}s. (attempt), {duration_all:.2f}s (total).",
                )

                if visualise[0] is not None or visualise[1] is not None:
                    print(
                        "Visualisations enabled. For faster runtimes, disable visualisations."
                    )

                # Write outputs
                write_outputs(
                    c_g_dict, c_name, edge_pths, lat_nodes, lat_edges, out_dir_pth
                )

                # Visualise result
                if visualise[0] or visualise[1]:

                    final_nx_g, _ = lattice_to_g(lat_nodes, lat_edges, nx_g)

                    if visualise[0]:
                        if (
                            visualise[0].lower() == "final"
                            or visualise[0].lower() == "detail"
                        ):

                            vis_3d_g(
                                final_nx_g,
                                edge_pths,
                                hide_ports=hide_ports,
                                fig_data=fig_data,
                            )

                    # Animate
                    if visualise[1]:
                        if (
                            visualise[1].lower() == "gif"
                            or visualise[1].lower() == "mp4"
                        ):
                            vis_3d_g(
                                final_nx_g,
                                edge_pths,
                                hide_ports=hide_ports,
                                save_to_file=True,
                                filename=f"{c_name}{c:03d}",
                                fig_data=fig_data,
                            )

                            create_animation(
                                filename_prefix=c_name,
                                restart_delay=5000,
                                duration=2500,
                                video=True if visualise[1] == "MP4" else False,
                            )

                # End loop
                if stop_on_first_success:
                    break

        except ValueError as e:

            # Stop timer
            duration_iter = (datetime.now() - t1_inner).total_seconds()
            duration_all = (datetime.now() - t1).total_seconds()

            # Update user
            if log_stats:
                print(
                    Colors.RED + f"ATTEMPT FAILED.\n{e}" + Colors.RESET,
                    f"Duration: {duration_iter:.2f}s. (attempt), {duration_all:.2f}s (total).",
                )
                if visualise[0] is not None or visualise[1] is not None:
                    print(
                        "Visualisations enabled. For faster runtimes, disable visualisations."
                    )

        # Delete temporary files
        try:
            if temp_dir_pth.exists():
                shutil.rmtree(temp_dir_pth)
        except (ValueError, FileNotFoundError) as e:
            print("Unable to delete temp files or temp folder does not exist", e)

    # RETURN: original ZX graph, edge_pths, nodes and edges of result
    return c_g_dict, edge_pths, lat_nodes, lat_edges
