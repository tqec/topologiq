"""Call and communicate with the inner pathfinder algorithm.

Usage:
    Call any function/class from a separate script.

"""

from typing import Any

import matplotlib
import networkx as nx

from topologiq.core.beams import CubeBeams
from topologiq.core.graph_manager.utils import reindex_path_dict
from topologiq.core.pathfinder.pathfinder import pathfinder
from topologiq.core.paths import PathBetweenNodes
from topologiq.dzw.common.components import ZxNode
from topologiq.utils.classes import StandardBlock, StandardCoord
from topologiq.utils.read_write import prep_stats_n_log
from topologiq.vis.animation import create_animation
from topologiq.vis.blockgraph import vis_3d
from topologiq.vis.common import lattice_to_g

from topologiq.dzw.common.attributes_zx import NodeId
from topologiq.dzw.augmented_nx_graph import AugmentedNxGraph

##############
# PATHFINDER #
##############
def call_pathfinder(
    ang: AugmentedNxGraph, source: ZxNode, target: ZxNode,
    init_step: int,
    critical_beams: dict[int, tuple[StandardCoord, int, CubeBeams, CubeBeams]] = {},
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
        ang: The AugmentedNxGraph which will track the construction process, relating the ZX-graph to the BG-graph.
        source: the source of the path we are looking for
        target: the target of the path we are looking for
        init_step: The ideal/intended (Manhattan) distance between source and target blocks.
        critical_beams (optional): Annotated beams object with details about minimum number of beams needed per node.
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

    Returns:
        clean_paths: A list of paths each containing the 3D cubes and pipes needed to connect source and target in the 3D space.
        pathfinder_vis_data: A list containing data for visualisation of a given pathfinder run.

    """

    # Edge path management
    pathfinder_vis_data = [None, None, None, None]
    valid_paths: dict[StandardBlock, list[StandardBlock]] | None = None
    clean_paths = []

    src_coords = source.realising_cube.position
    tgt_coords = target.realising_cube.position if target.is_realised() else None

    step = init_step

    # Copy taken to avoid accidental overwrites
    taken_cc = set(ang.occupied)
    if src_coords in taken_cc:
        taken_cc.remove(src_coords)
    if tgt_coords:
        taken_cc.remove(tgt_coords)

    # Loop call the inner pathfinder in case there is a need to re-run the pathfinder
    max_step = init_step if target.is_realised() else 15
    while step <= max_step:
        # Generate tentative coordinates for current step or use target node
        if tgt_coords:
            tent_coords = [tgt_coords]
        else:
            tent_coords = _gen_tent_tgt_coords(
                src_coords,
                step,
                list(ang.occupied),  # Real occupied coords: position cannot overlap start node
            )

        # Try finding paths to each tentative coordinates
        if tent_coords:
            valid_paths, pathfinder_vis_data = pathfinder(
                ang, source, target,
                tent_coords,
                critical_beams=critical_beams,
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


def _gen_tent_tgt_coords(
    src_c: StandardCoord,
    max_manhattan: int = 3,
    taken: list[StandardCoord] = [],
) -> list[StandardCoord]:
    """Generate a number of potential placement positions for target node.

    Args:
        src_c: The (x, y, z) coordinates for the originating block.
        max_manhattan: Max. (Manhattan) distance between origin and target blocks.
        taken: A list of coordinates already taken by previous operations.

    Returns:
        all_coords_at_distance: A list of tentative target coordinates that make good candidates for placing the target block.

    """

    # EXTRACT SOURCE COORDS
    sx, sy, sz = src_c
    base_for_next_layer = []
    tent_coords = {}

    # SINGLE MOVES
    tgts = [
        (sx + 3, sy, sz),
        (sx - 3, sy, sz),
        (sx, sy + 3, sz),
        (sx, sy - 3, sz),
        (sx, sy, sz + 3),
        (sx, sy, sz - 3),
    ]
    tent_coords[3] = [t for t in tgts if t not in taken]
    base_for_next_layer = [t for t in tgts]

    # MANHATTAN 6
    if max_manhattan > 3:
        tent_coords[6] = []
        for dx, dy, dz in [c for c in base_for_next_layer]:
            tgts = [
                (dx + 3, dy, dz),
                (dx - 3, dy, dz),
                (dx, dy + 3, dz),
                (dx, dy - 3, dz),
                (dx, dy, dz + 3),
                (dx, dy, dz - 3),
            ]
            tent_coords[6].extend([t for t in tgts if t not in taken and t != src_c])
            base_for_next_layer.extend([t for t in tgts])

    # MANHATTAN 9
    if max_manhattan > 6:
        tent_coords[9] = []
        for dx, dy, dz in [c for c in base_for_next_layer]:
            tgts = [
                (dx + 3, dy, dz),
                (dx - 3, dy, dz),
                (dx, dy + 3, dz),
                (dx, dy - 3, dz),
                (dx, dy, dz + 3),
                (dx, dy, dz - 3),
            ]
            tent_coords[9].extend([t for t in tgts if t not in taken and t != src_c])
            base_for_next_layer.extend([t for t in tgts])

    # > MANHATTAN 9
    if max_manhattan > 9:
        tent_coords[max_manhattan] = []
        num_loops = int((max_manhattan - 9) / 3)

        for _ in [i + 1 for i in range(num_loops)]:
            for dx, dy, dz in [c for c in base_for_next_layer]:
                tgts = [
                    (dx + 3, dy, dz),
                    (dx - 3, dy, dz),
                    (dx, dy + 3, dz),
                    (dx, dy - 3, dz),
                    (dx, dy, dz + 3),
                    (dx, dy, dz - 3),
                ]
                tent_coords[max_manhattan].extend(
                    [t for t in tgts if t not in taken and t != src_c]
                )
                base_for_next_layer.extend([t for t in tgts])

    all_coords_at_distance = tent_coords[min(max_manhattan, 15)]
    return all_coords_at_distance


################
# STATS LOGGER #
################
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
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

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


##################
# VISUALISATIONS #
##################
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
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

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
