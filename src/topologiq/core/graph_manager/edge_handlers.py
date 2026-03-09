"""Creation/generation utils to assist the primary graph managemer BFS.

Usage:
    Call any function/class from a separate script.

"""

from collections import deque
from typing import cast

import matplotlib
import networkx as nx

from topologiq.core.graph_manager.beams import check_path_to_beam_clashes, check_tgt_beam_clashes
from topologiq.core.graph_manager.callers import call_debug_vis, call_pathfinder
from topologiq.core.graph_manager.utils import get_node_degree, prune_beams, update_edge_paths
from topologiq.core.pathfinder.spatial import get_taken_coords
from topologiq.core.pathfinder.symbolic import check_exits
from topologiq.core.beams import CubeBeams
from topologiq.core.paths import PathBetweenNodes
from topologiq.utils.classes import (
    Colors,
    StandardBlock,
    StandardCoord,
)
from topologiq.utils.core import datetime_manager


##############################
# STANDARD EDGES / DISCOVERY #
##############################
def handle_std_edge(
    src_id: int,
    tgt_id: int,
    nx_g: nx.Graph,
    taken: list[StandardCoord],
    edge_paths: dict,
    circuit_name: str = "circuit",
    init_step: int = 3,
    fig_data: matplotlib.figure.Figure | None = None,
    twin_mode: bool = False,
    ids_to_twin: list[int] | None = None,
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
        twin_mode (optional): True when this function is used to create a twin of a given cube.
        ids_to_twin (optional): Cube IDs flagged as potentially problematic, passed only when function is used to create a twin node.
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

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
        clean_paths, pathfinder_vis_data = call_pathfinder(
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
                    nx_g,
                    src_id,
                    tgt_id,
                    coords_in_path,
                    twin_mode=twin_mode,
                    ids_to_twin=ids_to_twin,
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
                if not path_to_beam_clashes and not tgt_beam_clashes:
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

        # Call visualisation if applicable
        if kwargs["debug"] > 1 or kwargs["vis_options"][0] == "detail" or kwargs["vis_options"][1]:
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

    nx_g = prune_beams(nx_g, taken)
    return nx_g, taken, edge_paths, edge_success


####################################################
# CROSS EDGES / JOIN PREVIOUSLY DISCOVERED SPIDERS #
####################################################
def handle_cross_edge(
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
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

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
            critical_beams = _assemble_critical_beams(nx_g)

            # Check if edge is hadamard
            zx_edge_type = nx_g.get_edge_data(src_id, tgt_id).get("type")
            hdm: bool = True if zx_edge_type == "HADAMARD" else False

            # Call pathfinder using optional parameters that flag second pass nature of operation
            v_kind: str | None = nx_g.nodes[tgt_id].get("kind")

            if v_coords and v_kind:
                clean_paths, pathfinder_vis_data = call_pathfinder(
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

                # Finish timer before popping up visualisation
                _, t_total_iter = datetime_manager(t_1=t_1)

                # For visualisation, create a new graph on each step irrespective of outcome
                if (
                    kwargs["debug"] > 1
                    or kwargs["vis_options"][0] == "detail"
                    or kwargs["vis_options"][1]
                    # or (src_id == 75 and (tgt_id in [14, 15]))
                ):
                    # kwargs["debug"] = 3
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


def _assemble_critical_beams(
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


#####################################################
# RECOVERY EDGES / CREATE SPIDERS TO AVOID SHUTDOWN #
#####################################################
def add_twin(
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
):
    """Create a twin spider for any given number of priority spiders."""

    hold_for_edge_removal = []

    for priority_id in priority_ids:
        # Define new ID
        twin_id = max(nx_g.nodes) + 1
        twins[priority_id] = twin_id

        # Add the twin
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

        # Get neighbours pending for original and transfer to twin
        twin_pending_neighs = [
            n
            for n in nx_g.neighbors(priority_id)
            if tuple(sorted((n, priority_id))) not in list(edge_paths.keys())
        ]
        twin_pending_neighs = [n if n not in twins else twins[n] for n in twin_pending_neighs]

        nx_g.add_edge(priority_id, twin_id, type="SIMPLE")
        for twin_neigh_id in twin_pending_neighs:
            edge_type = nx_g.get_edge_data(priority_id, twin_neigh_id)
            hold_for_edge_removal.append((priority_id, twin_neigh_id))
            nx_g.add_edge(twin_id, twin_neigh_id, type=edge_type)

        # Try to place twin slightly away from current blockgraph
        taken = list(set(taken))

        step, max_step = (6, 15)
        while step <= max_step:
            nx_g, taken, edge_paths, edge_success = handle_std_edge(
                priority_id,
                twin_id,
                nx_g,
                taken,
                edge_paths,
                circuit_name=circuit_name,
                init_step=step,
                fig_data=fig_data,
                twin_mode=True,
                ids_to_twin=priority_ids,
                **kwargs,
            )

            # Move to next if there is a succesful placement
            if edge_success:
                visited.add(twin_id)
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
    new_queue: deque[int] = deque([src_id])
    while queue:
        next_in_queue = queue.popleft()
        if next_in_queue in priority_ids:
            new_queue.append(twins[next_in_queue])
            visited.add(twins[next_in_queue])
        else:
            new_queue.append(next_in_queue)
    queue.extend(new_queue)

    priority_ids = []

    return nx_g, queue, visited, twins, taken, priority_ids, hold_for_edge_removal
