"""Core script for the inner pathfinder algorithm (BFS).

This file contains functions that altogether create topologically-correct 3D edge paths
between a given source cube with pre-determined position and kind and one or more target cubes.
The algorithm is flexible enough to accomodate different kinds of requests. If it gets more
than one tentative coordinates for the target cube, it assumes the target cube has not yet
been placed in the 3D space and creates tentative paths to a user-determined max. % of
tentative coordinates (the max can be 100% but this has found to be unnecessary).
If it gets only one tentative position (and information for the target cube in that coordinates),
it assumes the target cube has already been placed in the 3D space and goes into single-path mode,
where it returns the shortest path between source and target cubes.

Usage:
    Call `pathfinder()` programmatically from a separate script, with an appropriate combination of optional parameters.

Notes:
    For now, none of the functions in this file are to be called individually.
    In the future, some of the functions could be called by variant algorithms that
        do not necessarily need or want to implement all separate features.

"""

from collections import deque

import networkx as nx
import numpy as np

from topologiq.core.blocks import PositionedZXBlock, ZXBlock, ZXBlockRegistry
from topologiq.core.pathfinder.aux import gen_exit_conditions, init_bfs
from topologiq.core.pathfinder.spatial import (
    check_clashes_parametrised_taken,
    check_skip_move,
    gen_bounding_box,
    get_coords_for_current_move,
)
from topologiq.utils.classes import CubeBeams, GraphBounds, StandardBlock, StandardCoord
from topologiq.utils.manhattan import get_manhattan


############################
# MAIN PATHFINDER WORKFLOW #
############################
def pathfinder(
    bgraph: nx.Graph,
    beams: CubeBeams,
    beams_short: CubeBeams,
    curr_src_id: int,
    curr_tgt_id: int,
    tent_coords: list[StandardCoord],
    cross_edge: bool,
    taken: set[StandardCoord],
    pruned_taken: set[StandardCoord],
    is_hadamard: bool,
    z_bounds: dict[str, int | None] = {},
    graph_bounds: GraphBounds | None = None,
    **kwargs,
) -> tuple[
    dict[PositionedZXBlock, list[PositionedZXBlock]] | None,
    tuple[
        list[str] | None,
        int,
        int,
        dict[PositionedZXBlock, list[PositionedZXBlock]] | None,
    ]
    | None,
]:
    """Call core pathfinder after generating list of possible kinds for the given operation.

    Please note is an edge fulfiller. The incoming BlockGraphManager is used to guide pathfinding,
    but the pathfinder does not change this object. Instead, the pathfinder returns the NX graph
    corresponding to self.blockgraph, which should be used in the calling class to update
    the class as needed for any subsequent iteration.

    Args:
        bgraph_manager: The BlockGraphManager currently managing the build.
        bgraph: The BlockGraph currently being built.
        beams: The beams for all the cubes in blockgraph that need beams.
        beams_short: The short beams for all the cubes in blockgraph that need beams.
        curr_src_id: The ID of the current source cube.
        curr_tgt_id: The ID of the current target cube.
        tent_coords: The list of tentative coords for the placement of the current target.
        cross_edge: True if the current edge is a cross-edge (as opposed to a standard edge).
        taken: The set of taken coordinates.
        pruned_taken: A pruned version of taken not containing source and target coordinates.
        is_hadamard: True if the current edge is a Hadamard in the input ZX graph.
        z_bounds: Min. and max. Z-coordinate possible for a given move, if either exists.
        graph_bounds (optional): A tuple of max_x and max_y coordinates to maintain build within bounds.
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

    Returns:
        bgraph: An updated blockgraph including the new path to be added to the calling BlockGraphManager.
        pathfinder_vis_data: A list containing data for visualisation of a given pathfinder run.

    """

    # Extract key info into easily accessible variables
    src_zx_block: ZXBlock = bgraph.nodes[curr_src_id]["zx_block"]
    tgt_zx_block: ZXBlock = bgraph.nodes[curr_tgt_id]["zx_block"]
    src_coords: StandardCoord = bgraph.nodes[curr_src_id]["coords"]

    # Parametrise taken
    parametrised_taken: dict[int, list[tuple[int, int]]] = {}
    for x, y, z in pruned_taken:
        if z in parametrised_taken:
            parametrised_taken[z].add((x, y))
        else:
            parametrised_taken[z] = set([(x, y)])

    # Generate kinds that could in theory be assigned to the target cube
    tent_tgt_kinds = tgt_zx_block.kind if cross_edge else tgt_zx_block.get_kind_family

    # Create bounding box to limit space search
    bounding_box, max_span = gen_bounding_box(
        taken, cross_edge=cross_edge, graph_bounds=graph_bounds
    )

    # Initialise BFS
    queue, visited, visit_attempts, path, valid_paths, all_search_paths = init_bfs(
        (src_coords, src_zx_block)
    )

    path_clashes = {}
    src_tgt_adjusts = {}
    out_pendings = {}

    # Define exit conditions in case something goes wrong
    tgts_to_fill, max_manhattan, src_tgt_manhattan = gen_exit_conditions(
        src_coords,
        tent_coords,
        pruned_taken,
        max_span,
        cross_edge,
        kwargs["min_succ_rate"],
        special_target_kind=tgt_zx_block.zx_type in ["T"],
    )
    tgts_to_fill = min(100, tgts_to_fill)

    # Manage queue
    hdm = is_hadamard
    break_for_success = False
    while queue:
        # Flag to exit prematurely due to success
        if break_for_success:
            break

        # Unpack current block (source for iteration)
        curr_block_positioned: PositionedZXBlock = queue.popleft()
        curr_coords: StandardCoord = curr_block_positioned[0]
        curr_zx_block: ZXBlock = curr_block_positioned[1]

        # Check skip/break tolerances
        curr_manhattan = get_manhattan(src_coords, curr_coords)
        if curr_manhattan > src_tgt_manhattan * 3:
            continue
        if curr_manhattan > max_manhattan:
            continue

        # Check for success
        if curr_coords in tent_coords:
            if cross_edge or curr_zx_block.zx_type in ["Y", "T", "O"]:
                continue

        # Avoid multiple special gates on same path
        if tgt_zx_block.zx_type in ["Y", "T"]:
            if any(
                [
                    zx_b.zx_type == tgt_zx_block.zx_type
                    for _, zx_b in path[curr_block_positioned][1:]
                ]
            ):
                continue

        # Check path for beam clashes before attempting move
        if _check_beams_clashes(
            bgraph,
            curr_block_positioned,
            cross_edge,
            path,
            beams,
            beams_short,
            curr_src_id,
            curr_tgt_id,
            path_clashes,
            out_pendings,
            src_tgt_adjusts,
            strict=not cross_edge,
        ):
            continue

        # Try moving in all directions
        for move in curr_zx_block.get_move_vectors:
            # Calculate next position and update paths accordingly
            nxt_coords, curr_path_coords = get_coords_for_current_move(
                curr_block_positioned, move, path
            )

            # Check if move can be skipped (for speed)
            if check_skip_move(
                nxt_coords,
                curr_path_coords,
                parametrised_taken,
                bounding_box=bounding_box,
            ):
                continue

            # Create a list of kinds that are valid for the next block
            possible_nxt_zx_blocks = curr_zx_block.nxt_kinds(
                move,
                is_hadamard=hdm,
                tgt_zx_type=tgt_zx_block.zx_type,
            )

            # Loop over all possible next types
            for possible_nxt_zx_block in possible_nxt_zx_blocks:
                # Check if next kind needs to be rotated due to Hadamard
                nxt_block: StandardBlock = (
                    nxt_coords,
                    possible_nxt_zx_block,
                )

                if possible_nxt_zx_block.zx_type == "T":
                    nxt_x, nxt_y, nxt_z = nxt_coords
                    skip_possible_nxt_zx_block = False
                    for i in [1, 2]:
                        if check_clashes_parametrised_taken(
                            (nxt_x, nxt_y, nxt_z - i), parametrised_taken
                        ):
                            skip_possible_nxt_zx_block = True
                        if (nxt_x, nxt_y, nxt_z - i) in taken or (
                            nxt_x,
                            nxt_y,
                            nxt_z - i,
                        ) in curr_path_coords:
                            skip_possible_nxt_zx_block = True
                    if skip_possible_nxt_zx_block:
                        continue

                # Log to visited and update path lengths if all conditions met
                queue, visited, path, visit_attempts, all_search_paths = _to_visit_or_not_to_visit(
                    curr_block_positioned,
                    nxt_block,
                    queue,
                    visited,
                    move,
                    path,
                    visit_attempts,
                    all_search_paths,
                )

                # Check for success
                if nxt_coords in tent_coords:
                    if _check_for_success(
                        (nxt_coords, possible_nxt_zx_block),
                        tgt_zx_block.zx_type,
                        tent_tgt_kinds,
                        path,
                        valid_paths,
                        tgts_to_fill,
                        pruned_taken,
                        z_bounds=z_bounds,
                    ):
                        break_for_success = True
                        break

            if break_for_success:
                break

        # Hadamards are introduce on very first move so once loop clears first
        # set of possible moves, there can be no more Hadamards in a specific edge.
        hdm = False

    # Exit by default
    return valid_paths, (tent_tgt_kinds, visit_attempts, len(visited), all_search_paths)


########
# AUX #
#######
def _check_beams_clashes(
    bgraph,
    curr_block_positioned,
    cross_edge,
    path,
    beams,
    beams_short,
    curr_src_id,
    curr_tgt_id,
    path_clashes,
    out_pendings,
    src_tgt_adjusts,
    strict=False,
):
    """Check if there are beam clashes."""

    if not cross_edge:
        return False

    beams_to_check = beams if strict else beams_short
    if len(path[curr_block_positioned]) > 1:
        path_clashes[curr_block_positioned] = path_clashes[path[curr_block_positioned][-2]].copy()

        # Check each cube against all other cubes
        for out_id, out_beams in beams_to_check.items():
            # Track outer beams in a way that remembers which beam is which

            broken_beams = [out_beam.contains(curr_block_positioned[0]) for out_beam in out_beams]

            path_clashes[curr_block_positioned][out_id] = path_clashes[
                path[curr_block_positioned][-2]
            ][out_id] + np.array(broken_beams)

            # Determine if out clashes are within tolerance
            if out_id not in src_tgt_adjusts:
                src_tgt_adjusts[out_id] = 1 if out_id in (curr_src_id, curr_tgt_id) else 0
            if out_id not in out_pendings:
                out_pendings[out_id] = (
                    min(1, bgraph.nodes[out_id]["completions"]["pending"])
                    if src_tgt_adjusts[out_id] == 0
                    else bgraph.nodes[out_id]["completions"]["pending"]
                )
            if (
                len(out_beams)
                + src_tgt_adjusts[out_id]
                - path_clashes[curr_block_positioned][out_id].sum()
                < out_pendings[out_id]
            ):
                return True

    else:
        path_clashes[curr_block_positioned] = {}
        for out_id, out_beams in beams_to_check.items():
            path_clashes[curr_block_positioned][out_id] = np.array([False for _ in out_beams])

    return False


def _to_visit_or_not_to_visit(
    curr_block_positioned: PositionedZXBlock,
    nxt_block: PositionedZXBlock,
    queue: deque,
    visited: dict[tuple[PositionedZXBlock, StandardCoord], int],
    move: tuple[int, int, int],
    path: dict[PositionedZXBlock, list[PositionedZXBlock]],
    visit_attempts: int,
    all_search_paths: dict[PositionedZXBlock, list[PositionedZXBlock]],
) -> tuple[
    deque,
    dict[tuple[PositionedZXBlock, StandardCoord], int],
    dict[PositionedZXBlock, list[PositionedZXBlock]],
    dict[PositionedZXBlock, int],
    int,
    dict[PositionedZXBlock, list[PositionedZXBlock]],
]:
    """Visit site if conditions are met.

    Args:
        curr_block_positioned: The positioned ZX block (coordinates, zx_block).
        nxt_block: The positioned ZX block for the next block.
        queue: The pathfinder's BFS primary queue.
        visited: All visited sites by the pathfinder BFS.
        move: The spatial displacement (aka. move) currently under consideration.
        path: The full path object for the entire BFS.
        visit_attempts:  Total number of visitation attempts made throughout the pathfinder BFS.
        all_search_paths: All paths searched throughout the pathfinder BFS including those not leading to a visit.

    Returns:
        queue: The pathfinder's BFS primary queue.
        visited: All visited sites by the pathfinder BFS.
        path: The full path object for the entire BFS.
        visit_attempt: Total number of visitation attempts made throughout the pathfinder BFS.
        all_search_paths: All paths searched throughout the pathfinder BFS including those not leading to a visit.

    """

    # Update counters and add path to all_search_paths
    visit_attempts += 1
    all_search_paths[nxt_block] = [*path[curr_block_positioned], nxt_block]

    # Check next coords not in visited or path no longer than equiv. path
    if ((nxt_block, move)) not in visited or len(all_search_paths[nxt_block]) < visited[
        (nxt_block, move)
    ]:
        # Log to visited
        visited[(nxt_block, move)] = len(all_search_paths[nxt_block])

        # Append to queue
        queue.append(nxt_block)

        # Add path
        path[nxt_block] = all_search_paths[nxt_block]

    return queue, visited, path, visit_attempts, all_search_paths


def _check_for_success(
    nxt_block_positioned: PositionedZXBlock,
    tgt_block_zx_type: str,
    tent_tgt_kinds: list[str],
    path: dict[PositionedZXBlock, list[PositionedZXBlock]],
    valid_paths: dict[PositionedZXBlock, list[PositionedZXBlock]],
    tgts_to_fill: int,
    pruned_taken: set[StandardCoord],
    z_bounds: dict[str, int | None] = {},
) -> tuple[dict[PositionedZXBlock, list[PositionedZXBlock]], bool]:
    """Check if iteration achieved success.

    Args:
        nxt_block_positioned: The positioned ZX block (coordinates, zx_block).
        tgt_block_zx_type: The ZX type of the target block.
        tent_tgt_kinds: A list of kinds matching the zx-type of target block.
        path: The full path object for the entire BFS.
        valid_paths: All paths found in round covering some or all tent_coords.
        tgts_to_fill: Min number of targets that need to be fulfilled for pathfinder to be successful.
        pruned_taken: A pruned version of taken not containing source and target coordinates.
        z_bounds: Min. and max. Z-coordinate possible for a given move, if either exists.

    Return:
        [bool]: True if success was achieved in this iteration, else False.

    """

    # Extract last block in path
    nxt_coords, _ = nxt_block_positioned

    # Separate checks into categories for readability
    fail_time_constraints = False
    fail_time_constraints_2 = False
    fail_special_cube_constraints = False

    # Check time constraints
    if z_bounds.get("min"):
        fail_time_constraints = z_bounds["min"] >= nxt_coords[2]

    if not fail_time_constraints and z_bounds.get("max"):
        fail_time_constraints_2 = z_bounds["max"] <= nxt_coords[2]

    if fail_time_constraints or fail_time_constraints_2:
        return False

    # Check special cube conditions
    if tgt_block_zx_type in ["Y", "T", "XZ", "O"]:
        fail_special_cube_constraints = nxt_coords in list([c for c, _ in valid_paths.keys()])
        if tent_tgt_kinds == ["TTO"]:
            curr_path_coords = [coords for coords, _ in path[nxt_block_positioned]]
            for i in range(1, 4):
                check_coords = (nxt_coords[0], nxt_coords[1], nxt_coords[2] - i)
                fail_special_cube_constraints = (
                    check_coords in pruned_taken or check_coords in curr_path_coords
                )
    if fail_special_cube_constraints:
        return False

    if tgt_block_zx_type == "T":
        goes_down = path[nxt_block_positioned][-2][0][2] > nxt_coords[2]
        if not goes_down:
            fail_special_cube_constraints = True

    if tgt_block_zx_type == "XZ":
        goes_up = path[nxt_block_positioned][-2][0][2] < nxt_coords[2]
        if not goes_up:
            fail_special_cube_constraints = True

    if fail_special_cube_constraints:
        return False

    # For all cases, return true only if standard checks clear
    if tgt_block_zx_type == "O" or nxt_block_positioned[1].kind in tent_tgt_kinds:
        if tgt_block_zx_type in ["XZ"]:
            nxt_kind = nxt_block_positioned[1].kind[:2] + "*"
            xz_block = ZXBlockRegistry.get_create(kind=nxt_kind)
            path[(nxt_coords, xz_block)] = [
                *path[nxt_block_positioned][:-1],
                (nxt_coords, xz_block),
            ]
            valid_paths[(nxt_coords, xz_block)] = path[(nxt_coords, xz_block)]
        else:
            if nxt_block_positioned not in valid_paths or len(path[nxt_block_positioned]) < len(valid_paths[nxt_block_positioned]):
                valid_paths[nxt_block_positioned] = path[nxt_block_positioned]

        if tgt_block_zx_type not in ["Y", "T", "O", "XZ"]:
            tgts_filled = len([p[0] for p in valid_paths.keys()])
            if tgts_filled >= tgts_to_fill:
                return True

    return False
