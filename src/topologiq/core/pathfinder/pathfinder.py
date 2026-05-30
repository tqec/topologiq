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

from topologiq.core.blocks import PositionedZXBlock, ZXBlock
from topologiq.core.pathfinder.spatial import (
    check_skip_move,
    gen_bounding_box,
    get_coords_for_current_move,
)
from topologiq.core.pathfinder.utils import gen_exit_conditions, init_bfs
from topologiq.utils.classes import CubeBeams, StandardBlock, StandardCoord
from topologiq.utils.misc import get_manhattan


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
    parametrised_taken: dict[tuple[int, int], set],
    is_hadamard: bool,
    time_deps: tuple[int, int] | None = None,
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
        parametrised_taken: A version of taken parametrised for more efficient clash detection.
        is_hadamard: True if the current edge is a Hadamard in the input ZX graph.
        time_deps (optional): A time-ordering dependency for the target spider (relative to another spider in the graph).
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
    time_deps_id, time_deps_diff = time_deps if time_deps else (None, None)
    time_deps_coords = bgraph.nodes[time_deps_id]["coords"] if time_deps_id else None

    # Generate kinds that could in theory be assigned to the target cube
    tent_tgt_kinds = tgt_zx_block.kind if cross_edge else tgt_zx_block.get_kind_family

    # Create bounding box to limit space search
    bounding_box, max_span = gen_bounding_box(taken, cross_edge=cross_edge)

    # Initialise BFS
    queue, visited, visit_attempts, path_len, path, valid_paths, all_search_paths = init_bfs(
        (src_coords, src_zx_block)
    )

    # Define exit conditions in case something goes wrong
    tgts_to_fill, max_manhattan, src_tgt_manhattan = gen_exit_conditions(
        src_coords,
        tent_coords,
        taken,
        max_span,
        cross_edge,
        kwargs["min_succ_rate"],
        special_target_kind=tgt_zx_block.zx_type in ["Y", "T", "XZ"],
    )

    # Manage queue
    hdm = is_hadamard
    while queue:
        # Unpack current block (source for iteration)
        curr_block_positioned: PositionedZXBlock = queue.popleft()
        curr_coords: StandardCoord = curr_block_positioned[0]
        curr_zx_block: ZXBlock = curr_block_positioned[1]

        # Check skip/break tolerances
        curr_manhattan = get_manhattan(src_coords, curr_coords)
        if curr_manhattan > src_tgt_manhattan * 3:
            continue
        if curr_manhattan > max_manhattan:
            pass  # Need to eventually delete, leaving it here for debugging purposes

        # Check for success
        if curr_coords in tent_coords:
            if _check_for_success(
                curr_block_positioned,
                tent_tgt_kinds,
                path,
                valid_paths,
                tgts_to_fill,
                pruned_taken,
                realised_time_deps=(time_deps_coords, time_deps_diff),
            ):
                break
            else:
                if cross_edge or curr_zx_block.zx_type in ["Y", "T", "XZ"]:
                    continue

        # Avoid multiple special gates on same path
        if tgt_zx_block.zx_type in ["Y", "T", "XZ"]:
            if any(
                [zx_b.zx_type == tgt_zx_block.zx_type for _, zx_b in path[curr_block_positioned]]
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
                bgraph,
                beams,
                beams_short,
                curr_src_id,
                curr_tgt_id,
                nxt_coords,
                tent_coords,
                bounding_box,
                curr_path_coords,
                pruned_taken,
                parametrised_taken,
                cross_edge,
                special_target_kind=tgt_zx_block.zx_type in ["Y", "T", "XZ"],
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

                # Log to visited and update path lengths if all conditions met
                queue, visited, path, path_len, visit_attempts, all_search_paths = (
                    _to_visit_or_not_to_visit(
                        curr_block_positioned,
                        nxt_block,
                        queue,
                        visited,
                        move,
                        path,
                        path_len,
                        visit_attempts,
                        all_search_paths,
                    )
                )

        # Hadamards are introduce on very first move so once loop clears first
        # set of possible moves, there can be no more Hadamards in a specific edge.
        hdm = False

    return valid_paths, (tent_tgt_kinds, visit_attempts, len(visited), all_search_paths)


########
# AUX #
#######
def _to_visit_or_not_to_visit(
    curr_block_positioned: PositionedZXBlock,
    nxt_block: PositionedZXBlock,
    queue: deque,
    visited: dict[tuple[PositionedZXBlock, StandardCoord], int],
    move: tuple[int, int, int],
    path: dict[PositionedZXBlock, list[PositionedZXBlock]],
    path_len: dict[PositionedZXBlock, int],
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
        path_len: The length of the current path.
        visit_attempts:  Total number of visitation attempts made throughout the pathfinder BFS.
        all_search_paths: All paths searched throughout the pathfinder BFS including those not leading to a visit.

    Returns:
        queue: The pathfinder's BFS primary queue.
        visited: All visited sites by the pathfinder BFS.
        path: The full path object for the entire BFS.
        path_len: The length of the current path.
        visit_attempt: Total number of visitation attempts made throughout the pathfinder BFS.
        all_search_paths: All paths searched throughout the pathfinder BFS including those not leading to a visit.

    """

    # Update counters and add path to all_search_paths
    visit_attempts += 1
    all_search_paths[nxt_block] = path[curr_block_positioned] + [nxt_block]

    # Determine length of new path
    new_path_len = path_len[curr_block_positioned] + 1

    # Check next coords not in visited or path no longer than equiv. path
    if ((nxt_block, move)) not in visited or new_path_len < visited[(nxt_block, move)]:
        # Log to visited & append to queue
        visited[(nxt_block, move)] = new_path_len
        queue.append(nxt_block)

        # Adjust path and path length
        path_len[nxt_block] = new_path_len
        path[nxt_block] = path[curr_block_positioned] + [nxt_block]

    return queue, visited, path, path_len, visit_attempts, all_search_paths


def _check_for_success(
    curr_block_positioned: PositionedZXBlock,
    tent_tgt_kinds: list[str],
    path: dict[PositionedZXBlock, list[PositionedZXBlock]],
    valid_paths: dict[PositionedZXBlock, list[PositionedZXBlock]],
    tgts_to_fill: int,
    pruned_taken: set[StandardCoord],
    realised_time_deps: tuple[StandardCoord | None, int | None],
) -> tuple[dict[PositionedZXBlock, list[PositionedZXBlock]], bool]:
    """Check if iteration achieved success.

    Args:
        curr_block_positioned: The positioned ZX block (coordinates, zx_block).
        tent_tgt_kinds: A list of kinds matching the zx-type of target block.
        path: The full path object for the entire BFS.
        valid_paths: All paths found in round covering some or all tent_coords.
        tgts_to_fill: Min number of targets that need to be fulfilled for pathfinder to be successful.
        pruned_taken: A pruned version of taken not containing source and target coordinates.
        realised_time_deps (optional): The z-coords and plus/minus diff between target and reference spider.

    Return:
        [bool]: True if success was achieved in this iteration, else False.

    """

    curr_coords, _ = curr_block_positioned
    time_deps_coords, time_deps_diff = realised_time_deps

    time_deps_check = True
    if time_deps_coords and time_deps_diff:
        if time_deps_diff == -1:
            time_deps_check = time_deps_coords[2] > curr_coords[2]
        elif time_deps_diff == 1:
            time_deps_check = time_deps_coords[2] < curr_coords[2]

        if not time_deps_check:
            return False

    if tent_tgt_kinds[0] in ["YYO", "TTO", "XZ*", "ZX*"]:
        if curr_coords in list([c for c, b in valid_paths.keys()]):
            return False

        if tent_tgt_kinds == ["TTO"]:
            curr_path_coords = [coords for coords, _ in path[curr_block_positioned]]
            for i in range(1, 4):
                check_coords = (curr_coords[0], curr_coords[1], curr_coords[2] - i)
                if check_coords in pruned_taken or check_coords in curr_path_coords:
                    return False

        if tent_tgt_kinds[0] in ["TTO", "ZX*", "XZ*"]:
            prev_coords = path[curr_block_positioned][-2][0]
            last_z_diff = abs(curr_coords[2]) - abs(prev_coords[2])
            last_z_diff = 1 if last_z_diff > 0 else -1 if last_z_diff < 0 else 0
            valid_last_diffs = [-1] if tent_tgt_kinds[0] == "TTO" else [1]
            if last_z_diff not in valid_last_diffs:
                return False

    if tent_tgt_kinds == ["OOO"] or curr_block_positioned[1].kind in tent_tgt_kinds:
        valid_paths[curr_block_positioned] = path[curr_block_positioned]

        tgts_filled = len(set([p[0] for p in valid_paths.keys()]))

        if tgts_filled >= tgts_to_fill:
            return True

    return False
