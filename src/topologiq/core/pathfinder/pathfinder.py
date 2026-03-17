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

from topologiq.core.beams import CubeBeams
from topologiq.core.pathfinder.spatial import (
    check_skip_move,
    gen_bounding_box,
    get_coords_for_current_move,
)
from topologiq.core.pathfinder.symbolic import (
    handle_kind_after_hadamard,
    nxt_kinds,
    validate_nxt_kind,
)
from topologiq.core.pathfinder.utils import (
    check_run_mode,
    gen_exit_conditions,
    gen_tent_tgt_kinds,
    get_manhattan,
    get_max_manhattan,
    init_bfs,
)
from topologiq.utils.classes import StandardBlock, StandardCoord
from topologiq.utils.core import datetime_manager
from topologiq.utils.read_write import prep_stats_n_log


############################
# MAIN PATHFINDER WORKFLOW #
############################
def pathfinder(
    src_block_info: StandardBlock,
    tent_coords: list[StandardCoord],
    tgt_zx_type: str,
    tgt_block_info: tuple[StandardCoord | None, str | None] = (None, None),
    taken: list[StandardCoord] = [],
    hdm: bool = False,
    critical_beams: dict[int, tuple[StandardCoord, int, CubeBeams, CubeBeams]] = {},
    src_tgt_ids: tuple[int, int] | None = None,
    **kwargs,
) -> tuple[
    dict[StandardBlock, list[StandardBlock]] | None,
    tuple[
        list[StandardCoord] | None,
        list[str] | None,
        dict[StandardBlock, list[StandardBlock]] | None,
        dict[StandardBlock, list[StandardBlock]] | None,
    ]
    | None,
]:
    """Call core pathfinder after generating list of possible kinds for the given operation.

    Args:
        src_block_info: The coords and kind of the source block.
        tent_coords: A list of tentative target coordinates to find paths to.
        tgt_zx_type: The ZX type of the target spider/cube.
        tgt_block_info (optional): The coords and type of a previously placed target block.
        taken (optional): A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process (updated regularly).
        hdm (optional): If True, it indicates that the original ZX-edge is a Hadamard edge.
        critical_beams (optional): Annotated beams object with details about minimum number of beams needed per node.
        src_tgt_ids (optional): The exact IDs of the source and target cubes.
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

    Returns:
        valid_paths: All paths found in round, covering some or all tent_coords.
        pathfinder_vis_data: A list containing data for visualisation of a given pathfinder run.

    """

    # Preliminaries
    t_1, _ = datetime_manager()

    # Unpack incoming data
    src_coords, _ = src_block_info
    _, tgt_kind = tgt_block_info
    taken_cc: list[StandardCoord] = taken[:]
    if taken_cc:
        if src_coords in taken_cc:
            taken_cc.remove(src_coords)

    # Generate kinds that could in theory be assigned to the target cube
    # Note. When handling many tent_coords, the kind for a given ZX type might differ
    tent_tgt_kinds = gen_tent_tgt_kinds(
        tgt_zx_type,
        tgt_kind=(tgt_kind if tgt_kind else None),
    )
    # Call pathfinder
    valid_paths, all_search_paths, visit_stats = core_pathfinder_bfs(
        src_block_info,
        tent_coords,
        tent_tgt_kinds,
        taken=taken_cc,
        hdm=hdm,
        critical_beams=critical_beams,
        src_tgt_ids=src_tgt_ids,
        **kwargs,
    )

    pathfinder_vis_data = [tent_coords, tent_tgt_kinds, all_search_paths, valid_paths]

    # Log stats if needed
    if kwargs["log_stats_id"] is not None:
        # End timers
        _, duration_pathfinder = datetime_manager(t_1=t_1)
        times = {"duration_pathfinder": duration_pathfinder}
        pathfinder_iter_success = True if valid_paths else False

        # Calculate key metrics
        len_longest_path = 0
        if valid_paths:
            for path in valid_paths.values():
                if path:
                    len_path = sum([2 if "o" in b[1] else 1 for b in path]) - 1
                    len_longest_path = max(len_longest_path, len_path)

        counts = {
            "num_tent_coords": len(tent_coords) if valid_paths else 0,
            "num_tent_coords_filled": (
                len(set([p[0] for p in valid_paths.keys()])) if valid_paths else 0
            ),
            "max_manhattan": get_max_manhattan(src_coords, tent_coords),
            "len_longest_path": len_longest_path if len_longest_path > 0 else 0,
        }

        # Log
        adjusted_target_info = (tent_coords, tent_tgt_kinds)
        prep_stats_n_log(
            "pathfinder",
            pathfinder_iter_success,
            counts,
            times,
            src_block_info=src_block_info,
            tgt_block_info=adjusted_target_info,
            tgt_zx_type=tgt_zx_type,
            visit_stats=visit_stats,
            cross_edge=len(tent_coords) == 1 and len(tent_tgt_kinds) == 1,
            **kwargs,
        )

    # Return valid paths and data for visualising round
    return valid_paths, pathfinder_vis_data


###############################
# CORE PATHFINDER SPATIAL BFS #
###############################
def core_pathfinder_bfs(
    src_block_info: StandardBlock,
    tent_coords: list[StandardCoord],
    tent_tgt_kinds: list[str],
    taken: list[StandardCoord] = [],
    hdm: bool = False,
    critical_beams: dict[int, tuple[StandardCoord, int, CubeBeams, CubeBeams]] = {},
    src_tgt_ids: tuple[int, int] | None = None,
    **kwargs,
) -> tuple[
    dict[StandardBlock, list[StandardBlock]] | None,
    dict[StandardBlock, list[StandardBlock]] | None,
    tuple[int, int],
]:
    """Create topologically-correct paths between a source and one or more target coordinates/kinds.

    This function is the core algorithm in the inner pathfinder BFS. It systematically explores a 4D space
    (x, y, z, kind) to find a topologically-correct path between a source cube with pre-existing coordinates
    and kind and one or more potential target cubes (given as a list of possibilities to test).

    Args:
        src_block_info: The coords and kind of the source block.
        tent_coords: A list of tentative target coordinates to find paths to.
        tent_tgt_kinds: A list of kinds matching the zx-type of target block.
        min_succ_rate: Minimum % of tentative coordinates that must be filled for each edge.
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        hdm (optional): If True, it indicates that the original ZX-edge is a Hadamard edge.
        critical_beams (optional): An object containing beams considered critical for future operations.
        src_tgt_ids (optional): The exact IDs of the source and target cubes.
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

    Returns:
        valid_paths: All paths found in round covering some or all tent_coords.

    """

    # Preliminaries
    src_coords, _ = src_block_info
    tgt_coords = tent_coords
    second_pass, taken = check_run_mode(src_coords, taken, tgt_coords, tent_tgt_kinds)
    bounding_box, max_span = gen_bounding_box(taken, second_pass=second_pass)

    # Initialise BFS
    queue, visited, visit_attempts, path_len, path, valid_paths, all_search_paths, moves = init_bfs(
        src_block_info
    )

    # Define exit conditions in case something goes wrong
    tgts_to_fill, max_manhattan, src_tgt_manhattan = gen_exit_conditions(
        src_coords, tent_coords, taken, max_span, second_pass, **kwargs
    )

    # Manage queue
    while queue:
        # Unpack current block (source for iteration)
        current_block: StandardBlock = queue.popleft()
        curr_coords, curr_kind = current_block
        scale = 2 if "o" in curr_kind else 1  # Block is pipe if "o" in kind

        # Check skip/break tolerances
        curr_manhattan = get_manhattan(src_coords, curr_coords)
        if curr_manhattan > src_tgt_manhattan + 6:
            continue
        if curr_manhattan > max_manhattan:
            pass  # Need to eventually delete, leaving it here for debugging purposes

        # Check for success
        if curr_coords in tgt_coords:
            if _check_for_success(current_block, tent_tgt_kinds, path, valid_paths, tgts_to_fill):
                break
            else:
                continue

        # Try moving in all directions
        for move in moves:
            # Calculate next position and update paths accordingly
            nxt_coords, curr_path_coords, full_path_coords, mid_coords = (
                get_coords_for_current_move((curr_coords, curr_kind), move, scale, path)
            )

            # Check if move can be skipped (for speed)
            if check_skip_move(
                nxt_coords,
                tgt_coords,
                taken,
                critical_beams,
                src_tgt_ids,
                second_pass,
                bounding_box,
                full_path_coords,
                curr_kind,
                curr_path_coords,
                mid_coords,
            ):
                continue

            # Rotate if current kind is a Hadamard
            alt_curr_kind, hdm = handle_kind_after_hadamard(current_block, nxt_coords, hdm)

            # Create a list of kinds that are valid for the next block
            possible_nxt_kinds = nxt_kinds(curr_kind if not alt_curr_kind else alt_curr_kind, move)

            # Loop over all possible next types
            for possible_nxt_kind in possible_nxt_kinds:
                # Check if next kind needs to be rotated due to Hadamard
                nxt_type = validate_nxt_kind(current_block, nxt_coords, possible_nxt_kind, hdm)
                nxt_block: StandardBlock = (nxt_coords, nxt_type)

                # Log to visited and update path lengths if all conditions met
                queue, visited, path, path_len, visit_attempts, all_search_paths = (
                    _to_visit_or_not_to_visit(
                        current_block,
                        nxt_block,
                        mid_coords,
                        queue,
                        visited,
                        move,
                        path,
                        path_len,
                        visit_attempts,
                        all_search_paths,
                    )
                )

    return valid_paths, all_search_paths, (visit_attempts, len(visited))


########
# AUX #
#######
def _to_visit_or_not_to_visit(
    current_block: StandardBlock,
    nxt_block: StandardBlock,
    mid_coords: tuple[int, int, int] | None,
    queue: deque,
    visited: dict[tuple[StandardBlock, StandardCoord], int],
    move: tuple[int, int, int],
    path: dict[StandardBlock, list[StandardBlock]],
    path_len: dict[StandardBlock, int],
    visit_attempts: int,
    all_search_paths: dict[StandardBlock, list[StandardBlock]],
) -> tuple[
    deque,
    dict[tuple[StandardBlock, StandardCoord], int],
    dict[StandardBlock, list[StandardBlock]],
    dict[StandardBlock, int],
    int,
    dict[StandardBlock, list[StandardBlock]],
]:
    """Visit site if conditions are met.

    Args:
        current_block: The coordinates and kind of the current block.
        nxt_block: The coordinates and kind of the current block.
        mid_coords: Any intermediate coordinates skipped due to scaler/multiplier.
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
    all_search_paths[nxt_block] = path[current_block] + [nxt_block]

    # Avoid clashes with mid_coords
    if mid_coords is None or nxt_block[0] != mid_coords:
        # Determine length of new path
        new_path_len = path_len[current_block] + 1

        # Check next coords not in visited or path no longer than equiv. path
        if ((nxt_block, move)) not in visited or new_path_len < visited[(nxt_block, move)]:
            # Log to visited & append to queue
            visited[(nxt_block, move)] = new_path_len
            queue.append(nxt_block)

            # Adjust path and path length
            path_len[nxt_block] = new_path_len
            path[nxt_block] = path[current_block] + [nxt_block]

    return queue, visited, path, path_len, visit_attempts, all_search_paths


def _check_for_success(
    current_block: StandardCoord,
    tent_tgt_kinds: list[str],
    path: dict[StandardBlock, list[StandardBlock]],
    valid_paths: dict[StandardBlock, list[StandardBlock]],
    tgts_to_fill: int,
) -> tuple[
    dict[StandardBlock, list[StandardBlock]], dict[StandardBlock, list[StandardBlock]], int, bool
]:
    """Check if iteration achieved success.

    Args:
        current_block: The coordinates and kind of the current block.
        tent_tgt_kinds: A list of kinds matching the zx-type of target block.
        path: The full path object for the entire BFS.
        valid_paths: All paths found in round covering some or all tent_coords.
        tgts_to_fill: Min number of targets that need to be fulfilled for pathfinder to be successful.

    Return:
        [bool]: True if success was achieved in this iteration, else False.

    """

    _, curr_kind = current_block
    if tent_tgt_kinds == ["ooo"] or curr_kind in tent_tgt_kinds:
        valid_paths[current_block] = path[current_block]
        tgts_filled = len(set([p[0] for p in valid_paths.keys()]))
        if tgts_filled >= tgts_to_fill:
            return True

    return False
