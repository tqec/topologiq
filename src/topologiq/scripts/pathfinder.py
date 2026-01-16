"""Manage the inner pathfinder BFS algorithm.

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
from datetime import datetime

from topologiq.utils.classes import CubeBeams, StandardBlock, StandardCoord
from topologiq.utils.utils_greedy_bfs import get_bounding_box
from topologiq.utils.utils_misc import prep_stats_n_log
from topologiq.utils.utils_pathfinder import get_max_manhattan, nxt_kinds, prune_visited, rot_o_kind


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
    min_succ_rate: int = 60,
    critical_beams: dict[int, tuple[StandardCoord, int, CubeBeams]] = {},
    src_tgt_ids: tuple[int, int] | None = None,
    log_stats_id: str | None = None,
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
        min_succ_rate (optional): Minimum % of tentative coordinates that must be filled for each edge.
        critical_beams (optional): Annotated beams object with details about minimum number of beams needed per node.
        src_tgt_ids (optional): The exact IDs of the source and target cubes.
        log_stats_id (optional): A unique datetime-based identifier for the purposes of logging stats for an specific run.

    Returns:
        valid_paths: all paths found in round, covering some or all tent_coords.

    """

    # Preliminaries
    t_start_pathfinder = datetime.now()
    t_end_pathfinder = None

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
        min_succ_rate,
        taken=taken_cc,
        hdm=hdm,
        critical_beams=critical_beams,
        src_tgt_ids=src_tgt_ids,
    )

    pathfinder_vis_data = [tent_coords, tent_tgt_kinds, all_search_paths, valid_paths]

    # Log stats if needed
    if log_stats_id is not None:
        # End timers
        t_end_pathfinder = datetime.now()
        duration_pathfinder = (t_end_pathfinder - t_start_pathfinder).total_seconds()
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
            log_stats_id,
            pathfinder_iter_success,
            counts,
            times,
            src_block_info=src_block_info,
            tgt_block_info=adjusted_target_info,
            tgt_zx_type=tgt_zx_type,
            visit_stats=visit_stats,
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
    min_succ_rate: int,
    taken: list[StandardCoord] = [],
    hdm: bool = False,
    critical_beams: dict[int, tuple[StandardCoord, int, CubeBeams]] = {},
    src_tgt_ids: tuple[int, int] | None = None,
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

    Returns:
        valid_paths: All paths found in round covering some or all tent_coords.

    """

    # Unpack incoming data
    src_coords, _ = src_block_info
    src_x, src_y, src_z = src_coords
    tgt_coords = tent_coords
    second_pass = False
    bounding_box, _ = get_bounding_box(taken, second_pass=second_pass)
    unbreakable_beams, negotiable_beams, forbidden_paths = (None, None, [])
    exit_flag = False

    if src_coords in taken:
        taken.remove(src_coords)
    if len(tgt_coords) == 1 and len(tent_tgt_kinds) == 1:
        second_pass = True
        if tgt_coords[0] in taken:
            taken.remove(tgt_coords[0])
        unbreakable_beams, negotiable_beams = split_critical_beams(critical_beams)

    # BFS management variables
    queue = deque([src_block_info])
    visited: dict[tuple[StandardBlock, StandardCoord], int] = {(src_block_info, (0, 0, 0)): 0}
    visit_attempts = 0
    path_len = {src_block_info: 0}
    path = {src_block_info: [src_block_info]}
    valid_paths: dict[StandardBlock, list[StandardBlock]] | None = {}
    all_search_paths: dict[StandardBlock, list[StandardBlock]] | None = {}
    moves = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
    ]

    # Define exit conditions in case something goes wrong
    tgts_filled = 0
    tgts_to_fill = int(len(tent_coords) * min_succ_rate / 100) if len(tent_coords) > 1 else 1
    if not second_pass:
        max_manhattan = get_max_manhattan(src_coords, tent_coords) * 2
        src_tgt_manhattan = max_manhattan
    else:
        src_tgt_manhattan = get_max_manhattan(src_coords, tent_coords)
        max_manhattan = max(
            get_max_manhattan(src_coords, taken) * 2,
            src_tgt_manhattan * 3,
        )
    src_tgt_manhattan = get_max_manhattan(src_coords, tent_coords)

    # Manage queue
    while queue:
        # Unpack current block (source for iteration)
        current_block: StandardBlock = queue.popleft()
        curr_coords, curr_kind = current_block
        x, y, z = curr_coords

        # Check exit conditions in case something's gone wrong
        curr_manhattan = abs(x - src_x) + abs(y - src_y) + abs(z - src_z)
        if curr_manhattan > src_tgt_manhattan + 6:
            continue
        if curr_manhattan > max_manhattan:
            break
        if second_pass:
            if current_block not in path:
                continue

        if curr_coords in tgt_coords:
            if tent_tgt_kinds == ["ooo"] or curr_kind in tent_tgt_kinds:
                if not second_pass:
                    valid_paths[current_block] = path[current_block]
                    tgts_filled = len(set([p[0] for p in valid_paths.keys()]))
                    if tgts_filled >= tgts_to_fill:
                        break

                #elif second_pass and path[current_block] not in forbidden_paths:
                    #all_coords_valid_path = get_taken_coords(path[current_block])
                    #if len(all_coords_valid_path) >= max_manhattan:
                        #continue
                    #if check_unbreakable_beams(unbreakable_beams, all_coords_valid_path, src_tgt_ids) and check_negotiable_beams(negotiable_beams, all_coords_valid_path, src_tgt_ids):
                        #print("path ok")
                        #valid_paths[current_block] = path[current_block]
                        #break
                    #else:
                        #print("path not ok")
                        #visited = prune_visited(visited, path, remove_block_info=current_block)
                        #visited = {}
                        #forbidden_paths.append(path[current_block])
                        #del path[current_block]
                        #del path_len[current_block]
            else:
                continue

        # Try moving in all directions
        scale = 2 if "o" in curr_kind else 1  # Block is pipe if "o" in kind
        for dx, dy, dz in moves:
            # Set exploration parameters to next position in moves
            nxt_x, nxt_y, nxt_z = x + dx * scale, y + dy * scale, z + dz * scale
            nxt_coords = (nxt_x, nxt_y, nxt_z)
            curr_path_coords = [n[0] for n in path[current_block]]
            #try:
                #full_path_coords = get_taken_coords(path[current_block])
            #except Exception as _:
                #full_path_coords = curr_path_coords

            # Skip if next position has been taken or is in current path
            if nxt_coords in taken or nxt_coords in curr_path_coords:
                continue

            if second_pass and bounding_box:
                if (
                    (nxt_x < bounding_box["x"]["min"] or nxt_x > bounding_box["x"]["max"])
                    or (nxt_y < bounding_box["y"]["min"] or nxt_y > bounding_box["y"]["max"])
                    or (nxt_z < bounding_box["z"]["min"] or nxt_x > bounding_box["z"]["max"])
                ):
                    continue

            # Adjust coords and taken coords for pipes
            mid_coords = None
            if "o" in curr_kind and scale == 2:
                mid_coords = (x + dx, y + dy, z + dz)
                if mid_coords in curr_path_coords or mid_coords in taken:
                    continue

            # Skip if beam clashes arise with nodes that need all their exits
            #if unbreakable_beams and "o" not in curr_kind:
                #if not check_unbreakable_beams(unbreakable_beams, full_path_coords, src_tgt_ids):
                    #continue

            # Skip if number of beam clashes with nodes that need only some exits > than tolerable
            #if negotiable_beams and "o" not in curr_kind:
                #if not check_negotiable_beams(negotiable_beams, full_path_coords, src_tgt_ids):
                    #continue

            # Rotate if current kind is a Hadamard
            # NB! The raw kinds of Hadamards correspond to unrotated colours.
            # As the next kind latches to rotated end of hadamard, kind must be rotated
            alt_curr_kind = None
            if "h" in curr_kind:
                hdm = False
                direction = sum(
                    [p[1] - p[0] if p[0] != p[1] else 0 for p in list(zip(curr_coords, nxt_coords))]
                )
                if direction < 0:
                    pass
                else:
                    alt_curr_kind = rot_o_kind(curr_kind)

            # Create a list of kinds that are valid for the next block
            possible_nxt_types = nxt_kinds(
                curr_coords, curr_kind if not alt_curr_kind else alt_curr_kind, nxt_coords
            )
            for possible_nxt_type in possible_nxt_types:
                # Create a copy of next type to avoid re-writting the actual loop variable
                nxt_type = possible_nxt_type

                # Place Hadamard instead of regular pipe if all corresponding flags are present
                if hdm and "o" in nxt_type:
                    nxt_type += "h"
                    direction = sum(
                        [
                            p[1] - p[0] if p[0] != p[1] else 0
                            for p in list(zip(curr_coords, nxt_coords))
                        ]
                    )
                    if direction < 0:
                        nxt_type = rot_o_kind(nxt_type)

                # Log to visited and update path lengths if all conditions met
                # Note. If conditions not met, move would break topology
                # Increase counter of times pathfinder tries visits something new
                visit_attempts += 1

                # Undertake checks and visit if move is valid
                # Do NOT visit otherwise so site is not taken for a different path
                if mid_coords is None or nxt_coords != mid_coords:
                    new_path_len = path_len[current_block] + 1
                    nxt_b_info: StandardBlock = (nxt_coords, nxt_type)

                    if ((nxt_b_info, (dx, dy, dz))) not in visited or new_path_len < visited[
                        (nxt_b_info, (dx, dy, dz))
                    ]:
                        visited[(nxt_b_info, (dx, dy, dz))] = new_path_len
                        queue.append(nxt_b_info)
                        path_len[nxt_b_info] = new_path_len
                        path[nxt_b_info] = path[current_block] + [nxt_b_info]

                        if nxt_coords in tgt_coords and (
                            tent_tgt_kinds == ["ooo"] or nxt_type in tent_tgt_kinds
                        ):
                            all_search_paths[nxt_b_info] = path[nxt_b_info]

                            if not second_pass:
                                valid_paths[nxt_b_info] = path[nxt_b_info]
                                tgts_filled = len(set([p[0] for p in valid_paths.keys()]))
                                if tgts_filled >= tgts_to_fill:
                                    break

                            elif second_pass and path[nxt_b_info] not in forbidden_paths:
                                all_coords_valid_path = get_taken_coords(path[nxt_b_info])
                                exit_flag = False
                                unbreakable_ok, problem_coords = check_unbreakable_beams(unbreakable_beams, all_coords_valid_path, src_tgt_ids)
                                negotiable_ok = check_negotiable_beams(negotiable_beams, all_coords_valid_path, src_tgt_ids)
                                if unbreakable_ok and negotiable_ok:

                                    valid_paths[nxt_b_info] = path[nxt_b_info]
                                    exit_flag = True
                                    break
                                else:
                                    #print("path not ok")
                                    exit_flag = False
                                    taken.extend(problem_coords)
                                    visited = prune_visited(visited, path, remove_block_info=nxt_b_info, problem_coords=problem_coords)
                                    visited = {}
                                    #if problem_coords:
                                        #for k, p in path.items():
                                            #check = any([block_coords in problem_coords for block_coords, _ in p])
                                            #if check:
                                                #print(check)


                                    forbidden_paths.append(path[nxt_b_info])
                                    del path[nxt_b_info]
                                    del path_len[nxt_b_info]

                        else:
                            all_search_paths[nxt_b_info] = path[nxt_b_info]

            if exit_flag:
                break

        if exit_flag:
           break

    return valid_paths, all_search_paths, (visit_attempts, len(visited))


##################
# AUX OPERATIONS #
##################
def gen_tent_tgt_kinds(tgt_zx_type: str, tgt_kind: str | None = None) -> list[str]:
    """Generate all possible valid kinds for a given ZX type.

    This function takes the ZX type of a potential new block in a 3D path and returns
    a list of block (cube or pipe) kinds that could fulfill that ZX type. Rather than
    seeing the function as creating kinds to check, the function should be seen as
    reducing the number of kinds to check in any given iteration.

    Args:
        tgt_zx_type: The ZX type of the target spider/cube.
        tgt_kind (optional): A specific kind used to override the function.

    Returns:
        kind_family: a list of applicable kinds for the given ZX type.

    """

    # Return override value if present
    if tgt_kind:
        return [tgt_kind]

    # Get family of kinds corresponding to the ZX type of target
    if tgt_zx_type in ["X", "Z"]:
        kind_family = ["zzx", "zxz", "xzz"] if tgt_zx_type == "X" else ["xxz", "xzx", "zxx"]
    elif tgt_zx_type == "O":
        kind_family = ["ooo"]
    elif tgt_zx_type == "SIMPLE":
        kind_family = ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"]
    elif tgt_zx_type == "HADAMARD":
        kind_family = ["zxoh", "xzoh", "oxzh", "ozxh", "xozh", "zoxh"]
    else:
        return [tgt_zx_type]

    return kind_family


def get_taken_coords(all_blocks: list[StandardBlock]) -> list[StandardCoord]:
    """Convert a series of blocks into a list of coordinates occupied by the blocks.

    Args:
        all_blocks: A list of blocks (cubes and pipes).

    Returns:
        taken: A list of coordinates taken by the incoming blocks (cubes and pipes).

    """

    # Refuse operation if not given a list of blocks
    if not all_blocks:
        return []

    # Add coords of first block
    taken_set = set()
    first_block = all_blocks[0]
    if first_block:
        b_1_coords = first_block[0]
        taken_set.add(b_1_coords)

    # Iterate from 2nd block onwards
    for i, _ in enumerate(all_blocks):
        if i > 0:
            # Get info for current and previous blocks
            current_block = all_blocks[i]
            prev_block = all_blocks[i - 1]

            # Extract coords from block info
            if current_block and prev_block:
                curr_coords, curr_kind = current_block
                prev_coords, _ = prev_block

                # Add current node's coordinates
                taken_set.add(curr_coords)

                # Add mid_position if block is a pipe
                if "o" in curr_kind:
                    current_x, current_y, current_z = curr_coords
                    prev_x, prev_y, prev_z = prev_coords
                    ext_cs = None

                    if current_x == prev_x + 1:
                        ext_cs = (current_x + 1, current_y, current_z)
                    elif current_x == prev_x - 1:
                        ext_cs = (current_x - 1, current_y, current_z)
                    elif current_y == prev_y + 1:
                        ext_cs = (current_x, current_y + 1, current_z)
                    elif current_y == prev_y - 1:
                        ext_cs = (current_x, current_y - 1, current_z)
                    elif current_z == prev_z + 1:
                        ext_cs = (current_x, current_y, current_z + 1)
                    elif current_z == prev_z - 1:
                        ext_cs = (current_x, current_y, current_z - 1)

                    if ext_cs:
                        taken_set.add(ext_cs)

    # Make sure taken is unique
    taken = list(taken_set)

    return taken


def split_critical_beams(
    critical_beams: dict[int, tuple[int, CubeBeams]],
) -> tuple[dict[int, tuple[int, CubeBeams]], dict[int, tuple[int, CubeBeams]]]:
    """Split critical beams into simple and verbose object containing different kinds of beams.

    This function separates the `critical_beams` object into a quickly iterable object containing
    coordinates of beams for nodes that needs absolutely all beams they have and a more verbose
    dictionary containing the beams for nodes that can lose some beams.

    Args:
        critical_beams: Beams considered critical for future operations.
        max_span: the longest edge of the bounding box, equivalent to largest beam needed to clear box.

    Returns:
        unbreakable_beams: The joint beam coordinates for nodes that need all beams they currently have.
        negotiable_beams: A minified `critical_beams` object containing beams for nodes that can lose some beams.

    """

    unbreakable_beams = {}
    negotiable_beams = {}
    for node_id, (node_coords, min_exit_num, node_beams) in critical_beams.items():
        if min_exit_num == len(node_beams):
            unbreakable_beams[node_id] = (node_coords, min_exit_num, [beam for beam in node_beams])
        else:
            negotiable_beams[node_id] = (node_coords, min_exit_num, [beam for beam in node_beams])

    return unbreakable_beams, negotiable_beams


def check_unbreakable_beams(
    unbreakable_beams: tuple[dict[int, tuple[StandardCoord, int, CubeBeams]]],
    full_path_coords: list[StandardCoord],
    src_tgt_ids: tuple[int, int],
):
    """Check that move does not break any beams of cubes that need all their exits.

    Args:
        unbreakable_beams: The joint beam coordinates for nodes that need all beams they currently have.
        full_path_coords: All coordinates occupied by current path.
        src_tgt_ids: The exact IDs of the source and target cubes.

    Return:
        (bool): True if move clears all checks, False otherwise

    """

    problem_coords = []
    for node_id, (_, _, node_beams) in unbreakable_beams.items():
        broken_beams = 0
        for single_beam in node_beams:
            clash_coords = [coord for coord in full_path_coords if single_beam.contains(coord)]
            if clash_coords:
                problem_coords.extend(clash_coords)
                # Reject if beam is of nodes other src and tgt
                if node_id not in src_tgt_ids:
                    return False, problem_coords
                # Reject if more than one beam of src and tgt cubes is broken
                if broken_beams == 1:
                    return False, problem_coords
                # Add to broken beams
                broken_beams += 1
    return True, problem_coords


def check_negotiable_beams(
    negotiable_beams: tuple[dict[int, tuple[StandardCoord, int, CubeBeams]]],
    full_path_coords: list[StandardCoord],
    src_tgt_ids: tuple[int, int],
):
    """Check that move does not break any beams of cubes that need all their exits.

    Args:
        negotiable_beams: A minified `critical_beams` object containing beams for nodes that can lose some beams.
        full_path_coords: All coordinates occupied by current path.
        src_tgt_ids: The exact IDs of the source and target cubes.

    Return:
        (bool): True if move clears all checks, False otherwise.

    """

    for node_id, (_, min_exit_num, cube_beams) in negotiable_beams.items():
        # For each beam of current cube, check if path breaks the beam
        broken_beams = 0
        for single_beam in cube_beams:
            if any([single_beam.contains(coord) for coord in full_path_coords]):
                broken_beams += 1
                # If beam is broken, add pre-existing beam-to-beam clashes,
                # as broken beam might eat into allowances already used
                for _, _, all_beams in negotiable_beams.values():
                    intersections = [single_beam.intersects(negotiable_beam) for negotiable_beam in all_beams]
                    if intersections:
                        broken_beams += sum(intersections)
                        broken_beams -= 1 if node_id in src_tgt_ids else 0

        # Adjust to consider the broken beam of outgoing/incoming edge in src and tgt cubes
        adjust = 1 if node_id in src_tgt_ids else 0

        # Flip check to false if number of broken beams is beyond tolerance
        if len(cube_beams) - broken_beams < (min_exit_num - adjust):
            return False
    return True
