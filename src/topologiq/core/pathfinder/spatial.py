"""Key/common 3D/spatial operations used by the pathfinder.

Usage:
    Call any function/class from a separate script.

"""

import sys

from topologiq.core.beams import CubeBeams
from topologiq.core.pathfinder.beams import check_critical_beams
from topologiq.utils.classes import StandardBlock, StandardCoord


#######################
# PATHS & COORDINATES #
#######################
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


def get_coords_for_current_move(
    current_block: StandardBlock,
    move: tuple[int, int, int],
    scale: int,
    path: dict[StandardBlock, list[StandardBlock]],
) -> tuple[StandardCoord, list[StandardCoord], list[StandardCoord], tuple[int, int, int] | None]:
    """Update paths and generate the next coordinates for the current move.

    Args:
        current_block: The current/source block at any given time in the pathfinder BFS.
        move: The spatial displacement (aka. move) currently under consideration.
        scale: A multiplier to increase the size of the displacement.
        path: The full path object for the entire BFS.

    Returns:
        nxt_coords: The exact coordinates where the move would lead, i.e., current_block coords + move.
        curr_path_coords: The coordinates for the current path.
        full_path_coords: The coordinates for the current path including any intermediate coordinates.
        mid_coords: Any intermediate coordinates skipped due to scaler/multiplier.

    """

    # Extract current coordinates and kind
    (x, y, z), curr_kind = current_block

    # Extract move
    dx, dy, dz = move

    # Calculate mid-coords (coordinate in between a pipe and next cube) if any
    mid_coords = (x + dx, y + dy, z + dz) if "o" in curr_kind and scale == 2 else None

    # Calculate next coordinates
    nxt_x, nxt_y, nxt_z = x + dx * scale, y + dy * scale, z + dz * scale
    nxt_coords = (nxt_x, nxt_y, nxt_z)

    # Calculate patch at current points
    curr_path_coords = [n[0] for n in path[current_block]]

    # Calculate coordinates for the full path
    try:
        full_path_coords = get_taken_coords(path[current_block])
    except Exception as _:
        full_path_coords = curr_path_coords

    return nxt_coords, curr_path_coords, full_path_coords, mid_coords


###############
# CONSTRAINTS #
###############
def gen_bounding_box(
    taken: list[StandardCoord], second_pass: bool = False
) -> tuple[dict[str, dict[str, int]], int]:
    """Determine min/max coordinates for any second pass search.

    Args:
        taken: A list of all coordinates occupied by any previously-placed blocks/pipes.
        second_pass: A boolean flag to determine if search is a primary or `second_pass` search.

    Returns:
        bounding_box: A box made of min. and max. coordinates for each axis, which make a box
            declaring the space inside which the pathfinder is allowed to search for paths.
        max_span: the longest edge of the bounding box, equivalent to largest axes needed for box.

    """

    # Get the bounds of pre-existing blocks.
    bounds_x = [x for (x, _, _) in taken] if taken else [0, 0, 0]
    bounds_y = [y for (_, y, _) in taken] if taken else [0, 0, 0]
    bounds_z = [z for (_, _, z) in taken] if taken else [0, 0, 0]

    # Add small leeway depending on type of search
    margin = 30 if second_pass else 21
    min_x, max_x = (min(bounds_x) - margin, max(bounds_x) + margin)
    min_y, max_y = (min(bounds_y) - margin, max(bounds_y) + margin)
    min_z, max_z = (min(bounds_z) - margin, max(bounds_z) + margin)
    bounding_box = {
        "x": {"min": min_x - margin, "max": max_x + margin},
        "y": {"min": min_y - margin, "max": max_y + margin},
        "z": {"min": min_z - margin, "max": max_z + margin},
    }

    # Calculate maximum span across all axes
    max_span = max(
        [
            abs((min_x + margin) - (max_x - margin)),
            abs((min_y + margin) - (max_y - margin)),
            abs((min_z + margin) - (max_z - margin)),
        ]
    )

    return bounding_box, max_span


##########
# CHECKS #
##########
def check_skip_move(
    nxt_coords: StandardCoord,
    tgt_coords: list[StandardCoord],
    taken: list[StandardCoord],
    critical_beams: dict[StandardCoord, int, tuple[int, CubeBeams], tuple[int, CubeBeams]],
    src_tgt_ids: tuple[int, int],
    second_pass: bool,
    bounding_box: dict[str, dict[str, int]],
    full_path_coords: list[StandardCoord],
    curr_kind: str,
    curr_path_coords: list[StandardCoord],
    mid_coords: tuple[int, int, int] | None,
) -> bool:
    """Check if current move should be skipped to speed up pathfinding process.

    Args:
        nxt_coords: The coordinates being checked as potential next position to place a block.
        tgt_coords: The final "target" coordinates at which path should arrive.
        taken: A list of all coordinates occupied by any blocks/pipes placed throughout the algorithmic process.
        critical_beams: An object containing beams considered critical for future operations.
        src_tgt_ids: The exact IDs of the source and target cubes.
        second_pass: Whether the current BFS is part of a cross-edge operation.
        bounding_box: The coordinates determining the bounding box outside of which moves are not allowed.
        full_path_coords: The coordinates for the current path including any intermediate coordinates.
        curr_kind: The kind of the current source block.
        curr_path_coords: The coordinates for the current path.
        mid_coords: Any intermediate coordinates skipped due to scaler/multiplier.
        clash_coords: A list of coordinates considered problematic, if any.

    """

    if nxt_coords in taken or nxt_coords in full_path_coords:
        return True

    nxt_x, nxt_y, nxt_z = nxt_coords
    if second_pass and bounding_box:
        if (
            nxt_x < bounding_box["x"]["min"]
            or nxt_x > bounding_box["x"]["max"]
            or nxt_y < bounding_box["y"]["min"]
            or nxt_y > bounding_box["y"]["max"]
            or nxt_z < bounding_box["z"]["min"]
            or nxt_z > bounding_box["z"]["max"]
        ):
            return True

    # Adjust coords and taken coords for pipes
    if mid_coords and (mid_coords in curr_path_coords or mid_coords in taken):
        return True

    if critical_beams and "o" not in curr_kind:
        if len(tgt_coords) > 1:
            sys.stderr.write("Warning: check_skip_move tgt_coords > 1")

        if not check_critical_beams(
            critical_beams, full_path_coords, nxt_coords, tgt_coords[0], src_tgt_ids
        ):
            return True

    return False
