"""Key/common 3D/spatial operations used by the pathfinder.

Usage:
    Call any function/class from a separate script.

"""

import networkx as nx

from topologiq.core.blocks import PositionedZXBlock
from topologiq.core.pathfinder.beams import check_beams
from topologiq.utils.classes import CubeBeams, StandardBlock, StandardCoord


#######################
# PATHS & COORDINATES #
#######################
def get_coords_for_current_move(
    curr_block_positioned: PositionedZXBlock,
    move: tuple[int, int, int],
    path: dict[StandardBlock, list[StandardBlock]],
) -> tuple[StandardCoord, list[StandardCoord], tuple[int, int, int] | None]:
    """Update paths and generate the next coordinates for the current move.

    Args:
        curr_block_positioned: The positioned ZX block (coordinates, zx_block).
        move: The spatial displacement (aka. move) currently under consideration.
        path: The full path object for the entire BFS.

    Returns:
        nxt_coords: The exact coordinates where the move would lead, i.e., current_block coords + move.
        curr_path_coords: The coordinates for the current path.

    """

    # Extract current coordinates and kind
    (x, y, z), _ = curr_block_positioned

    # Extract move
    dx, dy, dz = move

    # Calculate next coordinates
    nxt_coords = (x + dx, y + dy, z + dz)

    # Calculate patch at current points
    curr_path_coords = [n[0] for n in path[curr_block_positioned]]

    return nxt_coords, curr_path_coords


###############
# CONSTRAINTS #
###############
def gen_bounding_box(
    taken: list[StandardCoord], cross_edge: bool = False
) -> tuple[dict[str, dict[str, int]], int]:
    """Determine min/max coordinates for any second pass search.

    Args:
        taken: A list of all coordinates occupied by any previously-placed blocks/pipes.
        cross_edge: A boolean flag to determine if search is a primary or `cross_edge` search.

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
    margin = 30 if cross_edge else 21
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
    bgraph: nx.Graph,
    beams: CubeBeams,
    beams_short: CubeBeams,
    curr_src_id: int,
    curr_tgt_id: int,
    nxt_coords: StandardCoord,
    tent_coords: list[StandardCoord],
    bounding_box: dict[str, dict[str, int]],
    curr_path_coords: list[StandardCoord],
    pruned_taken: set[StandardCoord],
    cross_edge: bool,
) -> bool:
    """Check if current move should be skipped to speed up pathfinding process.

    Args:
        bgraph: The BlockGraph currently being built.
        beams: The beams for all the cubes in blockgraph that need beams..
        beams_short: The short beams for all the cubes in blockgraph that need beams..
        curr_src_id: The ID of the current source cube.
        curr_tgt_id: The ID of the current target cube.
        nxt_coords: The coordinates being checked as potential next position to place a block.
        tent_coords: The final "target" coordinates at which path should arrive.
        bounding_box: The coordinates determining the bounding box outside of which moves are not allowed.
        curr_path_coords: The coordinates for the current path.
        cross_edge: True if the current edge is a cross-edge (as opposed to a standard edge).
        pruned_taken: A pruned version of taken not containing source and target coordinates.

    """

    if nxt_coords in pruned_taken or nxt_coords in curr_path_coords:
        return True

    if cross_edge:
        if bounding_box:
            nxt_x, nxt_y, nxt_z = nxt_coords
            if (
                nxt_x < bounding_box["x"]["min"]
                or nxt_x > bounding_box["x"]["max"]
                or nxt_y < bounding_box["y"]["min"]
                or nxt_y > bounding_box["y"]["max"]
                or nxt_z < bounding_box["z"]["min"]
                or nxt_x > bounding_box["z"]["max"]
            ):
                return True

        if not check_beams(
            bgraph,
            beams,
            beams_short,
            curr_src_id,
            curr_tgt_id,
            nxt_coords,
            tent_coords,
            curr_path_coords,
        ):
            return True

    return False
