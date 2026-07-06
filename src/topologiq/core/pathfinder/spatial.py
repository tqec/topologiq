"""Key/common 3D/spatial operations used by the pathfinder.

Usage:
    Call any function/class from a separate script.

"""

from topologiq.core.blocks import PositionedZXBlock
from topologiq.utils.classes import GraphBounds, StandardBlock, StandardCoord


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
    taken: list[StandardCoord],
    cross_edge: bool = False,
    graph_bounds: GraphBounds | None = None,
) -> tuple[dict[str, dict[str, int]], int]:
    """Determine min/max coordinates for any second pass search.

    Args:
        taken: A list of all coordinates occupied by any previously-placed blocks/pipes.
        cross_edge: A boolean flag to determine if search is a primary or `cross_edge` search.
        graph_bounds (optional): A tuple of max_x and max_y coordinates to maintain build within bounds.

    Returns:
        bounding_box: A box made of min. and max. coordinates for each axis, which make a box
            declaring the space inside which the pathfinder is allowed to search for paths.
        max_span: the longest edge of the bounding box, equivalent to largest axes needed for box.

    """

    margin = 0
    if graph_bounds and graph_bounds.x and graph_bounds.y:
        # Get the bounds of pre-existing blocks.
        bounds_z = [z for (_, _, z) in taken] if taken else [0, 0, 0]

        # Add small leeway depending on type of search
        min_x, max_x = (0, graph_bounds.x)
        min_y, max_y = (0, graph_bounds.y)
        min_z, max_z = (min(bounds_z), max(bounds_z))
        bounding_box = {
            "x": {"min": min_x, "max": max_x},
            "y": {"min": min_y, "max": max_y},
            "z": {"min": min_z - 6, "max": max_z + 6},
        }

    # Calculate bounds from taken
    else:
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
    nxt_coords: StandardCoord,
    curr_path_coords: list[StandardCoord],
    parametrised_taken: dict[int, dict[int, set[int]]],
    bounding_box: dict[str, dict[str, int]] | None = None,
) -> bool:
    """Check if current move should be skipped to speed up pathfinding process.

    Args:
        nxt_coords: The coordinates being checked as potential next position to place a block.
        bounding_box: The coordinates determining the bounding box outside of which moves are not allowed.
        curr_path_coords: The coordinates for the current path.
        parametrised_taken: A version of taken parametrised for more efficient clash detection.
        cross_edge: True if the current edge is a cross-edge (as opposed to a standard edge).
        special_target_kind (optional): True if final target is a Y, conditional, or cultivation block.

    """

    if (
        check_clashes_parametrised_taken(nxt_coords, parametrised_taken)
        or nxt_coords in curr_path_coords
    ):
        return True

    if bounding_box:
        nxt_x, nxt_y, nxt_z = nxt_coords
        if (
            nxt_x < bounding_box["x"]["min"]
            or nxt_x > bounding_box["x"]["max"]
            or nxt_y < bounding_box["y"]["min"]
            or nxt_y > bounding_box["y"]["max"]
            or nxt_z < bounding_box["z"]["min"]
            or nxt_z > bounding_box["z"]["max"]
        ):
            return True

    return False

def check_clashes_parametrised_taken(
    nxt_coords: StandardCoord,
    parametrised_taken: dict[int, set[StandardCoord]],
) -> bool:
    """Check for clashes between an arbitrary set of coordinates and the pruned version of taken.

    Args:
        nxt_coords: The coordinates being checked as potential next position to place a block.
        parametrised_taken: A version of taken that has been organised into layers for more efficient clash detection.

    Returns:
        clash: False if no clashes are found, True otherwise.

    """

    x, y, z = nxt_coords
    if z in parametrised_taken:
        return (x, y) in parametrised_taken[z]
    return False
