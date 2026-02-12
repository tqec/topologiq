"""Key/common 3D/spatial operations used by the pathfinder.

Usage:
    Call any function/class from a separate script.

"""

import numpy as np

from topologiq.utils.classes import StandardCoord


#######################
# MANHATTAN DISTANCES #
#######################
def get_manhattan(src_coords: StandardCoord, tgt_coords: StandardCoord) -> int:
    """Calculate the Manhattan distance between any two (x, y, z) coordinates.

    Args:
        src_coords: The (x, y, z) coordinates for the source block.
        tgt_coords: The (x, y, z) coordinates for the target block.

    Returns:
        int: The Manhattan distance between the given coordinates.

    """

    return np.sum(np.abs(np.array(src_coords) - np.array(tgt_coords)))


def get_max_manhattan(src_coord: StandardCoord, all_coords: list[StandardCoord]) -> int:
    """Calculate the maximum Manhattan distance between a coordinate and a list of coordinates.

    Args:
        src_coord: The (x, y, z) coordinates for the source block.
        all_coords: A list of (x, y, z) coordinates of any arbitrary length, which may include src_coord.

    Returns:
        int: The max Manhattan distance between the source coordinate and all coordinates in the list of coordinates.

    """

    if all_coords:
        return max([get_manhattan(src_coord, c) for c in all_coords])

    return 0

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
