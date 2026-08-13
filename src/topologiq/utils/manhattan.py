"""Utilities to assist calculation of Manhattan distances.

Usage:
    Call any function/class from a separate script.

"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from topologiq.utils.classes import StandardCoord


#######################
# MANHATTAN DISTANCES #
#######################
@lru_cache
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


def get_min_manhattan(src_coord: StandardCoord, all_coords: list[StandardCoord]) -> int:
    """Calculate the maximum Manhattan distance between a coordinate and a list of coordinates.

    Args:
        src_coord: The (x, y, z) coordinates for the source block.
        all_coords: A list of (x, y, z) coordinates of any arbitrary length, which may include src_coord.

    Returns:
        int: The max Manhattan distance between the source coordinate and all coordinates in the list of coordinates.

    """

    if all_coords:
        return min([get_manhattan(src_coord, c) for c in all_coords])

    return 0
