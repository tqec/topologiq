import numpy as np

from typing import List
from utils.classes import StandardCoord


def get_manhattan(src_coords: StandardCoord, tgt_coords: StandardCoord) -> int:
    """Gets the Manhattan distance between any two (x, y, z) coordinates.

    Args:
        - src_coords: (x, y, z) coordinates for the source block.
        - tgt_coords: (x, y, z) coordinates for the target block.

    Returns:
        - [int]: the Manhattan distance between the two incoming coordinate tuples

    """

    return np.sum(np.abs(np.array(src_coords) - np.array(tgt_coords)))


def get_max_manhattan(src_coords: StandardCoord, all_coords: List[StandardCoord]) -> int:
    """Determines the maximum Manhattan distance and a list of (x, y, z) coordinates of any arbitrary length.
    
    Args:
        - src_coords: (x, y, z) coordinates for the source block.
        - all_coords: a list of (x, y, z) coordinates of any arbitrary length, which may include src_coords.
    
    Returns:
        - [int]: the max. Manhattan distance between the given source coordinate and all coordinate tuples in the list of coordinates.
    """
    
    if all_coords:
        distances = [get_manhattan(src_coords, coord) for coord in all_coords]
        return max(distances)
    
    return 0