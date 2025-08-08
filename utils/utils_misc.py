import csv
import numpy as np

from pathlib import Path
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


def get_max_manhattan(
    src_coords: StandardCoord, all_coords: List[StandardCoord]
) -> int:
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


def log_stats_to_file(stats_line: List, stats_type: str):
    """Writes statistics to one of the statistics files in `./assets/stats/`

    Args:
        - stats_line: the line of statistics to be logged to file.
        - stats_type: the type of statistics being logged to file, which also matches the name of recipient file.

    Returns:
        - n/a: stats are written to .csv files in `./assets/stats/`

    """

    repo_root: Path = Path(__file__).resolve().parent.parent
    stats_dir_path = repo_root / "assets/stats"

    with open(f"{stats_dir_path}/{stats_type}.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(stats_line)
        f.close()
