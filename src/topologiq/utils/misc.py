"""Misc utils of various sorts.

Usage:
    Call any function/class from a separate script.

"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyzx as zx

if TYPE_CHECKING:
    from topologiq.utils.classes import StandardCoord


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


def kind_to_zx_type(kind: str) -> str:
    """Get the ZX type corresponding to a given block or pipe kind.

    Args:
        kind: the /kind of a given block.

    Returns:
        zx_type: the ZX type corresponding to the kind.

    """

    if kind == "OOO":
        zx_type = "O"
    elif kind[0] in ["Y", "T"]:
        zx_type = kind[0]
    elif "*" in kind:
        zx_type = "ZX"
    elif "O" in kind:
        zx_type = "HADAMARD" if "h" in kind else "SIMPLE"
    else:
        zx_type = min(set(kind), key=lambda c: kind.count(c)).capitalize()
    return zx_type


def write_zx_to_json_file(zx_graph: zx.Graph, path_to_output_file: Path):
    """Write a PyZX graph to a JSON file."""

    json_data = zx_graph.to_json()
    with open("zx_cnots.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)
