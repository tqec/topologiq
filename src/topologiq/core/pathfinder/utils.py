"""Utilities to assist the management of the pathfinder.

Usage:
    Call any function/class from a separate script.

"""

from collections import deque

import numpy as np

from topologiq.utils.classes import StandardBlock, StandardCoord

#################
# HEALTH CHECKS #
#################


########
# INIT #
########
def init_bfs(
    src_block_info: StandardBlock,
) -> tuple[
    deque,
    dict[tuple[StandardBlock, StandardCoord], int],
    int,
    dict[StandardBlock, int],
    dict[StandardBlock, list[StandardBlock]],
    dict[StandardBlock, list[StandardBlock]],
    dict[StandardBlock, list[StandardBlock]],
    list[tuple[int, int, int]],
]:
    """Initialise BFS variables."""

    queue = deque([src_block_info])
    visited = {(src_block_info, (0, 0, 0)): 0}
    visit_attempts = 0
    path_len = {src_block_info: 0}
    path = {src_block_info: [src_block_info]}
    valid_paths = {}
    all_search_paths = {}
    moves = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
    ]

    return queue, visited, visit_attempts, path_len, path, valid_paths, all_search_paths, moves


def gen_exit_conditions(
    src_coords, tent_coords, taken, max_span, second_pass, **kwargs
) -> tuple[int, int, int, int]:
    """Calculate conditions that need to be met to exit the pathfinder BFS."""

    # Min numbers of targets that need to be filled for a run to be deemed successful
    tgts_filled = 0
    tgts_to_fill = (
        int(len(tent_coords) * kwargs["min_succ_rate"] / 100) if len(tent_coords) > 1 else 1
    )

    # Manhattan distances to skip iterations and exit BFS in the event of failure
    if not second_pass:
        max_manhattan = get_max_manhattan(src_coords, tent_coords) * 2
        src_tgt_manhattan = max_manhattan
    else:
        src_tgt_manhattan = get_max_manhattan(src_coords, tent_coords)
        max_manhattan = max(
            get_max_manhattan(src_coords, taken) * 2,
            max_span,
        )
    src_tgt_manhattan = get_max_manhattan(src_coords, tent_coords)

    return tgts_filled, tgts_to_fill, max_manhattan, src_tgt_manhattan


##########
# MANAGE #
##########
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


############
# MISC/AUX #
############
def check_run_mode(src_coords, taken, tgt_coords, tent_tgt_kinds):
    """Check if edge is standard or cross and update taken accordingly."""

    second_pass = False

    if src_coords in taken:
        taken.remove(src_coords)
    if len(tgt_coords) == 1 and len(tent_tgt_kinds) == 1:
        second_pass = True
        if tgt_coords[0] in taken:
            taken.remove(tgt_coords[0])

    return second_pass, taken


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

