"""Symbolic operations that provide heuristics used for pathfinding.

Usage:
    Call any function/class from a separate script.

"""

from functools import lru_cache

import numpy as np

from topologiq.core.beams import BeamAxisComponent, CubeBeams, SingleBeam
from topologiq.utils.classes import StandardBlock, StandardCoord


####################
# COMPOSITE CHECKS #
####################
def cube_match(src_kind: str, move: StandardCoord, tgt_kind: str) -> bool:
    """Check if two cubes have valid exits facing one another.

    Args:
        src_kind: The kind of the source block being checked.
        move: The (x, y, z) displacement between current and target position.
        tgt_kind: The kind of the target block being checked.

    Returns:
        (bool): True if cubes match else False.

    """
    # Check source -> target
    if not check_is_exit(src_kind.lower(), move):
        return False
    # Check target -> source
    if not check_is_exit(tgt_kind.lower(), tuple([-i for i in (move)])):
        return False

    return True


def check_exits(
    src_coords: StandardCoord,
    src_kind: str | None,
    taken: list[StandardCoord],
    coords_in_path: list[StandardCoord],
) -> tuple[int, CubeBeams, CubeBeams]:
    """Find the number of unobstructed exits for an arbitrary block and attach beams to them.

    Args:
        src_coords: The (x, y, z) coordinates for the block.
        src_kind: The kind of the block.
        taken: A list of coordinates taken by any blocks placed as a result of previous operations.
        coords_in_path: The coordinates taken by the path under current evaluation.

    Returns:
        unobstr_exits_n: The number of unobstructed exist for the block.
        cube_beams: The beams emanating from the block.
        cube_beams_short: The short beams emanating from the block.

    """

    unobstr_exits_n = 0
    cube_beams = []
    cube_beams_short = []

    diffs = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    for d in diffs:
        tgt_coords = (
            src_coords[0] + d[0],
            src_coords[1] + d[1],
            src_coords[2] + d[2],
        )

        if check_is_exit(src_kind, d):
            is_unobstr, single_beam, single_beam_short = check_unobstructed(
                src_coords, tgt_coords, taken
            )
            if is_unobstr and not any([single_beam.contains(coord) for coord in coords_in_path]):
                unobstr_exits_n += 1
                cube_beams.append(single_beam)
                cube_beams_short.append(single_beam_short)

    # Reset number of unobstructed exits
    unobstr_exits_n = len(cube_beams)
    return unobstr_exits_n, cube_beams, cube_beams_short


#################
# SIMPLE CHECKS #
#################
@lru_cache
def check_is_exit(src_kind: str, move: StandardCoord) -> bool:
    """Check if a face is an exit.

    Args:
        src_kind: The kind of the source block being checked.
        move: The (x, y, z) displacement between current and target position.

    Returns:
        (bool): True if face is an exit else False.

    """
    # Identify exit indexes
    kind_3d = list(src_kind)[:3]
    marker = "o" if "o" in kind_3d else [i for i in set(kind_3d) if kind_3d.count(i) >= 2][0]
    exit_idxs = [i for i, char in enumerate(kind_3d) if char == marker]

    # Identify exit axis
    diff_idx = int(np.nonzero(move)[0])

    # Return True for exits
    return diff_idx in exit_idxs


def check_unobstructed(
    src_coords: StandardCoord,
    tgt_coords: StandardCoord,
    taken: list[StandardCoord],
) -> tuple[bool, SingleBeam, SingleBeam]:
    """Check if a face is unobstructed and attach beams to it if this is the case.

    Args:
        src_coords: The (x, y, z) coordinates for the current block/pipe.
        tgt_coords: The coordinates for the target block/pipe.
        taken: The list of coordinates taken by any blocks/pipes placed as a result of previous operations.

    Returns:
        (bool): True if face is unobstructed else False.
        single_beam: If face unobstructed, the infinite beams for it.
        single_beam_short: If face unobstructed, the short beams for it.

    """

    diffs = [target - source for source, target in zip(src_coords, tgt_coords)]
    diffs = [1 if d > 0 else -1 if d < 0 else 0 for d in diffs]

    x_start, x_end, x_direction = (
        src_coords[0],
        src_coords[0] if diffs[0] == 0 else diffs[0] * np.inf,
        diffs[0],
    )
    y_start, y_end, y_direction = (
        src_coords[1],
        src_coords[1] if diffs[1] == 0 else diffs[1] * np.inf,
        diffs[1],
    )
    z_start, z_end, z_direction = (
        src_coords[2],
        src_coords[2] if diffs[2] == 0 else diffs[2] * np.inf,
        diffs[2],
    )

    single_beam = SingleBeam(
        BeamAxisComponent(x_start, x_end, x_direction),
        BeamAxisComponent(y_start, y_end, y_direction),
        BeamAxisComponent(z_start, z_end, z_direction),
    )

    x_start, x_end, x_direction = (
        src_coords[0],
        src_coords[0] if diffs[0] == 0 else src_coords[0] + diffs[0] * 9,
        diffs[0],
    )
    y_start, y_end, y_direction = (
        src_coords[1],
        src_coords[1] if diffs[1] == 0 else src_coords[1] + diffs[1] * 9,
        diffs[1],
    )
    z_start, z_end, z_direction = (
        src_coords[2],
        src_coords[2] if diffs[2] == 0 else src_coords[2] + diffs[2] * 9,
        diffs[2],
    )

    single_beam_short = SingleBeam(
        BeamAxisComponent(x_start, x_end, x_direction),
        BeamAxisComponent(y_start, y_end, y_direction),
        BeamAxisComponent(z_start, z_end, z_direction),
    )

    if not taken:
        return True, single_beam, single_beam_short

    if any([single_beam.contains(coord) for coord in taken]):
        return False, single_beam, single_beam_short

    return True, single_beam, single_beam_short


def face_match(src_kind: str, move: StandardCoord, tgt_kind: str) -> bool:
    """Check if the faces of two adjacent blocks match.

    NB! Call twice to robustly determine if two cubes match, (1) src_kind -> tgt_kind
    and then (2) tgt_kind -> src_kind with inverted `move` vector.

    Args:
        src_kind: The kind of the source block being checked.
        move: The (x, y, z) displacement between current and target position.
        tgt_kind: The kind of the target block being checked.

    Returns:
        (boolean): True if an available exit points towards target coordinate else False.

    """
    # Extract axis of displacement from kinds
    idx = int(np.nonzero(move)[0])
    src_kind_new = src_kind[:idx] + src_kind[idx + 1 :]
    tgt_kind_new = tgt_kind[:idx] + tgt_kind[idx + 1 :]

    # Return match
    return src_kind_new[:3] == tgt_kind_new


@lru_cache
def nxt_kinds(src_kind: str, move: StandardCoord) -> list[str]:
    """Reduce the number of possible kinds for next block.

    Args:
        src_kind: The kind of the current (source) cube or pipe.
        move: The (x, y, z) displacement between current and target position.

    Returns:
        ok_kinds: A list kinds that would constitute a topologically-correct placement.

    """
    cube_kinds = ["xxz", "zzx", "xzz", "zxx", "zxz", "xzx"]
    pipe_kinds = ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"]
    ok_kinds = [
        tgt_kind
        for tgt_kind in (cube_kinds if "o" in src_kind else pipe_kinds)
        if cube_match(src_kind[:3], move, tgt_kind)
    ]
    return [tgt_kind for tgt_kind in ok_kinds if face_match(src_kind[:3], move, tgt_kind)]


###################
# TRANSFORMATIONS #
###################
@lru_cache
def validate_nxt_kind(move: StandardCoord, nxt_kind: str, hdm: bool) -> str:
    """Return next kind after assessing if it needs to be rotated or not."""
    if hdm and "o" in nxt_kind:
        nxt_kind += "h"
        direction = sum(move)
        if direction < 0:
            return rotate_pipe(nxt_kind)
    return nxt_kind


def handle_kind_after_hadamard(curr_kind: StandardBlock, move: StandardCoord, hdm: bool) -> tuple[str | None, bool]:
    """Rotate hadamard if current kind is a hadamard."""
    if "h" in curr_kind:
        hdm = False
        direction = sum(move)
        if direction > 0:
            return rotate_pipe(curr_kind), hdm
    return None, hdm


def rotate_pipe(kind: str) -> str:
    """Rotate a pipe around its length.

    NB! Function might be a tad hard to understand. 1st step is to create a
    rotation matrix using the exit marker in an arbitrary kind. 2nd step is
    to rotate the incoming kind using symbolic multiplication.

    Args:
        kind: the kind of the pipe that needs rotation.

    Returns:
        rotated_kind: a kind with the rotation incorporated into the new name.

    """

    h_flag = False
    if "h" in kind:
        h_flag = True
        kind.replace("h", "")

    # Build rotation matrix based on placement of "o" node
    idxs = [0, 1, 2]
    idxs.remove(kind.index("o"))

    new_matrix = {
        kind.index("o"): np.eye(3, dtype=int)[kind.index("o")],
        idxs[0]: np.eye(3, dtype=int)[idxs[1]],
        idxs[1]: np.eye(3, dtype=int)[idxs[0]],
    }

    rot_matrix = np.array([new_matrix[0], new_matrix[1], new_matrix[2]])

    # Rotate kind
    rotated_kind = ""
    for r in rot_matrix:
        entry = ""
        for j, ele in enumerate(r):
            entry += abs(int(ele)) * kind[j]
        rotated_kind += entry

    if h_flag:
        rotated_kind += "h"

    return rotated_kind
