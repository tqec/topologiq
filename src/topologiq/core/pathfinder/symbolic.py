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

        This function checks if two cubes can spatially converge into one another
        by ensuring both cubes have a valid exit toward one another.

    Args:
        src_kind: The kind of the source block being checked.
        move: The (x, y, z) displacement between current and target position.
        tgt_kind: The kind of the target block being checked.

    Returns:
        (bool): True if cubes match else False.

    """

    # CHECK SOURCE TO TARGET
    if not check_is_exit(src_kind.lower(), move):
        return False

    # CHECK TARGET TO SOURCE
    if not check_is_exit(tgt_kind.lower(), tuple([-i for i in (move)])):
        return False

    return True


def check_exits(
    src_coords: StandardCoord,
    src_kind: str | None,
    taken: list[StandardCoord],
    coords_in_path: list[StandardCoord],
) -> tuple[int, CubeBeams, CubeBeams]:
    """Find the number of unobstructed exits for an arbitrary block.

    This function manages calls to other functions that determine if
    each face of a block is or is not an exit and whether the said
    exit is unobstructed.

    Args:
        src_coords: The (x, y, z) coordinates for the block.
        src_kind: The kind of the block.
        taken: A list of coordinates taken by any blocks placed as a result of previous operations.
        coords_in_path: The coordinates taken by the path under current evaluation.

    Returns:
        unobstr_exits_n: the number of unobstructed exist for the block.
        cube_beams: the beams emanating from the block.

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


def check_move(src_c: StandardCoord, tgt_c: StandardCoord) -> bool:
    """Determine if a given move is allowed/possible.

    This function uses a Manhattan distance to quickly check if a
    potential (source, target) combination is possible given the
    standard cube/pipe sizes and necessary relative placements.

    Args:
        src_c: (x, y, z) coordinates for the originating block.
        tgt_c: (x, y, z) coordinates for the potential placement of the target block.

    Returns:
        (bool): True if move is theoretically possible else False.

    """

    sx, sy, sz = src_c
    nx, ny, nz = tgt_c
    manhattan = abs(nx - sx) + abs(ny - sy) + abs(nz - sz)
    return manhattan % 3 == 0


#################
# SIMPLE CHECKS #
#################
@lru_cache
def check_is_exit(src_kind: str, move: StandardCoord) -> bool:
    """Check if a face is an exit.

    This function works by matching exit markers in a block's kind against a displacement
    array symbolising the direction of the target block/pipe.

    Args:
        src_kind: The kind of the source block being checked.
        move: The (x, y, z) displacement between current and target position.

    Returns:
        (bool): True if face is an exit else False.

    """

    src_kind = src_kind.lower()[:3]
    kind_3d = [src_kind[0], src_kind[1], src_kind[2]]

    if "o" in kind_3d:
        marker = "o"
    else:
        marker = [i for i in set(kind_3d) if kind_3d.count(i) >= 2][0]

    exit_idxs = [i for i, char in enumerate(kind_3d) if char == marker]

    diff_idx = -1
    for i, diff in enumerate(move):
        if diff != 0:
            diff_idx = i
            break

    if diff_idx != -1 and diff_idx in exit_idxs:
        return True
    else:
        return False


def check_unobstructed(
    src_c: StandardCoord,
    tgt_c: StandardCoord,
    taken: list[StandardCoord],
) -> tuple[bool, SingleBeam]:
    """Check if a face is unobstructed.

    This function should typically be called after verifying a face is exit.

    Args:
        src_c: The (x, y, z) coordinates for the current block/pipe.
        tgt_c: The coordinates for the target block/pipe.
        taken: The list of coordinates taken by any blocks/pipes placed as a result of previous operations.

    Returns:
        (bool): True if face is unobstructed else False.
        single_beam: If the face is unobstructed, its corresponding beam.

    """

    diffs = [target - source for source, target in zip(src_c, tgt_c)]
    diffs = [1 if d > 0 else -1 if d < 0 else 0 for d in diffs]

    x_start, x_end, x_direction = (
        src_c[0],
        src_c[0] if diffs[0] == 0 else diffs[0] * np.inf,
        diffs[0],
    )
    y_start, y_end, y_direction = (
        src_c[1],
        src_c[1] if diffs[1] == 0 else diffs[1] * np.inf,
        diffs[1],
    )
    z_start, z_end, z_direction = (
        src_c[2],
        src_c[2] if diffs[2] == 0 else diffs[2] * np.inf,
        diffs[2],
    )

    single_beam = SingleBeam(
        BeamAxisComponent(x_start, x_end, x_direction),
        BeamAxisComponent(y_start, y_end, y_direction),
        BeamAxisComponent(z_start, z_end, z_direction),
    )

    x_start, x_end, x_direction = (
        src_c[0],
        src_c[0] if diffs[0] == 0 else src_c[0] + diffs[0] * 9,
        diffs[0],
    )
    y_start, y_end, y_direction = (
        src_c[1],
        src_c[1] if diffs[1] == 0 else src_c[1] + diffs[1] * 9,
        diffs[1],
    )
    z_start, z_end, z_direction = (
        src_c[2],
        src_c[2] if diffs[2] == 0 else src_c[2] + diffs[2] * 9,
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
    """Check if block has an available exit pointing towards a target coordinate.

    This function checks if a block has an available exit that could be used to reach an
    arbitrary target coordinate. The function can also be used to check if two cubes match
    by calling it twice using current->target and target->current coordinates. That said,
    the function does not test if target coordinate is available or an exit is unobstructed.

    Args:
        src_kind: The kind of the source block being checked.
        move: The (x, y, z) displacement between current and target position.
        tgt_kind: The kind of the target block being checked.

    Returns:
        (boolean): True if an available exit points towards target coordinate else False.

    """

    # Sanitise kind in case of mixed case inputs and remove any Hadamard flag
    src_kind = src_kind.lower().replace("h", "")
    tgt_kind = tgt_kind.lower()

    # Extract axis of displacement from kinds
    idx = int(np.nonzero(move)[0])

    src_kind_new = src_kind[:idx] + src_kind[idx + 1 :]
    tgt_kind_new = tgt_kind[:idx] + tgt_kind[idx + 1 :]

    if not src_kind_new == tgt_kind_new:
        return False

    return True


@lru_cache
def nxt_kinds(src_kind: str, move: StandardCoord) -> list[str]:
    """Reduce the number of possible kinds for next block.

    This function reduces the number of kinds to check as potential next
    through a few quick pre-match expectations that consider the direction
    of the current move. It returns only kinds that would constitute a
    topologically-correct placement.

    Args:
        src_kind: The current node's kind.
        move: The (x, y, z) displacement between current and target position.

    Returns:
        reduced_valid_kinds: a subset of kinds applicable to next move.

    """

    # HELPER VARIABLES
    c_ks = ["xxz", "zzx", "xzz", "zxx", "zxz", "xzx"]
    p_ks = ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"]

    # Sanitise kind in case of mixed case inputs and remove any hadamard flag
    src_kind = src_kind.lower().replace("h", "")

    # If current kind has an "o", the next kind is a cube
    if "o" in src_kind:
        ok = [tgt_k for tgt_k in c_ks if cube_match(src_kind, move, tgt_k)]
    else:  # Else, the next kind is a pipe
        ok = [tgt_k for tgt_k in p_ks if cube_match(src_kind, move, tgt_k)]

    # Discard kinds where there is no colour match, and return
    ok_min = [tgt_k for tgt_k in ok if face_match(src_kind, move, tgt_k)]
    return ok_min


###################
# TRANSFORMATIONS #
###################
def validate_nxt_kind(
    current_block: StandardBlock, nxt_coords: StandardCoord, nxt_kind: str, hdm: bool
) -> str:
    """Return next kind after assessing if it needs to be rotated or not."""

    curr_coords, _ = current_block

    if hdm and "o" in nxt_kind:
        nxt_kind += "h"
        direction = sum(
            [p[1] - p[0] if p[0] != p[1] else 0 for p in list(zip(curr_coords, nxt_coords))]
        )

        if direction < 0:
            nxt_kind = rotate_pipe(nxt_kind)

    return nxt_kind


def handle_kind_after_hadamard(
    current_block: StandardBlock, nxt_coords: StandardCoord, hdm: bool
) -> str:
    """Rotate hadamard if current kind is a hadamard."""

    curr_coords, curr_kind = current_block

    alt_curr_kind = None
    if "h" in curr_kind:
        hdm = False
        direction = sum(
            [p[1] - p[0] if p[0] != p[1] else 0 for p in list(zip(curr_coords, nxt_coords))]
        )
        if direction < 0:
            pass
        else:
            alt_curr_kind = rotate_pipe(curr_kind)

    return alt_curr_kind, hdm


def rotate_pipe(k: str) -> str:
    """Rotate a pipe around its length.

    This function enables pipe rotation by using the exit marker in their kind
    to create a rotational matrix, which is then used to rotate the original kind
    using symbolic multiplication.

    Args:
        k: the kind of the pipe that needs rotation.

    Returns:
        rot_k: a kind with the rotation incorporated into the new name.

    """

    h_flag = False
    if "h" in k:
        h_flag = True
        k.replace("h", "")

    # Build rotation matrix based on placement of "o" node
    idxs = [0, 1, 2]
    idxs.remove(k.index("o"))

    new_matrix = {
        k.index("o"): np.eye(3, dtype=int)[k.index("o")],
        idxs[0]: np.eye(3, dtype=int)[idxs[1]],
        idxs[1]: np.eye(3, dtype=int)[idxs[0]],
    }

    rot_matrix = np.array([new_matrix[0], new_matrix[1], new_matrix[2]])

    # Rotate kind
    rot_k = ""
    for r in rot_matrix:
        entry = ""
        for j, ele in enumerate(r):
            entry += abs(int(ele)) * k[j]
        rot_k += entry

    if h_flag:
        rot_k += "h"

    return rot_k


def flip_hadamard(k: str) -> str:
    """Flip a Hadamard for the opposite Hadamard with length on the same axis.

    Args:
        k: the kind of the Hadamard that needs inverting.

    Returns:
        k_2: the kind of the corresponding/inverted Hadamard.

    """

    # list of hadamard equivalences
    equivs = {"zxoh": "xzoh", "xozh": "zoxh", "oxzh": "ozxh"}

    # Match to equivalent block given direction
    if k in equivs.keys():
        new_k = equivs[k]
    else:
        inv_equivs = {v: k for k, v in equivs.items()}
        new_k = inv_equivs[k]

    # Return revised kind
    return new_k
