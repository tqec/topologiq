import os
import numpy as np

from datetime import datetime
from typing import Tuple, List, Optional


from utils.utils_greedy_bfs import gen_tent_tgt_coords
from utils.classes import StandardCoord, NodeBeams, StandardBeam

#############################
# PATHFINDER AUX OPERATIONS #
#############################
def check_is_exit(src_c: StandardCoord, src_k: Optional[str], tgt_c: StandardCoord) -> bool:
    """Checks if a face is an exit by matching exit markers in the block/pipe's symbolic name
    and an displacement array symbolising the direction of the target block/pipe.

    Args:
        - src_c: (x, y, z) coordinates for the current block/pipe.
        - src_k: current block's/pipe's kind.
        - tgt_c: coordinates for the target block/pipe.

    Returns:
        - bool:
            - True: face is an exist
            - False: face is not an exit.

    """

    src_k = src_k.lower()[:3] if isinstance(src_k, str) else ""
    kind_3D = [src_k[0], src_k[1], src_k[2]]

    if "o" in kind_3D:
        marker = "o"
    else:
        marker = [i for i in set(kind_3D) if kind_3D.count(i) == 2][0]

    exit_idxs = [i for i, char in enumerate(kind_3D) if char == marker]
    diffs = [tgt - src for src, tgt in zip(src_c, tgt_c)]

    diff_idx = -1
    for i, diff in enumerate(diffs):
        if diff != 0:
            diff_idx = i
            break

    if diff_idx != -1 and diff_idx in exit_idxs:
        return True
    else:
        return False


def check_unobstr(
    src_c: StandardCoord,
    tgt_c: StandardCoord,
    taken: List[StandardCoord],
    all_beams: List[NodeBeams],
    beams_len: int,
) -> Tuple[bool, StandardBeam]:
    """Checks if a face (typically an exit: call after verifying face is exit) is unobstructed.

    Args:
        - src_c: (x, y, z) coordinates for the current block/pipe.
        - tgt_c: coordinates for the target block/pipe.
        - taken: list of coordinates taken by any blocks/pipes placed as a result of previous operations.
        - all_beams: list of coordinates taken by the beams of all blocks in original ZX-graph
        - beams_len: int representing how long does each beam spans from its originating block.

    Returns:
        - bool:
            - True: face is unobstructed
            - False: face is obstructed.

    """

    beam: StandardBeam = []

    diffs = [target - source for source, target in zip(src_c, tgt_c)]
    diffs = [1 if d > 0 else -1 if d < 0 else 0 for d in diffs]

    for i in range(1, beams_len):
        dx, dy, dz = (diffs[0] * i, diffs[1] * i, diffs[2] * i)
        beam.append((src_c[0] + dx, src_c[1] + dy, src_c[2] + dz))

    if not taken:
        return True, beam

    for c in beam:
        if c in taken or c in all_beams:
            return False, beam

    return True, beam


def check_exits(
    src_c: StandardCoord,
    src_k: str | None,
    taken: List[StandardCoord],
    all_beams: List[NodeBeams],
    beams_len: int,
) -> Tuple[int, NodeBeams]:
    """Finds the number of unobstructed exits for any given block/pipe by calling other functions that use the
    block's/pipe's symbolic name to determine if each face is or is not
    and whether the said exit is unobstructed.

    Args:
        - src_c: (x, y, z) coordinates for the block/pipe of interest.
        - src_k: kind/type of the block/pipe of interest.
        - taken: list of coordinates taken by any blocks/pipes placed as a result of previous operations.
        - all_beams: list of coordinates taken by the beams of all blocks in original ZX-graph
        - beams_len: int representing how long does each beam spans from its originating block.

    Returns:
        - unobstrexits_n: the number of unobstructed exist for the block/pipe of interest.
        - n_beams: the beams emanating from the block/pipe of interest.

    """

    unobstr_exits_n = 0
    n_beams: NodeBeams = []

    diffs = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    for d in diffs:
        tgt_c = (
            src_c[0] + d[0],
            src_c[1] + d[1],
            src_c[2] + d[2],
        )

        if check_is_exit(src_c, src_k, tgt_c):
            is_unobstr, exit_beam = check_unobstr(
                src_c, tgt_c, taken, all_beams, beams_len
            )
            if is_unobstr:
                unobstr_exits_n += 1
                n_beams.append(exit_beam)

    return unobstr_exits_n, n_beams


def check_move(src_c: StandardCoord, tgt_c: StandardCoord) -> bool:
    """Uses a Manhattan distance to quickly checks if a potential (source, target) combination
    is possible given the standard size of a block and pipe and that all blocks are followed by a pipe.

    Args:
        - src_c: (x, y, z) coordinates for the originating block.
        - tgt_c: (x, y, z) coordinates for the potential placement of the target block.

    Returns:
        - bool:
            - True: move is theoretically possible,
            - False: move is not theoretically possible.

    """

    sx, sy, sz = src_c
    nx, ny, nz = tgt_c
    manhattan = abs(nx - sx) + abs(ny - sy) + abs(nz - sz)
    return manhattan % 3 == 0


def face_match(
    src_c: StandardCoord, src_k: str, tgt_c: StandardCoord, tgt_k: str
) -> bool:
    """Checks if block or pipe has an available exit pointing towards a target coordinate
    by matching exit marker in the block's or pipe's symbolic name to the direction of target coordinate.

    ! Function does not test if target coordinate is available.
    ! Function does not test if exit is unobstructed.
    ! To check if two cubes match, run this function twice: current to target, target to current.

    Args:
        - src_c: (x, y, z) coords for source node.
        - src_k: kind for the source node.
        - tgt_c: (x, y, z) coords for target node.

    Returns:
        - (boolean):
            - True: available exit towards target coordinate,
            - False: NO available exit towards target coordinate.
    """

    # Sanitise kind in case of mixed case inputs
    src_k = src_k.lower()
    if "h" in src_k:
        src_k = src_k.replace("h", "")
    tgt_k = tgt_k.lower()

    # Extract axis of displacement from kinds
    diffs = [p[1] - p[0] for p in list(zip(src_c, tgt_c))]
    ax_diffs = [True if ax != 0 else False for ax in diffs]

    idx = ax_diffs.index(True)

    src_k_new = src_k[:idx] + src_k[idx + 1 :]
    tgt_k_new = tgt_k[:idx] + tgt_k[idx + 1 :]

    if not src_k_new == tgt_k_new:
        return False

    return True


def cube_match(
    src_c: StandardCoord, src_k: str, tgt_pos: StandardCoord, tgt_k: str
) -> bool:
    """Checks if two cubes match by comparing the symbols of their colours.

    ! Note. Function does not handle HADAMARDS.
    ! Note. To handle hadamards in "tgt_pos", strip the "h" from name, run as a regular pipe, add "h" back after match is found.
    ! Note. To handle hadamards in "src_c", rotate it, then run as regular pipe.

    Args:
        - src_c: (x, y, z) coordinates for the current node.
        - src_k: current node's kind.
        - tgt_pos: (x, y, z) coordinates for the next node.
        - tgt_k: target node's kind.

    Returns:
        - bool:
            - True: cubes match
            - False: no match.

    """

    # CHECK SOURCE TO TARGET
    if not check_is_exit(src_c, src_k.lower(), tgt_pos):
        return False

    # CHECK TARGET TO SOURCE
    if not check_is_exit(tgt_pos, tgt_k.lower(), src_c):
        return False

    return True


def nxt_kinds(src_c: StandardCoord, src_k: str, tgt_pos: StandardCoord) -> List[str]:
    """Reduces the number of possible types for next block/pipe by quickly running the current kind by a pre-match operations.

    Args:
        - src_c: (x, y, z) coordinates for the current node.
        - src_k: current node's kind.
        - tgt_pos: (x, y, z) coordinates for the next node.

    Returns:
        - reduced_valid_kinds: a subset of kinds applicable to next move.

    """

    # HELPER VARIABLES
    c_ks = ["xxz", "xzz", "xzx", "zzx", "zxx", "zxz"]
    p_ks = ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"]

    # CHECK FOR ALL POSSIBLE NEXT KINDS IN DISPLACEMENT AXIS
    # If current kind has an "o", the next kind is a cube
    if "o" in src_k:                
        ok = [tgt_k for tgt_k in c_ks if cube_match(src_c, src_k, tgt_pos, tgt_k)]            
    # If current kind does not have an "o", then current kind is cube and the next kind is a pipe
    else:
        ok = [tgt_k for tgt_k in p_ks if cube_match(src_c, src_k, tgt_pos, tgt_k)]

    # Discard kinds where there is no colour match, and return
    ok_min = [tgt_k for tgt_k in ok if face_match(src_c, src_k, tgt_pos, tgt_k)]
    return ok_min


def rot_o_kind(k: str) -> str:
    """Rotates a pipe around its length by using the exit marker in its symbolic name to create a rotational matrix,
    which is then used to rotate the original name with a symbolic multiplication.

    Args:
        - k: the kind of the pipe that needs rotation.

    Returns:
        - rot_k: a kind with the rotation incorporated into the new name.

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


def flip_hdm(k: str) -> str:
    """Quickly flips a Hadamard for the opposite Hadamard with length on the same axis.

    Args:
        - k: the kind of the Hadamard that needs inverting.

    Returns:
        - k_2: the kind of the corresponding/inverted Hadamard.


    """
    # List of hadamard equivalences
    equivs = {"zxoh": "xzoh", "xozh": "zoxh", "oxzh": "ozxh"}

    # Match to equivalent block given direction
    if k in equivs.keys():
        new_k = equivs[k]
    else:
        inv_equivs = {v: k for k, v in equivs.items()}
        new_k = inv_equivs[k]

    # Return revised kind
    return new_k
