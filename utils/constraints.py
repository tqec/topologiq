from typing import List
from utils.classes import StandardCoord
from utils.utils_greedy_bfs import is_exit


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
    if not is_exit(src_c, src_k.lower(), tgt_pos):
        return False

    # CHECK TARGET TO SOURCE
    if not is_exit(tgt_pos, tgt_k.lower(), src_c):
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
