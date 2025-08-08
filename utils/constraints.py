from typing import List
from utils.classes import StandardCoord
from utils.utils_greedy_bfs import check_is_exit


def check_face_match(
    src_coord: StandardCoord,
    src_kind: str,
    tgt_coord: StandardCoord,
    tgt_kind: str,
) -> bool:
    """Checks if a block or pipe has an available exit pointing towards a target coordinate
    by matching exit marker in the block's or pipe's symbolic name to the direction of target coordinate.

    ! Note. Function does not test if target coordinate is available.
    ! Note. Function does not test if exit is unobstructed.
    ! Note. To check if two cubes match, run this function twice: current to target, target to current.

    Args:
        - src_coord: (x, y, z) coordinates for source node.
        - src_kind: kind for the source node.
        - tgt_coord: (x, y, z) coordinates for target node.

    Returns:
        - boolean:
            - True: available exit towards target coordinate,
            - False: NO available exit towards target coordinate.

    """

    # Sanitise kind in case of mixed case inputs
    src_kind = src_kind.lower()
    if "h" in src_kind:
        src_kind = src_kind.replace("h", "")
    tgt_kind = tgt_kind.lower()

    # Extract axis of displacement from kinds
    diffs = [p[1] - p[0] for p in list(zip(src_coord, tgt_coord))]
    axes_diffs = [True if axis != 0 else False for axis in diffs]

    idx = axes_diffs.index(True)

    new_src_kind = src_kind[:idx] + src_kind[idx + 1 :]
    new_tgt_kind = tgt_kind[:idx] + tgt_kind[idx + 1 :]

    # Fail if two other dimensions do not match
    if not new_src_kind == new_tgt_kind:
        return False

    # Pass otherwise
    return True


def check_cube_match(
    src_coords: StandardCoord,
    src_kind: str,
    tgt_pos: StandardCoord,
    tgt_kind: str,
) -> bool:
    """Checks if two cubes match by comparing the symbols of their colours.

    ! Note. Function does not handle HADAMARDS.
    ! Note. To handle hadamards in "tgt_pos", strip the "h" from name, run as a regular pipe, add "h" back after match is found.
    ! Note. To handle hadamards in "src_coords", rotate it, then run as regular pipe.

    Args:
        - src_coords: (x, y, z) coordinates for the current node.
        - src_kind: current node's kind.
        - tgt_pos: (x, y, z) coordinates for the next node.
        - tgt_kind: target node's kind.

    Returns:
        - bool:
            - True: cubes match
            - False: no match.

    """

    # SANITISE
    src_kind = src_kind.lower()
    tgt_kind = tgt_kind.lower()

    # CHECK SOURCE TO TARGET
    # Connection takes place on a valid exit of source
    if not check_is_exit(src_coords, src_kind, tgt_pos):
        return False

    # CHECK TARGET TO SOURCE
    # Connection takes place on a valid exit of target
    if not check_is_exit(tgt_pos, tgt_kind, src_coords):
        return False

    return True


def get_valid_nxt_kinds(
    src_coords: StandardCoord, src_kind: str, tgt_pos: StandardCoord
) -> List[str]:
    """Reduces the number of possible types for next block/pipe by quickly running the current kind by a pre-match operations.

    Args:
        - src_coords: (x, y, z) coordinates for the current node.
        - src_kind: current node's kind.
        - tgt_pos: (x, y, z) coordinates for the next node.

    Returns:
        - reduced_valid_kinds: a subset of kinds applicable to next move.

    """

    # HELPER VARIABLES
    valid_kinds = []
    cube_kinds = ["xxz", "xzz", "xzx", "zzx", "zxx", "zxz"]
    pipe_kinds = ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"]

    # CHECK FOR ALL POSSIBLE NEXT KINDS IN DISPLACEMENT AXIS
    # If current kind has an "o", the next kind is a cube
    if "o" in src_kind:
        for tgt_kind in cube_kinds:
            cube_match = check_cube_match(src_coords, src_kind, tgt_pos, tgt_kind)
            if cube_match:
                valid_kinds.append(tgt_kind)

    # If current kind does not have an "o", then current kind is cube and the next kind is a pipe
    else:
        for tgt_kind in pipe_kinds:
            cube_match = check_cube_match(src_coords, src_kind, tgt_pos, tgt_kind)
            if cube_match:
                valid_kinds.append(tgt_kind)

    # Now discard possible kinds where there is no colour match for all non-connection faces
    valid_kinds_min = []
    for tgt_kind in valid_kinds:
        if check_face_match(src_coords, src_kind, tgt_pos, tgt_kind):
            valid_kinds_min.append(tgt_kind)

    # RETURN ARRAY OF POSSIBLE NEXT KINDS
    return valid_kinds_min
