from typing import List

from utils.classes import StandardCoord
from utils.utils_greedy_bfs import check_is_exit


def check_face_match(
    source_coord: StandardCoord,
    source_kind: str,
    target_coord: StandardCoord,
    target_kind: str,
) -> bool:
    """Checks if a block or pipe has an available exit pointing towards a target coordinate
    by matching exit marker in the block's or pipe's symbolic name to the direction of target coordinate.

    ! Note. Function does not test if target coordinate is available.
    ! Note. Function does not test if exit is unobstructed.
    ! Note. To check if two cubes match, run this function twice: current to target, target to current.

    Args:
        - source_coord: (x, y, z) coordinates for source node.
        - source_kind: kind for the source node.
        - target_coord: (x, y, z) coordinates for target node.

    Returns:
        - boolean:
            - True: available exit towards target coordinate,
            - False: NO available exit towards target coordinate.

    """

    # Sanitise kind in case of mixed case inputs
    source_kind = source_kind.lower()
    if "h" in source_kind:
        source_kind = source_kind.replace("h", "")
    target_kind = target_kind.lower()

    # Extract axis of displacement from kinds
    displacements = [p[1] - p[0] for p in list(zip(source_coord, target_coord))]
    axis_displacement = [True if axis != 0 else False for axis in displacements]

    idx = axis_displacement.index(True)

    new_source_kind = source_kind[:idx] + source_kind[idx + 1 :]
    new_target_kind = target_kind[:idx] + target_kind[idx + 1 :]

    # Fail if two other dimensions do not match
    if not new_source_kind == new_target_kind:
        return False

    # Pass otherwise
    return True


def check_cube_match(
    current_pos: StandardCoord,
    current_kind: str,
    next_pos: StandardCoord,
    next_kind: str,
) -> bool:
    """Checks if two cubes match by comparing the symbols of their colours.

    ! Note. Function does not handle HADAMARDS.
    ! Note. To handle hadamards in "next_pos", strip the "h" from name, run as a regular pipe, add "h" back after match is found.
    ! Note. To handle hadamards in "current_pos", rotate it, then run as regular pipe.

    Args:
        - current_pos: (x, y, z) coordinates for the current node.
        - current_kind: current node's kind.
        - next_pos: (x, y, z) coordinates for the next node.
        - next_kind: target node's kind.

    Returns:
        - bool:
            - True: cubes match
            - False: no match.

    """

    # SANITISE
    current_kind = current_kind.lower()
    next_kind = next_kind.lower()

    # CHECK SOURCE TO TARGET
    # Connection takes place on a valid exit of source
    if not check_is_exit(current_pos, current_kind, next_pos):
        return False

    # CHECK TARGET TO SOURCE
    # Connection takes place on a valid exit of target
    if not check_is_exit(next_pos, next_kind, current_pos):
        return False

    return True


def get_valid_nxt_kinds(
    current_pos: StandardCoord,
    current_kind: str,
    next_pos: StandardCoord,
    hadamard_flag: bool = False,
) -> List[str]:
    """Reduces the number of possible types for next block/pipe by quickly running the current kind by a pre-match operations.

    Args:
        - current_pos: (x, y, z) coordinates for the current node.
        - current_kind: current node's kind.
        - next_pos: (x, y, z) coordinates for the next node.

    Returns:
        - reduced_possible_kinds: a subset of kinds applicable to next move.

    """

    # HELPER VARIABLES
    possible_kinds = []
    all_cube_kinds = ["xxz", "xzz", "xzx", "zzx", "zxx", "zxz"]
    all_pipe_kinds = ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"]

    # CHECK FOR ALL POSSIBLE NEXT KINDS IN DISPLACEMENT AXIS
    # If current kind has an "o", the next kind is a cube
    if "o" in current_kind:
        for next_kind in all_cube_kinds:
            cube_match = check_cube_match(
                current_pos, current_kind, next_pos, next_kind
            )
            if cube_match:
                possible_kinds.append(next_kind)

    # If current kind does not have an "o", then current kind is cube and the next kind is a pipe
    else:
        for next_kind in all_pipe_kinds:
            cube_match = check_cube_match(
                current_pos, current_kind, next_pos, next_kind
            )
            if cube_match:
                possible_kinds.append(next_kind)

    # Now discard possible kinds where there is no colour match for all non-connection faces
    reduced_possible_kinds = []
    for next_kind in possible_kinds:
        if check_face_match(current_pos, current_kind, next_pos, next_kind):
            reduced_possible_kinds.append(next_kind)

    # RETURN ARRAY OF POSSIBLE NEXT KINDS
    return reduced_possible_kinds
