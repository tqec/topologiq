"""Util facilities to assist in 3D pathfinding operations.

Usage:
    Call any function/class from a separate script.

"""

import networkx as nx
import numpy as np

from topologiq.utils.classes import NodeBeams, StandardBeam, StandardBlock, StandardCoord


##################
# BFS MANAGEMENT #
##################
def prune_visited(
    visited: dict[tuple[StandardBlock, StandardCoord], int],
    curr_block_info: StandardBlock
) -> dict[tuple[StandardBlock, StandardCoord], int]:
    """Prune the visited dictionary from the pathfinder.

    Args:
        visited: The dictionary the pathfinder algorithm uses to keep track of visited sites.
        curr_block_info: The coordinates and kind of the current block.

    Returns:
        new_visited: A pruned version of the incoming dictionary, which allows revisiting some sites.

    """

    new_visited = {}
    for k, v in visited.items():
        block_info = k[0]
        new_visited[(block_info, (0,0,0))] = v
        new_visited[(curr_block_info, (0,0,0))] = 0

    return new_visited


##########################
# STANDARD 3D OPERATIONS #
##########################
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


#######################
# SYMBOLIC OPERATIONS #
#######################
def check_is_exit(src_c: StandardCoord, src_k: str | None, tgt_c: StandardCoord) -> bool:
    """Check if a face is an exit.

    This function works by matching exit markers in a block's kind against a displacement
    array symbolising the direction of the target block/pipe.

    Args:
        src_c: (x, y, z) coordinates for the current block/pipe.
        src_k: current block's/pipe's kind.
        tgt_c: coordinates for the target block/pipe.

    Returns:
        (bool): True if face is an exit else False.

    """

    src_k = src_k.lower()[:3] if isinstance(src_k, str) else ""
    kind_3d = [src_k[0], src_k[1], src_k[2]]

    if "o" in kind_3d:
        marker = "o"
    else:
        marker = [i for i in set(kind_3d) if kind_3d.count(i) >= 2][0]

    exit_idxs = [i for i, char in enumerate(kind_3d) if char == marker]
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
    taken: list[StandardCoord],
    all_beams: list[NodeBeams],
    beams_len: int,
) -> tuple[bool, StandardBeam]:
    """Check if a face is unobstructed.

    This function should typically be called after verifying a face is exit.

    Args:
        src_c: The (x, y, z) coordinates for the current block/pipe.
        tgt_c: The coordinates for the target block/pipe.
        taken: The list of coordinates taken by any blocks/pipes placed as a result of previous operations.
        all_beams: The list of coordinates taken by the beams of all blocks in original ZX-graph
        beams_len: The int representing how long does each beam spans from its originating block.

    Returns:
        (bool): True if face is unobstructed else False.

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
    taken: list[StandardCoord],
    coords_in_path: list[StandardCoord],
    nx_g: nx.Graph,
    beams_len: int,
) -> tuple[int, NodeBeams]:
    """Find the number of unobstructed exits for an arbitrary block.

    This function manages calls to other functions that determine if
    each face of a block is or is not an exit and whether the said
    exit is unobstructed.

    Args:
        src_c: The (x, y, z) coordinates for the block.
        src_k: The kind of the block.
        taken: A list of coordinates taken by any blocks placed as a result of previous operations.
        coords_in_path: The coordinates taken by the path under current evaluation.
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        beams_len: An integer representing how far each beam spans from its originating block.

    Returns:
        unobstr_exits_n: the number of unobstructed exist for the block.
        beams_for_block: the beams emanating from the block.

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
                src_c, tgt_c, taken, [], beams_len
            )
            if is_unobstr and not any([coord in coords_in_path for coord in exit_beam]):
                unobstr_exits_n += 1
                n_beams.append(exit_beam)

    # Remove any beams with beam clashes
    delete_beams = []
    for i, target_node_beam in enumerate(n_beams):
        for node_id in nx_g.nodes():
            node_beams = nx_g.nodes[node_id]["beams"]
            if node_beams != [] and node_beams is not None:
                for beam in node_beams:
                    if not any([coord in coords_in_path for coord in beam[:9]]):
                        if any([(coord in target_node_beam[:9]) for coord in beam]):
                            delete_beams.extend([i])
    delete_idxs = list(set(delete_beams))
    if n_beams and delete_idxs:
        for idx in sorted(delete_idxs, reverse=True):
            del n_beams[idx]

    # Reset number of unobstructed exits
    unobstr_exits_n = len(n_beams)

    return unobstr_exits_n, n_beams


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


def face_match(src_c: StandardCoord, src_k: str, tgt_c: StandardCoord, tgt_k: str) -> bool:
    """Check if block has an available exit pointing towards a target coordinate.

    This function checks if a block has an available exit that could be used to reach an
    arbitrary target coordinate. The function can also be used to check if two cubes match
    by calling it twice using current->target and target->current coordinates. That said,
    the function does not test if target coordinate is available or an exit is unobstructed.

    Args:
        src_c: (x, y, z) coords for source node.
        src_k: kind for the source node.
        tgt_c: (x, y, z) coords for target node.
        tgt_k: kind for the target node.

    Returns:
        (boolean): True if an available exit points towards target coordinate else False.

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


def cube_match(src_c: StandardCoord, src_k: str, tgt_pos: StandardCoord, tgt_k: str) -> bool:
    """Check if two cubes match.

    This function checks if two cubes match by comparing the symbols of their colours.
    To handle hadamards as `tgt_k`, strip the "h" from the block kind and run as a regular pipe,
    adding the "h" after match is determined. To handle hadamards in `src_k`, rotate it,
    then run as regular pipe.

    Args:
        src_c: (x, y, z) coordinates for the current node.
        src_k: current node's kind.
        tgt_pos: (x, y, z) coordinates for the next node.
        tgt_k: target node's kind.

    Returns:
        (bool): True if cubes match else False.

    """

    # CHECK SOURCE TO TARGET
    if not check_is_exit(src_c, src_k.lower(), tgt_pos):
        return False

    # CHECK TARGET TO SOURCE
    if not check_is_exit(tgt_pos, tgt_k.lower(), src_c):
        return False

    return True


def nxt_kinds(src_c: StandardCoord, src_k: str, tgt_pos: StandardCoord) -> list[str]:
    """Reduce the number of possible kinds for next block.

    This function reduces the total number of potential kinds to check as plausible next
    kinds by quickly running the current kind through a few quick pre-match expectations.

    Args:
        src_c: (x, y, z) coordinates for the current node.
        src_k: current node's kind.
        tgt_pos: (x, y, z) coordinates for the next node.

    Returns:
        reduced_valid_kinds: a subset of kinds applicable to next move.

    """

    # HELPER VARIABLES
    c_ks = ["xxz", "zzx", "xzz", "zxx", "zxz", "xzx"]
    p_ks = ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"]

    # CHECK FOR ALL POSSIBLE NEXT KINDS IN DISPLACEMENT AXIS
    # Remove Hadamard flag if present
    if "h" in src_k:
        src_k = src_k[:3]
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


def flip_hdm(k: str) -> str:
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
