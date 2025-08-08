from collections import deque
from typing import List, Tuple, Optional, Union

from utils.classes import StandardCoord, StandardBlock
from utils.utils_greedy_bfs import flip_hdm, rot_o_kind
from utils.constraints import get_valid_nxt_kinds
from utils.utils_misc import get_max_manhattan


#########################
# MAIN WORKFLOW MANAGER #
#########################
def pthfinder(
    src: StandardBlock,
    tent_coords: List[StandardCoord],
    tgt_zx_type: str,
    tgt: Tuple[Optional[StandardCoord], Optional[str]] = (None, None),
    taken: List[StandardCoord] = [],
    hdm: bool = False,
) -> Union[None, dict[StandardBlock, List[StandardBlock]]]:
    """
    Runs core pathfinder on a loop until path is found within predetermined distance of source node or max distance is reached.

    Args:
        - src: source block coordinates and kind.
        - tent_coords: list of tentative target coords to find paths to.
        - tgt_zx_type: ZX type of the target node, taken from a ZX chart.
        - tgt: the information of a specific block including its position and kind,
            used to override placement when target block has already been placed in 3D space in previous operations.
        - taken: list of coordinates already taken in previous operations.
        - hdm: flag to highlights current operation corresponds to a Hadamard edge.

    Returns:
        - valid_pths: all paths found in round, covering some or all tent_coords.

    """

    # UNPACK INCOMING DATA
    s_coords, _ = src
    _, tgt_kind = tgt

    taken_cc: List[StandardCoord] = taken[:]
    if taken_cc:
        if s_coords in taken_cc:
            taken_cc.remove(s_coords)

    # Generate all possible target types at tentative position
    tent_tgt_kinds = gen_tent_tgt_kinds(
        tgt_zx_type,
        tgt_kind=(tgt_kind if tgt_kind else None),
    )

    # Find paths to all potential target kinds
    valid_pths = core_pthfinder_bfs(
        src,
        tent_coords,
        tent_tgt_kinds,
        taken=taken_cc,
        hdm=hdm,
    )

    # Return boolean for success of path finding, lenght of winner path, and winner path
    return valid_pths


##################################
# CORE PATHFINDER BFS OPERATIONS #
##################################
def core_pthfinder_bfs(
    src: StandardBlock,
    tent_coords: List[StandardCoord],
    tent_tgt_kinds: List[str],
    taken: List[StandardCoord] = [],
    hdm: bool = False,
    min_succ_rate: int = 50,
) -> Union[None, dict[StandardBlock, List[StandardBlock]]]:
    """Core pathfinder BFS. Determines if topologically-correct paths are possible between a source and one or more target coordinates/kinds.

    Args:
        - src: source block's coordinates (tuple) and kind (str).
        - tent_coords: list of tentative target coords to find paths to.
        - tent_tgt_kinds: list of kinds matching the zx-type of target block
        - taken: list of coordinates that have already been occupied as part of previous operations.
        - hdm: a flag that highlights the current operation corresponds to a Hadamard edge.
        - min_succ_rate: min % of tent_coords to find a path to, used as exit condition.

    Returns:
        - valid_pths: all paths found in round, covering some or all tent_coords.

    """

    # UNPACK INCOMING DATA
    s_coords, _ = src
    sx, sy, sz = s_coords
    end_coords = tent_coords

    if s_coords in taken:
        taken.remove(s_coords)
    if (
        len(end_coords) == 1
        and len(tent_tgt_kinds) == 1
        and end_coords[0] in taken
    ):
        taken.remove(end_coords[0])

    # KEY BFS VARS
    queue = deque([src])
    visited: dict[Tuple[StandardBlock, Tuple[int, int, int]], int] = {
        (src, (0, 0, 0)): 0
    }

    pth_len = {src: 0}
    pth = {src: [src]}
    valid_pths: Union[None, dict[StandardBlock, List[StandardBlock]]] = {}
    moves = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    # EXIT CONDITIONS
    num_tgts_filled = 0
    num_tgts_to_fill = int(len(tent_coords) * min_succ_rate / 100)
    if len(tent_coords) > 1:
        max_manhattan = get_max_manhattan(s_coords, tent_coords) * 2
    else:
        max_manhattan = max(
            get_max_manhattan(s_coords, taken) * 2,
            get_max_manhattan(s_coords, tent_coords) * 2,
        )

    # CORE PATHFINDER BFS LOOP
    while queue:
        curr_b_info: StandardBlock = queue.popleft()
        curr_coords, curr_kind = curr_b_info
        x, y, z = [x for x in curr_coords]

        current_manhattan = abs(x - sx) + abs(y - sy) + abs(z - sz)
        if current_manhattan > max_manhattan:
            break

        if curr_coords in end_coords and (
            tent_tgt_kinds == ["ooo"] or curr_kind in tent_tgt_kinds
        ):
            valid_pths[curr_b_info] = pth[curr_b_info]
            num_tgts_filled = len(set([p[0] for p in valid_pths.keys()]))
            if num_tgts_filled >= num_tgts_to_fill:
                break

        scale = 2 if "o" in curr_kind else 1
        for dx, dy, dz in moves:
            nxt_x, nxt_y, nxt_z = x + dx * scale, y + dy * scale, z + dz * scale
            nxt_coords = (nxt_x, nxt_y, nxt_z)
            curr_pth_coords = [n[0] for n in pth[curr_b_info]]

            mid_pos = None
            if "o" in curr_kind and scale == 2:
                mid_x = x + dx * 1
                mid_y = y + dy * 1
                mid_z = z + dz * 1
                mid_pos = (mid_x, mid_y, mid_z)
                if mid_pos in curr_pth_coords or mid_pos in taken:
                    continue

            if "h" in curr_kind:
                hdm = False
                if (
                    sum(
                        [
                            p[0] + p[1] if p[0] != p[1] else 0
                            for p in list(zip(src[0], curr_coords))
                        ]
                    )
                    < 0
                ):
                    curr_kind = flip_hdm(curr_kind)
                    curr_kind = rot_o_kind(curr_kind)
                else:
                    rotated_type = rot_o_kind(curr_kind)
                    curr_kind = rotated_type

                curr_kind = curr_kind[:3]

            possible_nxt_types = get_valid_nxt_kinds(curr_coords, curr_kind, nxt_coords)

            for nxt_type in possible_nxt_types:

                # If hadamard flag is on and the block being placed is "o", place a hadamard instead of regular pipe
                if hdm and "o" in nxt_type:
                    nxt_type += "h"
                    if (
                        sum(
                            [
                                p[0] + p[1] if p[0] != p[1] else 0
                                for p in list(zip(curr_coords, nxt_coords))
                            ]
                        )
                        < 0
                    ):
                        nxt_type = rot_o_kind(nxt_type)

                if (
                    nxt_coords not in curr_pth_coords
                    and nxt_coords not in taken
                    and (mid_pos is None or nxt_coords != mid_pos)
                ):
                    new_pth_len = pth_len[curr_b_info] + 1
                    nxt_b_info: StandardBlock = (nxt_coords, nxt_type)

                    if (
                        (nxt_b_info, (dx, dy, dz))
                    ) not in visited or new_pth_len < visited[
                        (nxt_b_info, (dx, dy, dz))
                    ]:
                        visited[(nxt_b_info, (dx, dy, dz))] = new_pth_len
                        queue.append(nxt_b_info)
                        pth_len[nxt_b_info] = new_pth_len
                        pth[nxt_b_info] = pth[curr_b_info] + [nxt_b_info]

    return valid_pths


##################
# AUX OPERATIONS #
##################
def gen_tent_tgt_kinds(tgt_zx_type: str, tgt_kind: Optional[str] = None) -> List[str]:
    """Returns all possible valid kinds/types for a given ZX type,
    typically needed when a new block is being added to the 3D space,
    as each ZX type can be fulfilled with more than one block types/kinds.

    Args:
        - tgt_zx_type: the ZX type of the target node.
        - tgt_kind: a specific block/pipe type/kind to return irrespective of ZX type,
            used when the target block was already placed as part of previous operations and therefore already has an assigned kind.

    Returns:
        - fam: a list of applicable types/kinds for the given ZX type.

    """

    if tgt_kind:
        return [tgt_kind]

    # TYPE/KIND FAMILIES
    X = ["xxz", "xzx", "zxx"]
    Z = ["xzz", "zzx", "zxz"]
    B = ["ooo"]
    S = ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"]
    HDM = ["zxoh", "xzoh", "oxzh", "ozxh", "xozh", "zoxh"]

    if tgt_zx_type in ["X", "Z"]:
        fam = X if tgt_zx_type == "X" else Z
    elif tgt_zx_type == "O":
        fam = B
    elif tgt_zx_type == "SIMPLE":
        fam = S
    elif tgt_zx_type == "HADAMARD":
        fam = HDM
    else:
        return [tgt_zx_type]

    return fam


def get_taken_coords(all_blocks: List[StandardBlock]):
    """Converts a series of blocks into a list of all coordinates occupied by the blocks.

    Args:
        - all_blocks: a list of blocks and pipes that altogether make a space-time diagram.

    Returns:
        - list(taken): a list of coordinates taken by pre-existing cubes and pipes (blocks).

    """

    taken = set()

    if not all_blocks:
        return []

    # Add first block's coordinates
    b_1 = all_blocks[0]
    if b_1:
        b_1_coords = b_1[0]
        taken.add(b_1_coords)

    # Iterate from the second node
    for i, b in enumerate(all_blocks):

        if i > 0:

            curr_b = all_blocks[i]
            prev_b = all_blocks[i - 1]

            if curr_b and prev_b:
                curr_b_coords, curr_b_kind = curr_b
                prev_b_coords, _ = prev_b

                # Add current node's coordinates
                taken.add(curr_b_coords)

                if "o" in curr_b_kind:
                    cx, cy, cz = curr_b_coords
                    px, py, pz = prev_b_coords
                    ext_coords = None

                    if cx == px + 1:
                        ext_coords = (cx + 1, cy, cz)
                    elif cx == px - 1:
                        ext_coords = (cx - 1, cy, cz)
                    elif cy == py + 1:
                        ext_coords = (cx, cy + 1, cz)
                    elif cy == py - 1:
                        ext_coords = (cx, cy - 1, cz)
                    elif cz == pz + 1:
                        ext_coords = (cx, cy, cz + 1)
                    elif cz == pz - 1:
                        ext_coords = (cx, cy, cz - 1)

                    if ext_coords:
                        taken.add(ext_coords)

    return list(taken)
