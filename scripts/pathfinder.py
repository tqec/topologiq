from datetime import datetime
from collections import deque
from typing import List, Tuple, Optional, Union

from utils.classes import StandardCoord, StandardBlock
from utils.utils_greedy_bfs import flip_hdm, rot_o_kind
from utils.constraints import nxt_kinds
from utils.utils_misc import get_max_manhattan, log_stats_to_file


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
    min_succ_rate: int = 100,
    log_stats_id: Union[str, None] = None,
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
        - log_stats_id: unique identifier for logging stats to CSV files in `.assets/stats/` (`None` keeps logging is off).

    Returns:
        - valid_pths: all paths found in round, covering some or all tent_coords.

    """

    # PRELIMS
    t1 = None
    if log_stats_id is not None:
        t1 = datetime.now()

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
    valid_pths, visit_stats = core_pthfinder_bfs(
        src,
        tent_coords,
        tent_tgt_kinds,
        taken=taken_cc,
        hdm=hdm,
        min_succ_rate=min_succ_rate,
    )

    if log_stats_id is not None:
        if t1 is not None:
            max_len = 0
            num_tent_coords = 0
            num_tent_coords_filled = 0
            max_manhattan = get_max_manhattan(s_coords, tent_coords)

            pth_found = False
            if valid_pths:
                pth_found = True
                num_tent_coords = len(tent_coords)
                num_tent_coords_filled = len(set([p[0] for p in valid_pths.keys()]))
                for pth in valid_pths.values():
                    if pth:
                        len_pth = sum([2 if "o" in b[1] else 1 for b in pth]) - 1
                        max_len = max(max_len, len_pth)

            iter_stats = [
                log_stats_id,
                "creation" if not tgt[1] else "discovery",
                pth_found,
                src[0],
                src[1],
                tgt[0] if tgt[0] else "n/a",
                tgt_zx_type,
                tgt[1] if tgt[1] else "TBD",
                num_tent_coords,
                num_tent_coords_filled,
                max_manhattan,
                max_len if max_len > 0 else "n/a",
                visit_stats[0],
                visit_stats[1],
                (datetime.now() - t1).total_seconds(),
            ]

            opt_header = [
                "unique_run_id",
                "iter_type",
                "iter_success",
                "src_coords",
                "src_kind",
                "tgt_coords",
                "tgt_zx_type",
                "tgt_kind",
                "num_tent_coords_received",
                "num_tent_coords_filled",
                "max_manhattan_src_to_any_tent_coord",
                "len_longest_path",
                "num_visitation_attempts",
                "num_sites_visited",
                "iter_duration",
            ]

            log_stats_to_file(
                iter_stats,
                f"pathfinder_iterations{"_tests" if log_stats_id.endswith("*") else ""}",
                opt_header=opt_header,
            )

    # Return valid paths
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
    min_succ_rate: int = 100,
) -> Tuple[Union[None, dict[StandardBlock, List[StandardBlock]]], Tuple[int, int]]:
    """Core pathfinder BFS. Determines if topologically-correct paths are possible between a source and one or more target coordinates/kinds.

    Args:
        - src: source block's coordinates (tuple) and kind (str).
        - tent_coords: list of tentative target coords to find paths to.
        - tent_tgt_kinds: list of kinds matching the zx-type of target block
        - taken: list of coordinates that have already been occupied as part of previous operations.
        - hdm: a flag that highlights the current operation corresponds to a Hadamard edge.
        - min_succ_rate: min % of tent_coords that need to be filled, used as exit condition.

    Returns:
        - valid_pths: all paths found in round, covering some or all tent_coords.

    """

    # UNPACK INCOMING DATA
    s_coords, _ = src
    sx, sy, sz = s_coords
    end_coords = tent_coords

    if s_coords in taken:
        taken.remove(s_coords)
    if len(end_coords) == 1 and len(tent_tgt_kinds) == 1 and end_coords[0] in taken:
        taken.remove(end_coords[0])

    # KEY BFS VARS
    queue = deque([src])
    visited: dict[Tuple[StandardBlock, StandardCoord], int] = {(src, (0, 0, 0)): 0}
    visit_attempts = 0

    pth_len = {src: 0}
    pth = {src: [src]}
    valid_pths: Union[None, dict[StandardBlock, List[StandardBlock]]] = {}
    moves = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    # EXIT CONDITIONS
    tgts_filled = 0
    tgts_to_fill = int(len(tent_coords) * min_succ_rate / 100)
    if len(tent_coords) > 1:
        max_manhattan = get_max_manhattan(s_coords, tent_coords) * 2
    else:
        max_manhattan = max(
            get_max_manhattan(s_coords, taken) * 2,
            get_max_manhattan(s_coords, tent_coords) * 2,
        )

    # CORE PATHFINDER BFS LOOP
    while queue:
        curr: StandardBlock = queue.popleft()
        curr_coords, curr_kind = curr
        x, y, z = curr_coords

        curr_manhattan = abs(x - sx) + abs(y - sy) + abs(z - sz)
        if curr_manhattan > max_manhattan:
            break

        if curr_coords in end_coords and (
            tent_tgt_kinds == ["ooo"] or curr_kind in tent_tgt_kinds
        ):
            valid_pths[curr] = pth[curr]
            tgts_filled = len(set([p[0] for p in valid_pths.keys()]))
            if tgts_filled >= tgts_to_fill:
                break

        scale = 2 if "o" in curr_kind else 1
        for dx, dy, dz in moves:
            nxt_x, nxt_y, nxt_z = x + dx * scale, y + dy * scale, z + dz * scale
            nxt_coords = (nxt_x, nxt_y, nxt_z)
            curr_pth_coords = [n[0] for n in pth[curr]]

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

            possible_nxt_types = nxt_kinds(curr_coords, curr_kind, nxt_coords)

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
                    new_pth_len = pth_len[curr] + 1
                    nxt_b_info: StandardBlock = (nxt_coords, nxt_type)

                    if (
                        (nxt_b_info, (dx, dy, dz))
                    ) not in visited or new_pth_len < visited[
                        (nxt_b_info, (dx, dy, dz))
                    ]:
                        visited[(nxt_b_info, (dx, dy, dz))] = new_pth_len
                        queue.append(nxt_b_info)
                        pth_len[nxt_b_info] = new_pth_len
                        pth[nxt_b_info] = pth[curr] + [nxt_b_info]

                        if nxt_coords in end_coords and (
                            tent_tgt_kinds == ["ooo"] or nxt_type in tent_tgt_kinds
                        ):
                            valid_pths[nxt_b_info] = pth[nxt_b_info]
                            tgts_filled = len(set([p[0] for p in valid_pths.keys()]))
                            if tgts_filled >= tgts_to_fill:
                                break

            # Increase counter of times pathfinder tries visits something new
            visit_attempts += 1

    return valid_pths, (visit_attempts, len(visited))


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

    if tgt_zx_type in ["X", "Z"]:
        return ["xxz", "xzx", "zxx"] if tgt_zx_type == "X" else ["xzz", "zzx", "zxz"]
    elif tgt_zx_type == "O":
        return ["ooo"]
    elif tgt_zx_type == "SIMPLE":
        return ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"]
    elif tgt_zx_type == "HADAMARD":
        return ["zxoh", "xzoh", "oxzh", "ozxh", "xozh", "zoxh"]
    else:
        return [tgt_zx_type]


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
    b_1st = all_blocks[0]
    if b_1st:
        b_1_coords = b_1st[0]
        taken.add(b_1_coords)

    # Iterate from the second node
    for i, _ in enumerate(all_blocks):

        if i > 0:

            curr = all_blocks[i]
            prev = all_blocks[i - 1]

            if curr and prev:
                curr_c, curr_k = curr
                prev_c, _ = prev

                # Add current node's coordinates
                taken.add(curr_c)

                if "o" in curr_k:
                    cx, cy, cz = curr_c
                    px, py, pz = prev_c
                    ext_cs = None

                    if cx == px + 1:
                        ext_cs = (cx + 1, cy, cz)
                    elif cx == px - 1:
                        ext_cs = (cx - 1, cy, cz)
                    elif cy == py + 1:
                        ext_cs = (cx, cy + 1, cz)
                    elif cy == py - 1:
                        ext_cs = (cx, cy - 1, cz)
                    elif cz == pz + 1:
                        ext_cs = (cx, cy, cz + 1)
                    elif cz == pz - 1:
                        ext_cs = (cx, cy, cz - 1)

                    if ext_cs:
                        taken.add(ext_cs)

    return list(taken)
