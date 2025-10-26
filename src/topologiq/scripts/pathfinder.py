import os

from datetime import datetime
from collections import deque
from typing import List, Tuple, Optional, Union

from topologiq.utils.classes import NodeBeams, StandardCoord, StandardBlock
from topologiq.utils.utils_greedy_bfs import gen_tent_tgt_coords
from topologiq.utils.utils_pathfinder import (
    check_is_exit,
    prune_visited,
    rot_o_kind,
    nxt_kinds,
)
from topologiq.utils.utils_misc import get_max_manhattan, prep_stats_n_log


############################
# MAIN PATHFINDER WORKFLOW #
############################
def pthfinder(
    src: StandardBlock,
    tent_coords: List[StandardCoord],
    tgt_zx_type: str,
    tgt: Tuple[Optional[StandardCoord], Optional[str]] = (None, None),
    taken: List[StandardCoord] = [],
    hdm: bool = False,
    min_succ_rate: int = 60,
    critical_beams: dict[int, Tuple[int, NodeBeams]] = {},
    u_v_ids: Optional[Tuple[int,int]] = None,
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
        - min_succ_rate: min % of tent_coords that need to be filled, used as exit condition.
        - log_stats_id: unique identifier for logging stats to CSV files in `.assets/stats/` (`None` keeps logging is off).

    Returns:
        - valid_pths: all paths found in round, covering some or all tent_coords.

    """

    # PRELIMS
    t1 = None
    t_end = None
    if log_stats_id is not None:
        t1 = datetime.now()

    # UNPACK INCOMING DATA
    s_coords, _ = src
    _, tgt_kind = tgt

    taken_cc: List[StandardCoord] = taken[:]
    if taken_cc:
        if s_coords in taken_cc:
            taken_cc.remove(s_coords)

    # GENERATE ALL POSSIBLE TENTATIVE TARGET TYPES
    tent_tgt_kinds = gen_tent_tgt_kinds(
        tgt_zx_type,
        tgt_kind=(tgt_kind if tgt_kind else None),
    )

    # CALL PATHFINDER ON ALL TENT COORD AND TENT KINDS
    valid_pths, visit_stats = core_pthfinder_bfs(
        src,
        tent_coords,
        tent_tgt_kinds,
        min_succ_rate,
        taken=taken_cc,
        hdm=hdm,
        critical_beams=critical_beams,
        u_v_ids=u_v_ids,
    )

    # LOG STATS IF NEEDED
    if log_stats_id is not None:

        t_end = datetime.now()
        times = {"t1": t1, "t_end": t_end}
        iter_success = True if valid_pths else False

        len_longest_pth = 0
        if valid_pths:
            for pth in valid_pths.values():
                if pth:
                    len_pth = sum([2 if "o" in b[1] else 1 for b in pth]) - 1
                    len_longest_pth = max(len_longest_pth, len_pth)

        counts = {
            "num_tent_coords": len(tent_coords) if valid_pths else 0,
            "num_tent_coords_filled": (
                len(set([p[0] for p in valid_pths.keys()])) if valid_pths else 0
            ),
            "max_manhattan": get_max_manhattan(s_coords, tent_coords),
            "len_longest_path": len_longest_pth if len_longest_pth > 0 else 0,
        }

        prep_stats_n_log(
            "pathfinder",
            log_stats_id,
            iter_success,
            counts,
            times,
            src=src,
            tgt=tgt,
            tgt_zx_type=tgt_zx_type,
            visit_stats=visit_stats,
        )

    # RETURN VALID PATHS (ONE OR MORE IF SUCCESSFUL)
    return valid_pths


###############################
# CORE PATHFINDER SPATIAL BFS #
###############################
def core_pthfinder_bfs(
    src: StandardBlock,
    tent_coords: List[StandardCoord],
    tent_tgt_kinds: List[str],
    min_succ_rate,
    taken: List[StandardCoord] = [],
    hdm: bool = False,
    critical_beams: dict[int, Tuple[int, NodeBeams]] = {},
    u_v_ids: Optional[Tuple[int,int]] = None,
) -> Tuple[Union[None, dict[StandardBlock, List[StandardBlock]]], Tuple[int, int]]:
    """Core pathfinder BFS. Determines if topologically-correct paths are possible between a source and one or more target coordinates/kinds.

    Args:
        - src: source block's coordinates (tuple) and kind (str).
        - tent_coords: list of tentative target coords to find paths to.
        - tent_tgt_kinds: list of kinds matching the zx-type of target block
        - min_succ_rate: min % of tent_coords that need to be filled, used as exit condition.
        - taken: list of coordinates that have already been occupied as part of previous operations.
        - hdm: a flag that highlights the current operation corresponds to a Hadamard edge.

    Returns:
        - valid_pths: all paths found in round, covering some or all tent_coords.

    """

    # UNPACK INCOMING DATA
    s_coords, s_kind = src
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

    moves_unadjusted = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    # EXIT CONDITIONS
    tgts_filled = 0
    tgts_to_fill = (
        int(len(tent_coords) * min_succ_rate / 100) if len(tent_coords) > 1 else 1
    )

    if len(tent_coords) > 1:
        max_manhattan = get_max_manhattan(s_coords, tent_coords) * 2
        src_tgt_manhattan = max_manhattan
    else:
        src_tgt_manhattan = get_max_manhattan(s_coords, tent_coords)
        max_manhattan = max(
            get_max_manhattan(s_coords, taken) * 2,
            src_tgt_manhattan * 2,
        )

    src_tgt_manhattan = get_max_manhattan(s_coords, tent_coords)
    prune_distance = src_tgt_manhattan
    # CORE LOOP

    while queue:
        curr: StandardBlock = queue.popleft()
        curr_coords, curr_kind = curr
        x, y, z = curr_coords

        curr_manhattan = abs(x - sx) + abs(y - sy) + abs(z - sz)

        if curr_manhattan < prune_distance - 6:
            visited = prune_visited(visited)
            prune_distance = curr_manhattan

        if curr_manhattan > src_tgt_manhattan + 6:
            continue

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

        moves_adjusted = []
        remaining_moves = []

        for move in moves_unadjusted:
            valid_exit = check_is_exit((0, 0, 0), curr_kind, move)
            if valid_exit:
                moves_adjusted.append(move)
            else:
                remaining_moves.append(move)
        moves_adjusted += remaining_moves
        if len(tent_coords) == 1:
            moves = moves_adjusted
        else: 
            moves = moves_unadjusted
        moves = moves_unadjusted

        for dx, dy, dz in moves:
            nxt_x, nxt_y, nxt_z = x + dx * scale, y + dy * scale, z + dz * scale
            nxt_coords = (nxt_x, nxt_y, nxt_z)
            curr_pth_coords = [n[0] for n in pth[curr]]

            if nxt_coords in taken:
                continue

            if critical_beams:
                nodes_with_critical_beams_id = critical_beams.keys()
                if nodes_with_critical_beams_id:
                    continue_flag = False
                    for node_id in nodes_with_critical_beams_id:
                        min_exit_num = critical_beams[node_id][0]
                        beams = critical_beams[node_id][1]
                        beams_broken_for_node = sum(
                            [nxt_coords in beam[:3] for beam in beams]
                        )
                        adjust_for_source_node = 1 if node_id in u_v_ids else 0
                        if len(beams) - beams_broken_for_node < (min_exit_num - adjust_for_source_node):
                            continue_flag = True
                            break
                    if continue_flag:
                        continue

            mid_pos = None
            if "o" in curr_kind and scale == 2:
                mid_x = x + dx * 1
                mid_y = y + dy * 1
                mid_z = z + dz * 1
                mid_pos = (mid_x, mid_y, mid_z)
                if mid_pos in curr_pth_coords or mid_pos in taken:
                    continue

                if critical_beams:
                    nodes_with_critical_beams_id = critical_beams.keys()
                    if nodes_with_critical_beams_id:
                        continue_flag = False
                        for node_id in nodes_with_critical_beams_id:
                            min_exit_num = critical_beams[node_id][0]
                            beams = critical_beams[node_id][1]
                            beams_broken_for_node = sum(
                                [mid_pos in beam[:3] for beam in beams]
                            )
                            adjust_for_source_node = 1 if node_id in u_v_ids else 0
                            if len(beams) - beams_broken_for_node < (min_exit_num - adjust_for_source_node):
                                continue_flag = True
                                break
                        if continue_flag:
                            continue
            
            alt_curr_kind = None
            if "h" in curr_kind:
                hdm = False
                direction = sum([p[1] - p[0] if p[0] != p[1] else 0 for p in list(zip(curr_coords, nxt_coords))])
                if direction < 0:
                    pass
                else:
                    alt_curr_kind = rot_o_kind(curr_kind)

            possible_nxt_types = nxt_kinds(curr_coords, curr_kind if not alt_curr_kind else alt_curr_kind, nxt_coords)
            for nxt_type in possible_nxt_types:
                
                # If hadamard flag is on and the block being placed is "o", place a hadamard instead of regular pipe
                if hdm and "o" in nxt_type:
                    nxt_type += "h"
                    direction = sum([p[1] - p[0] if p[0] != p[1] else 0 for p in list(zip(curr_coords, nxt_coords))])
                    if direction < 0:
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

    # RETURN VALID PATHS
    return valid_pths, (visit_attempts, len(visited))


##################
# AUX OPERATIONS #
##################
def gen_tent_tgt_kinds(tgt_zx_type: str, tgt_kind: Optional[str] = None) -> List[str]:
    """Returns all possible valid kinds for a given ZX type,
    typically needed when a new block is being added to the 3D space,
    as each ZX type can be fulfilled with more than one block kind.

    Args:
        - tgt_zx_type: the ZX type of the target node.
        - tgt_kind: a specific kind to return irrespective of ZX type,
            used when the target block was already placed as part of previous operations and therefore already has an assigned kind.

    Returns:
        - fam: a list of applicable kinds for the given ZX type.

    """

    if tgt_kind:
        return [tgt_kind]

    if tgt_zx_type in ["X", "Z"]:
        return ["zzx", "zxz", "xzz"] if tgt_zx_type == "X" else ["xxz", "xzx", "zxx"]
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

    # ADD 1ST BLOCK COORDS
    b_1st = all_blocks[0]
    if b_1st:
        b_1_coords = b_1st[0]
        taken.add(b_1_coords)

    # ITERATE FROM 2ND BLOCK ONWARDS
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

    # RETURN LIST OF TAKEN COORDS
    return list(taken)


##########
# TESTS  #
##########
def test_pthfinder(
    stats_dir: str, min_succ_rate, max_test_step: int = 3, num_repetitions: int = 1
):
    """Checks runtimes for creation of paths by pathfinder algorithm.

    Args:
        - stats_dir: the directory where states are to be saved
        - min_succ_rate: min % of tent_coords that need to be filled, used as exit condition.
        - max_test_step: Sets the maximum distance for target test node
        - num_repetitions: Sets the number of times that the test loop will be repeated.

    """

    if os.path.exists(f"{stats_dir}/pathfinder_tests.csv"):
        os.remove(f"{stats_dir}/pathfinder_tests.csv")

    taken: List[StandardCoord] = []
    hdm: bool = False
    src_coords: StandardCoord = (0, 0, 0)
    all_valid_start_kinds: List[str] = ["zzx", "zxz", "xzz", "xxz", "xzx", "zxx"]
    all_valid_zx_types: List[str] = ["X", "Z"]
    unique_run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "*"

    i = 0
    while i < num_repetitions:
        i += 1
        step: int = 3
        print(f"\nRunning tests. Loop {i} of {num_repetitions}.")
        while step <= max_test_step:
            tent_coords = gen_tent_tgt_coords(src_coords, step, taken)
            for start_kind in all_valid_start_kinds:
                src = ((0, 0, 0), start_kind)
                for zx_type in all_valid_zx_types:
                    clean_paths = pthfinder(
                        src,
                        tent_coords,
                        zx_type,
                        tgt=(None, None),
                        taken=taken,
                        hdm=hdm,
                        min_succ_rate=min_succ_rate,
                        log_stats_id=unique_run_id,
                    )
                    if clean_paths:
                        print(".", end="")
            step += 3
