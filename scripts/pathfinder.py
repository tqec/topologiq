from collections import deque
from typing import List, Tuple, Optional, Union

from utils.classes import StandardCoord, StandardBlock, Colors
from utils.utils_greedy_bfs import adjust_hadamards_direction, rotate_o_types
from utils.constraints import get_valid_nxt_kinds


#########################
# MAIN WORKFLOW MANAGER #
#########################
def run_bfs_for_all_potential_target_nodes(
    src_node: StandardBlock,
    tent_positions: List[StandardCoord],
    tgt_zx_type: str,
    overwrite_tgt: Tuple[Optional[StandardCoord], Optional[str]] = (None, None),
    occupied_coords: List[StandardCoord] = [],
    hadamard_flag: bool = False,
) -> Union[None, dict[StandardBlock, List[StandardBlock]]]:
    """
    Runs core pathfinder on a loop until path is found within predetermined distance of source node or max distance is reached.

    Args:
        - src_node: source node's coordinates (tuple) and type (str).
        - tgt_zx_type: ZX type of the target node, taken from a ZX chart.
        - distance: current allowed distance between source and target nodes.
        - max_distance: maximum allowed distance between source and target nodes.
        - attempts_per_distance: number of random target positions to try at each distance.
        - overwrite_tgt: the information of a specific block including its position and kind,
            used to override placement of a new node when the target node/block has already been placed in 3D space as part of previous operations.
        - occupied_coords: list of coordinates that have already been occupied as part of previous operations.
        - hadamard_flag: a flag that highlights the current operation corresponds to a Hadamard edge.

    Returns:
        - path_found:
            True: path was found (success)
            False: path was not found (fail)
        - length: the lenght of the best path of round
        - path: the best path of round
        - all_paths_from_round: an object containing all paths found in round

    """

    # UNPACK INCOMING DATA
    start_coords, _ = src_node
    _, overwrite_tgt_kind = overwrite_tgt

    obstacle_coords: List[StandardCoord] = occupied_coords[:]
    if obstacle_coords:
        if start_coords in obstacle_coords:
            obstacle_coords.remove(start_coords)

    # Generate all possible target types at tentative position
    tent_tgt_kinds = gen_tent_tgt_kinds(
        tgt_zx_type,
        overwrite_tgt_kind=(
            overwrite_tgt_kind if overwrite_tgt_kind else None
        ),
    )

    # Find paths to all potential target kinds
    valid_paths = bfs_extended_3d(
        src_node,
        tent_positions,
        tent_tgt_kinds,
        forbidden_cords=obstacle_coords,
        hadamard_flag=hadamard_flag,
    )

    # Return boolean for success of path finding, lenght of winner path, and winner path
    return valid_paths


##################################
# CORE PATHFINDER BFS OPERATIONS #
##################################
def bfs_extended_3d(
    src_node: StandardBlock,
    tent_positions: List[StandardCoord],
    tent_tgt_kinds: List[str],
    forbidden_cords: List[StandardCoord] = [],
    hadamard_flag: bool = False,
    completion_target: int = 100,
    max_manhattan: int = 30,
) -> Union[None, dict[StandardBlock, List[StandardBlock]]]:
    """Core pathfinder function. Takes a source and target node (given to it as part of a loop with many possible combinations)
    and a list of obstacle coordinates to avoid and determines if a topologically-correct path is possible between the source and target nodes.

    Args:
        - src_node: source block's coordinates (tuple) and kind (str).
        - target_node: target block's node's coordinates (tuple) and kind (str).
        - forbidden_cords: list of coordinates that have already been occupied as part of previous operations.
        - hadamard_flag: a flag that highlights the current operation corresponds to a Hadamard edge.

    Returns:
        - bool:
            - True: path found (success),
            - False: path NOT found (fail).
        - path_length: the lenght of the path found, or -1 if path not found.
        - path: the topologically-correct path between source and target blocks.

    """

    # UNPACK INCOMING DATA
    start_coords, _ = src_node
    start_x, start_y, start_z = [int(x) for x in start_coords]
    end_coords = tent_positions
    end_types = tent_tgt_kinds

    if start_coords in forbidden_cords:
        forbidden_cords.remove(start_coords)
    if (
        len(end_coords) == 1
        and len(end_types) == 1
        and end_coords[0] in forbidden_cords
    ):
        forbidden_cords.remove(end_coords[0])

    # KEY BFS VARS
    queue = deque([src_node])
    visited = {tuple(src_node): 0}
    path_length = {tuple(src_node): 0}
    path = {tuple(src_node): [src_node]}
    valid_paths: Union[None, dict[StandardBlock, List[StandardBlock]]] = {}

    # EXIT CONDITIONS
    min_num_positions_to_fill = int(len(tent_positions) * completion_target / 100)

    # CORE PATHFINDER BFS LOOP
    while queue:
        current_node_info: StandardBlock = queue.popleft()
        current_coords, current_type = current_node_info
        x, y, z = [int(x) for x in current_coords]

        current_manhattan = abs(x - start_x) + abs(y - start_y) + abs(z - start_z)
        if current_manhattan > max_manhattan:
            break

        if current_coords in end_coords and (
            end_types == ["ooo"] or current_type in end_types
        ):
            valid_paths[current_node_info] = path[current_node_info]
            num_positions_filled = len(set([p[0] for p in valid_paths.keys()]))
            if num_positions_filled >= min_num_positions_to_fill:
                break

        scale = 2 if "o" in current_type else 1
        spatial_moves = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]

        for dx, dy, dz in spatial_moves:
            nxt_x, nxt_y, nxt_z = x + dx * scale, y + dy * scale, z + dz * scale
            nxt_coords = (nxt_x, nxt_y, nxt_z)
            current_path_coords = [node[0] for node in path[current_node_info]]

            intermediate_pos = None
            if "o" in current_type and scale == 2:
                intermediate_x = x + dx * 1
                intermediate_y = y + dy * 1
                intermediate_z = z + dz * 1
                intermediate_pos = (intermediate_x, intermediate_y, intermediate_z)
                if (
                    intermediate_pos in current_path_coords
                    or intermediate_pos in forbidden_cords
                ):
                    continue

            if "h" in current_type:

                hadamard_flag = False
                if (
                    sum(
                        [
                            p[0] + p[1] if p[0] != p[1] else 0
                            for p in list(zip(src_node[0], current_coords))
                        ]
                    )
                    < 0
                ):
                    current_type = adjust_hadamards_direction(current_type)
                    current_type = rotate_o_types(current_type)
                else:
                    rotated_type = rotate_o_types(current_type)
                    current_type = rotated_type

                current_type = current_type[:3]

            possible_nxt_types = get_valid_nxt_kinds(
                current_coords, current_type, nxt_coords, hadamard_flag=hadamard_flag
            )

            for nxt_type in possible_nxt_types:

                # If hadamard flag is on and the block being placed is "o", place a hadamard instead of regular pipe
                if hadamard_flag and "o" in nxt_type:
                    nxt_type += "h"
                    if (
                        sum(
                            [
                                p[0] + p[1] if p[0] != p[1] else 0
                                for p in list(zip(current_coords, nxt_coords))
                            ]
                        )
                        < 0
                    ):
                        nxt_type = rotate_o_types(nxt_type)

                nxt_node_info: StandardBlock = (nxt_coords, nxt_type)

                if (
                    nxt_coords not in current_path_coords
                    and nxt_coords not in forbidden_cords
                    and (intermediate_pos is None or nxt_coords != intermediate_pos)
                ):
                    new_path_length = path_length[current_node_info] + 1
                    if (
                        nxt_node_info not in visited
                        or new_path_length < visited[nxt_node_info]
                    ):
                        visited[nxt_node_info] = new_path_length
                        queue.append(nxt_node_info)
                        path_length[nxt_node_info] = new_path_length
                        path[nxt_node_info] = path[current_node_info] + [
                            nxt_node_info
                        ]

    return valid_paths


##################
# AUX OPERATIONS #
##################
def determine_grid_size(
    start_coords: StandardCoord,
    end_coords: StandardCoord,
    obstacle_coords: Optional[List[StandardCoord]] = None,
    margin: int = 5,
) -> Tuple[int, ...]:
    """Determines the bounding box of the search space.

    Args:
        - start_coords: (x, y, z) position of the source node.
        - end_coords: (x, y, z) position of the target node
        - obstacle_coords: list of coordinates that have already been occupied as part of previous operations.
        - margin: the margin to leave beyond the bounding box made by start_coord and end_coords.

    Returns:
        - min_x, max_x, min_y, max_y, min_z, max_z: min and max coordinates for all axes in 3D space.

    """

    all_coords = [start_coords, end_coords]
    if obstacle_coords:
        all_coords.extend(obstacle_coords)

    min_x = min(coord[0] for coord in all_coords) - margin
    max_x = max(coord[0] for coord in all_coords) + margin
    min_y = min(coord[1] for coord in all_coords) - margin
    max_y = max(coord[1] for coord in all_coords) + margin
    min_z = min(coord[2] for coord in all_coords) - margin
    max_z = max(coord[2] for coord in all_coords) + margin

    return min_x, max_x, min_y, max_y, min_z, max_z


def gen_tent_tgt_kinds(
    tgt_zx_type: str, overwrite_tgt_kind: Optional[str] = None
) -> List[str]:
    """Returns all possible valid kinds/types for a given ZX type,
    typically needed when a new block is being added to the 3D space,
    as each ZX type can be fulfilled with more than one block types/kinds.

    Args:
        - tgt_zx_type: the ZX type of the target node.
        - overwrite_tgt_kind: a specific block/pipe type/kind to return irrespective of ZX type,
            used when the target block was already placed as part of previous operations and therefore already has an assigned kind.

    Returns:
        - family: a list of applicable types/kinds for the given ZX type.

    """

    if overwrite_tgt_kind:
        return [overwrite_tgt_kind]

    # NODE TYPE FAMILIES
    X = ["xxz", "xzx", "zxx"]
    Z = ["xzz", "zzx", "zxz"]
    BOUNDARY = ["ooo"]
    SIMPLE = ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"]
    HADAMARD = ["zxoh", "xzoh", "oxzh", "ozxh", "xozh", "zoxh"]

    if tgt_zx_type in ["X", "Z"]:
        family = X if tgt_zx_type == "X" else Z
    elif tgt_zx_type == "O":
        family = BOUNDARY
    elif tgt_zx_type == "SIMPLE":
        family = SIMPLE
    elif tgt_zx_type == "HADAMARD":
        family = HADAMARD
    else:
        return [tgt_zx_type]

    return family


def get_coords_occupied_by_blocks(preexistent_structure: List[StandardBlock]):
    """Converts a series of blocks into a list of all coordinates occupied by the blocks.

    Args:
        - preexistent_structure: a list of blocks and pipes that altogether make a space-time diagram.

    Returns:
        - list(obstacle_coords): a list of coordinates taken by the blocks and pipes in the preexistent_structure.

    """

    obstacle_coords = set()

    if not preexistent_structure:
        return []

    # Add first block's coordinates
    first_block = preexistent_structure[0]
    if first_block:
        first_block_coords = first_block[0]
        obstacle_coords.add(first_block_coords)

    # Iterate from the second node
    for i, block in enumerate(preexistent_structure):

        if i > 0:

            current_node = preexistent_structure[i]
            prev_node = preexistent_structure[i - 1]

            if current_node and prev_node:
                current_node_coords, current_node_type = current_node
                prev_node_coords, prev_node_type = prev_node

                # Add current node's coordinates
                obstacle_coords.add(current_node_coords)

                if "o" in current_node_type:
                    cx, cy, cz = current_node_coords
                    px, py, pz = prev_node_coords
                    extended_coords = None

                    if cx == px + 1:
                        extended_coords = (cx + 1, cy, cz)
                    elif cx == px - 1:
                        extended_coords = (cx - 1, cy, cz)
                    elif cy == py + 1:
                        extended_coords = (cx, cy + 1, cz)
                    elif cy == py - 1:
                        extended_coords = (cx, cy - 1, cz)
                    elif cz == pz + 1:
                        extended_coords = (cx, cy, cz + 1)
                    elif cz == pz - 1:
                        extended_coords = (cx, cy, cz - 1)

                    if extended_coords:
                        obstacle_coords.add(extended_coords)

    return list(obstacle_coords)
