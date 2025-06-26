import sys
from pathlib import Path
from collections import deque
from utils.utils import is_move_allowed
from utils.constraints import get_valid_next_kinds
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.classes import StandardCoord, StandardBlock


#########################
# MAIN WORKFLOW MANAGER #
#########################
def run_bfs_for_all_potential_target_nodes(
    source_node: StandardBlock,
    target_node_zx_type: str,
    distance: int,
    max_distance: int = 30,
    attempts_per_distance: int = 10,
    overwrite_target_node: Tuple[Optional[StandardCoord], Optional[str]] = (None, None),
    occupied_coords: List[StandardCoord] = [],
):
    """
    Runs BFS on a loop until path is found within predetermined distance of source node or max distance is reached.

    Args:
        - source_node: source node's coordinates (tuple) and type (str).
        - target_node_zx_type: ZX type of the target node, taken from a ZX chart.
        - distance: current allowed distance between source and target nodes.
        - max_distance: maximum allowed distance between source and target nodes.
        - attempts_per_distance: number of random target positions to try at each distance.

    """

    # HELPER VARIABLES
    all_paths_from_round: List[Optional[List[StandardBlock]]] = []
    start_coords, _ = source_node
    min_path_length: Optional[int] = None
    path_found: bool = False
    length: Optional[int] = None
    path = None
    found_path_at_current_distance: bool = False
    overwrite_target_coords, overwrite_target_type = overwrite_target_node

    obstacle_coords: List[StandardCoord] = occupied_coords[:]
    if obstacle_coords:
        if start_coords in obstacle_coords:
            obstacle_coords.remove(start_coords)

    min_x_bb, max_x_bb, min_y_bb, max_y_bb, min_z_bb, max_z_bb = determine_grid_size(
        start_coords,
        overwrite_target_coords if overwrite_target_coords else (0, 0, 0),
        obstacle_coords=obstacle_coords,
    )

    # PATH FINDING LOOP W. MULTIPLE BFS ROUNDS WITH INCREASING DISTANCE FROM SOURCE NODE
    for attempt in range(attempts_per_distance):

        # Break if path is found in previous run of loop
        if distance > max_distance or path_found:
            break

        # Generate tentative position for target using the new function
        tentative_target_position = generate_tentative_target_position(
            source_node,
            min_x_bb,
            max_x_bb,
            min_y_bb,
            max_y_bb,
            min_z_bb,
            max_z_bb,
            obstacle_coords=obstacle_coords,
            overwrite_target_coords=overwrite_target_node[0],
        )

        if tentative_target_position is None:
            continue

        # Generate all possible target types at tentative position
        potential_target_types = generate_tentative_target_types(
            target_node_zx_type,
            overwrite_target_type=(
                overwrite_target_type if overwrite_target_type else None
            ),
        )

        # Find paths to all potential target kinds
        for potential_target_type in potential_target_types:
            target_node: StandardBlock = (
                tentative_target_position,
                potential_target_type,
            )
            candidate_path_found, candidate_length, candidate_path = bfs_extended_3d(
                source_node,
                target_node,
                forbidden_cords=obstacle_coords,
            )

            # If path found, keep shortest path of round
            if candidate_path_found:
                path_found = True
                all_paths_from_round.append(candidate_path)
                if min_path_length is None or candidate_length < min_path_length:
                    min_path_length = candidate_length
                    length = candidate_length
                    path = candidate_path

        # Inform user of outcome at this distance
        if found_path_at_current_distance:
            path_found = True

    # Return boolean for success of path finding, lenght of winner path, and winner path
    return path_found, length, path, all_paths_from_round


##################################
# CORE PATHFINDER BFS OPERATIONS #
##################################
def bfs_extended_3d(
    source_node: StandardBlock,
    target_node: StandardBlock,
    forbidden_cords: List[StandardCoord] = [],
):

    # Unpack information for source and target nodes
    start_coords, _ = source_node
    end_coords, end_type = target_node

    if start_coords in forbidden_cords:
        forbidden_cords.remove(start_coords)
    if end_coords in forbidden_cords:
        forbidden_cords.remove(end_coords)

    queue = deque([source_node])
    visited = {tuple(source_node): 0}
    path_length = {tuple(source_node): 0}
    path = {tuple(source_node): [source_node]}

    start_x, start_y, start_z = [int(x) for x in start_coords]
    end_x, end_y, end_z = [int(x) for x in end_coords]
    initial_manhattan_distance = (
        abs(start_x - end_x) + abs(start_y - end_y) + abs(start_z - end_z)
    )

    while queue:
        current_node_info = queue.popleft()
        current_coords, current_type = current_node_info
        x, y, z = [int(x) for x in current_coords]

        current_manhattan_distance = abs(x - end_x) + abs(y - end_y) + abs(z - end_z)
        if current_manhattan_distance > 6 * initial_manhattan_distance:
            print(f"- Terminated.")
            return False, -1, None

        if current_coords == end_coords and current_type == end_type:
            print(f"- SUCCESS!")

            return (
                True,
                path_length[current_node_info],
                path[current_node_info],
            )

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
            next_x, next_y, next_z = x + dx * scale, y + dy * scale, z + dz * scale
            next_coords = (next_x, next_y, next_z)
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

            possible_next_types = get_valid_next_kinds(
                current_coords, current_type, next_coords
            )

            for next_type in possible_next_types:
                next_node_info: StandardBlock = (next_coords, next_type)

                if (
                    next_coords not in current_path_coords
                    and next_coords not in forbidden_cords
                    and (intermediate_pos is None or next_coords != intermediate_pos)
                ):
                    new_path_length = path_length[current_node_info] + 1
                    if (
                        next_node_info not in visited
                        or new_path_length < visited[next_node_info]
                    ):
                        visited[next_node_info] = new_path_length
                        queue.append(next_node_info)
                        path_length[next_node_info] = new_path_length
                        path[next_node_info] = path[current_node_info] + [
                            next_node_info
                        ]

    return False, -1, None


##################
# AUX OPERATIONS #
##################
def generate_tentative_target_position(
    source_node: StandardBlock,
    min_x: int,
    max_x: int,
    min_y: int,
    max_y: int,
    min_z: int,
    max_z: int,
    obstacle_coords: Optional[List[StandardCoord]] = None,
    overwrite_target_coords: Optional[StandardCoord] = None,
) -> StandardCoord | None:
    """Generates a tentative coordinate for next target, favouring closer targets favoured, and checking position's validity.
    Note. Function is not really used in current flow (returns overwrite_target_coords). Here in case of future need.
    """


    if overwrite_target_coords:
        return overwrite_target_coords

    obstacle_coords = obstacle_coords if obstacle_coords else []
    source_coords, _ = source_node
    sx, sy, sz = source_coords

    # Level 1: Single Axis Displacement (+/- 3)
    potential_targets_level_1 = [
        (sx + 3, sy, sz),
        (sx - 3, sy, sz),
        (sx, sy + 3, sz),
        (sx, sy - 3, sz),
        (sx, sy, sz + 3),
        (sx, sy, sz - 3),
    ]
    for tx, ty, tz in potential_targets_level_1:
        if min_x <= tx <= max_x and min_y <= ty <= max_y and min_z <= tz <= max_z:
            if (tx, ty, tz) not in obstacle_coords and is_move_allowed(
                source_coords, (tx, ty, tz)
            ):
                print(f"=>> Returning potential target at Level 1: {(tx, ty, tz)}")
                return (tx, ty, tz)

    # Level 2: Two Axis Displacement (+/- 3 on two axes)
    potential_targets_level_2 = []
    for dx in [-3, 3]:
        for dy in [-3, 3]:
            potential_targets_level_2.extend(
                [(sx + dx, sy + dy, sz), (sx + dy, sy + dx, sz)]
            )
        for dz in [-3, 3]:
            potential_targets_level_2.extend(
                [(sx + dx, sy, sz + dz), (sx + dz, sy, sz + dx)]
            )
        for dy in [-3, 3]:
            for dz in [-3, 3]:
                potential_targets_level_2.extend(
                    [(sx, sy + dy, sz + dz), (sx, sy + dz, sz + dy)]
                )

    # Remove duplicates
    potential_targets_level_2 = list(set(potential_targets_level_2))

    for tx, ty, tz in potential_targets_level_2:
        if min_x <= tx <= max_x and min_y <= ty <= max_y and min_z <= tz <= max_z:
            if (tx, ty, tz) not in obstacle_coords and is_move_allowed(
                source_coords, (tx, ty, tz)
            ):
                print(f"=>> Returning potential target at Level 2: {(tx, ty, tz)}")
                return (tx, ty, tz)

    # Level 3: Three Axis Displacement (+/- 3 on all three axes)
    potential_targets_level_3 = []
    for dx in [-3, 3]:
        for dy in [-3, 3]:
            for dz in [-3, 3]:
                if dx != 0 or dy != 0 or dz != 0:  # Exclude the source itself
                    potential_targets_level_3.append((sx + dx, sy + dy, sz + dz))

    for tx, ty, tz in potential_targets_level_3:
        if min_x <= tx <= max_x and min_y <= ty <= max_y and min_z <= tz <= max_z:
            if (tx, ty, tz) not in obstacle_coords and is_move_allowed(
                source_coords, (tx, ty, tz)
            ):
                print(f"=>> Returning potential target at Level 3: {(tx, ty, tz)}")
                return (tx, ty, tz)

    print(
        "=> Could not generate a valid tentative target within the prioritized distances."
    )

    return None


def determine_grid_size(
    start_coords: StandardCoord,
    end_coords: StandardCoord,
    obstacle_coords: Optional[List[StandardCoord]] = None,
    margin: int = 5,
) -> Tuple[int, ...]:
    """Determines the bounding box of the search space."""


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


def generate_tentative_target_types(
    target_node_zx_type: str, overwrite_target_type: Optional[str] = None
) -> List[str]:
    """Returns all possible valid blockgraph types for a given ZX type."""


    if overwrite_target_type:
        return [overwrite_target_type]

    # NODE TYPE FAMILIES
    X = ["xxz", "xzx", "zxx"]
    Z = ["xzz", "zzx", "zxz"]
    SIMPLE = ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"]
    HADAMARD = ["zxoh", "xzoh", "oxzh", "ozxh", "xozh", "zoxh"]

    if target_node_zx_type in ["X", "Z"]:
        family = X if target_node_zx_type == "X" else Z
    elif target_node_zx_type == "SIMPLE":
        family = SIMPLE
    elif target_node_zx_type == "HADAMARD":
        family = HADAMARD
    else:
        return [target_node_zx_type]

    return family


def get_coords_occupied_by_blocks(preexistent_structure: List[StandardBlock]):
    """Converts a series of blocks into a list of all coordinates occupied by the blocks."""


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
                    else:
                        print("Could not determine extended coords for 'o' type.")

    return list(obstacle_coords)
