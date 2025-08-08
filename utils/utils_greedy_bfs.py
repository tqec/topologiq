import numpy as np
from typing import Tuple, List, Optional

from utils.classes import StandardCoord, StandardBeam, NodeBeams, StandardBlock


def check_is_exit(
    src_coord: StandardCoord, src_kind: Optional[str], tgt_coord: StandardCoord
) -> bool:
    """Checks if a face is an exit by matching exit markers in the block/pipe's symbolic name
    and an displacement array symbolising the direction of the target block/pipe.

    Args:
        - src_coord: (x, y, z) coordinates for the current block/pipe.
        - src_kind: current block's/pipe's kind.
        - tgt_coord: coordinates for the target block/pipe.

    Returns:
        - bool:
            - True: face is an exist
            - False: face is not an exit.

    """

    src_kind = src_kind.lower()[:3] if isinstance(src_kind, str) else ""
    kind_3D = [src_kind[0], src_kind[1], src_kind[2]]

    if "o" in kind_3D:
        marker = "o"
    else:
        marker = [i for i in set(kind_3D) if kind_3D.count(i) == 2][0]

    exit_idxs = [i for i, char in enumerate(kind_3D) if char == marker]
    diffs = [tgt - src for src, tgt in zip(src_coord, tgt_coord)]

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
    src_coord: StandardCoord,
    tgt_coord: StandardCoord,
    taken: List[StandardCoord],
    all_beams: List[NodeBeams],
    beams_len: int,
) -> Tuple[bool, StandardBeam]:
    """Checks if a face (typically an exit: call after verifying face is exit) is unobstructed.

    Args:
        - src_coord: (x, y, z) coordinates for the current block/pipe.
        - tgt_coord: coordinates for the target block/pipe.
        - taken: list of coordinates taken by any blocks/pipes placed as a result of previous operations.
        - all_beams: list of coordinates taken by the beams of all blocks in original ZX-graph
        - beams_len: int representing how long does each beam spans from its originating block.

    Returns:
        - bool:
            - True: face is unobstructed
            - False: face is obstructed.

    """

    single_beam: StandardBeam = []

    diffs = [target - source for source, target in zip(src_coord, tgt_coord)]
    diffs = [1 if d > 0 else -1 if d < 0 else 0 for d in diffs]

    for i in range(1, beams_len):
        dx, dy, dz = (diffs[0] * i, diffs[1] * i, diffs[2] * i)
        single_beam.append((src_coord[0] + dx, src_coord[1] + dy, src_coord[2] + dz))

    if not taken:
        return True, single_beam

    for coord in single_beam:
        if coord in taken or coord in all_beams:
            return False, single_beam

    return True, single_beam


def check_for_exits(
    src_coords: StandardCoord,
    src_kind: str | None,
    taken: List[StandardCoord],
    all_beams: List[NodeBeams],
    beams_len: int,
) -> Tuple[int, NodeBeams]:
    """Finds the number of unobstructed exits for any given block/pipe by calling other functions that use the
    block's/pipe's symbolic name to determine if each face is or is not
    and whether the said exit is unobstructed.

    Args:
        - src_coords: (x, y, z) coordinates for the block/pipe of interest.
        - src_kind: kind/type of the block/pipe of interest.
        - taken: list of coordinates taken by any blocks/pipes placed as a result of previous operations.
        - all_beams: list of coordinates taken by the beams of all blocks in original ZX-graph
        - beams_len: int representing how long does each beam spans from its originating block.

    Returns:
        - unobstrexits_n: the number of unobstructed exist for the block/pipe of interest.
        - node_beams: the beams emanating from the block/pipe of interest.

    """

    unobstr_exits_n = 0
    node_beams: NodeBeams = []

    diffs = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    for d in diffs:
        tgt_coords = (
            src_coords[0] + d[0],
            src_coords[1] + d[1],
            src_coords[2] + d[2],
        )

        if check_is_exit(src_coords, src_kind, tgt_coords):
            is_unobstr, exit_beam = check_unobstr(
                src_coords, tgt_coords, taken, all_beams, beams_len
            )
            if is_unobstr:
                unobstr_exits_n += 1
                node_beams.append(exit_beam)

    return unobstr_exits_n, node_beams


def is_move_allowed(src_coords: StandardCoord, tgt_coords: StandardCoord) -> bool:
    """Uses a Manhattan distance to quickly checks if a potential (source, target) combination
    is possible given the standard size of a block and pipe and that all blocks are followed by a pipe.

    Args:
        - src_coords: (x, y, z) coordinates for the originating block.
        - tgt_coords: (x, y, z) coordinates for the potential placement of the target block.

    Returns:
        - bool:
            - True: move is theoretically possible,
            - False: move is not theoretically possible.

    """

    sx, sy, sz = src_coords
    nx, ny, nz = tgt_coords
    manhattan = abs(nx - sx) + abs(ny - sy) + abs(nz - sz)
    return manhattan % 3 == 0


def gen_tent_tgt_coords(
    src_coords: StandardCoord,
    max_manhattan: int = 3,
    taken: List[StandardCoord] = [],
) -> List[StandardCoord]:
    """Generates a number of potential placement positions for target node.

    Args:
        - src_coords: (x, y, z) coordinates for the originating block.
        - max_manhattan: Max. (Manhattan) distance between origin and target blocks.
        - taken: a list of coordinates already taken by previous operations.

    Returns:
        - tent_coords: a list of tentative target coordinates that make good candidates for placing the target block.

    """

    # EXTRACT SOURCE COORDS
    sx, sy, sz = src_coords
    tent_coords = {}

    # SINGLE MOVES
    targets = [
        (sx + 3, sy, sz),
        (sx - 3, sy, sz),
        (sx, sy + 3, sz),
        (sx, sy - 3, sz),
        (sx, sy, sz + 3),
        (sx, sy, sz - 3),
    ]
    tent_coords[3] = [t for t in targets if t not in taken]

    # MANHATTAN 6
    if max_manhattan > 3:
        tent_coords[6] = []
        for dx, dy, dz in [c for c in tent_coords[3]]:
            targets = [
                (dx + 3, dy, dz),
                (dx - 3, dy, dz),
                (dx, dy + 3, dz),
                (dx, dy - 3, dz),
                (dx, dy, dz + 3),
                (dx, dy, dz - 3),
            ]

            tent_coords[6].extend(
                [
                    t
                    for t in targets
                    if all(
                        [
                            abs(t[0]) - abs(sx) != 6,
                            abs(t[1]) - abs(sy) != 6,
                            abs(t[2]) - abs(sz) != 6,
                        ]
                    )
                    and sum(
                        [abs(t[0]) - abs(sx), abs(t[1]) - abs(sy), abs(t[2]) - abs(sz)]
                    )
                    == 6
                    and t not in taken
                ]
            )

    # MANHATTAN 9
    if max_manhattan > 6:
        tent_coords[9] = []
        for dx, dy, dz in [c for c in tent_coords[6]]:
            targets = [
                (dx + 3, dy, dz),
                (dx - 3, dy, dz),
                (dx, dy + 3, dz),
                (dx, dy - 3, dz),
                (dx, dy, dz + 3),
                (dx, dy, dz - 3),
            ]

            tent_coords[9].extend(
                [
                    t
                    for t in targets
                    if all(
                        [
                            abs(t[0]) - abs(sx) != 9,
                            abs(t[1]) - abs(sy) != 9,
                            abs(t[2]) - abs(sz) != 9,
                        ]
                    )
                    and sum(
                        [abs(t[0]) - abs(sx), abs(t[1]) - abs(sy), abs(t[2]) - abs(sz)]
                    )
                    == 9
                    and t not in taken
                ]
            )

    # RETURN ALL COORDS WITHIN DISTANCE
    return tent_coords[min(max_manhattan, 9)]


def prune_beams(
    all_beams: List[NodeBeams], taken: List[StandardCoord]
) -> List[NodeBeams]:
    """Removes beams that have already been broken, as these are no longer indicative of anything.

    Args:
        - all_beams: list of coordinates taken by the beams of all blocks in original ZX-graph
        - taken: list of coordinates taken by any blocks/pipes placed as a result of previous operations.

    Returns:
        - new_all_beams: list of beams without any beams where any of the coordinates in the path of the beam overlap with taken coords.

    """

    try:
        new_all_beams = []
        for node_beams in all_beams:
            new_node_beams = []
            for single_beam in node_beams:
                if all([coord not in taken for coord in single_beam]):
                    new_node_beams.append(single_beam)
            if new_node_beams:
                new_all_beams.append(new_node_beams)
    except:
        new_all_beams = all_beams

    return new_all_beams


def reindex_pth_dict(
    edge_pths: dict,
) -> Tuple[dict[int, StandardBlock], dict[Tuple[int, int], List[str]]]:
    """Distils an edge_pth object into a final list of nodes/blocks and edges/pipes for the space-time diagram.

    Args:
        - edge_pths: a dictionary containing a number of edge paths, i.e., full paths between two blocks, each path made of 3D blocks and pipes.

    Returns:
        - lat_nodes: the nodes/blocks of the resulting space-time diagram / lattice surgery (without redundant blocks)
        - lat_edges: the edges/pipes of the resulting space-time diagram / lattice surgery (without redundant pipes)

    """

    max_id = 0
    idx_pths = {}
    for edge_pth in edge_pths.values():
        max_id = max(max_id, edge_pth["src_tgt_ids"][0], edge_pth["src_tgt_ids"][1])
    next_id = max_id + 1

    for edge_pth in edge_pths.values():

        indexed_pth = {}
        start_key, end_key = edge_pth["src_tgt_ids"]
        nodes_in_pth = edge_pth["pth_nodes"]

        indexed_pth[start_key] = nodes_in_pth[0]

        for i in range(1, len(nodes_in_pth) - 1):
            intermediate_node = nodes_in_pth[i]
            indexed_pth[next_id] = intermediate_node
            next_id += 1

        if len(nodes_in_pth) > 1:
            indexed_pth[end_key] = nodes_in_pth[-1]

        idx_pths[(start_key, end_key)] = indexed_pth

    final_edges = {}
    for original_edge_key, pth_id_value_map in idx_pths.items():
        node_ids = list(pth_id_value_map.keys())
        block_ids = []
        edge_ids = []

        for i in range(len(node_ids)):
            if i % 2 == 0:
                block_ids.append(node_ids[i])
            else:
                edge_ids.append(node_ids[i])

        if len(block_ids) >= 2:
            for i in range(len(block_ids) - 1):
                node1 = block_ids[i]
                node2 = block_ids[i + 1]
                edge_type = pth_id_value_map[edge_ids[i]][1]

                final_edges[(node1, node2)] = [edge_type, original_edge_key]

    lat_nodes: dict[int, StandardBlock] = {}
    lat_edges: dict[Tuple[int, int], List[str]] = {}
    for pth in idx_pths.values():
        keys = list(pth.keys())
        i = 0
        for key, info in pth.items():
            if i % 2 == 0:
                lat_nodes[key] = info
            else:
                edge_key = (keys[i - 1], keys[i + 1])
                edge_type = info[1]
                lat_edges[edge_key] = [edge_type, final_edges[edge_key][1]]
            i += 1

    return lat_nodes, lat_edges


def rot_o_kind(kind: str) -> str:
    """Rotates a pipe around its length by using the exit marker in its symbolic name to create a rotational matrix,
    which is then used to rotate the original name with a symbolic multiplication.

    Args:
        - kind: the type/kind/name of the pipe that needs rotation.

    Returns:
        - rot_kind: a new type/kind/name with the rotation incorporated into the new name.

    """

    h_flag = False
    if "h" in kind:
        h_flag = True
        kind.replace("h", "")

    # Build rotation matrix based on placement of "o" node
    available_idxs = [0, 1, 2]
    available_idxs.remove(kind.index("o"))

    new_matrix = {
        kind.index("o"): np.eye(3, dtype=int)[kind.index("o")],
        available_idxs[0]: np.eye(3, dtype=int)[available_idxs[1]],
        available_idxs[1]: np.eye(3, dtype=int)[available_idxs[0]],
    }

    rot_matrix = np.array([new_matrix[0], new_matrix[1], new_matrix[2]])

    # Rotate kind
    rot_kind = ""
    for row in rot_matrix:
        entry = ""
        for j, element in enumerate(row):
            entry += abs(int(element)) * kind[j]
        rot_kind += entry

    if h_flag:
        rot_kind += "h"

    return rot_kind


def flip_hdm(kind: str) -> str:
    """Quickly flips a Hadamard for the opposite Hadamard with length on the same axis.

    Args:
        - kind: the type/kind/name of the Hadamard that needs inverting.

    Returns:
        - kind: the type/kind/name of the corresponding/inverted Hadamard.


    """
    # List of hadamard equivalences
    hdm_equivs = {"zxoh": "xzoh", "xozh": "zoxh", "oxzh": "ozxh"}

    # Match to equivalent block given direction
    if kind in hdm_equivs.keys():
        kind = hdm_equivs[kind]
    else:
        inv_equivs = {value: key for key, value in hdm_equivs.items()}
        kind = inv_equivs[kind]

    # Return revised kind
    return kind
