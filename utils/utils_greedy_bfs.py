import numpy as np
from typing import Tuple, List, Optional

from utils.classes import StandardCoord, StandardBeam, NodeBeams, StandardBlock


def is_exit(src_c: StandardCoord, src_k: Optional[str], tgt_c: StandardCoord) -> bool:
    """Checks if a face is an exit by matching exit markers in the block/pipe's symbolic name
    and an displacement array symbolising the direction of the target block/pipe.

    Args:
        - src_c: (x, y, z) coordinates for the current block/pipe.
        - src_k: current block's/pipe's kind.
        - tgt_c: coordinates for the target block/pipe.

    Returns:
        - bool:
            - True: face is an exist
            - False: face is not an exit.

    """

    src_k = src_k.lower()[:3] if isinstance(src_k, str) else ""
    kind_3D = [src_k[0], src_k[1], src_k[2]]

    if "o" in kind_3D:
        marker = "o"
    else:
        marker = [i for i in set(kind_3D) if kind_3D.count(i) == 2][0]

    exit_idxs = [i for i, char in enumerate(kind_3D) if char == marker]
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
    taken: List[StandardCoord],
    all_beams: List[NodeBeams],
    beams_len: int,
) -> Tuple[bool, StandardBeam]:
    """Checks if a face (typically an exit: call after verifying face is exit) is unobstructed.

    Args:
        - src_c: (x, y, z) coordinates for the current block/pipe.
        - tgt_c: coordinates for the target block/pipe.
        - taken: list of coordinates taken by any blocks/pipes placed as a result of previous operations.
        - all_beams: list of coordinates taken by the beams of all blocks in original ZX-graph
        - beams_len: int representing how long does each beam spans from its originating block.

    Returns:
        - bool:
            - True: face is unobstructed
            - False: face is obstructed.

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
    taken: List[StandardCoord],
    all_beams: List[NodeBeams],
    beams_len: int,
) -> Tuple[int, NodeBeams]:
    """Finds the number of unobstructed exits for any given block/pipe by calling other functions that use the
    block's/pipe's symbolic name to determine if each face is or is not
    and whether the said exit is unobstructed.

    Args:
        - src_c: (x, y, z) coordinates for the block/pipe of interest.
        - src_k: kind/type of the block/pipe of interest.
        - taken: list of coordinates taken by any blocks/pipes placed as a result of previous operations.
        - all_beams: list of coordinates taken by the beams of all blocks in original ZX-graph
        - beams_len: int representing how long does each beam spans from its originating block.

    Returns:
        - unobstrexits_n: the number of unobstructed exist for the block/pipe of interest.
        - n_beams: the beams emanating from the block/pipe of interest.

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

        if is_exit(src_c, src_k, tgt_c):
            is_unobstr, exit_beam = check_unobstr(
                src_c, tgt_c, taken, all_beams, beams_len
            )
            if is_unobstr:
                unobstr_exits_n += 1
                n_beams.append(exit_beam)

    return unobstr_exits_n, n_beams


def is_move_allowed(src_c: StandardCoord, tgt_c: StandardCoord) -> bool:
    """Uses a Manhattan distance to quickly checks if a potential (source, target) combination
    is possible given the standard size of a block and pipe and that all blocks are followed by a pipe.

    Args:
        - src_c: (x, y, z) coordinates for the originating block.
        - tgt_c: (x, y, z) coordinates for the potential placement of the target block.

    Returns:
        - bool:
            - True: move is theoretically possible,
            - False: move is not theoretically possible.

    """

    sx, sy, sz = src_c
    nx, ny, nz = tgt_c
    manhattan = abs(nx - sx) + abs(ny - sy) + abs(nz - sz)
    return manhattan % 3 == 0


def gen_tent_tgt_coords(
    src_c: StandardCoord,
    max_manhattan: int = 3,
    taken: List[StandardCoord] = [],
) -> List[StandardCoord]:
    """Generates a number of potential placement positions for target node.

    Args:
        - src_c: (x, y, z) coordinates for the originating block.
        - max_manhattan: Max. (Manhattan) distance between origin and target blocks.
        - taken: a list of coordinates already taken by previous operations.

    Returns:
        - tent_coords: a list of tentative target coordinates that make good candidates for placing the target block.

    """

    # EXTRACT SOURCE COORDS
    sx, sy, sz = src_c
    tent_coords = {}

    # SINGLE MOVES
    tgts = [
        (sx + 3, sy, sz),
        (sx - 3, sy, sz),
        (sx, sy + 3, sz),
        (sx, sy - 3, sz),
        (sx, sy, sz + 3),
        (sx, sy, sz - 3),
    ]
    tent_coords[3] = [t for t in tgts if t not in taken]

    # MANHATTAN 6
    if max_manhattan > 3:
        tent_coords[6] = []
        for dx, dy, dz in [c for c in tent_coords[3]]:
            tgts = [
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
                    for t in tgts
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
            tgts = [
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
                    for t in tgts
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
        - new_beams: list of beams without any beams where any of the coordinates in the path of the beam overlap with taken coords.
    """

    try:
        new_beams = []
        for beams in all_beams:
            iter_beams = [
                beam for beam in beams if all([coord not in taken for coord in beam])
            ]
            if iter_beams:
                new_beams.append(iter_beams)
    except:
        new_beams = all_beams

    return new_beams


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
    for pth in edge_pths.values():
        max_id = max(max_id, pth["src_tgt_ids"][0], pth["src_tgt_ids"][1])
    nxt_id = max_id + 1

    for pth in edge_pths.values():

        idxd_pth = {}
        key_1, key_2 = pth["src_tgt_ids"]
        pth_nodes = pth["pth_nodes"]

        idxd_pth[key_1] = pth_nodes[0]

        for i in range(1, len(pth_nodes) - 1):
            mid_node = pth_nodes[i]
            idxd_pth[nxt_id] = mid_node
            nxt_id += 1

        if len(pth_nodes) > 1:
            idxd_pth[key_2] = pth_nodes[-1]

        idx_pths[(key_1, key_2)] = idxd_pth

    final_edges = {}
    for orig_key, pth_id_value_map in idx_pths.items():
        n_ids = list(pth_id_value_map.keys())
        b_ids = []
        e_ids = []

        for i in range(len(n_ids)):
            if i % 2 == 0:
                b_ids.append(n_ids[i])
            else:
                e_ids.append(n_ids[i])

        if len(b_ids) >= 2:
            for i in range(len(b_ids) - 1):
                n1 = b_ids[i]
                n2 = b_ids[i + 1]
                e_type = pth_id_value_map[e_ids[i]][1]

                final_edges[(n1, n2)] = [e_type, orig_key]

    lat_nodes: dict[int, StandardBlock] = {}
    lat_edges: dict[Tuple[int, int], List[str]] = {}
    for pth in idx_pths.values():
        keys = list(pth.keys())
        i = 0
        for key, info in pth.items():
            if i % 2 == 0:
                lat_nodes[key] = info
            else:
                e_key = (keys[i - 1], keys[i + 1])
                e_type = info[1]
                lat_edges[e_key] = [e_type, final_edges[e_key][1]]
            i += 1

    return lat_nodes, lat_edges


def rot_o_kind(k: str) -> str:
    """Rotates a pipe around its length by using the exit marker in its symbolic name to create a rotational matrix,
    which is then used to rotate the original name with a symbolic multiplication.

    Args:
        - k: the kind of the pipe that needs rotation.

    Returns:
        - rot_k: a kind with the rotation incorporated into the new name.

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
    """Quickly flips a Hadamard for the opposite Hadamard with length on the same axis.

    Args:
        - k: the kind of the Hadamard that needs inverting.

    Returns:
        - k_2: the kind of the corresponding/inverted Hadamard.


    """
    # List of hadamard equivalences
    equivs = {"zxoh": "xzoh", "xozh": "zoxh", "oxzh": "ozxh"}

    # Match to equivalent block given direction
    if k in equivs.keys():
        new_k = equivs[k]
    else:
        inv_equivs = {v: k for k, v in equivs.items()}
        new_k = inv_equivs[k]

    # Return revised kind
    return new_k
