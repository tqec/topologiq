"""Symbolic operations that provide heuristics used for pathfinding.

Usage:
    Call any function/class from a separate script.

"""

import numpy as np

from topologiq.core.blocks import ZXBlock
from topologiq.utils.classes import (
    BeamAxisComponent,
    CubeBeams,
    SingleBeam,
    StandardCoord,
)


#########################
# BLOCK TRANSFORMATIONS #
#########################
def rotate_pipe(kind: str, move: StandardCoord) -> str:
    """Rotate a pipe around its length.

    This function enables pipe rotation by using the exit marker in their kind
    to create a rotational matrix, which is then used to rotate the original kind
    using symbolic multiplication.

    Args:
        kind: the kind of the block that needs rotation.
        move: The (x, y, z) displacement between current and target position.

    Returns:
        rot_k: a kind with the rotation incorporated into the new name.

    """

    h_flag = False
    if "H" in kind:
        h_flag = True
        kind.replace("H", "")

    # Build rotation matrix based on the direction of the move vector
    idxs = [0, 1, 2]
    idxs.remove(int(np.nonzero(move)[0]))

    new_matrix = {
        int(np.nonzero(move)[0]): np.eye(3, dtype=int)[int(np.nonzero(move)[0])],
        idxs[0]: np.eye(3, dtype=int)[idxs[1]],
        idxs[1]: np.eye(3, dtype=int)[idxs[0]],
    }

    rot_matrix = np.array([new_matrix[0], new_matrix[1], new_matrix[2]])

    # Rotate kind
    rot_k = ""
    for r in rot_matrix:
        entry = ""
        for j, ele in enumerate(r):
            entry += abs(int(ele)) * kind[j]
        rot_k += entry

    if h_flag:
        rot_k += "H"

    return rot_k


###################
# BEAM GENERATION #
###################
def check_exits_add_beams(
    zx_block: ZXBlock,
    src_coords: StandardCoord,
    taken: set[StandardCoord],
    coords_in_path: list[StandardCoord],
) -> tuple[int, CubeBeams, CubeBeams]:
    """Find the number of unobstructed exits for an arbitrary block and attach beams to them.

    Args:
        zx_block: The ZX block being checked.
        src_coords: The (x, y, z) coordinates for the ZX block.
        taken: The coordinates taken by any blocks placed as a result of previous operations.
        coords_in_path: The coordinates taken by the path under current evaluation.

    Returns:
        unobstr_exits_n: The number of unobstructed exist for the block.
        cube_beams: The beams emanating from the block.
        cube_beams_short: The short beams emanating from the block.

    """

    unobstr_exits_n = 0
    cube_beams = []
    cube_beams_short = []

    diffs = zx_block.get_move_vectors

    for d in diffs:
        tgt_coords = (
            src_coords[0] + d[0],
            src_coords[1] + d[1],
            src_coords[2] + d[2],
        )

        is_unobstr, single_beam, single_beam_short = check_unobstructed(
            src_coords, tgt_coords, taken
        )
        if is_unobstr and not any([single_beam.contains(coord) for coord in coords_in_path]):
            unobstr_exits_n += 1
            cube_beams.append(single_beam)
            cube_beams_short.append(single_beam_short)

    # Reset number of unobstructed exits
    unobstr_exits_n = len(cube_beams)
    return unobstr_exits_n, cube_beams, cube_beams_short


def check_unobstructed(
    src_c: StandardCoord,
    tgt_c: StandardCoord,
    taken: set[StandardCoord],
) -> tuple[bool, SingleBeam]:
    """Check if a face is unobstructed.

    This function should typically be called after verifying a face is exit.

    Args:
        src_c: The (x, y, z) coordinates for the current block/pipe.
        tgt_c: The coordinates for the target block/pipe.
        taken: The coordinates taken by any blocks/pipes placed as a result of previous operations.

    Returns:
        (bool): True if face is unobstructed else False.
        single_beam: If the face is unobstructed, its corresponding beam.

    """

    diffs = [target - source for source, target in zip(src_c, tgt_c)]
    diffs = [1 if d > 0 else -1 if d < 0 else 0 for d in diffs]

    x_start, x_end, x_direction = (
        src_c[0],
        src_c[0] if diffs[0] == 0 else diffs[0] * np.inf,
        diffs[0],
    )
    y_start, y_end, y_direction = (
        src_c[1],
        src_c[1] if diffs[1] == 0 else diffs[1] * np.inf,
        diffs[1],
    )
    z_start, z_end, z_direction = (
        src_c[2],
        src_c[2] if diffs[2] == 0 else diffs[2] * np.inf,
        diffs[2],
    )

    single_beam = SingleBeam(
        BeamAxisComponent(x_start, x_end, x_direction),
        BeamAxisComponent(y_start, y_end, y_direction),
        BeamAxisComponent(z_start, z_end, z_direction),
    )

    x_start, x_end, x_direction = (
        src_c[0],
        src_c[0] if diffs[0] == 0 else src_c[0] + diffs[0] * 9,
        diffs[0],
    )
    y_start, y_end, y_direction = (
        src_c[1],
        src_c[1] if diffs[1] == 0 else src_c[1] + diffs[1] * 9,
        diffs[1],
    )
    z_start, z_end, z_direction = (
        src_c[2],
        src_c[2] if diffs[2] == 0 else src_c[2] + diffs[2] * 9,
        diffs[2],
    )

    single_beam_short = SingleBeam(
        BeamAxisComponent(x_start, x_end, x_direction),
        BeamAxisComponent(y_start, y_end, y_direction),
        BeamAxisComponent(z_start, z_end, z_direction),
    )

    if not taken:
        return True, single_beam, single_beam_short

    if any([single_beam.contains(coord) for coord in taken]):
        return False, single_beam, single_beam_short

    return True, single_beam, single_beam_short
