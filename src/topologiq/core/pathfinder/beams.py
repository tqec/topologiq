"""Functions used for building and checking beams.

Usage:
    Call any function/class from a separate script.

Note:
    Not all functions in this file are in use. Some functions are historical
    but not yet removed as they can provide clues to improve the performance of
    the current implementation.

"""

import numpy as np

from topologiq.core.beams import CubeBeams
from topologiq.utils.classes import StandardCoord

##################
# STANDARD EDGES #
##################
# Beam management for standard edges currently happens in the graph manager.
# There is a need to either move standard edges to the pathfinder or
# cross-edges beam management to the graph manager, or place them both somewhere
# common.
# At the moment, I favour moving cross edges out of the pathfinder.
# It is cheaper to test a full path once at the end than to check many times
# on each move. Not everyone agrees with this assessment.


###############
# CROSS EDGES #
###############
def check_critical_beams(
    critical_beams: dict[int, tuple[StandardCoord, int, CubeBeams, CubeBeams]],
    full_path_coords: list[StandardCoord],
    nxt_coords: StandardCoord,
    tgt_coords: StandardCoord,
    src_tgt_ids: tuple[int, int],
) -> bool:
    """Check that move does not break any beams of cubes that need all their exits.

    Args:
        critical_beams: A dictionary containing beam information for cubes with beams.
        full_path_coords: All coordinates occupied by current path.
        nxt_coords: The coordinates being checked as potential next position to place a block.
        tgt_coords: The final "target" coordinates at which path should arrive.
        src_tgt_ids: The exact IDs of the source and target cubes.

    Return:
        (bool): True if move clears all checks, False otherwise.

    """

    # Check each cube against all other cubes
    for out_id, (_, out_min_exit_num, _, out_beams_short) in critical_beams.items():
        # Track outer beams in a way that remembers which beam is which
        out_clash_tracker = np.array([False for _ in out_beams_short])

        # Look for clashes against path
        broken_beams = [
            any([out_beam.contains(coord) for coord in full_path_coords])
            for out_beam in out_beams_short
        ]

        out_clash_tracker = out_clash_tracker + np.array(broken_beams)

        # Look for clashes against the beams of other cubes
        if any(broken_beams) and nxt_coords == tgt_coords and out_id not in src_tgt_ids:
            for in_id, (_, in_min_exit_num, _, in_beams_short) in critical_beams.items():
                in_clash_tracker = 0
                for in_beam in in_beams_short:
                    intersections = [out_beam.intersects(in_beam) for out_beam in out_beams_short]
                    # out_clash_tracker = out_clash_tracker + np.array(intersections)
                    in_clash_tracker += any(intersections)

                src_tgt_adjust = 1 if in_id in src_tgt_ids else 0
                in_pending = 1 if in_id not in src_tgt_ids else in_min_exit_num
                if len(in_beams_short) + src_tgt_adjust - in_clash_tracker < min(in_pending, 1):
                    return False

        # Determine if clashes are within tolerance
        src_tgt_adjust = 1 if out_id in src_tgt_ids else 0
        out_pending = 1 if out_id not in src_tgt_ids else out_min_exit_num
        if len(out_beams_short) + src_tgt_adjust - sum(out_clash_tracker) < min(out_pending, 1):
            return False

    return True


###########################################################
# CROSS EDGES NOT CURRENTLY IN USED BUT NOT DISCARDED YET #
###########################################################
def split_critical_beams(
    critical_beams: dict[int, tuple[StandardCoord, int, CubeBeams, CubeBeams]],
    src_tgt_ids: tuple[int, int],
) -> tuple[
    dict[int, tuple[StandardCoord, int, CubeBeams, CubeBeams]],
    dict[int, tuple[StandardCoord, int, CubeBeams, CubeBeams]],
]:
    """Split critical beams into simple and verbose object containing different kinds of beams.

    This function separates the `critical_beams` object into a quickly iterable object containing
    coordinates of beams for nodes that needs absolutely all beams they have and a more verbose
    dictionary containing the beams for nodes that can lose some beams.

    Args:
        critical_beams: Beams considered critical for future operations.
        src_tgt_ids: The exact IDs of the source and target cubes.
        max_span: the longest edge of the bounding box, equivalent to largest beam needed to clear box.

    Returns:
        unbreakable_beams: The joint beam coordinates for nodes that need all beams they currently have.
        negotiable_beams: A minified `critical_beams` object containing beams for nodes that can lose some beams.

    """

    unbreakable_beams = {}
    negotiable_beams = {}
    for node_id, (
        node_coords,
        min_exit_num,
        node_beams,
        node_beams_short,
    ) in critical_beams.items():
        if node_id not in src_tgt_ids and min_exit_num == len(node_beams):
            unbreakable_beams[node_id] = (
                node_coords,
                min_exit_num,
                [beam for beam in node_beams],
                [beam for beam in node_beams_short],
            )
        else:
            negotiable_beams[node_id] = (
                node_coords,
                min_exit_num,
                [beam for beam in node_beams],
                [beam for beam in node_beams_short],
            )

    return unbreakable_beams, negotiable_beams


def check_unbreakable_beams(
    unbreakable_beams: dict[int, tuple[StandardCoord, int, CubeBeams, CubeBeams]],
    full_path_coords: list[StandardCoord],
    src_tgt_ids: tuple[int, int],
) -> tuple[bool, list[StandardCoord]]:
    """Check that move does not break any beams of cubes that need all their exits.

    Args:
        unbreakable_beams: The joint beam coordinates for nodes that need all beams they currently have.
        full_path_coords: All coordinates occupied by current path.
        src_tgt_ids: The exact IDs of the source and target cubes.

    Return:
        (bool): True if move clears all checks, False otherwise.
        clash_coords: A list of coordinates where unbreakable beams get broken.

    """

    clash_coords = []
    for node_id, (_, _, _, node_beams_short) in unbreakable_beams.items():
        broken_beams = 0
        for single_beam in node_beams_short:
            clash_coords = [coord for coord in full_path_coords if single_beam.contains(coord)]
            if clash_coords:
                # Reject if beam is of nodes other src and tgt
                if node_id not in src_tgt_ids:
                    return False, clash_coords

                # Reject if more than one beam of src and tgt cubes is broken
                if broken_beams == 1:
                    return False, clash_coords
                # Add to broken beams if dealing with src or tgt cube
                broken_beams += 1

    return True, clash_coords
