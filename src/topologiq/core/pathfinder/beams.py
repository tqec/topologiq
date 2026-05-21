"""Functions used for building and checking beams.

Usage:
    Call any function/class from a separate script.

Note:
    Not all functions in this file are in use. Some functions are historical
    but not yet removed as they can provide clues to improve the performance of
    the current implementation.

"""

import networkx as nx
import numpy as np

from topologiq.utils.classes import CubeBeams, StandardCoord


#######################
# UNIFIED BEAM CHECKS #
#######################
def check_beams(
    bgraph: nx.Graph,
    beams: CubeBeams,
    beams_short: CubeBeams,
    curr_src_id: int,
    curr_tgt_id: int,
    nxt_coords: StandardCoord,
    tent_coords: list[StandardCoord],
    curr_path_coords: list[StandardCoord],
    strict: bool = False,
) -> bool:
    """Check that move does not break any beams of cubes that need all their exits.

    Args:
        bgraph: The BlockGraph currently being built..
        beams: The beams for all the cubes in blockgraph that need beams..
        beams_short: The short beams for all the cubes in blockgraph that need beams..
        curr_src_id: The ID of the current source cube.
        curr_tgt_id: The ID of the current target cube.
        nxt_coords: The coordinates being checked as potential next position to place a block.
        tent_coords: The final "target" coordinates at which path should arrive.
        curr_path_coords: The coordinates for the current path.
        strict (optional): Whether to check against full or short beams.

    Return:
        (bool): True if move clears all checks, False otherwise.

    """

    # Select short of full beams as appropriate
    all_beams = beams if strict else beams_short

    # Check each cube against all other cubes
    for out_id, out_beams in all_beams.items():
        # Track outer beams in a way that remembers which beam is which
        out_clash_tracker = np.array([False for _ in out_beams])

        # Look for clashes against path
        broken_beams = [
            any([out_beam.contains(coord) for coord in curr_path_coords]) for out_beam in out_beams
        ]

        out_clash_tracker = out_clash_tracker + np.array(broken_beams)

        # Look for clashes against the beams of other cubes
        # if (
        # any(broken_beams)
        # and nxt_coords in tent_coords
        # and out_id not in (curr_src_id, curr_tgt_id)
        # ):
        # for in_id, in_beams in all_beams.items():
        # Track inner beams in a way that remembers which beam is which
        # in_clash_tracker = 0

        # Look for inner clashes
        # for in_beam in in_beams:
        # intersections = [out_beam.intersects(in_beam, 9) for out_beam in out_beams]
        # out_clash_tracker = out_clash_tracker + np.array(intersections)
        # in_clash_tracker += any(intersections)

        # Determine if in clashes are within tolerance
        # src_tgt_adjust = 1 if in_id in (curr_src_id, curr_tgt_id) else 0
        # in_pending = (
        # 1 if src_tgt_adjust == 0 else bgraph.nodes[in_id]["completions"]["pending"]
        # )
        # if len(in_beams) + src_tgt_adjust - in_clash_tracker < min(in_pending, 1):
        # return False

        # Determine if out clashes are within tolerance
        src_tgt_adjust = 1 if out_id in (curr_src_id, curr_tgt_id) else 0
        out_pending = (
            min(1, bgraph.nodes[out_id]["completions"]["pending"])
            if src_tgt_adjust == 0
            else bgraph.nodes[out_id]["completions"]["pending"]
        )
        if len(out_beams) + src_tgt_adjust - sum(out_clash_tracker) < out_pending:
            return False

    return True
