"""Constrain and heuristics used to gauge and select between paths.

Usage:
    Call any function/class from a separate script.

"""

import networkx as nx
import numpy as np

from topologiq.utils.classes import CubeBeams, StandardCoord


##############
# LOOK-AHEAD #
##############
def check_beam_clashes(
    bgraph: nx.Graph,
    tgt_beams: CubeBeams,
    tgt_beams_short: CubeBeams,
    all_beams: CubeBeams,
    all_beams_short: CubeBeams,
    curr_src_id: int,
    curr_tgt_id: int,
    coords_in_path: list[StandardCoord],
    strict: bool = False,
    twin_mode: bool = False,
    ids_to_twin: list[int] = [],
) -> tuple[bool, int]:
    """Determine if placement triggers critical multi-beam clashes.

    This function checks if a given placement blocks more beams that tolerable.
    A single beam being broken is not necessarily a problem, as some cubes can lose
    some beams. However, if a new placement breaks more beams than what any one cube
    can lose, it will become impossible to make all connections for the said cube.

    Args:
        bgraph: The BlockGraph currently being built.
        tgt_beams: The beams of the potential target cube.
        tgt_beams_short: The short beams of the potential target cube.
        all_beams: The beams for all the cubes in blockgraph that need beams.
        all_beams_short: The short beams for all the cubes in blockgraph that need beams.
        curr_src_id: The ID of the current source cube.
        curr_tgt_id: The ID of the current target cube.
        coords_in_path: All coords in the current path.
        strict (optional): Whether to perform a strict or loose check.
        twin_mode (optional): True if the current edge is part of a twin creation cycle.
        ids_to_twin (optional): A pre-existing list of IDs that need twins.

    Returns:
        clash: False if no critical beam clashed found, else True.
        beams_broken_by_path: Accumulated total number of beams for which path creates some kind of problem.

    """

    # Aux params
    clash = False
    beams_broken_by_path = 0
    tgt_beams_to_check = tgt_beams if strict else tgt_beams_short
    all_beams_to_check = all_beams if strict else all_beams_short

    # Check target against beams of each other cube in 3D space
    tgt_clash_tracker = np.array([False for _ in tgt_beams_to_check])
    for cube_id in bgraph.nodes():
        # Loop only cares for target,
        # reset trackers on every cube irrespectively
        clash = False
        cube_clash_count = 0

        # Count clashes for cubes with beams
        if cube_id not in (curr_src_id, curr_tgt_id) and cube_id in all_beams_to_check:
            for cube_beam in all_beams_to_check[cube_id]:
                # Only consider up and down beams on boundary nodes
                if (
                    bgraph.nodes[curr_tgt_id]["zx_block"].zx_type == "O"
                    or bgraph.nodes[cube_id]["zx_block"].zx_type == "O"
                ):
                    continue

                intersections = [tgt_beam.intersects(cube_beam) for tgt_beam in tgt_beams_to_check]
                tgt_clash_tracker = tgt_clash_tracker + np.array(intersections)
                cube_clash_count += 1 if any(intersections) else 0

            src_tgt_adjust = 1 if cube_id in [curr_src_id, curr_tgt_id] else 0
            in_threshold = (
                min(1, bgraph.nodes[cube_id]["completions"]["pending"])
                if twin_mode or not strict
                else bgraph.nodes[cube_id]["completions"]["pending"]
            )
            if len(all_beams_to_check[cube_id]) - cube_clash_count + src_tgt_adjust < in_threshold:
                beams_broken_by_path += 1
                clash = True

    out_threshold = bgraph.nodes[curr_tgt_id]["completions"]["pending"] - 1
    if len(tgt_beams_to_check) - sum(tgt_clash_tracker) < out_threshold:
        beams_broken_by_path += 1
        clash = True

    # Early return if a clash is detected
    if clash:
        return clash, beams_broken_by_path

    # Loop over all cubes in 3D space looking for conflicts with path
    for cube_id in bgraph.nodes():
        # Loop is cumulative,
        # skip rest of sequence if clash is detected
        if clash:
            break

        # Check if cube has beams
        if cube_id in all_beams_to_check:
            # Counter for number of beams broken
            cube_broken_count = 0

            for beam in all_beams_to_check[cube_id]:
                if any([beam.contains(coord) for coord in coords_in_path]):
                    beams_broken_by_path += 1
                    cube_broken_count += 1

            # Append to priority IDs for all cubes with problems
            # Flip check if even ONE cube has problems
            threshold = (
                min(1, bgraph.nodes[cube_id]["completions"]["pending"])
                if twin_mode
                else bgraph.nodes[cube_id]["completions"]["pending"]
            )
            src_tgt_adjust = (
                1 if (cube_id in [curr_src_id, curr_tgt_id] and curr_src_id != curr_tgt_id) else 0
            )

            if len(all_beams_to_check[cube_id]) - cube_broken_count + src_tgt_adjust < threshold:
                clash = True
                break

    return clash, beams_broken_by_path


def check_beam_clashes_for_twins(
    bgraph: nx.Graph,
    all_beams: CubeBeams,
    all_beams_short: CubeBeams,
    curr_src_id: int,
    curr_tgt_id: int,
    taken: set[StandardCoord],
    ids_to_twin: list[int] = [],
    strict: bool = True,
):
    """Determine if the number of beam clashes is such that a twin is required for any given cube.

    Args:
        bgraph: The BlockGraph currently being built.
        all_beams: The beams for all the cubes in blockgraph that need beams.
        all_beams_short: The short beams for all the cubes in blockgraph that need beams.
        curr_src_id: The ID of the current source cube.
        curr_tgt_id: The ID of the current target cube.
        taken: The set of taken coordinates.
        ids_to_twin (optional): A pre-existing list of IDs that need twins.
        strict (optional): Whether to perform a strict or loose check.

    """

    # Select short or full beams as appropriate
    all_beams_to_check = all_beams if strict else all_beams_short

    # Check beams of all cubes against target beams
    for out_id, out_beams in all_beams_to_check.items():
        if bgraph.nodes[out_id]["coords"]:
            # Axis-aware tracker for broken beams
            out_tracker = np.array([False for beam in out_beams])

            # Check beams against beams of other cubes
            for in_id, in_beams in all_beams_to_check.items():
                if in_id != out_id:
                    inner_count = 0
                    if bgraph.nodes[in_id]["coords"]:
                        for beam in in_beams:
                            broken_beams = [
                                beam.intersects(out_beam, short_beams=False)
                                for out_beam in out_beams
                            ]
                            out_tracker = out_tracker + np.array(broken_beams)
                            inner_count += sum(broken_beams)

                            in_pending = (
                                0
                                if in_id in [*ids_to_twin, curr_src_id, curr_tgt_id]
                                else bgraph.nodes[in_id]["completions"]["pending"]
                            )

                            if len(in_beams) - inner_count < in_pending:
                                ids_to_twin.add(in_id)

            # Also check beams against taken
            broken_beams = [
                any([out_beam.contains(coord) for coord in taken]) for out_beam in out_beams
            ]
            out_tracker = out_tracker + np.array(broken_beams)
            out_pending = (
                0 if out_id in [*ids_to_twin] else bgraph.nodes[out_id]["completions"]["pending"]
            )

            if len(out_beams) - sum(out_tracker) < out_pending:
                ids_to_twin.add(out_id)

    return ids_to_twin


###########
# TIDY UP #
###########
def prune_beams(
    bgraph: nx.Graph, beams: CubeBeams, beams_short: CubeBeams, taken: set[StandardCoord]
):
    """Remove beams broken by recent placements.

    Args:
        bgraph: The BlockGraph currently being built.
        beams: The full beams of the BlockGraphManager.
        beams_short: The short beams of the BlockGraphManager.
        taken: The set of taken coordinates.

    Returns:
        beams: The full beams of the BlockGraphManager.
        beams_short: The short beams of the BlockGraphManager.

    """

    # Undertake all pruning within a fail-safe TRY block
    try:
        for n_id in bgraph.nodes():
            # Do not bother if node not in primary beam tracker (irrelevant or completed node)
            if n_id not in beams:
                pass

            # Eliminate any recently completed nodes from both beam trackers
            elif bgraph.nodes[n_id]["completions"]["pending"] <= 0:
                del beams[n_id]
                del beams_short[n_id]

            # Prune node still needing completion
            else:
                # Infinite beams
                if beams[n_id]:
                    new_beams = []
                    for single_beam in beams[n_id]:
                        if not any([single_beam.contains(coord) for coord in taken]):
                            new_beams += [single_beam]
                    if new_beams:
                        beams[n_id] = new_beams
                    else:
                        del beams[n_id]

                # Short beams
                if beams_short[n_id]:
                    new_beams_short = []
                    for single_beam_short in beams_short[n_id]:
                        if not any([single_beam_short.contains(coord) for coord in taken]):
                            new_beams_short += [single_beam_short]
                    if new_beams_short:
                        beams_short[n_id] = new_beams_short
                    else:
                        del beams_short[n_id]

    # Fail silently and hope for the best
    except (IndexError, ValueError, LookupError, KeyError):
        pass

    return beams, beams_short
