"""Constrain and heuristics used to gauge and select between paths.

Usage:
    Call any function/class from a separate script.

"""

import networkx as nx
import numpy as np

from topologiq.core.graph_manager.utils import get_node_degree
from topologiq.core.pathfinder.spatial import get_manhattan
from topologiq.utils.classes import CubeBeams, StandardCoord


#######################
# PATH / BEAM CLASHES #
#######################
def check_path_to_beam_clashes(
    nx_g: nx.Graph,
    src_id: int,
    tgt_id: int,
    coords_in_path: list[StandardCoord],
    beams_broken_by_path: int | None = None,
    priority_ids: list[int | None] | None = None,
    strict: bool = True,
    twin_mode: bool = False,
    ids_to_twin: tuple[int] | None = None,
) -> tuple[bool, int, list[int | None] | None]:
    """Determine if placement triggers critical multi-beam clashes.

    This function checks if a given placement blocks more beams that tolerable.
    A single beam being broken is not necessarily a problem, as some cubes can lose
    some beams. However, if a new placement breaks more beams than what any one cube
    can lose, it will become impossible to make all connections for the said cube.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        src_id: The ID of the current source cube.
        tgt_id: The ID of the potential target cube.
        coords_in_path: All coords in the current path.
        beams_broken_by_path (optional): A pre-existent count of broken beams.
        priority_ids (optional): A list of spider IDs with one or another kind of conflict.
        strict (optional): Whether to perform a strict or loose check.
        twin_mode (optional): Skip calculation for the src cube, when used to create a twin of a given cube.
        ids_to_twin (optional): Cube IDs flagged as potentially problematic, passed only when function is used to create a twin node.

    Returns:
        clash: False if no critical beam clashed found, else True.
        beams_broken_by_path: Accumulated total number of beams for which path creates some kind of problem.
        priority_ids: Cube IDs flagged as potentially problematic.

    """

    # Initialise beams broken by path if it hasn't been initialised
    beams_broken_by_path = 0 if beams_broken_by_path is None else beams_broken_by_path

    # Defaults
    clash = False
    priority_ids = priority_ids if priority_ids else []

    # Loop over all cubes in 3D space
    for cube_id in nx_g.nodes():
        if ids_to_twin and cube_id in ids_to_twin:
            if ids_to_twin and src_id in ids_to_twin:
                if ids_to_twin.index(src_id) < ids_to_twin.index(cube_id):
                    continue

        # Use infinite beams if `strict`
        if nx_g.nodes[cube_id]["beams"] if strict else nx_g.nodes[cube_id]["beams_short"]:
            beams_to_check = (
                nx_g.nodes[cube_id]["beams"] if strict else nx_g.nodes[cube_id]["beams_short"]
            )

            cube_broken_count = 0
            if twin_mode and ids_to_twin and nx_g.neighbors(cube_id):
                cube_neighs = []
                for i in list(nx_g.neighbors(cube_id)):
                    if i not in ids_to_twin:
                        cube_neighs.append(i)
                    elif ids_to_twin.index(src_id) < ids_to_twin.index(i):
                        cube_neighs.append(i)
                cube_degree = len(cube_neighs)
            else:
                cube_degree = get_node_degree(nx_g, cube_id)
            cube_pending_edges = cube_degree - nx_g.nodes[cube_id]["completed"]

            if twin_mode and ids_to_twin and cube_id in ids_to_twin:
                if ids_to_twin.index(src_id) > ids_to_twin.index(cube_id):
                    cube_pending_edges = 0

            for beam in beams_to_check:
                if any([beam.contains(coord) for coord in coords_in_path]):
                    beams_broken_by_path += 1
                    cube_broken_count += 1

            # Append to priority IDs for all cubes with problems
            # Flip check if even ONE cube has problems
            src_tgt_adjust = 1 if (cube_id in [src_id, tgt_id] and src_id != tgt_id) else 0
            if len(beams_to_check) - cube_broken_count + src_tgt_adjust < min(
                cube_pending_edges, 1
            ):
                priority_ids.append(cube_id)
                clash = True

    return clash, beams_broken_by_path, priority_ids


def check_tgt_beam_clashes(
    nx_g: nx.Graph,
    src_id: int,
    tgt_id: int,
    tgt_beams: CubeBeams,
    tgt_beams_short: CubeBeams,
    tgt_degree: int,
    beams_broken_by_path: int = 0,
    strict: bool = True,
    **kwargs,
) -> bool:
    """Determine if placement triggers critical multi-beam clashes.

    This function checks if a given placement blocks more beams that tolerable.
    A single beam being broken is not necessarily a problem, as some cubes can lose
    some beams. However, if a new placement breaks more beams than what any one cube
    can lose, it will become impossible to make all connections for the said cube.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        src_id: The ID of the current source cube.
        tgt_id: The ID of the potential target cube.
        tgt_beams: The beams of the potential target cube.
        tgt_beams_short: The short beams of the potential target cube.
        tgt_degree: The number of neighbours of the potential target cube.
        beams_broken_by_path (optional): A pre-existent count of broken beams.
        strict (optional): Whether to perform a strict or loose check.
        **kwargs: !

    Returns:
        clash: False if no critical beam clashed found, else True.
        beams_broken_by_path: Accumulated total number of beams for which path creates some kind of problem.

    """

    # Aux params
    clash = False
    tgt_beams_to_check = tgt_beams if strict else tgt_beams_short

    # Check tgt against beams of each other cube in 3D space
    if tgt_beams_to_check:
        for cube_id in nx_g.nodes():
            # Reset trackers on every cube irrespectively
            tgt_clash_tracker = np.array([False for _ in tgt_beams_short])
            clash = False
            cube_clash_count = 0

            # Count clashes for cubes with beams
            if cube_id not in (src_id, tgt_id) and (
                nx_g.nodes[cube_id]["beams"] if strict else nx_g.nodes[cube_id]["beams_short"]
            ):
                cube_degree = get_node_degree(nx_g, cube_id)
                cube_pending_edges = cube_degree - nx_g.nodes[cube_id]["completed"]
                for cube_beam in nx_g.nodes[cube_id]["beams_short"]:
                    if nx_g.nodes[cube_id]["beams_short"]:
                        intersections = [
                            tgt_beam.intersects(cube_beam, kwargs["beams_len_short"])
                            for tgt_beam in tgt_beams_short
                        ]
                        tgt_clash_tracker = tgt_clash_tracker + np.array(intersections)
                        cube_clash_count += 1 if any(intersections) else 0

                src_tgt_adjust = 1 if cube_id in [src_id, tgt_id] else 0
                if (
                    len(nx_g.nodes[cube_id]["beams"]) - cube_clash_count + src_tgt_adjust
                    < cube_pending_edges
                ):
                    beams_broken_by_path += 1
                    clash = True

        if len(tgt_beams) - sum(tgt_clash_tracker) < tgt_degree - 1:
            beams_broken_by_path += 1
            clash = True

    return clash, beams_broken_by_path


##############
# LOOK-AHEAD #
##############
def check_need_for_twins(
    nx_g: nx.Graph,
    src_id: int,
    tgt_id: int,
    taken: list[StandardCoord],
    priority_ids: list[int | None] | None = None,
    strict: bool = True,
) -> list[int | None] | None:
    """Determine if there is a need to create twin cubes.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        src_id: The ID of the current source cube.
        tgt_id: The ID of the potential target cube.
        taken: All coords taken by all paths.
        priority_ids (optional): A list of spider IDs with one or another kind of conflict.
        strict (optional): Whether to perform a strict or loose check.

    Returns:
        priority_ids: Cube IDs flagged as potentially problematic.

    """

    _, _, priority_ids = check_need_twins_path(
        nx_g, src_id, tgt_id, taken, priority_ids=[], strict=strict
    )
    priority_ids = check_need_twins_beams(
        nx_g, (src_id, tgt_id), priority_ids=priority_ids, strict=strict
    )
    priority_ids = list(set(priority_ids))

    return priority_ids


def check_need_twins_path(
    nx_g: nx.Graph,
    src_id: int,
    tgt_id: int,
    taken: list[StandardCoord],
    beams_broken_by_path: int | None = None,
    priority_ids: list[int | None] | None = None,
    strict: bool = True,
) -> tuple[bool, int, list[int | None] | None]:
    """Determine if placement triggers critical multi-beam clashes.

    This function checks if a given placement blocks more beams that tolerable.
    A single beam being broken is not necessarily a problem, as some cubes can lose
    some beams. However, if a new placement breaks more beams than what any one cube
    can lose, it will become impossible to make all connections for the said cube.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        src_id: The ID of the current source cube.
        tgt_id: The ID of the potential target cube.
        taken: All coords taken by all paths.
        beams_broken_by_path (optional): A pre-existent count of broken beams.
        priority_ids (optional): A list of spider IDs with one or another kind of conflict.
        strict (optional): Whether to perform a strict or loose check.
        ids_to_twin (optional): Cube IDs flagged as potentially problematic, passed only when function is used to create a twin node.

    Returns:
        clash: False if no critical beam clashed found, else True.
        beams_broken_by_path: Accumulated total number of beams for which path creates some kind of problem.
        priority_ids: Cube IDs flagged as potentially problematic.

    """

    # Initialise beams broken by path if it hasn't been initialised
    beams_broken_by_path = 0 if beams_broken_by_path is None else beams_broken_by_path

    # Default to False in case no cubes with beams exist
    clash = False
    priority_ids = priority_ids if priority_ids else []

    # Loop over all cubes in 3D space
    for cube_id in nx_g.nodes():
        if nx_g.nodes[cube_id]["beams"] if strict else nx_g.nodes[cube_id]["beams_short"]:
            # Use infinite or short beams according to `strict`
            beams_to_check = (
                nx_g.nodes[cube_id]["beams"] if strict else nx_g.nodes[cube_id]["beams_short"]
            )

            cube_broken_count = 0
            cube_degree = get_node_degree(nx_g, cube_id)
            cube_pending_edges = cube_degree - nx_g.nodes[cube_id]["completed"]

            for beam in beams_to_check:
                if any([beam.contains(coord) for coord in taken]):
                    beams_broken_by_path += 1
                    cube_broken_count += 1

            # Append to priority IDs for all cubes with problems
            # Flip check if even ONE cube has problems
            src_tgt_adjust = 1 if (cube_id in [src_id, tgt_id] and src_id != tgt_id) else 0

            if len(beams_to_check) - cube_broken_count + src_tgt_adjust < cube_pending_edges:
                priority_ids.append(cube_id)
                clash = True

    return clash, beams_broken_by_path, priority_ids


def check_need_twins_beams(
    nx_g: nx.Graph,
    last_src_tgt_ids: tuple[int, int],
    priority_ids: list[int | None] | None = None,
    strict: bool = True,
) -> list[int | None] | None:
    """Determine if there are critical beam clashes for any given node.

    Args:
        nx_g: A nx_graph initially like the input ZX graph but with 3D-amicable structure, updated regularly.
        last_src_tgt_ids: The current or last (src_id, tgt_id) pair processed.
        priority_ids (optional): A list of spider IDs with one or another kind of conflict.
        strict (optional): Whether to perform a strict or loose check.
        twin_mode (optional): Skip calculation for the src cube, when used to create a twin of a given cube.

    Returns:
        priority_ids: Cube IDs flagged as potentially problematic.

    """

    # Check beams of all cubes against target beams
    priority_ids = priority_ids if priority_ids else []
    for out_id in nx_g.nodes():
        if (
            nx_g.nodes[out_id]["beams"] if strict else nx_g.nodes[out_id]["beams_short"]
        ) and nx_g.nodes[out_id]["coords"]:
            out_coords = nx_g.nodes[out_id]["coords"]
            out_beams = nx_g.nodes[out_id]["beams"] if strict else nx_g.nodes[out_id]["beams_short"]
            out_beams_num = len(out_beams)
            out_degree = get_node_degree(nx_g, out_id)
            out_pending = out_degree - nx_g.nodes[out_id]["completed"]

            out_tracker = np.array([False for beam in out_beams])
            for in_id in nx_g.nodes():
                if in_id != out_id:
                    inner_count = 0  # Tracks each "in" cube
                    if (
                        nx_g.nodes[out_id]["beams"] if strict else nx_g.nodes[in_id]["beams_short"]
                    ) and nx_g.nodes[in_id]["coords"]:
                        in_coords = nx_g.nodes[in_id]["coords"]
                        in_beams = (
                            nx_g.nodes[in_id]["beams"]
                            if strict
                            else nx_g.nodes[in_id]["beams_short"]
                        )
                        in_beams_num = len(in_beams)
                        in_degree = get_node_degree(nx_g, in_id)
                        in_pending = in_degree - nx_g.nodes[in_id]["completed"]
                        manhattan_between = get_manhattan(out_coords, in_coords)

                        for beam in in_beams:
                            broken_beams = [
                                beam.intersects(out_beam, manhattan_between)
                                for out_beam in out_beams
                            ]
                            out_tracker = out_tracker + np.array(broken_beams)

                            inner_count += sum(broken_beams)
                            in_pending = (
                                0 if in_id in priority_ids + list(last_src_tgt_ids) else in_pending
                            )
                            if in_beams_num - inner_count < in_pending:
                                priority_ids.append(in_id)

            out_pending = 0 if out_id in priority_ids + list(last_src_tgt_ids) else out_pending
            if out_beams_num - sum(out_tracker) < out_pending:
                priority_ids.append(out_id)

    return priority_ids
