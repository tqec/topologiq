"""Auxiliary functions for the management of the pathfinder.

Usage:
    Call any function/class from a separate script.

"""

from topologiq.utils.manhattan import get_max_manhattan


########
# INIT #
########
def gen_exit_conditions(
    src_coords,
    tent_coords,
    taken,
    max_span: int,
    cross_edge: bool,
    min_success_rate: int,
    special_target_kind: bool = False,
) -> tuple[int, int, int]:
    """Calculate conditions that need to be met to exit the pathfinder BFS.

    Args:
        src_coords: The source coords for the current edge.
        tent_coords: A list of tentative coords for final edge target cube.
        taken: The set of  taken coordinates.
        max_span: the longest edge of the bounding box, equivalent to largest axes needed for box.
        cross_edge: A boolean flag to determine if search is a primary or `cross_edge` search.
        min_success_rate: The minimum percentage of edges (relative to total possible) after which to exit.
        special_target_kind (optional): True if final target is a Y, conditional, or cultivation block.

    """

    # Manhattan distances to skip iterations and exit BFS in the event of failure
    if cross_edge or special_target_kind:
        tgts_to_fill = 1
        src_tgt_manhattan = get_max_manhattan(src_coords, tent_coords)
        if cross_edge:
            max_manhattan = max(
                get_max_manhattan(src_coords, taken) * 2,
                max_span,
            )
        else:
            max_manhattan = src_tgt_manhattan + 6

    else:
        tgts_to_fill = int(len(tent_coords) * min_success_rate / 100)
        max_manhattan = get_max_manhattan(src_coords, tent_coords) * 2
        src_tgt_manhattan = max_manhattan

    src_tgt_manhattan = get_max_manhattan(src_coords, tent_coords)
    tgts_to_fill = min(100, tgts_to_fill)

    return tgts_to_fill, max_manhattan, src_tgt_manhattan


##########
# MANAGE #
##########
def gen_tent_tgt_kinds(tgt_zx_type: str, tgt_kind: str | None = None) -> list[str]:
    """Generate all possible valid kinds for a given ZX type.

    This function takes the ZX type of a potential new block in a 3D path and returns
    a list of block (cube or pipe) kinds that could fulfill that ZX type. Rather than
    seeing the function as creating kinds to check, the function should be seen as
    reducing the number of kinds to check in any given iteration.

    Args:
        tgt_zx_type: The ZX type of the target spider/cube.
        tgt_kind (optional): A specific kind used to override the function.

    Returns:
        kind_family: a list of applicable kinds for the given ZX type.

    """

    # Return override value if present
    if tgt_kind:
        return [tgt_kind]

    # Get family of kinds corresponding to the ZX type of target
    if tgt_zx_type in ["X", "Z"]:
        kind_family = ["zzx", "zxz", "xzz"] if tgt_zx_type == "X" else ["xxz", "xzx", "zxx"]
    elif tgt_zx_type == "O":
        kind_family = ["ooo"]
    elif tgt_zx_type == "SIMPLE":
        kind_family = ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"]
    elif tgt_zx_type == "HADAMARD":
        kind_family = ["zxoh", "xzoh", "oxzh", "ozxh", "xozh", "zoxh"]
    else:
        return [tgt_zx_type]

    return kind_family


############
# MISC/AUX #
############
def check_run_mode(src_coords, taken, tgt_coords, tent_tgt_kinds):
    """Check if edge is standard or cross and update taken accordingly."""

    second_pass = False

    if src_coords in taken:
        taken.remove(src_coords)
    if len(tgt_coords) == 1 and len(tent_tgt_kinds) == 1:
        second_pass = True
        if tgt_coords[0] in taken:
            taken.remove(tgt_coords[0])

    return second_pass, taken
