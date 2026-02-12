"""Creation/generation utils to assist the primary graph managemer BFS.

Usage:
    Call any function/class from a separate script.

"""

from topologiq.utils.classes import StandardCoord


def gen_tent_tgt_coords(
    src_c: StandardCoord,
    max_manhattan: int = 3,
    taken: list[StandardCoord] = [],
) -> list[StandardCoord]:
    """Generate a number of potential placement positions for target node.

    Args:
        src_c: The (x, y, z) coordinates for the originating block.
        max_manhattan: Max. (Manhattan) distance between origin and target blocks.
        taken: A list of coordinates already taken by previous operations.

    Returns:
        all_coords_at_distance: A list of tentative target coordinates that make good candidates for placing the target block.

    """

    # EXTRACT SOURCE COORDS
    sx, sy, sz = src_c
    base_for_next_layer = []
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
    base_for_next_layer = [t for t in tgts]

    # MANHATTAN 6
    if max_manhattan > 3:
        tent_coords[6] = []
        for dx, dy, dz in [c for c in base_for_next_layer]:
            tgts = [
                (dx + 3, dy, dz),
                (dx - 3, dy, dz),
                (dx, dy + 3, dz),
                (dx, dy - 3, dz),
                (dx, dy, dz + 3),
                (dx, dy, dz - 3),
            ]
            tent_coords[6].extend([t for t in tgts if t not in taken and t != src_c])
            base_for_next_layer.extend([t for t in tgts])

    # MANHATTAN 9
    if max_manhattan > 6:
        tent_coords[9] = []
        for dx, dy, dz in [c for c in base_for_next_layer]:
            tgts = [
                (dx + 3, dy, dz),
                (dx - 3, dy, dz),
                (dx, dy + 3, dz),
                (dx, dy - 3, dz),
                (dx, dy, dz + 3),
                (dx, dy, dz - 3),
            ]
            tent_coords[9].extend([t for t in tgts if t not in taken and t != src_c])
            base_for_next_layer.extend([t for t in tgts])

    # > MANHATTAN 9
    if max_manhattan > 9:
        tent_coords[max_manhattan] = []
        num_loops = int((max_manhattan - 9) / 3)

        for _ in [i+1 for i in range(num_loops)]:
            for dx, dy, dz in [c for c in base_for_next_layer]:
                tgts = [
                    (dx + 3, dy, dz),
                    (dx - 3, dy, dz),
                    (dx, dy + 3, dz),
                    (dx, dy - 3, dz),
                    (dx, dy, dz + 3),
                    (dx, dy, dz - 3),
                ]
                tent_coords[max_manhattan].extend([t for t in tgts if t not in taken and t != src_c])
                base_for_next_layer.extend([t for t in tgts])

    all_coords_at_distance = tent_coords[min(max_manhattan, 15)]
    return all_coords_at_distance
