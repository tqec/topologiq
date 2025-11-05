"""
Contains objects and functions used by several visualisation files. 

This file contains auxiliary objects that are used to create different kinds of 
visualisations. Do NOT call anything in this file directly.

Usage:
    Call the specific object from a separate script.
"""

from typing import Any, List, Tuple
from topologiq.utils.classes import StandardCoord
from topologiq.utils.grapher import get_faces, get_vertices
from topologiq.utils.utils_pathfinder import check_is_exit
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#############
# CONSTANTS #
#############
node_hex_map = {
    "xxz": ["#d7a4a1", "#d7a4a1", "#b9cdff"],
    "xzz": ["#d7a4a1", "#b9cdff", "#b9cdff"],
    "xzx": ["#d7a4a1", "#b9cdff", "#d7a4a1"],
    "zzx": ["#b9cdff", "#b9cdff", "#d7a4a1"],
    "zxx": ["#b9cdff", "#d7a4a1", "#d7a4a1"],
    "zxz": ["#b9cdff", "#d7a4a1", "#b9cdff"],
    "zxo": ["#b9cdff", "#d7a4a1", "gray"],
    "xzo": ["#d7a4a1", "#b9cdff", "gray"],
    "oxz": ["gray", "#d7a4a1", "#b9cdff"],
    "ozx": ["gray", "#b9cdff", "#d7a4a1"],
    "xoz": ["#d7a4a1", "gray", "#b9cdff"],
    "zox": ["#b9cdff", "gray", "#d7a4a1"],
    "zxoh": ["#b9cdff", "#d7a4a1", "gray"],
    "xzoh": ["#d7a4a1", "#b9cdff", "gray"],
    "oxzh": ["gray", "#d7a4a1", "#b9cdff"],
    "ozxh": ["gray", "#b9cdff", "#d7a4a1"],
    "xozh": ["#d7a4a1", "gray", "#b9cdff"],
    "zoxh": ["#b9cdff", "gray", "#d7a4a1"],
    "xxx": ["red", "red", "red"],
    "yyy": ["green", "green", "green"],
    "zzz": ["blue", "blue", "blue"],
    "X": ["red", "red", "red"],
    "Y": ["green", "green", "green"],
    "Z": ["blue", "blue", "blue"],
    "aaa": ["silver", "silver", "silver"],
    "hadamard": ["yellow"],
    "beam": ["yellow"],
}


# FUNCTIONS
def render_block(
    ax: Any,
    node_id: int,
    coords: Tuple[int, int, int],
    size: list[float],
    node_type: str,
    node_hex_map: dict[str, list[str]],
    alpha: float = 1.0,
    edge_col: None | str = None,
    border_width: float = 0.5,
    current_nodes: Tuple[int, int] | None = None,
    taken: List[StandardCoord] = [],
):
    """Renders a regular (non-'h') block.

    Args:
        - ax: Matplotlib's 3D subplot object.
        - node_id: the ID of the node
        - coords: (x, y, z) coordinates of the block.
        - size: (size_x, size_y, size_z) of the block.
        - node_type: block's kind.
        - node_hex_map: map of (HEX) colours for block.
        - edge_col: color for the edges of blocks.
        - border_width: width for borders of block.
        - current_nodes: optional parameter to pass ID of the last (src, tgt) nodes connected by the algorithm,
        - taken: list of coordinates occupied by any blocks/pipes placed as a result of previous operations.
    """

    x, y, z = coords
    size_x, size_y, size_z = size

    vertices = get_vertices(x, y, z, size_x, size_y, size_z)
    faces = get_faces(vertices)

    # ADD COLORS AS PER MAP
    cols = node_hex_map.get(node_type, ["gray"] * 3)
    face_cols = [cols[2]] * 2 + [cols[1]] * 2 + [cols[0]] * 2

    # JOIN
    poly_collection = Poly3DCollection(
        faces,
        facecolors=face_cols if "_visited" not in node_type else "red",
        linewidths=border_width,
        edgecolors=edge_col,
        alpha=alpha,
        picker=True,
        label=node_id,
    )

    diffs = [
        (2, 0, 0),
        (0, 0, 2),
        (0, 0, -2),
        (0, 2, 0),
        (-2, 0, 0),
        (0, -2, 0),
    ]

    for d in diffs:
        label_pos = (
            coords[0] + d[0],
            coords[1] + d[1],
            coords[2] + d[2],
        )

        if (
            check_is_exit(coords, node_type, label_pos) is not True
            or node_type == "ooo"
        ) and label_pos not in taken:
            ax.text(
                label_pos[0],
                label_pos[1],
                label_pos[2],
                s=f"{node_id}: {node_type}",
                color="black",
                visible=True if current_nodes and node_id in current_nodes else False,
            )

            ax.quiver(
                coords[0],
                coords[1],
                coords[2],
                label_pos[0] - coords[0],
                label_pos[1] - coords[1],
                label_pos[2] - coords[2],
                color="black",
                lw=1,
                label=node_id,
                visible=True if current_nodes and node_id in current_nodes else False,
            )

            break

    # ADD TO PLOT
    ax.add_collection3d(poly_collection)