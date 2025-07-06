"""NetworkX / Matplotlib functions to create quick 3D visualisations of algorithmic progress into a blockgraph.
This file is an absolute mess at the moment. It works though.
"""

import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Annotated, Literal, Any, Tuple
from numpy.typing import NDArray

from utils.utils_greedy_bfs import rotate_o_types

# CONSTANTS
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
}


# MISC FUNCTIONS
def get_vertices(
    x: int, y: int, z: int, size_x: float, size_y: float, size_z: float
) -> Annotated[NDArray[np.float64], Literal[..., 3]]:
    """
    Calculates the coordinates of the eight vertices of a cuboid.

    Args:
        - x: x-coordinate of the centre of the cuboid.
        - y: y-coordinate of the centre of the cuboid.
        - z: z-coordinate of the centre of the cuboid.
        - size_x: length of the cuboid along the x-axis.
        - size_y: length of the cuboid along the y-axis.
        - size_z: length of the cuboid along the z-axis.

    Returns:
        numpy.ndarray: A NumPy array of shape (8, 3) where each row represents the
                       (x, y, z) coordinates of a vertex of the cuboid. The order
                       of the vertices is consistent with how Matplotlib's
                       Poly3DCollection expects them for defining faces.
    """

    half_size_x = size_x / 2
    half_size_y = size_y / 2
    half_size_z = size_z / 2
    return np.array(
        [
            [x - half_size_x, y - half_size_y, z - half_size_z],
            [x + half_size_x, y - half_size_y, z - half_size_z],
            [x + half_size_x, y + half_size_y, z - half_size_z],
            [x - half_size_x, y + half_size_y, z - half_size_z],
            [x - half_size_x, y - half_size_y, z + half_size_z],
            [x + half_size_x, y - half_size_y, z + half_size_z],
            [x + half_size_x, y + half_size_y, z + half_size_z],
            [x - half_size_x, y + half_size_y, z + half_size_z],
        ]
    )


def get_faces(vertices: Annotated[NDArray[np.float64], Literal[..., 3]]):
    """
    Defines the faces of a cuboid based on its vertices.

    Args:
        - vertices: A NumPy array of shape (8, 3) containing the
                                  coordinates of the cuboid's vertices, as returned
                                  by the `get_vertices` function. The order of
                                  vertices is assumed to be consistent with that
                                  function's output.

    Returns:
        list: A list of lists, where each inner list represents a face of the
              cuboid and contains the coordinates of the four vertices that form
              that face. The order of faces typically corresponds to:
              [bottom, top, front, back, right, left].
    """

    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
    ]


def render_block(
    ax: Any,
    position: Tuple[int, int, int],
    size: list[float],
    node_type: str,
    node_hex_map: dict[str, list[str]],
    alpha: float = 1.0,
    edge_col: None | str = None,
    line_width: float = 1.0,
):
    """
    Renders a regular (non-'h') node.

    Args:
        - ax: The Matplotlib 3D subplot object.
        - position: The (x, y, z) coordinates of the node.
        - size: The (size_x, size_y, size_z) of the node.
        - node_type: The type string of the node.
        - node_hex_map: The map of (HEX) colours for the nodes.
        - edge_col: The color of the edges.
    """

    x, y, z = position
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
        linewidths=1,
        edgecolors=edge_col,
        alpha=alpha,
    )

    # ADD TO PLOT
    ax.add_collection3d(poly_collection)


def render_edge(
    ax: Any,
    centre: NDArray[np.float64],
    size: list[float],
    face_cols: list[str],
    edge_col: str,
    alpha: float | int,
):

    # ESTABLISH CUBOID'S CENTRE & SIZE
    x, y, z = centre
    sx, sy, sz = size

    # DETERMINE VERTICES
    vertices = np.array(
        [
            [x - sx / 2, y - sy / 2, z - sz / 2],
            [x + sx / 2, y - sy / 2, z - sz / 2],
            [x + sx / 2, y + sy / 2, z - sz / 2],
            [x - sx / 2, y + sy / 2, z - sz / 2],
            [x - sx / 2, y - sy / 2, z + sz / 2],
            [x + sx / 2, y - sy / 2, z + sz / 2],
            [x + sx / 2, y + sy / 2, z + sz / 2],
            [x - sx / 2, y + sy / 2, z + sz / 2],
        ]
    )

    # ADD FACES
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ]
    face_list = [vertices[face] for face in faces]

    # MAKE COLLECTION
    poly = Poly3DCollection(
        face_list,
        facecolors=face_cols,
        edgecolors=edge_col,
        linewidths=1,
        alpha=alpha,
    )

    # ADD TO PLOT
    ax.add_collection3d(poly)


def visualise_3d_graph(
    graph: nx.Graph,
    hide_boundaries: bool = False,
    node_hex_map: dict[str, list[str]] = node_hex_map,
    save_to_file: bool = False,
    filename: str | None = None,
):

    # HELPER VARIABLES
    gray_hex = "gray"
    yellow_hex = "#e0e317"

    # CREATE FOUNDATIONAL MATPLOTLIB
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # GET POSITIONS AND TYPES
    node_positions = nx.get_node_attributes(graph, "pos")
    node_types = nx.get_node_attributes(graph, "type")
    edge_types = nx.get_edge_attributes(graph, "pipe_type")

    # RENDER CUBES (NODES)
    for node_id in graph.nodes():
        node_type = node_types.get(node_id)
        if (
            node_type
            and "o" not in node_type
            or (not hide_boundaries and node_type == "ooo")
        ):
            position = node_positions.get(node_id)
            if position:
                size = [1.0, 1.0, 1.0]
                edge_col = "black"
                alpha = 1.0

                if node_type == "ooo":
                    size = [0.9, 0.9, 0.9]
                    edge_col = "white"
                    alpha = 0.5

                if "*" in node_type:
                    edge_col = "white"
                    alpha = 0.7

                elif "_visited" in node_type:
                    size = [0.5, 0.5, 0.5]
                    edge_col = gray_hex
                    alpha = 0.8  # Adjust alpha as needed

                else:
                    render_block(
                        ax,
                        position,
                        size,
                        node_type[:3],
                        node_hex_map,
                        alpha=alpha,
                        edge_col=edge_col,
                        line_width=0.5 if "_visited" in node_type else 1.0,
                    )

    # RENDER PIPES (EDGES)
    for u, v in graph.edges():
        pos_u = np.array(node_positions.get(u))
        pos_v = np.array(node_positions.get(v))
        if pos_u is not None and pos_v is not None:
            midpoint = (pos_u + pos_v) / 2

            delta = pos_v - pos_u
            original_length = np.linalg.norm(delta)
            adjusted_length = original_length - 1.0

            if adjusted_length > 0:
                orientation = np.argmax(np.abs(delta))
                size = [1.0, 1.0, 1.0]
                size[orientation] = float(adjusted_length)

                # Initialize all colours to gray
                pipe_type = edge_types.get((u, v), "gray")
                face_cols = [gray_hex] * 6

                if pipe_type:

                    alpha = 0.7 if "*" in pipe_type else 1
                    edge_col = "white" if "*" in pipe_type else "black"

                    col = node_hex_map.get(pipe_type.replace("*", ""), ["gray"] * 3)
                    face_cols = [col[2]] * 2 + [col[1]] * 2 + [col[0]] * 2

                    if "h" in pipe_type:

                        # Hadamards split into three: two coloured ends and a yellow ring at the middle
                        if adjusted_length > 0:
                            yellow_length = 0.1 * adjusted_length
                            colored_length = 0.45 * adjusted_length

                            # Skip if lengths are invalid
                            if colored_length < 0 or yellow_length < 0:
                                continue

                            size_col = [1.0, 1.0, 1.0]
                            size_yellow = [1.0, 1.0, 1.0]
                            size_col[orientation] = float(colored_length)
                            size_yellow[orientation] = float(yellow_length)

                            offset1 = np.zeros(3)
                            offset3 = np.zeros(3)

                            offset1[orientation] = -(
                                yellow_length / 2 + colored_length / 2
                            )
                            offset3[orientation] = (
                                yellow_length / 2 + colored_length / 2
                            )

                            centre1 = midpoint + offset1
                            centre2 = midpoint
                            centre3 = midpoint + offset3

                            # Base of hadamard
                            face_cols_1 = list(face_cols)
                            
                            # Middle yellow ring
                            face_cols_yellow = [yellow_hex] * 6

                            # Far end of the hadamard
                            # Note. Keeping track of the correct rotations proved tricky
                            # Keep this bit spread out across lines – easier
                            face_cols_2 = [gray_hex] * 6
                            rotated_pipe_type = rotate_o_types(pipe_type[:3]) + "h"
                            col = node_hex_map.get(rotated_pipe_type, ["gray"] * 3)
                            face_cols_2[4] = col[0]  # right (+x)
                            face_cols_2[5] = col[0]  # left (-x)
                            face_cols_2[2] = col[1]  # front (-y)
                            face_cols_2[3] = col[1]  # back (+y)
                            face_cols_2[0] = col[2]  # bottom (-z)
                            face_cols_2[1] = col[2]  # top (+z)

                            render_edge(ax, centre1, size_col, face_cols_1, edge_col, alpha)
                            render_edge(ax, centre2, size_yellow, face_cols_yellow, edge_col, alpha)
                            render_edge(ax, centre3, size_col, face_cols_2, edge_col, alpha)
                    else:
                        render_edge(ax, midpoint, size, face_cols, edge_col, alpha)

    # Adjust plot limits
    all_positions = np.array(list(node_positions.values()))
    if all_positions.size > 0:
        max_range = np.ptp(all_positions, axis=0).max() / 2.0
        mid = np.mean(all_positions, axis=0)
        ax.set_xlim(mid[0] - max_range - 1, mid[0] + max_range + 1)
        ax.set_ylim(mid[1] - max_range - 1, mid[1] + max_range + 1)
        ax.set_zlim(mid[2] - max_range - 1, mid[2] + max_range + 1)
    else:
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Pop visualisation or save to file
    if save_to_file:
        temp_folder_path = "./outputs/temp"
        Path(temp_folder_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{temp_folder_path}/{filename}.png")
        plt.close()
    else:
        plt.show()


def make_graph_from_edge_paths(edge_paths: dict[Any, Any]) -> nx.Graph:
    final_graph = nx.Graph()
    node_counter = 0
    for edge, path_data in edge_paths.items():
        primary_node_and_edges = []
        path_nodes = path_data["path_nodes"]
        if path_nodes == "error":
            continue
        node_index_map = {}

        for pos, kind in path_nodes:
            if (pos, kind) not in node_index_map:
                node_index_map[(pos, kind)] = node_counter
                primary_node_and_edges.append([node_counter, pos, kind])
                node_counter += 1
            else:

                index_to_use = node_index_map[(pos, kind)]

                found = False
                for entry in primary_node_and_edges:
                    if entry[0] == index_to_use:
                        entry[1] = pos
                        found = True
                        break
                if not found:
                    primary_node_and_edges.append([index_to_use, pos, kind])

        # Add nodes
        for index, pos, node_type in primary_node_and_edges:
            if index not in final_graph:
                final_graph.add_node(index, pos=pos, type=node_type)

        # Add edges
        for i in range(len(primary_node_and_edges)):
            index, pos, node_type = primary_node_and_edges[i]
            if "o" in node_type:
                prev_index_path = i - 1
                next_index_path = i + 1
                if 0 <= prev_index_path < len(
                    primary_node_and_edges
                ) and 0 <= next_index_path < len(primary_node_and_edges):
                    prev_node_index = primary_node_and_edges[prev_index_path][0]
                    next_node_index = primary_node_and_edges[next_index_path][0]
                    if (
                        prev_node_index in final_graph
                        and next_node_index in final_graph
                    ):
                        final_graph.add_edge(
                            prev_node_index, next_node_index, pipe_type=node_type
                        )

    return final_graph
