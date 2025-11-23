"""Objects and functions used by several visualisation files. 

This file contains auxiliary objects that are used to create different kinds of 
visualisations. Do NOT call anything in this file directly.

Usage:
    Call any required object/function from a separate script.

NB! AI policy. If you use AI to modify this file, refer to `./README` for appropriate disclaimer guidelines.
"""

import io
import numpy as np
import networkx as nx

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.text as mtext
import matplotlib.path as mpath
import matplotlib.animation as animation

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from numpy.typing import NDArray
from typing import Any, Annotated, Callable, Dict, List, Literal, Tuple,  IO, Union


from topologiq.utils.utils_misc import get_manhattan
from topologiq.utils.classes import StandardBlock, StandardCoord
from topologiq.utils.utils_pathfinder import check_is_exit, rot_o_kind
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

###################
# TRANSFORMATIONS #
###################
def edge_paths_to_nx_graph(edge_paths: dict[StandardBlock, List[StandardBlock]]) -> nx.Graph:
    """Convert an edge_paths object into an nx.Graph.

    This function takes a list of 3D block-by-block `edge_paths` (which is how Topologiq stores progress)
    into a NetworkX graph. The `edge_paths` object is convenient for creating visualisation objects quickly. 
    However, the object may contain redundant blocks that are irrelevant for visualisation (they get rendered
    on top of each other) but would be highly inconvenient in final outputs. Accordingly, this function is 
    NOT for usage in operations that produce final results.

    Args:
        edge_paths: An edge-by-edge/block-by-block summary of the space-time diagram Topologiq builds.

    Returns:
        nx_graph: An nx.Graph with the information in `edge_paths` in a format more amicable for visualisation.
    """

    # Create foundational NX graph
    nx_graph = nx.Graph()
    block_count = 0

    # Iterate over `edge_paths` extracting objects
    for _, path_data in edge_paths.items():

        # Preliminaries
        primary_blocks_and_edges = []
        path_blocks = path_data
        
        if path_blocks == "error":
            continue
        path_index_map = {}

        # Organise objects
        for block_coords, block_kind in path_blocks:
            if (block_coords, block_kind) not in path_index_map:
                path_index_map[(block_coords, block_kind)] = block_count
                primary_blocks_and_edges.append([block_count, block_coords, block_kind])
                block_count += 1
            else:
                index_to_use = path_index_map[(block_coords, block_kind)]
                found = False
                for entry in primary_blocks_and_edges:
                    if entry[0] == index_to_use:
                        entry[1] = block_coords
                        found = True
                        break
                if not found:
                    primary_blocks_and_edges.append([index_to_use, block_coords, block_kind])

        # Add cubes
        for cube_index, cube_coords, cube_kind in primary_blocks_and_edges:
            if cube_index not in nx_graph:
                nx_graph.add_node(cube_index, coords=cube_coords, type=cube_kind)

        # Add pipes
        for i in range(len(primary_blocks_and_edges)):
            _, _, pipe_kind = primary_blocks_and_edges[i]
            if "o" in pipe_kind:
                prev_index_path = i - 1
                next_index_path = i + 1
                if 0 <= prev_index_path < len(
                    primary_blocks_and_edges
                ) and 0 <= next_index_path < len(primary_blocks_and_edges):
                    prev_cube_index = primary_blocks_and_edges[prev_index_path][0]
                    next_cube_index = primary_blocks_and_edges[next_index_path][0]
                    if (
                        prev_cube_index in nx_graph
                        and next_cube_index in nx_graph
                    ):
                        nx_graph.add_edge(
                            prev_cube_index, next_cube_index, pipe_type=pipe_kind
                        )

    return nx_graph


def lattice_to_g(
    lat_nodes: dict[int, StandardBlock],
    lat_edges: dict[Tuple[int, int], List[str]],
    nx_g: nx.Graph,
    pauli_webs: dict[Tuple[int, int], str] = {},
) -> Tuple[nx.Graph, nx.Graph]:
    """Convert a set of lattice nodes and edges into an nx.Graph.

    This function converts two dictionaries of lattice nodes and lattice edges into a single NX graph.
    The NX graph is easier to visualise than the separate dictionaries. The function is therefore meant
    as a pre-processing step to use before visualising progress and/or results.

    Args:
        lat_nodes: the cubes of the resulting space-time diagram with explicit position and kind information.
        lat_edges: the pipes of the resulting space-time diagram with explicit position and kind information.

    Returns:
        final_graph: an nx.Graph with all the information in the lattice nodes and edges but in a format amicable for visualisation
    """

    # Create foundational graph
    lattice_g = nx.Graph()
    pauli_webs_graph = nx.Graph()

    # Add cubes/nodes
    for key, node_info in lat_nodes.items():
        lattice_g.add_node(key, coords=node_info[0], type=node_info[1])
        if key in nx_g.nodes():
            lattice_g.nodes()[key]["beams"] = nx_g.nodes()[key]["beams"]
        else:
            lattice_g.nodes()[key]["beams"] = []

    # Add pipes/edges
    for key, edge_info in lat_edges.items():
        lattice_g.add_edge(key[0], key[1], pipe_type=edge_info[0])

    # Add pauli webs if present
    if pauli_webs:
        nodes_in_web = []
        for key, edge_info in pauli_webs.items():
            pauli_webs_graph.add_edge(key[0], key[1], pipe_type=edge_info)
            nodes_in_web += [key[0], key[1]]
        nodes_in_web = list(set(nodes_in_web))

        for node_id in nodes_in_web:
            pauli_webs_graph.add_node(
                node_id, coords=lat_nodes[node_id][0], type=lat_nodes[node_id][1]
            )

    return lattice_g, pauli_webs_graph


def figure_to_png(
    fig: matplotlib.figure.Figure,
    processed_ids: List[Union[str, int]],
    processed_edges: List[Tuple[int, int]],
    src_tgt_ids: Tuple[int, int] = None,
) -> IO[bytes]:
    """Converts a Matplotlib Figure object to an in-memory PNG file with a transparent background.

    This function takes the Matplotlib figure of the original ZX graph, deconstructs it, and 
    rebuilds it into a new figure that can be transformed into a PNG overlay to display 
    over 3D visualisations. 

    Args:
        fig: The Matplotlib Figure object.
        processed_ids: IDs of the nodes already processed by the algorithm.
        processed_edges: The (src, tgt) IDs composing the edges already processed by the algorithm.

    Returns:
        png_io: an in-memory binary stream (BytesIO object) containing the PNG data.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    # Preliminaries
    ax = fig.gca()
    fig.canvas.draw()
    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")

    artists_data = []
    max_r = 0

    # Deconstruct original figure
    # ...

    # Collect artists & calculate bounding box
    for artist in ax.get_children():
        artist_properties = {"type": None, "properties": {}}

        # Nodes/spiders
        if isinstance(artist, mpatches.Circle):
            x, y = artist.get_center()
            r = artist.get_radius()
            max_r = r if r > max_r else max_r
            min_x = min(min_x, x - r)
            max_x = max(max_x, x + r)
            min_y = min(min_y, y - r)
            max_y = max(max_y, y + r)

            artist_properties.update(
                {
                    "type": "circle",
                    "properties": {
                        "center": artist.get_center(),
                        "radius": artist.get_radius(),
                        "facecolor": artist.get_facecolor(),
                        "edgecolor": artist.get_edgecolor(),
                    },
                }
            )

        # Labels
        elif isinstance(artist, mtext.Text):
            x, y = artist.get_position()
            text = artist.get_text()

            if text.isnumeric():
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

                artist_properties.update(
                    {
                        "type": "text",
                        "properties": {
                            "position": artist.get_position(),
                            "text": text,
                            "fontsize": artist.get_fontsize(),
                            "horizontalalignment": artist.get_horizontalalignment(),
                            "verticalalignment": artist.get_verticalalignment(),
                        },
                    }
                )

        # Edges
        elif isinstance(artist, mlines.Line2D):
            x_data, y_data = artist.get_data()
            min_x = min(min_x, np.min(x_data))
            max_x = max(max_x, np.max(x_data))
            min_y = min(min_y, np.min(y_data))
            max_y = max(max_y, np.max(y_data))

            artist_properties.update(
                {
                    "type": "line",
                    "properties": {
                        "xdata": artist.get_xdata(),
                        "ydata": artist.get_ydata(),
                        "color": artist.get_color(),
                        "linestyle": artist.get_linestyle(),
                        "alpha": artist.get_alpha() if artist.get_alpha() else 1,
                    },
                }
            )

        # Curved edges
        elif isinstance(artist, mpatches.PathPatch):
            bbox_display = artist.get_window_extent()
            bbox_data = bbox_display.transformed(ax.transData.inverted())
            min_x = min(min_x, bbox_data.x0)
            max_x = max(max_x, bbox_data.x1)
            min_y = min(min_y, bbox_data.y0)
            max_y = max(max_y, bbox_data.y1)

            path = artist.get_path()
            if path:
                artist_properties.update(
                    {
                        "type": "path",
                        "properties": {
                            "path": path,
                            "facecolor": artist.get_facecolor(),
                            "edgecolor": artist.get_edgecolor(),
                            "linewidth": artist.get_linewidth(),
                        },
                    }
                )

        if artist_properties["type"]:
            artists_data.append(artist_properties)

    # Recreate figure
    # ...

    # IDs -> strings
    processed_ids_str = [str(i) for i in processed_ids]

    # Margins
    padding = 0.5
    width = max_x - min_x + padding * 2
    height = max_y - min_y + padding * 2

    # New figure object
    new_fig = plt.figure(figsize=(width, height), dpi=100)
    new_ax = new_fig.add_axes([0, 0, 1, 1])

    # Recreate artists
    label_positions = {}
    for data in artists_data:
        props = data["properties"]

        # Nodes/spiders
        if data["type"] == "circle":
            new_x = props["center"][0] - min_x + padding
            new_y = props["center"][1] - min_y + padding
            new_circle = mpatches.Circle(
                (new_x, new_y),
                props["radius"],
                facecolor=props["facecolor"] if props["facecolor"] != (0.8, 1.0, 0.8, 1.0) else "#b9cdff",
                edgecolor=props["edgecolor"],
                transform=new_ax.transData,
                zorder=1,
            )
            new_ax.add_patch(new_circle)

        # Labels
        elif data["type"] == "text":
            original_text = props["text"]

            is_processed = False
            if original_text.isnumeric() and original_text in processed_ids_str:
                is_processed = True

            is_part_of_current_edge = False
            if src_tgt_ids and int(original_text) in src_tgt_ids:
                is_part_of_current_edge = True

            new_x = props["position"][0] - min_x + padding
            new_y = props["position"][1] - min_y + padding
            new_text = mtext.Text(
                x=new_x,
                y=new_y,
                text=original_text,
                color="green" if is_processed else "black",
                horizontalalignment=props["horizontalalignment"],
                verticalalignment=props["verticalalignment"],
                transform=new_ax.transData,
                zorder=3,
                bbox=dict(
                    facecolor="gold" if is_part_of_current_edge else "white",
                    edgecolor="green" if is_processed else "black",
                    boxstyle="round",
                ),
            )
            new_text.set_fontsize(max(props["fontsize"], 12))
            new_ax.add_artist(new_text)

            label_positions[props["text"]] = props["position"]

        # Edges
        elif data["type"] == "line":
            x_data = [x - min_x + padding for x in props["xdata"]]
            y_data = [y - min_y + padding for y in props["ydata"]]
            new_line = mlines.Line2D(
                xdata=x_data,
                ydata=y_data,
                color=props["color"],
                linestyle=props["linestyle"],
                alpha=props["alpha"],
                transform=new_ax.transData,
                zorder=0,
            )
            new_ax.add_line(new_line)

        # Curved edges
        elif data["type"] == "path":
            path = props["path"]
            vertices = path.vertices
            if len(vertices) >= 2:
                start_point = vertices[0]
                end_point = vertices[-1]

                new_start_x = start_point[0] - min_x + padding
                new_start_y = start_point[1] - min_y + padding
                new_end_x = end_point[0] - min_x + padding
                new_end_y = end_point[1] - min_y + padding

                mid_x = (new_start_x + new_end_x) / 2
                mid_y = (new_start_y + new_end_y) / 2
                dx = new_end_x - new_start_x
                dy = new_end_y - new_start_y
                offset = max_r * 2

                if abs(dx) < 1e-6:
                    control_x = mid_x - offset
                    control_y = mid_y
                else:
                    perp_dx = -dy
                    perp_dy = dx
                    length = np.sqrt(perp_dx**2 + perp_dy**2)
                    if length != 0:
                        perp_dx /= length
                        perp_dy /= length
                    control_x = mid_x + perp_dx * offset
                    control_y = mid_y + perp_dy * offset

                path_data = [
                    (mpath.Path.MOVETO, (new_start_x, new_start_y)),
                    (mpath.Path.CURVE3, (control_x, control_y)),
                    (mpath.Path.CURVE3, (new_end_x, new_end_y)),
                ]
                codes, verts = zip(*path_data)

                new_path = mpath.Path(verts, codes)
                new_patch = mpatches.PathPatch(
                    new_path,
                    facecolor=props["facecolor"],
                    edgecolor=props["edgecolor"],
                    linewidth=props["linewidth"],
                    transform=new_ax.transData,
                    zorder=0,
                )
                new_ax.add_patch(new_patch)

    for u, v in processed_edges:
        u_str, v_str = str(u), str(v)

        if u_str in label_positions and v_str in label_positions:
            x1, y1 = label_positions[u_str]
            x2, y2 = label_positions[v_str]

            # Apply the same offset as for other artists
            new_x_data = [x1 - min_x + padding, x2 - min_x + padding]
            new_y_data = [y1 - min_y + padding, y2 - min_y + padding]

            new_edge = mlines.Line2D(
                xdata=new_x_data,
                ydata=new_y_data,
                color="black",
                linestyle=":",
                zorder=2,
                transform=new_ax.transData,
            )
            new_ax.add_line(new_edge)

    # Ax settings
    new_ax.set_xlim(0, width)
    new_ax.set_ylim(0, height)
    new_ax.set_aspect("equal")
    new_ax.axis("off")

    border_inset = 0.05  # A small offset to prevent clipping
    rect = mpatches.Rectangle(
        (border_inset, border_inset),
        width - 2 * border_inset,
        height - 2 * border_inset,
        facecolor="none",
        edgecolor="black",
        linestyle=":",
        transform=new_ax.transData,
        zorder=10,
    )
    new_ax.add_patch(rect)

    # Create a PNG buffer from recreated figure
    # ...

    # Buffer
    png = io.BytesIO()
    new_fig.savefig(png, format="png")
    png.seek(0)

    # Close for safety
    plt.close(fig)
    plt.close(new_fig)

    return png

#####################
# PRIMARY RENDERERS #
#####################
def render_block(
    ax: matplotlib.axes.Axes,
    node_id: int,
    coords: Tuple[int, int, int],
    size: list[float],
    node_type: str,
    node_hex_map: dict[str, list[str]],
    alpha: float = 1.0,
    edge_col: str = "black",
    border_width: float = 0.5,
    taken: List[StandardCoord] = [],
    **kwargs,
) -> Poly3DCollection:
    """Renders a regular (non-'h') block.

    This function creates a 3D cube to an existing Matplotlib ax. It takes the position,
    size and other graphical characteristics as parameters, applies specific face colors
    based on the `node_type`, and, if applicable, attaches invisible labels and direction
    quivers for debugging and interaction.

    Args:
        ax: Matplotlib's 3D subplot object.
        node_id: the ID of the node
        coords: (x, y, z) coordinates of the block.
        size: (size_x, size_y, size_z) of the block.
        node_type: block's kind.
        node_hex_map: map of (HEX) colours for block.
        edge_col: color for the edges of blocks.
        border_width: width for borders of block.
        taken: A list of coordinates occupied by any blocks/pipes placed as a result of previous operations.
    """

    # General dimensions
    x, y, z = coords
    size_x, size_y, size_z = size

    vertices = get_vertices(x, y, z, size_x, size_y, size_z)
    faces = get_faces(vertices)

    # Add colours as per map
    cols = node_hex_map.get(node_type, ["gray"] * 3)
    face_cols = [cols[2]] * 2 + [cols[1]] * 2 + [cols[0]] * 2

    # Join into Poly collection
    poly_collection = Poly3DCollection(
        faces,
        facecolors=face_cols if "_visited" not in node_type else "red",
        linewidths=border_width,
        edgecolors=edge_col,
        alpha=alpha,
        picker=True,
        label=node_id,
        **kwargs,
    )

    # Attach labels if node has ID
    if node_id != "TBD":

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
                    visible=False,
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
                    visible=False,
                )

                break

    # Add to plot
    ax.add_collection3d(poly_collection)

    # Return for usage in show/hide toggle features
    return [poly_collection]


def render_pipe(
        ax: matplotlib.axes.Axes,
        u_coords: StandardCoord,
        v_coords: StandardCoord,
        block_kind: str,
        edge_col: str = "black",
        border_width: float = 0.5,
        alpha: float = 1.0
) -> List[Poly3DCollection]:
    """Add a pipe to the Matplotlib ax.
    
    This function adds a pipe (regular or hadamard) to an existing Matplotlib ax.
    It takes the position, size and other graphical characteristics as parameters, 
    applies specific face colors based on the `node_type`, and, if applicable,
    attaches invisible labels and direction quivers for debugging and interaction.

    Args:
        ax: Matplotlib's 3D subplot object.
        u_coords: (x, y, z) coordinates of the source cube.
        v_coords: (x, y, z) coordinates of the target cube.
        block_kind: The type of the pipe block (e.g., 'Xh' for Hadamard).
        edge_col: color of the edges for the edge/pipe.
        border_width: width for borders of edge.
        alpha: any desired value for alpha (transparency).

    Returns:
        List[Poly3DCollection]: A list containing the Matplotlib artists for the pipe sections.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """
    
    # Convert positions to np.arrays
    u_coords = np.array(u_coords)
    v_coords = np.array(v_coords)

    # Establish midpoint and pipe length
    midpoint = (u_coords + v_coords) / 2
    original_length = np.linalg.norm(v_coords - u_coords)
    adjusted_length = original_length - 1.0

    # Process pipe
    if adjusted_length > 0:
        orientation = np.argmax(np.abs(v_coords - u_coords))
        size = [1.0, 1.0, 1.0]
        size[orientation] = float(adjusted_length)
        face_cols = ["gray"] * 6
        
        col = node_hex_map.get(block_kind.replace("*", ""), ["gray"] * 3)
        face_cols = [col[2]] * 2 + [col[1]] * 2 + [col[0]] * 2

        # Regular pipes
        if "h" not in block_kind:
            artists = render_pipe_section(
                ax,
                midpoint,
                size,
                face_cols,
                edge_col,
                alpha,
                border_width=border_width,
            )

        # Hadamard pipes
        elif "h" in block_kind:

            # Break into three sections
            #   2 * coloured ends
            #   1 * middle yellow ring
            if adjusted_length > 0:
                yellow_length = 0.1 * adjusted_length
                colored_length = 0.45 * adjusted_length

                # Skip if internal lengths are invalid
                if colored_length < 0 or yellow_length < 0:
                    return

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
                face_cols_yellow = ["yellow"] * 6

                # Far end of the hadamard
                # Note. Keeping track of the correct rotations proved tricky
                # Keep this bit spread out across lines – easier
                face_cols_2 = ["gray"] * 6
                rotated_kind = rot_o_kind(block_kind[:3]) + "h"
                col = node_hex_map.get(rotated_kind, ["gray"] * 3)
                face_cols_2[4] = col[0]  # right (+x)
                face_cols_2[5] = col[0]  # left (-x)
                face_cols_2[2] = col[1]  # front (-y)
                face_cols_2[3] = col[1]  # back (+y)
                face_cols_2[0] = col[2]  # bottom (-z)
                face_cols_2[1] = col[2]  # top (+z)

                artists_1 = render_pipe_section(
                    ax,
                    centre1,
                    size_col,
                    face_cols_1,
                    edge_col,
                    alpha,
                    border_width=border_width,
                )
                artists_2 = render_pipe_section(
                    ax,
                    centre2,
                    size_yellow,
                    face_cols_yellow,
                    edge_col,
                    alpha,
                    border_width=border_width,
                )
                artists_3 = render_pipe_section(
                    ax,
                    centre3,
                    size_col,
                    face_cols_2,
                    edge_col,
                    alpha,
                    border_width=border_width,
                )

                artists = artists_1 + artists_2 + artists_3

        # Return for usage in show/hide toggle features
        return artists


def render_pipe_section(
    ax: matplotlib.axes.Axes,
    centre: NDArray[np.float64],
    size: list[float],
    face_cols: list[str],
    edge_col: str,
    alpha: float | int,
    border_width: float = 0.5,
) -> Poly3DCollection:
    """Renders edges/pipes.

    This function takes care of rendering a section of a 3D pipe/edge. It takes the coordinates
    and size of the pipe alongside other visual formatting parameters and calculates all 
    geometric objects needed to render it in 3D. 

    Args:
        ax: Matplotlib's 3D subplot object.
        centre: (x, y, z) coordinates of the edge's centre (midpoint between connecting nodes).
        size: (size_x, size_y, size_z) of the edge/pipe.
        face_cols: colour pattern for the edge/pipe.
        edge_col: color of the edges for the edge/pipe.
        alpha: any desired value for alpha (transparency)
        border_width: width for borders of edge.
    
    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    # Determine centre and size
    x, y, z = centre
    sx, sy, sz = size

    # Determine vertices
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

    # Add faces
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ]
    face_list = [vertices[face] for face in faces]

    # Turn into a collection
    poly_collection = Poly3DCollection(
        face_list,
        facecolors=face_cols,
        edgecolors=edge_col,
        linewidths=border_width,
        alpha=alpha,
    )

    # Add to plot
    ax.add_collection3d(poly_collection)

    return [poly_collection]


def render_prox_paths_view(fig, edge_col="white", border_width=3):
    """Clears and redraws the proximate paths based on the current view mode.

    This function supports two distinct view modes: 'ALL' (shows all filtered paths) and 
    'SINGLE' (shows only the path at fig.prox_current_index with a highlight). 

    Args:
        fig: The Matplotlib Figure object storing view state and data.
        edge_col (optional): The default color for the path edges in 'ALL' mode.
        border_width (optional): The default border width for the rendered blocks.
    
    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    # Clear previous artists
    for artist in fig.prox_path_artists:
        try:
            artist.remove()
        except Exception:
            pass 
    fig.prox_path_artists.clear()

    if not fig.prox_filtered_paths:
        fig.canvas.draw_idle()
        return

    # Determine which paths to render
    alpha = 1
    if fig.prox_view_mode == 'ALL':
        paths_to_render = fig.prox_filtered_paths
        current_edge_col = edge_col
    elif fig.prox_view_mode == 'SINGLE':
        # Use modulo for safe looping through the index
        idx = fig.prox_current_index % len(fig.prox_filtered_paths)
        paths_to_render = [fig.prox_filtered_paths[idx]]
        current_edge_col = 'cyan' # Use a distinct color for the focused path
    else:
        paths_to_render = []
        current_edge_col = edge_col

    # Rendering loop
    size = [1.0, 1.0, 1.0]
    taken = getattr(fig, 'taken', [])
    for path_data in paths_to_render:
        full_path = path_data['full_path']
        block_artists = []

        # Render Blocks and Pipes
        for i, (block_coords, block_kind) in enumerate(full_path):
            # Cubes
            if "o" not in block_kind:
                if block_coords in taken:
                    continue 
                    
                artists = render_block(
                    fig.ax, f"P-{i}", block_coords, size, block_kind, node_hex_map,
                    edge_col=current_edge_col, 
                    border_width=border_width, 
                    alpha=alpha,
                )
                if artists:
                    block_artists.extend(artists)
            # Pipes
            else:
                u_coords = full_path[i-1][0]
                try:
                    v_coords = full_path[i+1][0]
                except Exception as _:
                    v_coords = block_coords + ((np.array(u_coords) - np.array(block_coords)) * 2)

                artists = render_pipe(
                    fig.ax, u_coords, v_coords, block_kind, 
                    edge_col=current_edge_col, 
                    border_width=border_width,
                )
                if artists:
                    block_artists.extend(artists)

        fig.prox_path_artists.extend(block_artists)
        
    fig.canvas.draw_idle()

#################
# AUXILIARY OPS #
#################
def get_vertices(
    x: int, y: int, z: int, size_x: float, size_y: float, size_z: float
) -> Annotated[NDArray[np.float64], Literal[..., 3]]:
    """Calculates the coordinates of the eight vertices of a cuboid.

    This function calculates the exact position of the vertices of a cuboind based on
    a central position and the desired dimensions for the cuboid.

    Args:
        x: x-coordinate of the centre of the cuboid.
        y: y-coordinate of the centre of the cuboid.
        z: z-coordinate of the centre of the cuboid.
        size_x: length of the cuboid along the x-axis.
        size_y: length of the cuboid along the y-axis.
        size_z: length of the cuboid along the z-axis.

    Returns:
        array: array (numpy) of shape (8, 3) where each row represents the (x, y, z) coordinates of a vertex of the cuboid.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
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
    """Defines the faces of a cuboid based on its vertices.

    This function takes an array of vertices and returns a list that defines the faces of a
    cuboid to render as part of a 3D visualisation.

    Args:
        vertices: array (numpy) of shape (8, 3) where each row represents the (x, y, z) coordinates of a vertex of a cuboid.

    Returns:
        list: list of lists where each inner list represents a face and contains the coords of the vertices for that face.
    
    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
    ]


def recalculate_and_sort_prox_paths(fig: matplotlib.figure.Figure, tent_coords):
    """Recalculate and sort search paths by the current MD threshold.

    This function filters all raw search paths by the current MD threshold and sorts them 
    by their minimum distance to any tentative target.

    Args:
        fig: The Matplotlib Figure object storing state and data.
        tent_coords: List of coordinates defining the tentative targets.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    # Preliminaries
    fig.prox_filtered_paths.clear()
    prox_paths_with_dist = []

    # Get Manhattan distances for all search paths
    for path_data in fig.all_search_paths_raw:
        path_coords = path_data['coords']

        min_dist = _get_min_prox_distance(path_coords, tent_coords)

        if min_dist <= fig.prox_distance_threshold:
            # Store the path data and its minimum distance
            prox_paths_with_dist.append((path_data, min_dist))

    # Sort by minimum distance (closest first)
    prox_paths_with_dist.sort(key=lambda item: item[1])

    # Store only the path data in the figure state
    fig.prox_filtered_paths = [item[0] for item in prox_paths_with_dist]

    # Reset view index and mode
    fig.prox_current_index = 0
    fig.prox_view_mode = 'ALL'

    # Clear previous artists
    for artist in fig.prox_path_artists:
        try:
            artist.remove()
        except Exception:
            pass 
    fig.prox_path_artists.clear()

    if not fig.prox_filtered_paths:
        fig.canvas.draw_idle()
        return

    # Render
    edge_col = "black"
    alpha = 1
    if fig.prox_view_mode == 'ALL':
        paths_to_render = fig.prox_filtered_paths
        current_edge_col = edge_col
    elif fig.prox_view_mode == 'SINGLE':
        # Use modulo for safe looping through the index
        idx = fig.prox_current_index % len(fig.prox_filtered_paths)
        paths_to_render = [fig.prox_filtered_paths[idx]]
        current_edge_col = 'cyan' # Use a distinct color for the focused path
    else:
        paths_to_render = []
        current_edge_col = edge_col

    # Rendering loop
    size = [1.0, 1.0, 1.0]
    taken = getattr(fig, 'taken', [])

    for path_data in paths_to_render:
        full_path = path_data['full_path']
        block_artists = []

        # Render Blocks and Pipes
        border_width = 1
        for i, (block_coords, block_kind) in enumerate(full_path):
            # Cubes
            if "o" not in block_kind:
                if block_coords in taken:
                    continue 
                    
                artists = render_block(
                    fig.ax, f"P-{i}", block_coords, size, block_kind, node_hex_map,
                    edge_col=current_edge_col, 
                    border_width=border_width, 
                    alpha=alpha,
                )
                if artists:
                    block_artists.extend(artists)
            # Pipes
            else:
                u_coords = full_path[i-1][0]
                try:
                    v_coords = full_path[i+1][0]
                except Exception as _:
                    v_coords = block_coords + ((np.array(u_coords) - np.array(block_coords)) * 2)

                artists = render_pipe(
                    fig.ax, u_coords, v_coords, block_kind, 
                    edge_col=current_edge_col, 
                    border_width=border_width,
                )
                if artists:
                    block_artists.extend(artists)

        fig.prox_path_artists.extend(block_artists)

    fig.canvas.draw_idle()


def _get_min_prox_distance(path_coords, tent_coords):
    """Calculate the minimum Manhattan Distance (MD) between a path and tentative targets.

    This function iterates through all block coordinates in a path and all 
    tentative target coordinates to find the shortest MD between them.

    Args:
        path_coords: The 3D coordinates of all blocks in a path.
        tent_coords: The 3D coordinates of the tentative target blocks.

    Returns:
        min_dist: The minimum Manhattan Distance found.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.    
    """

    # Return infinity if no targets defined
    min_dist = float('inf')
    if not tent_coords:
        return min_dist

    # Calculate MD if targets defined
    for path_coord in path_coords:
        for target_coord in tent_coords:
            dist = get_manhattan(path_coord, target_coord) 
            min_dist = min(min_dist, dist)

    return min_dist

##################
# EVENT HANDLERS #
#################
def onpick_handler(e: matplotlib.backend_bases.PickEvent, ax: matplotlib.axes.Axes):
    """Handle click events on a visualisation to toggle associated labels/artists.
    
    Upon clicking a cube, this function looks up the cube ID and toggles the visibility of 
    its label and the 3D line pointing to it.
    
    Args:
        e: The Matplotlib PickEvent object containing the clicked artist.
        ax: The Matplotlib Axes object (specifically Axes3D in this context) 
            containing all the children artists.
        
    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    # Get event artist ID
    artist = e.artist
    node_id = artist.get_label()

    # Loop over artists to find and label appropriate artist
    for child in ax.get_children():
        if isinstance(child, mtext.Text) and child.get_text() != "":
            label_id = child.get_text()[: child.get_text().find(":")]
            if label_id == node_id:
                child.set_visible(not child.get_visible())
        elif isinstance(child, Line3DCollection) and child.get_label() != "":
            if child.get_label() == node_id:
                child.set_visible(not child.get_visible())

    # Re-draw
    plt.draw()


def toggle_animation_handler(
    e: matplotlib.backend_bases.MouseEvent,
    fig: matplotlib.figure.Figure,
    btn_anim: matplotlib.widgets.Button,
    persistent_green_artists: List[Dict[str, Any]],
    update_func: Callable[[int], List[Any]],
    num_frames: int,
    animation_interval_ms: int,
    num_paths: int,
    target_duration_ms: int,
) -> None:
    """Manage the state and replay of the search path animation.

    Upon click, this function triggers and manages the sequence of visualisations that 
    show all paths searched by the pathfinder. 

    Args:
        e: The MouseEvent from the button click.
        fig: The main Matplotlib Figure object (holds state and canvas).
        btn_anim: The Button widget itself, required to update its label and visibility.
        persistent_green_artists: List of dictionary items storing the artists for valid paths.
        update_func: The animation update function defined in the main scope.
        num_frames: Total frames for the animation.
        animation_interval_ms: Delay between frames.
        num_paths: Total number of paths searched.
        target_duration_ms: The total expected duration of the animation.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    # Hide pre-existent static paths
    if fig.show_static_search_paths: 
        # Hide green paths
        for item in persistent_green_artists:
            if item:
                # Assuming the artist is stored in item['artist']
                item['artist'].set_alpha(0.0) 
            
        # Hide red paths (stored in fig state)
        for artist in fig.static_search_artists:
            artist.remove() 
        fig.static_search_artists = []
        
        # Update state and button label
        fig.show_static_search_paths = False
        btn_anim.label.set_text('Replay Path Search')
        
        # Redraw
        fig.canvas.draw_idle()
        return

    # Replay the animation
    else:  
        # Hide button before starting animation
        btn_anim.ax.set_visible(False)   
        btn_anim.label.set_visible(False)
        
        # Clear paths (in case there was a prior failed animation or lingering paths)
        for item in persistent_green_artists:
            if item:
                item['artist'].set_alpha(0.0)
        for artist in fig.static_search_artists:
            # Check if artist is still visible before removing (robustness)
            if artist.axes:
                artist.remove() 
        fig.static_search_artists = []
            
        # Create and start the animation
        anim = animation.FuncAnimation(
            fig, 
            update_func, # Use the passed function
            frames=num_frames,
            interval=animation_interval_ms,
            blit=True,
            repeat=False,
            init_func=lambda: []
        )
        fig.animation_handle = anim

        # Handle completion to ensure paths are drawn and button returns
        def restore_button():

            # Manually execute the final frame logic
            update_func(num_paths) # Use the passed function and num_paths
            
            # Update state flag and label
            fig.show_static_search_paths = True 
            btn_anim.label.set_text('Hide Search Paths')
            
            # Restore button visibility and activity
            btn_anim.ax.set_visible(True)
            btn_anim.label.set_visible(True)
            btn_anim.set_active(True)
            fig.canvas.draw_idle()

        # Set a timer to execute restore_button after animation completion
        fig.canvas.manager.window.after(target_duration_ms + 250, restore_button)          
        fig.canvas.draw_idle()


def toggle_winner_path_handler(e: matplotlib.backend_bases.MouseEvent, fig: matplotlib.figure.Figure, btn_win: matplotlib.widgets.Button, btn_valid: matplotlib.widgets.Button) -> None:
    """Toggle the visibility of the optimal winner path.

    This function handles the show/hide functionality for the winner path
    and applies related updates to valid paths and buttons.

    Args:
        e: The MouseEvent from the button click.
        fig: The main Matplotlib Figure object (holds state and canvas).
        btn_win: The "Winner Path" Button widget to update its label.
        btn_valid: The "Valid Paths" Button widget to update its state and label if needed.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """
    
    # Toggle the visibility state
    new_state: bool = not fig.show_winner_path
    fig.show_winner_path = new_state
    
    # Toggle the visibility of all stored winner path artists
    for artist in fig.winner_path_artists:
        artist.set_visible(new_state)

    # If showing the winner path, hide the valid paths
    if new_state and fig.show_valid_paths:
        # Call the hiding logic for the valid paths
        fig.show_valid_paths = False
        for artist in fig.valid_path_artists:
            artist.set_visible(False)
        # Update the valid path button text to "Show Valid Paths"
        btn_valid.label.set_text('Show Valid Paths')

    # Update button text
    btn_win.label.set_text('Hide Winner Path' if new_state else 'Show Winner Path')

    # Force a redraw
    fig.canvas.draw_idle()


def toggle_beams_handler(e: matplotlib.backend_bases.MouseEvent, fig: matplotlib.figure.Figure, btn_beams: matplotlib.widgets.Button) -> None:
    """Toggle beams visibility in visualisation.

    This function handles the show/hide functionality for the beams emanating 
    from any cube in the space-time diagram that still needs connections.

    Args:
        e: The MouseEvent from the button click.
        fig: The main Matplotlib Figure object (holds state and canvas).
        btn_beams: The Button widget itself to update its label.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """
    # Toggle the visibility state
    new_state: bool = not fig.show_beams
    fig.show_beams = new_state
    
    # Toggle the visibility of all stored beam artists
    for artist in fig.beam_artists:
        artist.set_visible(new_state)

    # Update button text
    btn_beams.label.set_text('Hide Beams' if new_state else 'Show Beams')

    # Force a redraw
    fig.canvas.draw_idle()


def toggle_targets_handler(e: matplotlib.backend_bases.MouseEvent, fig: matplotlib.figure.Figure, btn_tgt: matplotlib.widgets.Button) -> None:
    """Toggle the visibility of the blocks marking tentative target coordinates.

    This function handles the show/hide functionality for the cubes used to denote valid tentative
    positions and kinds for any new cube being added to space-time diagram. 

    Args:
        e: The MouseEvent from the button click.
        fig: The main Matplotlib Figure object (holds state and canvas).
        btn_tgt: The Button widget itself to update its label.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """
    # Toggle the visibility state
    new_state: bool = not fig.show_tent_tgt_blocks
    fig.show_tent_tgt_blocks = new_state
    
    # Toggle the visibility of all stored target artists
    for artist in fig.target_artists:
        artist.set_visible(new_state)

    # Update button text
    btn_tgt.label.set_text('Hide Targets' if new_state else 'Show Targets')

    # Force a redraw
    fig.canvas.draw_idle()


def toggle_valid_paths_handler(
    e: matplotlib.backend_bases.MouseEvent,
    fig: matplotlib.figure.Figure,
    btn_valid: matplotlib.widgets.Button,
    btn_win: matplotlib.widgets.Button
) -> None:
    """Toggle the visibility of all valid paths.

    This function handles the show/hide functionality for all paths returned by the pathfinder
    as valid paths between a given source block and one or more tentative targets.

    Args:
        e: The MouseEvent from the button click.
        fig: The main Matplotlib Figure object (holds state and canvas).
        btn_valid: The "Valid Paths" Button widget to update its label.
        btn_win: The "Winner Path" Button widget to update its state and label if needed.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """
    # Toggle the visibility state
    new_state: bool = not fig.show_valid_paths
    fig.show_valid_paths = new_state
    
    # Toggle the visibility of all stored valid path artists
    for artist in fig.valid_path_artists:
        artist.set_visible(new_state)

    # If showing valid paths, hide the winner path
    if new_state and fig.show_winner_path:
        # Call the hiding logic for the winner path
        fig.show_winner_path = False
        for artist in fig.winner_path_artists:
            artist.set_visible(False)
        # Update the winner path button text to "Show Winner Path"
        btn_win.label.set_text('Show Winner Path')
        
    # Update button text
    btn_valid.label.set_text('Hide Valid Paths' if new_state else 'Show Valid Paths')

    # Force a redraw
    fig.canvas.draw_idle()


def toggle_overlay_handler(
    e: matplotlib.backend_bases.MouseEvent,
    fig: matplotlib.figure.Figure,
    btn_overlay: matplotlib.widgets.Button,
    btn_pos: List[float],
) -> None:
    """Toggle the visibility of the ZX-graph overlay and dynamically update the button.

    Args:
        e: The MouseEvent from the button click (or simulated for hide_on_click).
        fig: The main Matplotlib Figure object (holds state and canvas).
        btn_overlay: The Button widget itself.
        btn_pos: Positioning constants required by the toggle function.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """
    # Only proceed if the artist object exists
    if fig.overlay_image_artist is None:
        return

    # Extract button positioning info
    BTN_W_MAX, BTN_W_MIN, BTN_BOTTOM, BTN_HEIGHT = btn_pos

    # Toggle the visibility state
    new_state: bool = not fig.show_overlay
    fig.show_overlay = new_state 

    # Toggle the alpha of the stored Image Artist
    new_alpha: float = 1.0 if new_state else 0.0
    fig.overlay_image_artist.set_alpha(new_alpha)

    # Dynamic Resize, Reposition, and Text Update
    new_text: str = 'X' if new_state else 'Show Input ZX Graph'
    new_width: float = BTN_W_MIN if new_state else BTN_W_MAX
    
    # Keep the button flush right (align right edge to 1.0)
    new_left: float = 1.0 - new_width 

    # Update the axis position and size
    btn_overlay.ax.set_position([new_left, BTN_BOTTOM, new_width, BTN_HEIGHT])
    
    # Update button text
    btn_overlay.label.set_text(new_text)

    # Force a redraw
    fig.canvas.draw_idle()


def hide_overlay_handler(
    e: matplotlib.backend_bases.MouseEvent,
    fig: matplotlib.figure.Figure,
    toggle_func: Callable[[matplotlib.backend_bases.MouseEvent, matplotlib.figure.Figure, matplotlib.widgets.Button, float, float, float, float], None],
    btn_overlay: matplotlib.widgets.Button,
    btn_pos: List[float],
) -> None:
    """Hide the overlay when the user clicks anywhere inside the overlay's plot area.

    Args:
        e: The MouseEvent from the canvas click.
        fig: The main Matplotlib Figure object (holds state and canvas).
        toggle_func: The primary function used to toggle the overlay visibility.
        btn_overlay: The button widget handle.
        btn_pos: Positioning constants required by the toggle function.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.   
    """

    # Extract positions
    BTN_W_MAX, BTN_W_MIN, BTN_BOTTOM, BTN_HEIGHT = btn_pos

    # Check if the click happened inside the stored overlay axis bounds AND it's currently visible
    if hasattr(fig, 'ax_overlay') and e.inaxes == fig.ax_overlay and fig.show_overlay:
        # Call the external toggle logic to hide it, passing all required arguments
        toggle_func(
            e,
            fig,
            btn_overlay,
            BTN_W_MAX,
            BTN_W_MIN,
            BTN_BOTTOM,
            BTN_HEIGHT
        )


def toggle_prox_paths_handler(e, fig, btn_prox, tent_coords):
    """Toggles the display of proximate search paths.
    
    This function toggles proximate paths on/off. A path is proximate
    if any of its blocks is within `fig.prox_distance_threshold` 
    Manhattan Distance (MD) of any coordinate in `tent_coords`.

    Args:
        e: The Matplotlib event object (unused but required by signature).
        fig: The Matplotlib Figure object storing state.
        btn_prox: The Matplotlib button object whose text label is updated.
        tent_coords: List of coordinates defining the tentative targets.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    # Toggle to target mode
    fig.show_prox_paths = not fig.show_prox_paths

    # Show
    if fig.show_prox_paths:    
        if not tent_coords:
            print("Cannot show proximate paths: No tentative target coordinates found.")
            fig.show_prox_paths = False
            btn_prox.label.set_text("Prox Paths")
            fig.canvas.draw_idle()
            return
            
        # Recalculate, filter, and sort all proximate paths based on current MD
        recalculate_and_sort_prox_paths(fig, tent_coords)
        
        # Reset mode to 'ALL' and render the initial view
        fig.prox_view_mode = 'ALL' 
        render_prox_paths_view(fig)
        
        # Update Button Text
        count = len(fig.prox_filtered_paths)
        btn_prox.label.set_text(f"{count} Prox Paths @ MD {fig.prox_distance_threshold}")

    # Hide
    else:
        for artist in fig.prox_path_artists:
            try:
                artist.remove()
            except Exception:
                pass 
        fig.prox_path_artists.clear()

        # 2. Reset view mode and update button text
        fig.prox_view_mode = 'ALL' 
        btn_prox.label.set_text('Prox Paths')

    fig.canvas.draw_idle()


def keypress_handler(e, fig, btn_prox, tent_coords):
    """Handles key presses for proximate paths view control.

    This function handles key press actions for the proximate paths:
        - Up/down keys adjust the Manhattan Distance (MD) threshold.
        - Left/right keys cycle through the filtered paths in 'SINGLE' view mode.

    Args:
        e: The Matplotlib key event object.
        fig: The Matplotlib Figure object storing state.
        btn_prox: The Matplotlib button object whose text label is updated.
        tent_coords: List of coordinates defining the tentative targets.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    # Only active when feature is ON
    if not fig.show_prox_paths:
        return

    # Handle MD Threshold Change (Up/down)
    if e.key in ['up', 'down']:
        old_md = fig.prox_distance_threshold
        if e.key == 'up':
            fig.prox_distance_threshold = min(100, old_md + 1) # Max 100
        elif e.key == 'down':
            fig.prox_distance_threshold = max(1, old_md - 1)  # Min 1

        # Recalculate, sort, and reset view to 'ALL'
        if fig.prox_distance_threshold != old_md:

            recalculate_and_sort_prox_paths(fig, tent_coords)
            render_prox_paths_view(fig)

            # Update button label
            count = len(fig.prox_filtered_paths)
            btn_prox.label.set_text(f"{count} Prox Paths @ MD: {fig.prox_distance_threshold}")

    # Handle Path Cycling (Left/right)
    elif e.key in ['left', 'right']:

        # Pass if no proximate paths
        if not fig.prox_filtered_paths:
            return

        # Switch to SINGLE view mode if currently in 'ALL'
        count = len(fig.prox_filtered_paths)        
        if fig.prox_view_mode == 'ALL':
            fig.prox_view_mode = 'SINGLE'

        # Calculate new index
        current_idx = fig.prox_current_index
        if e.key == 'right':
            fig.prox_current_index = (current_idx + 1) % count
        elif e.key == 'left':
            fig.prox_current_index = (current_idx - 1 + count) % count

        # Render the single path and update the button label
        render_prox_paths_view(fig)
        current_path = fig.prox_filtered_paths[fig.prox_current_index]
        min_dist = _get_min_prox_distance(current_path['coords'], tent_coords)

        # Update button
        btn_prox.label.set_text(f"Path {fig.prox_current_index + 1}/{count} @ MD: {min_dist}")

    fig.canvas.draw_idle()
