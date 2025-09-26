# NetworkX / Matplotlib functions to create quick 3D visualisations of algorithmic progress and a visualisation of final result.
# File is an absolute mess at the moment. It works though.
import io
import numpy as np
import networkx as nx
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.text as mtext
import matplotlib.path as mpath

import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.text import Text
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from typing import Annotated, Literal, Any, Optional, Tuple, List, IO, Union
from numpy.typing import NDArray

from utils.utils_pathfinder import check_is_exit, rot_o_kind
from utils.classes import StandardBlock, StandardCoord

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
    "X": ["red", "red", "red"],
    "Y": ["green", "green", "green"],
    "Z": ["blue", "blue", "blue"],
    "beam": ["yellow"],
}


def figure_to_png(
    fig: matplotlib.figure.Figure,
    processed_ids: List[Union[str, int]],
    processed_edges: List[Tuple[int, int]],
) -> IO[bytes]:
    """Converts a Matplotlib Figure object to an in-memory PNG file with a transparent background.

    Args:
        - fig: the Matplotlib Figure object to convert.
        - processed_ids: ids of the nodes already processed by the algorithm.
        - processed_edges: (src, tgt) ids composing the edges already processed by the algorithm.

    Returns:
        - png_io: an in-memory binary stream (BytesIO object) containing the PNG data.

    """

    # PRELIMS
    ax = fig.gca()
    fig.canvas.draw()
    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")
    background_color: Tuple = (0.95, 0.95, 0.95)

    artists_data = []
    max_r = 0

    # COLLECT ARTISTS DATA & CALC BOUNDING BOX
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

    # IDs -> STRING
    processed_ids_str = [str(i) for i in processed_ids]

    # MARGINS
    padding = 0.5
    width = max_x - min_x + padding * 2
    height = max_y - min_y + padding * 2

    # NEW FIGURE
    new_fig = plt.figure(figsize=(width, height), dpi=100)
    new_ax = new_fig.add_axes([0, 0, 1, 1])
    new_ax.set_facecolor(background_color)
    new_fig_patch = getattr(new_fig, "patch", None)
    if new_fig_patch:
        new_fig_patch.set_facecolor(background_color)

    # Recreate artists
    label_positions = {}
    for data in artists_data:
        props = data["properties"]

        # Nodes/spiders
        if data["type"] == "circle":
            new_x = props["center"][0] - min_x + padding
            new_y = props["center"][1] - min_y + padding
            print(props["facecolor"])
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
                    facecolor="white" if is_processed else "none",
                    edgecolor="green" if is_processed else "none",
                    boxstyle="round",
                ),
            )
            new_text.set_fontsize(props["fontsize"])
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

    # NEW AX CONFIG
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
        edgecolor="gray",
        linewidth=1,
        linestyle=":",
        transform=new_ax.transData,
        zorder=10,
    )
    new_ax.add_patch(rect)

    # PNG BUFFER
    png = io.BytesIO()
    new_fig.savefig(png, format="png")
    png.seek(0)

    # CLOSE FIGS TO AVOID OVERLOADS
    plt.close(fig)
    plt.close(new_fig)

    # RETURN PNG BUFFER
    return png


# MAIN VISUALISATION FUNCTION
def vis_3d_g(
    graph: nx.Graph,
    edge_pths: dict,
    current_nodes: Optional[Tuple[int, int]] = None,
    hide_ports: bool = False,
    node_hex_map: dict[str, list[str]] = node_hex_map,
    save_to_file: bool = False,
    filename: Optional[str] = None,
    pauli_webs_graph: Optional[nx.Graph] = None,
    debug: bool = False,
    taken: List[StandardCoord] = [],
    fig_data: Optional[matplotlib.figure.Figure] = None,
):
    """Manages the process of visualising a graph with many nodes/blocks and edges/pipes.

    Args:
        - graph: incoming graph formatted as an nx.Graph.
        - edge_pths: the raw set of 3D edges found by the algorithm (with redundant blocks for start and end positions of some edges).
        - current_nodes: ID of the last (src, tgt) nodes connected by the algorithm.
        - hide_ports:
            - True: do not display boundary nodes even if present in the incoming graph,
            - False: display boundary nodes if present.
        - node_hex_map: a hex map of colours covering all possible blocks and pipes.
        - save_to_file:
            - True: saves visualisation to file and does NOT show it on screen,
            - False: shows visualisation on screen and does NOT save it to file.
        - filename: filename to use if saving a visualisation.
        - pauli_webs_graph: optional graph containing a single Pauli web.
        - debug: optional parameter to turn debugging mode on (added details will be visualised on each step).
            - True: debugging mode on,
            - False: debugging mode off.
        - taken: list of coordinates occupied by any blocks/pipes placed as a result of previous operations.
        - fig_data: optional parameter to pass the original visualisation for input graph (currently only available for PyZX graphs).
    """

    def onpick(event):
        """Handles click events on visualisation pop-up frame"""

        artist = event.artist
        node_id = artist.get_label()

        for child in ax.get_children():
            if isinstance(child, Text) and child.get_text() != "":
                label_id = child.get_text()[: child.get_text().find(":")]
                if label_id == node_id:
                    child.set_visible(not child.get_visible())
            elif isinstance(child, Line3DCollection) and child.get_label() != "":
                if child.get_label() == node_id:
                    child.set_visible(not child.get_visible())

        plt.draw()

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
            or (not hide_ports and node_type == "ooo")
        ):
            position = node_positions.get(node_id)
            if position:
                size = [1.0, 1.0, 1.0]
                edge_col = "black"
                alpha = 0.5 if debug else 0.7 if pauli_webs_graph else 1

                if node_type == "ooo":
                    size = [0.9, 0.9, 0.9]
                    edge_col = "white"

                render_block(
                    ax,
                    node_id,
                    position,
                    size,
                    node_type[:3],
                    node_hex_map,
                    alpha=alpha,
                    edge_col=edge_col,
                    border_width=0.5 if not pauli_webs_graph else 0.1,
                    current_nodes=current_nodes,
                    debug=debug,
                    taken=taken,
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
                    alpha = 0.5 if debug else 0.4 if pauli_webs_graph else 1
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
                            rotated_pipe_type = rot_o_kind(pipe_type[:3]) + "h"
                            col = node_hex_map.get(rotated_pipe_type, ["gray"] * 3)
                            face_cols_2[4] = col[0]  # right (+x)
                            face_cols_2[5] = col[0]  # left (-x)
                            face_cols_2[2] = col[1]  # front (-y)
                            face_cols_2[3] = col[1]  # back (+y)
                            face_cols_2[0] = col[2]  # bottom (-z)
                            face_cols_2[1] = col[2]  # top (+z)

                            render_edge(
                                ax,
                                centre1,
                                size_col,
                                face_cols_1,
                                edge_col,
                                alpha,
                                border_width=0.5 if not pauli_webs_graph else 0,
                            )
                            render_edge(
                                ax,
                                centre2,
                                size_yellow,
                                face_cols_yellow,
                                edge_col,
                                alpha,
                                border_width=0.5 if not pauli_webs_graph else 0,
                            )
                            render_edge(
                                ax,
                                centre3,
                                size_col,
                                face_cols_2,
                                edge_col,
                                alpha,
                                border_width=0.5 if not pauli_webs_graph else 0,
                            )
                    else:
                        render_edge(
                            ax,
                            midpoint,
                            size,
                            face_cols,
                            edge_col,
                            alpha,
                            border_width=0.5 if not pauli_webs_graph else 0,
                        )

    # RENDER BEAMS
    if debug:
        for node_id in graph:
            node_beams = graph.nodes()[node_id]["beams"]
            if node_beams:
                for beam in node_beams:
                    for beam_pos in beam:
                        ax.scatter(
                            beam_pos[0],
                            beam_pos[1],
                            beam_pos[2],
                            c="yellow",
                            s=10,
                            edgecolors="black",
                            alpha=0.5,
                            depthshade=True,
                        )

    # RENDER PAULI WEBS
    if pauli_webs_graph:

        # Cubes (nodes)
        for node_id in pauli_webs_graph.nodes:
            pos = node_positions[node_id]
            ax.scatter(
                pos[0],
                pos[1],
                pos[2],
                c="black",
                s=0,
                edgecolors="white",
                depthshade=True,
            )

        # Pipes (edges)
        for (u, v), node_info in pauli_webs_graph.edges().items():
            pos_u = node_positions[u]
            pos_v = node_positions[v]
            col = node_hex_map[node_info["pipe_type"]][0]
            ax.plot(
                [pos_u[0], pos_v[0]],
                [pos_u[1], pos_v[1]],
                [pos_u[2], pos_v[2]],
                c=col,
                linewidth=3,
            )

    # ADJUST PLOT LIMITS
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

    # SET LABELS
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # RENDER OVERLAY
    if fig_data:
        png_buffer = figure_to_png(
            fig_data,
            processed_ids=list(graph),
            processed_edges=list(edge_pths.keys()),
        )
        overlay_image = Image.open(png_buffer)
        img_width, img_height = overlay_image.size
        aspect_ratio = img_width / img_height

        # Width ratio
        desired_height_ratio = 0.3
        calculated_width_ratio = (
            desired_height_ratio
            * (fig.get_figheight() / fig.get_figwidth())
            * aspect_ratio
        )

        # Width constraint (20% to 50%)
        min_width_ratio = 0.3
        max_width_ratio = 0.5
        if calculated_width_ratio < min_width_ratio:
            calculated_width_ratio = min_width_ratio
        elif calculated_width_ratio > max_width_ratio:
            calculated_width_ratio = max_width_ratio

        # Recalculate height to maintain ratio
        calculated_height_ratio = (
            calculated_width_ratio
            / aspect_ratio
            * (fig.get_figwidth() / fig.get_figheight())
        )

        # New axes for overlay, positioned bottom-right ([left, bottom, width, height])
        margin = 0.02
        left = 1 - calculated_width_ratio - margin  # Align to right
        bottom = 0.0 + margin  # Align to bottom

        ax_overlay = fig.add_axes(
            [left, bottom, calculated_width_ratio, calculated_height_ratio]
        )

        # HIDE UNNECESSARY FEATURES
        ax_overlay.set_xticks([])
        ax_overlay.set_yticks([])
        ax_overlay.axis("off")

        # OVERLAY IT
        overlay_array = np.asarray(overlay_image)
        ax_overlay.imshow(overlay_array)

    # POP UP OR SAVE VIS
    repository_root: Path = Path(__file__).resolve().parent.parent
    temp_folder_pth = repository_root / "outputs/temp"
    if save_to_file:
        Path(temp_folder_pth).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{temp_folder_pth}/{filename}.png")
        plt.close()
    else:
        fig.canvas.mpl_connect("pick_event", onpick)
        plt.show()


# TOP LEVEL FUNCTIONS TO PREPARE OBJECTS FOR VISUALISATION
def edge_pths_to_g(edge_pths: dict[Any, Any]) -> nx.Graph:
    """Converts an edge_pths object into an nx.Graph that can be visualised with `vis_3d_g`. It is worth noting
    that the function will create a graph with potentially redundant blocks, which is irrelevant for visualisation purposes
    but does mean the function should not be used when producing final results.

    Args:
        - edge_pths: a dictionary containing a number of edge paths, i.e., full paths between two blocks, each path made of 3D blocks and pipes.

    Returns:
        - final_graph: an nx.Graph with all the information in edge_pths but in a format more amicable for visualisation

    """

    final_graph = nx.Graph()
    node_counter = 0
    for edge, pth_data in edge_pths.items():
        primary_node_and_edges = []
        pth_nodes = pth_data["pth_nodes"]
        if pth_nodes == "error":
            continue
        node_index_map = {}

        for pos, kind in pth_nodes:
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
                prev_index_pth = i - 1
                next_index_pth = i + 1
                if 0 <= prev_index_pth < len(
                    primary_node_and_edges
                ) and 0 <= next_index_pth < len(primary_node_and_edges):
                    prev_node_index = primary_node_and_edges[prev_index_pth][0]
                    next_node_index = primary_node_and_edges[next_index_pth][0]
                    if (
                        prev_node_index in final_graph
                        and next_node_index in final_graph
                    ):
                        final_graph.add_edge(
                            prev_node_index, next_node_index, pipe_type=node_type
                        )

    return final_graph


def lattice_to_g(
    lat_nodes: dict[int, StandardBlock],
    lat_edges: dict[Tuple[int, int], List[str]],
    nx_g: nx.Graph,
    pauli_webs: dict[Tuple[int, int], str] = {},
) -> Tuple[nx.Graph, nx.Graph]:
    """Converts an set of lattice nodes and edges into an nx.Graph that can be visualised with `vis_3d_g`.

    Args:
        - lat_nodes: the nodes/blocks of the resulting space-time diagram (without redundant blocks)
        - lat_edges: the edges/pipes of the resulting space-time diagram (without redundant pipes)

    Returns:
        - final_graph: an nx.Graph with all the information in the lattice nodes and edges but in a format amicable for visualisation

    """

    lattice_g = nx.Graph()
    pauli_webs_graph = nx.Graph()

    for key, node_info in lat_nodes.items():
        lattice_g.add_node(key, pos=node_info[0], type=node_info[1])
        if key in nx_g.nodes():
            lattice_g.nodes()[key]["beams"] = nx_g.nodes()[key]["beams"]
        else:
            lattice_g.nodes()[key]["beams"] = []

    for key, edge_info in lat_edges.items():
        lattice_g.add_edge(key[0], key[1], pipe_type=edge_info[0])

    if pauli_webs:
        nodes_in_web = []
        for key, edge_info in pauli_webs.items():
            pauli_webs_graph.add_edge(key[0], key[1], pipe_type=edge_info)
            nodes_in_web += [key[0], key[1]]
        nodes_in_web = list(set(nodes_in_web))

        for node_id in nodes_in_web:
            pauli_webs_graph.add_node(
                node_id, pos=lat_nodes[node_id][0], type=lat_nodes[node_id][1]
            )

    return lattice_g, pauli_webs_graph


# MISC FUNCTIONS
def get_vertices(
    x: int, y: int, z: int, size_x: float, size_y: float, size_z: float
) -> Annotated[NDArray[np.float64], Literal[..., 3]]:
    """Calculates the coordinates of the eight vertices of a cuboid.

    Args:
        - x: x-coordinate of the centre of the cuboid.
        - y: y-coordinate of the centre of the cuboid.
        - z: z-coordinate of the centre of the cuboid.
        - size_x: length of the cuboid along the x-axis.
        - size_y: length of the cuboid along the y-axis.
        - size_z: length of the cuboid along the z-axis.

    Returns:
        - array: array (numpy) of shape (8, 3) where each row represents the (x, y, z) coordinates of a vertex of the cuboid.

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

    Args:
        - vertices: array (numpy) of shape (8, 3) where each row represents the (x, y, z) coordinates of a vertex of the cuboid,
            as returned by `get_vertices`.

    Returns:
        - list: list of lists, where each inner list represents a face and contains the coords of the vertices for that face.

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
    node_id: int,
    position: Tuple[int, int, int],
    size: list[float],
    node_type: str,
    node_hex_map: dict[str, list[str]],
    alpha: float = 1.0,
    edge_col: None | str = None,
    border_width: float = 0.5,
    current_nodes: Optional[Tuple[int, int]] = None,
    debug: bool = False,
    taken: List[StandardCoord] = [],
):
    """Renders a regular (non-'h') block.

    Args:
        - ax: Matplotlib's 3D subplot object.
        - node_id: the ID of the node
        - position: (x, y, z) coordinates of the block.
        - size: (size_x, size_y, size_z) of the block.
        - node_type: block's kind.
        - node_hex_map: map of (HEX) colours for block.
        - edge_col: color for the edges of blocks.
        - border_width: width for borders of block.
        - current_nodes: optional parameter to pass ID of the last (src, tgt) nodes connected by the algorithm,
        - taken: list of coordinates occupied by any blocks/pipes placed as a result of previous operations.
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
            position[0] + d[0],
            position[1] + d[1],
            position[2] + d[2],
        )

        if (
            check_is_exit(position, node_type, label_pos) is not True
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
                position[0],
                position[1],
                position[2],
                label_pos[0] - position[0],
                label_pos[1] - position[1],
                label_pos[2] - position[2],
                color="black",
                lw=1,
                label=node_id,
                visible=True if current_nodes and node_id in current_nodes else False,
            )

            break

    # ADD TO PLOT
    ax.add_collection3d(poly_collection)


def render_edge(
    ax: Any,
    centre: NDArray[np.float64],
    size: list[float],
    face_cols: list[str],
    edge_col: str,
    alpha: float | int,
    border_width: float = 0.5,
):
    """Renders edges/pipes.

    Args:
        - ax: Matplotlib's 3D subplot object.
        - centre: (x, y, z) coordinates of the edge's centre (midpoint between connecting nodes).
        - size: (size_x, size_y, size_z) of the edge/pipe.
        - face_cols: colour pattern for the edge/pipe.
        - edge_col: color of the edges for the edge/pipe.
        - alpha: any desired value for alpha (transparency)
        - border_width: width for borders of edge.
    """

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
        linewidths=border_width,
        alpha=alpha,
    )

    # ADD TO PLOT
    ax.add_collection3d(poly)
