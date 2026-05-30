"""Debug 3D visualisations for the graph manager.

This file contains functions that can help create full-detail visualisations of how
the pathfinder algorithm goes about resolving specific edges.

Usage:
    Call `draw_blockgraph()` programmatically with an appropriate parameter combination.

"""

from fractions import Fraction
from typing import Annotated, Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import networkx as nx
import numpy as np
from matplotlib import ticker
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from numpy.typing import NDArray

from topologiq.core.blocks import ZXBlock
from topologiq.core.pathfinder.symbolic import rotate_pipe
from topologiq.input.utils import ZXColors
from topologiq.utils.classes import StandardCoord
from topologiq.utils.vis import node_hex_map


#####################
# VIS ORCHESTRATION #
#####################
def draw_as_zx(
    bgraph: nx.Graph, in_qubits: dict[int, int], in_rows: dict[int, int], draw_style: str = "zx"
):
    """Draw the NX BlockGraph using ZX or NX styling.

    Args:
        bgraph: The BlockGraph being built.
        in_qubits: A dictionary containing spiders with explicit qubit number.
        in_rows: A dictionary containing spiders with explicit row number.
        draw_style: The style of drawing:
            zx: Positions nodes in a PyZX-like manner
            nx: Positions nodes using NX algorithms.

    """

    # Position nodes
    if draw_style == "zx":
        try:
            positions = {k: [in_rows[k], -q] for k, q in in_qubits.items()}
        except Exception as _:
            positions = nx.spring_layout(bgraph)
    else:
        positions = nx.spring_layout(bgraph)

    # Assemble color maps
    node_colors = [
        ZXColors.lookup(attrs["zx_block"].zx_type) for _, attrs in bgraph.nodes(data=True)
    ]
    edge_colors = [ZXColors.lookup(attrs["edge_type"]) for _, _, attrs in bgraph.edges(data=True)]

    # Styling
    options = {
        "font_size": 9,
        "edgecolors": "#111111",
    }

    # Call and display
    nx.draw(
        bgraph,
        pos=positions,
        node_color=node_colors,
        edge_color=edge_colors,
        with_labels=True,
        **options,
    )
    plt.show()


def draw_as_blockgraph(
    bgraph: nx.Graph,
    taken: set[StandardCoord],
    is_cross_edge: bool,
    in_zx: dict[str, Any],
    base_graph: dict[str, Any],
    is_final_vis: bool = True,
    iter_fail: bool = False,
    block_style: str = "pipe",
):
    """Draw the primary BlockGraph algon collapsable overlays with additional information.

    Args:
        bgraph: The BlockGraph being built.
        taken: The set of taken coordinates.
        is_cross_edge: True if the current edge is a cross edge.
        in_zx: The information of the input ZX packed as a dictionary.
        base_graph: The information of the base graph (input ZX after mandatory LS transformations) packed as a dictionary.
        is_final_vis: True if visualising the final BlockGraph.
        iter_fail: True if the visualisation is being called due to a build failure.
        block_style: str = "pipe"

    """

    # Create foundational Matplotlib objects
    fig, ax = init_vis(is_cross_edge, is_final_vis)

    # Shared settings
    cube_size = [0.33 if block_style == "pipe" else 1] * 3
    edge_col = "black"

    # Add cubes
    for cube_id in bgraph.nodes():
        # Get coordinates first
        cube_coords = bgraph.nodes[cube_id]["coords"]

        # Only visualise nodes that have been placed
        if cube_coords:
            zx_block = bgraph.nodes[cube_id]["zx_block"]
            if zx_block:
                edge_col = "black" if zx_block.kind != "OOO" else "white"
                _ = render_block(
                    ax,
                    cube_id,
                    cube_coords,
                    cube_size,
                    zx_block,
                    edge_col=edge_col,
                    taken=taken,
                )

    # Add pipes (edges)
    for u_id, v_id in bgraph.edges():
        u_coords = bgraph.edges[(u_id, v_id)]["start_coords"]
        v_coords = bgraph.edges[(u_id, v_id)]["end_coords"]
        pipe_kind = bgraph.edges[(u_id, v_id)]["kind"]

        if u_coords is not None and v_coords is not None:
            _ = render_pipe(ax, u_coords, v_coords, pipe_kind)

    # Define visualisation boundaries
    _ = adjust_plot_styles(fig, ax, bgraph, is_final_vis, iter_fail=iter_fail)

    # Setup overlays for input ZX and base_graph
    setup_overlays(fig, in_zx, base_graph)

    # Show
    fig.canvas.mpl_connect("pick_event", lambda e: onpick_handler(e, ax))
    plt.show()


############
# OVERLAYS #
############
def setup_overlays(fig: plt.Figure, in_zx: dict[str, Any], base_graph: dict[str, Any]):
    """Create 50% width buttons and click handlers for the graph overlays."""

    # Bottom button regions [left, bottom, width, height]
    ax_btn_zx = fig.add_axes([0.0, 0.0, 0.5, 0.08])
    ax_btn_base = fig.add_axes([0.5, 0.0, 0.5, 0.08])

    btn_zx = Button(ax_btn_zx, "RAW ZX INPUT", color="#f2f3fb", hovercolor="#b2b2b2")
    btn_base = Button(ax_btn_base, "BASE ZX GRAPH", color="#f2f3fb", hovercolor="#b2b2b2")

    # Toggle click logic
    def toggle_overlay(overlay_key: str, graph_data: dict[str, Any]):
        other_key = "base_graph" if overlay_key == "in_zx" else "in_zx"

        # Close the other overlay if it's currently open
        if fig._overlay_axes[other_key] is not None:
            fig._overlay_axes[other_key].remove()
            fig._overlay_axes[other_key] = None

        # If selected overlay is open, close it (toggle off)
        if fig._overlay_axes[overlay_key] is not None:
            fig._overlay_axes[overlay_key].remove()
            fig._overlay_axes[overlay_key] = None
        else:
            # Create a full-width overlay axis restricted to 30% height just above the button strip
            overlay_ax = fig.add_axes([0.0, 0.08, 1.0, 0.30])
            fig._overlay_axes[overlay_key] = overlay_ax

            # Render the ZX structure onto this new axis
            render_overlay_graph(overlay_ax, graph_data)

        fig.canvas.draw_idle()

    # Bind
    btn_zx.on_clicked(lambda event: toggle_overlay("in_zx", in_zx))
    btn_base.on_clicked(lambda event: toggle_overlay("base_graph", base_graph))

    # Reference to buttons (so they remain clickable)
    fig._buttons.extend([btn_zx, btn_base])


def render_overlay_graph(ax: plt.Axes, graph_data: dict[str, Any]):
    """Rebuilds and draws the NX layout onto the targeted overlay axis using existing rules."""

    node_colors = []
    positions = {}

    for spider_id, zx_type in graph_data["types"].items():
        row = graph_data["rows"].get(spider_id, -1)
        qubit = graph_data["qubits"].get(spider_id, -1)
        positions[spider_id] = [row, -qubit]

        try:
            node_colors.append(ZXColors.lookup(zx_type))
        except KeyError:
            node_colors.append("#FD1818")

    ax.patch.set_facecolor("#f2f3fb")
    ax.patch.set_alpha(0.7)  # Ensure full opacity

    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 2. Draw Edges (LineCollections / standard plots)
    # Group coordinates by line segments: [[x1, x2], [y1, y2]]
    for (u, v), edge_type in graph_data["edge_types"].items():
        if u in positions and v in positions:
            x_coords = [positions[u][0], positions[v][0]]
            y_coords = [positions[u][1], positions[v][1]]

            try:
                col = ZXColors.lookup(edge_type)
            except KeyError:
                col = "black"

            ax.plot(x_coords, y_coords, color=col, zorder=1)

    # 3. Draw Nodes (Scatter plot)
    # Extract structural arrays
    node_ids = list(positions.keys())
    x_nodes = [positions[n][0] for n in node_ids]
    y_nodes = [positions[n][1] for n in node_ids]

    ax.scatter(
        x_nodes,
        y_nodes,
        c=node_colors,
        edgecolors="#111111",
        s=300,  # Marker size matching NetworkX defaults
        zorder=2,  # Force node dots above the edge connections
    )

    # 4. Draw Labels (Text objects)
    for n_id, (x, y) in positions.items():
        # Main Node ID Label
        ax.text(
            x,
            y,
            str(n_id),
            fontsize=9,
            ha="center",
            va="center",
            zorder=3,  # Force text on top
            color="black",
        )

        # Add phase label underneath if it is non-zero
        phase = graph_data.get("phases", {}).get(n_id, 0)
        if phase != 0 and phase != Fraction(1 / 1):
            if hasattr(phase, "numerator"):
                num, den = phase.numerator, phase.denominator
                if num == 1:
                    phase_str = f"π/{den}" if den != 1 else "π"
            else:
                phase_str = f"{phase}"
            ax.text(
                x,
                y - 0.25,  # Slight vertical offset downwards
                phase_str,
                fontsize=8,
                ha="center",
                va="top",  # Top-aligned so it expands cleanly downwards
                zorder=3,
                color="black",
                weight="bold",  # Bold styling to distinguish it from the ID
            )

    # Auto-scale the limits of the white box to match the positions drawn
    if positions:
        ax.set_xlim(min(x_nodes) - 1, max(x_nodes) + 1)
        ax.set_ylim(min(y_nodes) - 1, max(y_nodes) + 1)


##################
# FIG MANAGEMENT #
##################
def init_vis(is_cross_edge: bool, is_final_vis: bool):
    """Initialise main visualisation and add state trackers to the main visualisation.

    This function handles the initialisation of a number of state trackers needed to track and
    manage states for the main visualisation function. These are added directly to the figure (fig).
    The function also initialises and returns several critical objects used across the visualisation.

    Args:
        bgraph: The BlockGraph being built.
        is_cross_edge: True if the current edge is a cross edge.
        is_final_vis: A boolean flag to tell if visualisation is the final built BlockGraph.

    Return:
        fig: The Matplotlib Figure object to which state trackers (e.g., fig.show_overlay) will be attached.
        ax: The Matplotlib ax.

    AI disclaimer:
        category: Coding partner (see CONTRIBUTING.md for details).
        model: Gemini, 2.5 Flash.

    """

    # Create figure and ax
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection="3d")

    # Push the 3D plot up slightly to leave comfortable room for a 30% height overlay at the bottom
    fig.subplots_adjust(bottom=0.15)
    fig.ax = ax

    # Beams
    fig.beam_artists = []
    fig.show_beams = False

    # Track references to prevent garbage collection and maintain toggle states
    fig._overlay_axes = {"in_zx": None, "base_graph": None}
    fig._buttons = []

    return fig, ax


def adjust_plot_styles(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    bgraph: nx.Graph,
    is_final_vis: bool,
    iter_fail: bool = False,
):
    """Adjust the dimensions of the matplotlib plot.

    This function adjusts the dimensions (and therefore "zoom") of the main matplotlib
    pane. It defines the optimal dimensions based on a list of coordinates sent to it,
    which should itself contain all objects being displayed.

    Args:
        fig: The Matplotlib Figure object.
        ax: Matplotlib's 3D subplot object.
        bgraph: The BlockGraph being built.
        is_final_vis: A boolean to flag if the visualisation is the very last output.
        iter_fail: Boolean to flag if current visualisation comes right after an iteration failure.

    AI disclaimer:
        category: Coding partner (see CONTRIBUTING.md for details).
        model: Gemini, 2.5 Flash.

    """

    # Calculate positions of all contents
    all_static_coords: Annotated[np.ndarray, Literal["N", 3]] = np.array(
        [c for c in nx.get_node_attributes(bgraph, "coords").values() if c]
    )

    max_range = 5
    if all_static_coords.size > 0:
        max_x, min_x = all_static_coords[:, 0].max(), all_static_coords[:, 0].min()
        max_y, min_y = all_static_coords[:, 1].max(), all_static_coords[:, 1].min()
        max_z, min_z = all_static_coords[:, 2].max(), all_static_coords[:, 2].min()

        max_range = max((max_x - min_x), (max_y - min_y), (max_z - min_z)) / 2.0
        mid = np.array([(max_x + min_x) / 2, (max_y + min_y) / 2, (max_z + min_z) / 2])

        ax.set_xlim(mid[0] - max_range - 1, mid[0] + max_range + 1)
        ax.set_ylim(mid[1] - max_range - 1, mid[1] + max_range + 1)
        ax.set_zlim(mid[2] - max_range - 1, mid[2] + max_range + 1)
    else:
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    # Clean string formatter to drop trailing decimals
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.zaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    # BG colours (yellow: progress, green: success, red: failure)
    bg_colour = "#353535" if is_final_vis else "#b8b8b8"
    if iter_fail:
        bg_colour = "#fcbbb8"
    fig.patch.set_facecolor(bg_colour)
    ax.patch.set_facecolor(bg_colour)

    # Return max range for subsequent calculation of beam length
    return max_range


###################
# BLOCK RENDERERS #
###################
def render_block(
    ax: matplotlib.axes.Axes,
    node_id: int,
    coords: tuple[int, int, int],
    size: list[float],
    zx_block: ZXBlock,
    alpha: float = 1.0,
    edge_col: str = "black",
    border_width: float = 0.5,
    taken: set[StandardCoord] = [],
) -> Poly3DCollection:
    """Render a regular (non-Hadamard) block.

    This function creates a 3D cube to an existing Matplotlib ax. It takes the position,
    size and other graphical characteristics as parameters, applies specific face colors
    based on the `node_type`, and, if applicable, attaches invisible labels and direction
    quivers for debugging and interaction.

    Args:
        ax: Matplotlib's 3D subplot object.
        node_id: The ID of the node
        coords: The (x, y, z) coordinates of the block.
        size: The (size_x, size_y, size_z) of the block.
        zx_block: The ZX block at the given coordinates.
        alpha: The transparency for the block
        edge_col: The color for the edges of blocks.
        border_width: The width for borders of block.
        taken: The coordinates occupied by any blocks/pipes placed as a result of previous operations.

    """

    # General dimensions
    x, y, z = coords
    size_x, size_y, size_z = size
    vertices = get_vertices(
        x,
        y,
        z if zx_block.zx_type != "T" else z - 1.33,
        size_x,
        size_y,
        size_z if zx_block.zx_type != "T" else 3,
        zx_block.zx_type,
    )
    faces = get_faces(vertices)

    # Add colours as per map
    cols = zx_block.get_face_colors
    if cols[0] == ZXColors.Y:
        cols = tuple([cols[0]] * 6)
    face_cols = [cols[2]] * 2 + [cols[1]] * 2 + [cols[0]] * 2

    # Join into Poly collection
    poly_collection = Poly3DCollection(
        faces,
        facecolors=face_cols,
        linewidths=border_width if zx_block.zx_type != "T" else 3,
        edgecolors=edge_col,
        alpha=alpha,
        picker=True,
        label=node_id,
    )

    # Attach labels if node has ID
    if node_id != "TBD":
        if zx_block.zx_type != "O":
            label_ax = [0 if i else 1 for i in zx_block.get_open_axes]
            diffs = [tuple(label_ax), tuple([-1 * i for i in label_ax])]
        else:
            diffs = zx_block.get_move_vectors

        for d in diffs:
            label_pos = (
                coords[0] + d[0] / 2,
                coords[1] + d[1] / 2,
                coords[2] + d[2] / 2,
            )

            label_pos_normalised = (
                coords[0] + d[0],
                coords[1] + d[1],
                coords[2] + d[2],
            )

            if label_pos_normalised not in taken:
                ax.text(
                    label_pos[0],
                    label_pos[1],
                    label_pos[2],
                    s=f"{node_id}: {zx_block.kind}",
                    color="black",
                    visible=False,
                    fontsize="small",
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
    kind: str,
    edge_col: str = "black",
    border_width: float = 0.5,
    alpha: float = 1.0,
) -> list[Poly3DCollection]:
    """Add a pipe to the Matplotlib ax.

    This function adds a pipe (regular or hadamard) to an existing Matplotlib ax.
    It takes the position, size and other graphical characteristics as parameters,
    applies specific face colors based on the `node_type`, and, if applicable,
    attaches invisible labels and direction quivers for debugging and interaction.

    Args:
        ax: Matplotlib's 3D subplot object.
        u_coords: (x, y, z) coordinates of the source cube.
        v_coords: (x, y, z) coordinates of the target cube.
        kind: The type of the pipe block (e.g., 'Xh' for Hadamard).
        edge_col: color of the edges for the edge/pipe.
        border_width: width for borders of edge.
        alpha: any desired value for alpha (transparency).

    Returns:
        list[Poly3DCollection]: A list containing the Matplotlib artists for the pipe sections.

    AI disclaimer:
        category: Coding partner (see CONTRIBUTING.md for details).
        model: Gemini, 2.5 Flash.

    """

    # Convert positions to np.arrays
    u_coords = np.array(u_coords)
    v_coords = np.array(v_coords)

    # Establish midpoint and pipe length
    midpoint = (u_coords + v_coords) / 2
    original_length = np.linalg.norm(v_coords - u_coords)
    adjusted_length = original_length - 0.33

    # Process pipe
    if adjusted_length > 0:
        orientation = np.argmax(np.abs(v_coords - u_coords))
        size = [0.33, 0.33, 0.33]
        size[orientation] = float(adjusted_length)
        face_cols = ["gray"] * 6

        col = node_hex_map.get(kind.replace("*", "").lower(), ["gray"] * 3)
        face_cols = [col[2]] * 2 + [col[1]] * 2 + [col[0]] * 2

        # Regular pipes
        if "H" not in kind:
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
        elif "H" in kind:
            # Break into three sections
            #   2 * coloured ends
            #   1 * middle yellow ring
            if adjusted_length > 0:
                yellow_length = 0.1 * adjusted_length
                colored_length = 0.45 * adjusted_length

                # Skip if internal lengths are invalid
                if colored_length < 0 or yellow_length < 0:
                    return

                size_col = list(size)
                size_yellow = list(size)
                size_col[orientation] = float(colored_length)
                size_yellow[orientation] = float(yellow_length)

                offset1 = np.zeros(3)
                offset3 = np.zeros(3)

                offset1[orientation] = -(yellow_length / 2 + colored_length / 2)
                offset3[orientation] = yellow_length / 2 + colored_length / 2

                centre1 = midpoint + offset1
                centre2 = midpoint
                centre3 = midpoint + offset3

                # Base of hadamard
                face_cols_1 = list(face_cols)

                # Middle yellow ring
                face_cols_yellow = ["yellow"] * 6

                # Far end of the hadamard
                # Note. Keeping track of the correct rotations proved tricky
                # Keep this bit spread out across lines for comprehensibility
                face_cols_2 = ["gray"] * 6
                rotated_kind = rotate_pipe(
                    kind[:3], tuple([0 if i != max(size) else 1 for i in size])
                )
                col = node_hex_map.get(rotated_kind.lower(), ["gray"] * 3)
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
    """Render edges/pipes.

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
        category: Coding partner (see CONTRIBUTING.md for details).
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


def get_vertices(
    x: int,
    y: int,
    z: int,
    size_x: float,
    size_y: float,
    size_z: float,
    zx_type: str | None = None,
) -> Annotated[NDArray[np.float64], Literal[..., 3]]:
    """Calculate the coordinates of the eight vertices of a cuboid.

    This function calculates the exact position of the vertices of a cuboind based on
    a central position and the desired dimensions for the cuboid.

    Args:
        x: x-coordinate of the centre of the cuboid.
        y: y-coordinate of the centre of the cuboid.
        z: z-coordinate of the centre of the cuboid.
        size_x: length of the cuboid along the x-axis.
        size_y: length of the cuboid along the y-axis.
        size_z: length of the cuboid along the z-axis.
        zx_type (optional): The ZX type of the cube being rendered.

    Returns:
        array: array (numpy) of shape (8, 3) where each row represents the (x, y, z) coordinates of a vertex of the cuboid.

    AI disclaimer:
        category: Coding partner (see CONTRIBUTING.md for details).
        model: Gemini, 2.5 Flash.

    """

    half_size_x = size_x / 2
    half_size_y = size_y / 2
    half_size_z = size_z / 2
    return np.array(
        [
            [
                x - (half_size_x if zx_type != "T" else 0),
                y - (half_size_y if zx_type != "T" else 0),
                z - half_size_z,
            ],
            [
                x + (half_size_x if zx_type != "T" else 0),
                y - (half_size_y if zx_type != "T" else 0),
                z - half_size_z,
            ],
            [
                x + (half_size_x if zx_type != "T" else 0),
                y + (half_size_y if zx_type != "T" else 0),
                z - half_size_z,
            ],
            [
                x - (half_size_x if zx_type != "T" else 0),
                y + (half_size_y if zx_type != "T" else 0),
                z - half_size_z,
            ],
            [x - half_size_x, y - half_size_y, z + half_size_z],
            [x + half_size_x, y - half_size_y, z + half_size_z],
            [x + half_size_x, y + half_size_y, z + half_size_z],
            [x - half_size_x, y + half_size_y, z + half_size_z],
        ]
    )


def get_faces(vertices: Annotated[NDArray[np.float64], Literal[..., 3]]):
    """Define the faces of a cuboid based on its vertices.

    This function takes an array of vertices and returns a list that defines the faces of a
    cuboid to render as part of a 3D visualisation.

    Args:
        vertices: array (numpy) of shape (8, 3) where each row represents the (x, y, z) coordinates of a vertex of a cuboid.

    Returns:
        list: list of lists where each inner list represents a face and contains the coords of the vertices for that face.

    AI disclaimer:
        category: Coding partner (see CONTRIBUTING.md for details).
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


##################
# EVENT MANAGERS #
##################
def onpick_handler(e: matplotlib.backend_bases.PickEvent, ax: matplotlib.axes.Axes):
    """Handle click events on a visualisation to toggle associated labels/artists.

    Upon clicking a cube, this function looks up the cube ID and toggles the visibility of
    its label and the 3D line pointing to it.

    Args:
        e: The Matplotlib PickEvent object containing the clicked artist.
        ax: The Matplotlib Axes object (specifically Axes3D in this context)
            containing all the children artists.

    AI disclaimer:
        category: Coding partner (see CONTRIBUTING.md for details).
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
