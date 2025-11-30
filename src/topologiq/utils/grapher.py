"""
This file contains Topologiq's main visualisation facility. 

This file contains functions that can help create full-detail visualisations of how
the pathfinder algorithm goes about resolving specific edges.

Usage: 
    Call `vis_3d()` programmatically with an appropriate parameter combination.

NB! AI policy. If you use AI to modify this file, refer to `./README` for appropriate disclaimer guidelines.
"""

from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.widgets import Button

from typing import List, Tuple, Union
from PIL import Image

from topologiq.utils.classes import PathBetweenNodes, StandardBlock, StandardCoord
from topologiq.utils.grapher_common import (
    node_hex_map,
    edge_paths_to_nx_graph,
    render_block,
    render_pipe,
    figure_to_png,
    onpick_handler,
    toggle_animation_handler,
    toggle_winner_path_handler,
    toggle_beams_handler,
    toggle_targets_handler,
    toggle_valid_paths_handler,
    toggle_overlay_handler,
    hide_overlay_handler,
    toggle_prox_paths_handler,
    keypress_handler,

)
from topologiq.utils.utils_zx_graphs import kind_to_zx_type


###############################
# MAIN VISUALISATION FUNCTION #
###############################
def vis_3d(
    nx_g: nx.Graph,
    partial_nx_g: nx.Graph,
    edge_paths: dict,
    valid_paths: dict[StandardBlock, List[StandardBlock]] | None,
    winner_path: PathBetweenNodes | List[StandardBlock] | None,
    src_block_info: StandardBlock | None,
    tent_coords: List[StandardCoord] | None,
    tent_tgt_kinds: List[str] | None,
    hide_ports: bool = False,
    all_search_paths: dict[StandardBlock, List[StandardBlock]] | None = [],
    debug: int = 1,
    vis_options: Tuple[Union[None, str], Union[None, str]] = (None, None),
    src_tgt_ids: Tuple[int, int] | None = None,
    fig_data: matplotlib.figure.Figure | None = None,
    filename_info: Tuple[str, int] | None = None,
):
    """Create a granular visualisation of a single iteration of the inner pathfinder algorithm.

    This function visualises several key aspects of a given pathfinder iteration, including the
    source and target cubes (one or many) for the iteration, the beams present during the iteration,
    all valid paths found in the iteration, and the path ultimately chosen for inclusion in the final
    output. Interactive buttons enable users to show/hide aspects of the visualisation.

    Args:
        nx_g: The main NX graph the algorithm uses to guide its discovery process.
        partial_nx_g: A partial NX graph with completed progress.
        edge_paths: An edge-by-edge/block-by-block summary of the space-time diagram Topologiq builds.
        valid_paths: A dictionary of topologically-correct paths found during a single pathfinder iteration.
        winner_path: The valid_path that was ultimately chosen for inclusion in the final result.
        src_block_info: The information of the source cube including its position in the 3D space and its kind.
        tent_coords: A list of tentative target coordinates to find paths to.
        tent_tgt_kinds: A list of kinds matching the zx-type of target block.
        hide_ports (optional): If True, boundary spiders are considered by Topologiq but not displayed in visualisations.
        all_search_paths (optional): A dictionary containing all paths searched by the inner pathfinder algorithm.
        debug (optional): Debug mode (0: off, 1: graph manager, 2: pathfinder, 3: pathfinder w. discarded paths).
        vis_options (optional): Visualisation settings provided as a Tuple.
            vis_options[0]: If enabled, triggers "final" or "detail" visualisations.
                (None): No visualisation.
                (str) "final" | "detail": A single visualisation of the final result or one visualisation per completed edge.
            vis_options[1]: If enabled, triggers creation of an animated summary for the entire process.
                (None): No animation.
                (str) "GIF" | "MP4": A step-by-step visualisation of the process in GIF or MP4 format.
        src_tgt_ids (optional): The IDs of the (src, tgt) spiders/blocks for the pathfinder iteration.
        fig_data (optional): the Matplotlib figure of the input graph (used as optional overlay).

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    # Preliminaries
    # ---

    # Create foundational Matplotlib objects
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection="3d")
    fig.ax = ax

    (
        is_final_vis,
        show_src_block, src_coords, src_kind,
        tgt_kind, valid_paths_mini_graph,
        valid_paths_block_positions, taken
    ) = _init_vis(fig, edge_paths, valid_paths, winner_path, src_block_info, tent_coords, tent_tgt_kinds, debug)

    
    # Static elements
    # ---
    
    # Shared settings
    size = [1.0, 1.0, 1.0]
    edge_col = "black"

    # Source
    if src_block_info and show_src_block:
        edge_col = "gold"
        _ = render_block(ax, src_tgt_ids[0], src_coords, size, src_kind, node_hex_map, edge_col=edge_col, border_width=3, taken=taken)

    # Winner path
    if winner_path and not is_final_vis:
        edge_col = "gold"
        _render_winner_path(
            fig,
            ax,
            winner_path,
            tent_coords,
            src_coords,
            node_hex_map,
            src_tgt_ids=src_tgt_ids,
            hide_ports=hide_ports,
            edge_col=edge_col,
            taken=taken,
        )
     
    # Pre-existing blocks / previous edges
    _render_nx_graph(
        fig,
        ax,
        partial_nx_g,
        src_coords,
        tent_coords,
        is_final_vis,
        hide_ports,
        taken=taken,
        valid_paths_block_positions=valid_paths_block_positions,
    )

    # Dynamic (toggle) elements
    # ---

    # Valid paths
    if valid_paths and not is_final_vis:
        _render_nx_graph(
            fig,
            ax,
            valid_paths_mini_graph,
            src_coords,
            tent_coords,
            is_final_vis,
            hide_ports,
            taken=taken,
            tgt_kind=tgt_kind,
        )

    # Targets
    if tent_coords and not is_final_vis:
        _render_tent_tgts(fig, ax, tent_coords, tgt_kind, node_hex_map, src_tgt_ids)

    # Beams
    if nx_g and not is_final_vis:
        _render_beams(fig, ax, nx_g)
    
    # ZX input graph overlay
    if fig_data:
        _render_zx_graph_overlay(fig, partial_nx_g, edge_paths, is_final_vis, fig_data, src_tgt_ids)
    
    # All paths searched by pathfinder algorithm
    def _update_fallback(frame):
        """ Placeholder update function when no paths are found or debug is low."""
        return []

    if all_search_paths and debug > 2:
        (
            animation_sequence, 
            num_paths, 
            num_frames, 
            animation_interval_ms, 
            TARGET_DURATION_MS
        ) = _prepare_search_paths_data(fig, all_search_paths, valid_paths)
        
        if num_paths > 0:
            update, persistent_green_artists = _setup_path_animation(
                fig, ax, animation_sequence, num_paths
            )
        else:  # Fallback if paths were provided but list was empty
            update = _update_fallback
            persistent_green_artists = []
            num_frames = 1
            animation_interval_ms = 500
            TARGET_DURATION_MS = 1000
            num_paths = 0 
    else:  # Define necessary variables for the button handlers to work when debug < 3
        update = _update_fallback
        persistent_green_artists = []
        num_frames = 1
        animation_interval_ms = 500
        TARGET_DURATION_MS = 1000
        num_paths = 0 

    # Plot adjustments
    # ---
    _adjust_plot_dimensions(fig, ax, partial_nx_g, valid_paths, tent_coords, src_block_info, is_final_vis, pathfinder_success=True if winner_path else False)

    # Interactive buttons & elements
    # ---

    # Base dimensions & paddings
    BTN_L, BTN_B, BTN_W, BTN_H = [0.01, 0.05, 0.18, 0.05]
    BTN_PAD = 0.01

    # Show in high-debug modes
    if debug > 2:
        # Replay path search
        ax_anim = fig.add_axes([BTN_L, BTN_B + (BTN_H + BTN_PAD)*5, BTN_W, BTN_H])
        btn_anim = Button(ax_anim, 'Replay Path Search')
        btn_anim.set_active(True)
        btn_anim.on_clicked(
            lambda e: toggle_animation_handler(
                e,
                fig,
                btn_anim,
                persistent_green_artists,
                update, # The locally defined update function from Section 5
                num_frames,
                animation_interval_ms,
                num_paths,
                TARGET_DURATION_MS
            )
        )

        # Show/hide winner path (NEW)
        ax_win = fig.add_axes([BTN_L, BTN_B + (BTN_H + BTN_PAD)*4, BTN_W, BTN_H])
        btn_win = Button(ax_win, 'Hide Winner Path')
        btn_win.on_clicked(
            lambda e: toggle_winner_path_handler(
                e,
                fig,
                btn_win,
                btn_valid
            )
        )
        
        # Show/hide beams
        ax_beams = fig.add_axes([BTN_L, BTN_B + (BTN_H + BTN_PAD)*3, BTN_W, BTN_H])
        btn_beams = Button(ax_beams, 'Show Beams')
        btn_beams.on_clicked(
            lambda e: toggle_beams_handler(
                e,
                fig,
                btn_beams
            )
        )

        # Show/hide targets
        ax_tgt = fig.add_axes([BTN_L, BTN_B + (BTN_H + BTN_PAD)*2, BTN_W, BTN_H])
        btn_tgt = Button(ax_tgt, 'Hide Targets' if fig.show_tent_tgt_blocks else 'Show Targets')
        btn_tgt.on_clicked(
            lambda e: toggle_targets_handler(
                e,
                fig,
                btn_tgt
            )
        )

        # Show/hide valid paths
        ax_valid = fig.add_axes([BTN_L, BTN_B + (BTN_H + BTN_PAD), BTN_W, BTN_H]) # Position next to Targets
        btn_valid = Button(ax_valid, 'Show Valid Paths')
        btn_valid.on_clicked(
            lambda e: toggle_valid_paths_handler(
                e,
                fig,
                btn_valid,
                btn_win # Dependency on the winner path button handle
            )
        )

        # Show/hide proximate paths
        ax_prox = fig.add_axes([BTN_L, BTN_B, BTN_W, BTN_H])
        btn_prox = Button(ax_prox, "Prox Paths")
        btn_prox.set_active(True)
        btn_prox.on_clicked(
            lambda e: toggle_prox_paths_handler(e, fig, btn_prox, tent_coords)
        )

    # Show always if constituent elements exist
    if fig_data:

        # Overlay of input ZX-graph
        BTN_W_MIN, BTN_B = (0.04, 0.0)
        current_width = BTN_W_MIN if fig.show_overlay else BTN_W
        ax_overlay_btn = fig.add_axes([1 - current_width, BTN_B, current_width, BTN_H])

        for spine in ax_overlay_btn.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linestyle("dotted")

        initial_btn_text = 'X' if fig.show_overlay else 'Show Input ZX Graph'
        btn_overlay = Button(ax_overlay_btn, initial_btn_text) 
        fig.overlay_button_handle = btn_overlay

        # Attach the primary button handler
        btn_overlay.on_clicked(
            lambda e: toggle_overlay_handler(e, fig, btn_overlay, [BTN_W, BTN_W_MIN, BTN_B, BTN_H])
        )

        # Overlay click-to-hide functionality
        if fig_data and hasattr(fig, 'ax_overlay'):
            btn_pos = [BTN_W, BTN_W_MIN, BTN_B, BTN_H]
            fig.canvas.mpl_connect(
                'button_press_event',
                lambda e: hide_overlay_handler(e, fig, toggle_overlay_handler, btn_overlay, btn_pos)
            )

    # Connect events
    fig.canvas.mpl_connect("pick_event", lambda e: onpick_handler(e, ax))
    fig.canvas.mpl_connect('key_press_event', lambda e: keypress_handler(e, fig, btn_prox, tent_coords))

    # Save to file if applicable
    repo_root: Path = Path(__file__).resolve().parent.parent
    temp_folder_path = repo_root / "output/temp"
    if filename_info:
        circuit_name, c = filename_info
        Path(temp_folder_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{temp_folder_path}/{circuit_name}{c:03d}.png")

        # Save visualisation data to TXT file if debug mode is at max (4).
        if debug == 4:
            temp_folder_path = repo_root / "output/txt"
            file_path = f"{temp_folder_path}/{circuit_name}-last-edge.txt"
            
            with open(file_path, "w") as f:
                f.write("# PATHFINDER ITERATION SUMMARY\n")
                f.write(f"Circuit name: {circuit_name if circuit_name else 'None'}\n")
                f.write(f"Edge: {src_tgt_ids if src_tgt_ids else 'None'}\n")

                f.write("\n## Iteration params\n")
                f.write(f"src_tgt_ids: {src_tgt_ids if src_tgt_ids else 'None'}\n")
                f.write(f"src_block_info: {src_block_info if src_block_info else 'None'}\n")
                f.write(f"tent_coords: {tent_coords if tent_coords else 'None'}\n")
                f.write(f"tent_tgt_kinds: {tent_tgt_kinds if tent_tgt_kinds else 'None'}\n")
                f.write(f"taken: {taken if taken else 'None'}\n")

                f.write("\n## Pre-existent paths at time of iteration\n")
                if edge_paths:
                    for k, edge_path in edge_paths.items():
                        f.write(f"{k}: {edge_path}\n")
                else:
                    f.write("None\n")

                f.write("\n## winner_path\n")
                f.write(repr(winner_path) if winner_path else "None")

                f.write("\n\n## valid_paths\n")
                if valid_paths:
                    for path in valid_paths.values():
                        f.write(f"{str(path)}\n")
                else:
                    f.write("None\n")

                f.write("\n## all_search_paths\n")
                if all_search_paths:
                    for path in all_search_paths.values():
                        f.write(f"{str(path)}\n")
                else:
                    f.write("None\n")
                f.write("\n")

                f.close()

    # Show
    if debug > 1 or vis_options[0] == "detail" or (vis_options[0] == "final" and is_final_vis):
        plt.show()
    else:
        plt.close()


#######################
# AUXILIARY FUNCTIONS #
#######################
def _init_vis(
    fig: matplotlib.figure.Figure,
    edge_paths: dict,
    valid_paths: dict[StandardBlock, List[StandardBlock]] | None,
    winner_path: PathBetweenNodes | List[StandardBlock] | None,
    src_block_info: StandardBlock | None,
    tent_coords: List[StandardCoord] | None,
    tent_tgt_kinds: List[str] | None,
    debug: int = 2,
):
    """Initialise main visualisation and add state trackers to the main visualisation.

    This function handles the initialisation of a number of state trackers needed to track and 
    manage states for the main visualisation function. These are added directly to the figure (fig). 
    The function also initialises and returns several critical objects used across the visualisation.

    Args:
        fig: The Matplotlib Figure object to which state trackers (e.g., fig.show_overlay) will be attached.
        edge_paths: An edge-by-edge/block-by-block summary of the space-time diagram Topologiq builds.
        valid_paths: A dictionary of topologically-correct paths found during a single pathfinder iteration.
        winner_path: The valid_path that was ultimately chosen for inclusion in the final result.
        src_block_info: The information of the source cube including its position in the 3D space and its kind.
        tent_coords: A list of tentative target coordinates to find paths to.
        tent_tgt_kinds: A list of kinds matching the zx-type of target block.
        debug (optional): Debug mode (0: off, 1: graph manager, 2: pathfinder, 3: pathfinder w. discarded paths).

    Returns:
        init_objects: A tuple containing initialisation objects and settings:
            is_final_vis: True if this is the final visualisation.
            show_src_block: True if the source block should be rendered.
            src_coords: The 3D coordinates of the source block.
            src_kind: The kind of the source block.
            tgt_kind: The kind of the target block(s).
            valid_paths_mini_graph: A networkx graph representing the valid paths found in a pathfinder iteration.
            valid_paths_block_positions: Node positions for the valid paths mini-graph.
            taken: A list of coordinates occupied by existing blocks, targets, and source block.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    # Determine type of visualisation
    is_final_vis = False if tent_coords and tent_tgt_kinds and debug > 0 else True
    is_single_target = False if is_final_vis else len(tent_coords) == 1 and len(tent_tgt_kinds) == 1
        
    # Source block
    show_src_block = False if is_final_vis else True

    # Tentative targets
    fig.target_artists = []
    fig.show_tent_tgt_blocks = is_single_target

    # Valid paths
    fig.valid_path_artists = []
    fig.show_valid_paths = False
    fig.animation_handle = None

    # Searched paths
    fig.show_static_search_paths = False
    fig.static_search_artists = []

    # Beams
    fig.beam_artists = []
    fig.show_beams = False

    # Winner paths
    fig.winner_path_artists = []
    fig.show_winner_path = True

    # Input ZX graph overlay
    fig.ax_overlay = None
    fig.show_overlay = debug > 0
    fig.overlay_image_artist = None
    fig.overlay_button_handle = None

    # Proximate paths
    fig.show_prox_paths = False
    fig.all_search_paths_raw = []
    fig.prox_path_artists = []
    fig.prox_distance_threshold = 1 
    fig.prox_filtered_paths = []
    fig.prox_view_mode = 'ALL'
    fig.prox_current_index = 0

    # Get source and target info for current pathfinder iteration
    src_coords, src_kind = src_block_info if src_block_info else (None, None)
    tgt_kind = None
    valid_paths_mini_graph = None
    valid_paths_block_positions = None
    
    if not is_final_vis:
        if src_block_info:
             src_coords, src_kind = src_block_info 
        if len(tent_tgt_kinds) == 1:
            tgt_kind = tent_tgt_kinds[0]
        else:
            # NOTE: Assumes tent_tgt_kinds is not empty.
            zx_type = kind_to_zx_type(tent_tgt_kinds[0])
            tgt_kind = zx_type.lower()*3 if zx_type in ["X", "Y", "Z"] else "ooo"
        
        # Valid paths mini-graph
        valid_paths_mini_graph = edge_paths_to_nx_graph(valid_paths) if valid_paths else nx.Graph()
        valid_paths_block_positions = nx.get_node_attributes(valid_paths_mini_graph, "coords")


    # Recalculate taken from incoming parameters
    taken = []
    if not is_final_vis:
        if tent_coords:
            taken.extend(tent_coords)
        if src_coords:
            taken.append(src_coords)
        
        # Check if winner_path exists before accessing its attributes.
        if winner_path:
            if isinstance(winner_path, PathBetweenNodes):
                taken.extend(winner_path.coords_in_path)
            else:
                taken.extend([p[0] for p in winner_path])
    
    # Use a standard for loop for clarity over list comprehension for side effects
    if edge_paths:
        for edge_path in edge_paths.values():
            if edge_path.get("path_coordinates") and edge_path["path_coordinates"] != "error":
                taken.extend(edge_path["path_coordinates"])
                
    taken = list(set(taken))

    # Wrap into a single tuple
    init_objects = (
        is_final_vis,
        show_src_block,
        src_coords,
        src_kind,
        tgt_kind,
        valid_paths_mini_graph,
        valid_paths_block_positions,
        taken
    )

    return init_objects


def _render_winner_path(
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        winner_path: PathBetweenNodes | List[StandardBlock] | None,
        tent_coords: List[StandardCoord] | None,
        src_coords: Tuple[int, int, int],
        node_hex_map: dict[str, list[str]],
        src_tgt_ids: Tuple[int, int] | None = None,
        hide_ports: bool = False,
        edge_col: str = "black",
        taken: List[StandardCoord] | List[None] | None = None,
        ):
    """Renders the blocks and pipes of the winner path.

    This function handles the rendering of blocks and pipe segments in the path selected as
    the best path for a given pathfinder iteration. The rendered artists are added to 
    the figure's state tracker for later interactive toggling.

    Args:
        fig: The Matplotlib Figure object used for storing rendered artists.
        ax: Matplotlib's 3D subplot object where the rendering occurs.
        winner_path: The valid_path that was ultimately chosen for inclusion in the final result.
        tent_coords: A list of tentative target coordinates to find paths to.
        src_coords: The coordinates for the source block.
        node_hex_map: A map of colors to use for the different faces of different blocks.
        src_tgt_ids (optional): The IDs of the (src, tgt) spiders/blocks for the pathfinder iteration.
        hide_ports (optional): If True, boundary spiders are considered by Topologiq but not displayed in visualisations.
        edge_col (optional): The color to use for edges. Defaults to "black".
        taken: A list of coordinates occupied by any blocks/pipes placed as a result of previous operations.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    # Preliminaries
    border_width = 1
    winner_path = winner_path.all_nodes_in_path if isinstance(winner_path, PathBetweenNodes) else winner_path

    if taken is None:
        taken = []

    if not winner_path:
        return

    # Loop over blocks in winner path and add
    path_length = len(winner_path)
    for i, (block_coords, block_kind) in enumerate(winner_path):
        current_artists = []

        # Cubes
        if "o" not in block_kind or (not hide_ports and block_kind == "ooo"):
            if block_coords != src_coords:
                size = [1.0, 1.0, 1.0] if block_kind != "ooo" else [0.9, 0.9, 0.9]
                block_id = src_tgt_ids[1] if block_coords in tent_coords else "TBD"
                current_artists = render_block(
                    ax,
                    block_id,
                    block_coords,
                    size,
                    block_kind,
                    node_hex_map,
                    edge_col=edge_col,
                    border_width=border_width,
                    taken=taken,
                )

        # Pipes
        elif "o" in block_kind:
            if i > 0 and i < path_length - 1:  # A pipe segment must have nodes before and after it.
                u_coords = winner_path[i-1][0]
                v_coords = winner_path[i+1][0]
                
                if u_coords is not None and v_coords is not None:
                    current_artists = render_pipe(ax, u_coords, v_coords, block_kind,  border_width=border_width, edge_col=edge_col)

        # Add to artists & manage visibility
        if current_artists:
            fig.winner_path_artists.extend(current_artists)
            for artist in current_artists:
                artist.set_visible(fig.show_winner_path)


def _render_tent_tgts(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    tent_coords: List[StandardCoord] | None,
    tgt_kind: str,
    node_hex_map: dict[str, list[str]],
    src_tgt_ids: Tuple[int, int] | None = None,
):
    """ Renders placeholder cubes for any number of tentative targets.

    This function handles the rendering of placeholder cubes in the positions sent to 
    the pathfinder algorithm as tentative targets. It can handle one or more tentative targets.
    If the tentative target kind is set, the cube will be drawn using that kind, else, placeholder
    cubes will be of a colour which corresponds with they ZX type. 

    Args:
        fig: The Matplotlib Figure object used for storing rendered artists.
        ax: Matplotlib's 3D subplot object where the rendering occurs.
        tent_coords: A list of tentative target coordinates to find paths to.
        tgt_kind: The kind of the target block (used to determine color and size).
        node_hex_map: A map of colors to use for the different faces of different blocks.
        src_tgt_ids (optional): The IDs of the (src, tgt) spiders/blocks for the pathfinder iteration.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    # Add loop
    for tent_coord in tent_coords:
            artists = render_block(
                ax,
                "TBD" if len(tent_coords) > 1 else src_tgt_ids[1],
                tent_coord,
                [0.9, 0.9, 0.9] if tgt_kind != "ooo" else [0.7, 0.7, 0.7],
                tgt_kind,
                node_hex_map,
                edge_col="violet",
                border_width=2,
            )
            
        # Add to artists & manage visibility
            fig.target_artists.extend(artists)
            for artist in artists:
                artist.set_visible(fig.show_tent_tgt_blocks)


def _render_beams(fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, nx_g: nx.Graph):
    """ Renders any beams saved to the nodes of the NX graph received as parameter.

    This function handles the rendering of any beams present in the information of the nodes 
    sent to the function as a parameter.

    Args:
        fig: the Matplotlib Figure object.
        ax: Matplotlib's 3D subplot object.
        nx_g: The main NX graph the algorithm uses to guide its discovery process.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    # Preliminaries
    all_beam_x = []
    all_beam_y = []
    all_beam_z = []

    # Add loop
    for node_id in nx_g:
        node_beams = nx_g.nodes()[node_id].get("beams", [])
        if node_beams:
            for beam in node_beams:
                for beam_coords in beam:
                    all_beam_x.append(beam_coords[0])
                    all_beam_y.append(beam_coords[1])
                    all_beam_z.append(beam_coords[2])

    # Create a single artist for all beams
    if all_beam_x:
        beam_artist = ax.scatter(
            all_beam_x,
            all_beam_y,
            all_beam_z,
            c="yellow",
            s=20,
            edgecolors="black",
            alpha=1,
            depthshade=True,
        )
        
        # Store the single artist for toggling
        fig.beam_artists.append(beam_artist)
        beam_artist.set_visible(fig.show_beams)


def _render_zx_graph_overlay(
    fig: matplotlib.figure.Figure,
    partial_nx_g: nx.Graph,
    edge_paths: dict,
    is_final_vis: bool,
    fig_data: matplotlib.figure.Figure,
    src_tgt_ids: Tuple[int, int] | None = None,
):
    """Generate PNG of the original input ZX graph to use as overlay over 3D visualisations.

    This function generates a PNG buffer of the input ZX graph and renders it as a bottom-right
    overlay on the main figure.

    Args:
        fig: The Matplotlib Figure object.
        partial_nx_g: A partial NX graph with completed progress.
        edge_paths: An edge-by-edge/block-by-block summary of the space-time diagram Topologiq builds.
        is_final_vis: a boolean to flag if the visualisation is the very last output.
        fig_data (optional): the Matplotlib figure of the input graph (used as optional overlay).
        src_tgt_ids (optional): The IDs of the (src, tgt) spiders/blocks for the pathfinder iteration.
    """

    # Prepare PNG buffer of the input ZX graph
    processed_edges = list(edge_paths.keys()) + [src_tgt_ids] if not is_final_vis else list(edge_paths.keys())
    
    png_buffer = figure_to_png(
        fig_data,
        processed_ids=partial_nx_g.nodes(),
        processed_edges=processed_edges,
        src_tgt_ids=src_tgt_ids if not is_final_vis else None,
    )
    overlay_image = Image.open(png_buffer)
    img_width, img_height = overlay_image.size
    aspect_ratio = img_width / img_height

    # Calculate optimal size and position for the overlay axes
    desired_height_ratio = 0.25
    calculated_width_ratio = (
        desired_height_ratio
        * (fig.get_figheight() / fig.get_figwidth())
        * aspect_ratio
    )

    # Width constraint (20% to 50%)
    min_width_ratio = 0.3
    max_width_ratio = 0.4
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
    left = 1 - calculated_width_ratio  # Align to right
    bottom = 0.0  # Align to bottom
    ax_overlay = fig.add_axes(
        [left, bottom, calculated_width_ratio, calculated_height_ratio]
    )

    # Hide unnecessary features
    ax_overlay.set_xticks([])
    ax_overlay.set_yticks([])
    ax_overlay.axis("off")

    # Overlay image
    overlay_array = np.asarray(overlay_image)
    overlay_artist = ax_overlay.imshow(overlay_array)

    # Store the artist and the axis
    fig.ax_overlay = ax_overlay 
    fig.overlay_image_artist = overlay_artist

    # Set initial alpha based on the correct state flag (fig.show_overlay)
    fig.ax_overlay.set_visible(True) # Keep the axis active
    fig.overlay_image_artist.set_alpha(1.0 if fig.show_overlay else 0.0)


def _render_nx_graph(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    nx_g: nx.Graph,
    src_coords: Tuple[int, int, int],
    tent_coords: List[StandardCoord] | None,
    is_final_vis: bool,
    hide_ports: bool = False,
    taken: List[StandardCoord] | List[None] | None = None,
    valid_paths_block_positions: dict[int, StandardCoord] | None = None,
    tgt_kind: str | None = None,
):
    """Renders the nodes/edges of an NX graph as blocks/pipes of a space-time diagram.

    This function visualizes a a NetworkX graph as a 3D space-time diagram using Matplotlib.
    It draws nodes as blocks and edges as pipes, distinguishing between two modes:
        1.  Static Mode (tgt_kind is None): Renders pre-existing components of the space-time diagram 
            that were completed in previous pathfinder iterations.
        2.  Dynamic Mode (tgt_kind is present): Renders the set of valid paths discovered during 
            the current pathfinder iteration, adding them to `fig.valid_path_artists` for interactive toggling.

    Args:
        fig: the Matplotlib Figure object.
        ax: Matplotlib's 3D subplot object.
        nx_g: An NX graph with appropriately formatted space-time diagram components.
        src_coords: The coordinates for the source block.
        tent_coords: A list of tentative target coordinates to find paths to.
        tgt_kind: The kind of the target block.
        is_final_vis: a boolean to flag if the visualisation is the very last output.
        hide_ports (optional): If True, boundary spiders are considered by Topologiq but not displayed in visualisations.
        taken (optional): A list of coordinates occupied by any blocks/pipes placed as a result of previous operations.
        valid_paths_block_positions (optional): The positions ocuppied by any valid paths rendered elsewhere in visualisation.

    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    # General
    artists = []
    edge_col = "black" if tgt_kind is None else "violet"
    alpha = 1 if tgt_kind is None else 0.5

    # Extract positions
    block_positions = nx.get_node_attributes(nx_g, "coords")
    cube_kinds = nx.get_node_attributes(nx_g, "type")
    pipe_kinds = nx.get_edge_attributes(nx_g, "pipe_type")

    # If there is no tgt_kind the call is for pre-existent paths
    if tgt_kind is None:
        for cube_id in nx_g.nodes():
            cube_kind = cube_kinds.get(cube_id)
            if cube_kind and ("o" not in cube_kind or (not hide_ports and cube_kind == "ooo")):
                cube_coords = block_positions.get(cube_id)
                if cube_coords and (is_final_vis or (cube_coords != src_coords and cube_coords not in valid_paths_block_positions)):
                    size = [1.0, 1.0, 1.0] if cube_kind != "ooo" else [0.9, 0.9, 0.9]
                    if not is_final_vis and len(tent_coords) == 1 and cube_coords in tent_coords:
                        edge_col = "gold"
                        border_width = 1
                    else:
                        edge_col = "black" if cube_kind != "ooo" else "white"
                        border_width = 0.5
                    _ = render_block(
                        ax,
                        cube_id,
                        cube_coords,
                        size,
                        cube_kind,
                        node_hex_map,
                        edge_col=edge_col,
                        border_width=border_width,
                        taken=taken,
                    )

        for u_id, v_id in nx_g.edges():
            u_coords = block_positions.get(u_id)
            v_coords = block_positions.get(v_id)
            pipe_kind = pipe_kinds.get((u_id, v_id), "gray")

            if u_coords is not None and v_coords is not None:
                if is_final_vis or (u_coords not in valid_paths_block_positions and v_coords not in valid_paths_block_positions):
                    _ = render_pipe(ax, u_coords, v_coords, pipe_kind)

    # If there is tgt_kind the call is for valid paths
    else:
        for cube_id in nx_g.nodes():
            cube_kind = cube_kinds.get(cube_id)

            if cube_kind and "o" not in cube_kind:
                cube_coords = block_positions.get(cube_id)

                if cube_coords:
                    border_width = 3 if cube_coords == src_coords else 2 if cube_coords in tent_coords else 0.5

                    if tgt_kind == "ooo" and cube_coords in tent_coords:
                        size = [0.9, 0.9, 0.9]
                        cube_kind = "ooo"
                    else:
                        size = [1.3, 1.3, 1.3] if cube_coords == src_coords else [1.0, 1.0, 1.0]

                    # Capture artists to show/hide
                    current_artists = render_block(
                        ax,
                        "TBD",
                        cube_coords,
                        size,
                        cube_kind,
                        node_hex_map,
                        edge_col="violet",
                        border_width=border_width,
                        alpha=alpha,
                    )
                    artists.extend(current_artists)

        # Pipes
        for u_id, v_id in nx_g.edges():
            u_coords = block_positions.get(u_id)
            v_coords = block_positions.get(v_id)
            pipe_kind = pipe_kinds.get((u_id, v_id), "gray")

            if u_coords is not None and v_coords is not None:
                current_artists = render_pipe(ax, u_coords, v_coords, pipe_kind, edge_col=edge_col, alpha=alpha)
                artists.extend(current_artists)

        # Apply initial visibility state
        for artist in artists:
            artist.set_visible(fig.show_valid_paths)

        fig.valid_path_artists.extend(artists)


def _adjust_plot_dimensions(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    nx_g: nx.Graph,
    valid_paths: dict[StandardBlock, List[StandardBlock]] | None,
    tent_coords: List[StandardCoord] | None,
    src_block_info: StandardBlock | None,
    is_final_vis: bool,
    pathfinder_success: bool = True,
):
    """Adjust the dimensions of the matplotlib plot.

    This function adjusts the dimensions (and therefore "zoom") of the main matplotlib
    pane. It defines the optimal dimensions based on a list of coordinates sent to it, 
    which should itself contain all objects being displayed. 

    Args:
        fig: The Matplotlib Figure object.
        ax: Matplotlib's 3D subplot object.
        nx_g: An NX graph with appropriately formatted space-time diagram components.
        valid_paths: A dictionary of topologically-correct paths found during a single pathfinder iteration.
        tent_coords: A list of tentative target coordinates to find paths to.
        src_block_info: The information of the source cube including its position in the 3D space and its kind.
        is_final_vis: A boolean to flag if the visualisation is the very last output.
        pathfinder_success (optional): A boolean to flag if the last pathfinder iteration succeeded.
    
    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """

    # Calculate positions of all contents
    if is_final_vis:
        all_static_coords = np.array(list(nx.get_node_attributes(nx_g, "coords").values()))
    else: 
        all_static_coords = np.array(
            list(nx.get_node_attributes(nx_g, "coords").values())
            + tent_coords
            + [src_block_info[0]]
            )

    # Apply dimensions
    if valid_paths:
        for path in valid_paths.values():
            all_static_coords = np.vstack([all_static_coords, np.array([block[0] for block in path])])

    if all_static_coords.size > 0:
        max_x, min_x = all_static_coords[:, 0].max(), all_static_coords[:, 0].min()
        max_y, min_y = all_static_coords[:, 1].max(), all_static_coords[:, 1].min()
        max_z, min_z = all_static_coords[:, 2].max(), all_static_coords[:, 2].min()

        max_range = max((max_x - min_x), (max_y - min_y), (max_z - min_z)) / 2.0
        mid = np.array([(max_x + min_x)/2, (max_y + min_y)/2, (max_z + min_z)/2])

        ax.set_xlim(mid[0] - max_range - 1, mid[0] + max_range + 1)
        ax.set_ylim(mid[1] - max_range - 1, mid[1] + max_range + 1)
        ax.set_zlim(mid[2] - max_range - 1, mid[2] + max_range + 1)
    else:
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # BG colours (yellow: progress, green: success, red: failure)
    bg_colour = "#c3ffb0" if is_final_vis else "#b8b8b8"
    if not is_final_vis and pathfinder_success is False:
        bg_colour = "#fcbbb8"
    fig.patch.set_facecolor(bg_colour)
    ax.patch.set_facecolor(bg_colour)


def _prepare_search_paths_data(fig, all_search_paths, valid_paths):
    """Prepare animation sequence for search paths.
    
    This function processes the raw search path data (all paths found by the inner pathfinder) 
    into two outputs:
        1. Figure state (`fig.all_search_paths_raw`): A persistent master list of all path coordinates and validity.
        2. Animation sequence: A sequential list of paths structured for frame-by-frame animation.

    Args:
        fig: The Matplotlib Figure object used for storing state.
        all_search_paths: Dictionary of all paths searched by the inner pathfinder.
        valid_paths: A dictionary of topologically-correct paths found during a single pathfinder iteration.

    Returns:
        tuple: (animation_sequence, num_paths, num_frames, animation_interval_ms, TARGET_DURATION_MS)
    
    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """
    animation_sequence = []
    
    if all_search_paths:
        # NOTE: Ensure fig.all_search_paths_raw is initialized (e.g., in _init_vis)
        for path in all_search_paths.values():
            
            # Convert path blocks to a numpy array of coordinates
            path_coords = np.array([block[0] for block in path])

            # Populate fig.all_search_paths_raw (master list)
            fig.all_search_paths_raw.append({
                'full_path': path,
                'coords': path_coords,
                # Check for validity using the actual path object
                'is_valid': path in valid_paths.values() if valid_paths else False
            })

            is_valid = path in valid_paths.values() if valid_paths else False
            animation_sequence.append({
                'coords': path_coords, 
                'color': 'green' if is_valid else 'red', 
                'persist': is_valid
            })

    num_paths = len(fig.all_search_paths_raw)
    num_frames = num_paths + 1 
    
    # Calculate timing metrics
    TARGET_DURATION_MS = 20000 if num_paths > 500 else 10000 if num_paths > 100 else 1000
    animation_interval_ms = int(TARGET_DURATION_MS / num_paths) if num_paths > 0 else 500
    
    return animation_sequence, num_paths, num_frames, animation_interval_ms, TARGET_DURATION_MS


def _setup_path_animation(fig, ax, animation_sequence, num_paths):
    """ Initialize the artists and functionality needed to animate a path search as a sequence. 

    This function creates the necessary Matplotlib artists: 
        1. `dynamic_red_path_line`: A single line artist used to draw the current path being processed (red/frame).
        2. `persistent_green_artists`: Pre-created, hidden artists for valid paths (green) that fade in at the end. 
        It also returns the `update` closure, which contains the core animation logic.

    Args:
        fig: The Matplotlib Figure object used to store persistent artists
        ax: The 3D Axes object.
        animation_sequence: List of path data for the animation frames.
        num_paths: The total number of paths.

    Returns:
        update_function: A closure that updates the plot for each animation frame.
        persistent_green_artists: A list of Matplotlib line artists for the valid paths.
    
    AI disclaimer:
        category: Coding partner (see README for details).
        model: Gemini, 2.5 Flash.
    """
    
    # Pre-create persistent paths (green, valid paths that will fade in)
    persistent_green_artists = []
    for item in animation_sequence:
        if item['persist']:
            path_artist, = ax.plot(
                item['coords'][:, 0], item['coords'][:, 1], item['coords'][:, 2], 
                color='green', linestyle=":", zorder=8, alpha=0,
            )
            persistent_green_artists.append({'artist': path_artist, 'index': len(persistent_green_artists)})
        else:
            persistent_green_artists.append(None) 
            
    # Initialize the Single Dynamic Path Line (for red/current path visualization)
    dynamic_red_path_line, = ax.plot(
        [], [], [], color='red', linestyle=":", zorder=8
    )

    # The Update function MUST be defined here to access the artists (closure)
    def update(frame):
        """Manages the dual animation mode, adding a static final frame showing all paths."""
        
        artists_to_return = []
   
        if frame < num_paths:  # Dynamic frames (animation)
            # Clear pre-existent line (if it was static from the last frame)
            dynamic_red_path_line.set_data([], [])
            dynamic_red_path_line.set_3d_properties([])
            artists_to_return.append(dynamic_red_path_line)
            
            # Update current item
            current_item = animation_sequence[frame]
            
            # Draw ALL paths (green or red) using the dynamic artist for visualization
            dynamic_red_path_line.set_data(current_item['coords'][:, 0], current_item['coords'][:, 1])
            dynamic_red_path_line.set_3d_properties(current_item['coords'][:, 2])

        else: # Static final frame (== num_paths) with full wireframe
            # Clear pre-existent dynamic line
            dynamic_red_path_line.set_data([], [])
            dynamic_red_path_line.set_3d_properties([])
            
            # Iterate through ALL paths and statically draw them
            for i in range(num_paths):

                item = animation_sequence[i]
                path_coords = item['coords']
                
                # Colour and z-order based on persistence
                color = 'green' if item['persist'] else 'red'
                zorder = 8 if item['persist'] else 7
                alpha = 1.0 if item['persist'] else 0.5

                # Plot
                static_path, = ax.plot(
                    path_coords[:, 0],
                    path_coords[:, 1],
                    path_coords[:, 2],
                    color=color,
                    linestyle=":",
                    zorder=zorder, 
                    alpha=alpha,
                    visible=True
                )
                artists_to_return.append(static_path)
                
                # Store static path in the unified list for later toggling
                fig.static_search_artists.append(static_path)

        # Return all artists that need to be drawn/re-drawn
        return artists_to_return

    return update, persistent_green_artists