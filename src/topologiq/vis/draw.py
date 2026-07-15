"""Debug 3D visualisations for the graph manager.

This file contains functions that can help create full-detail visualisations of how
the pathfinder algorithm goes about resolving specific edges.

Usage:
    Call `draw_blockgraph()` programmatically with an appropriate parameter combination.

AI disclaimer:
    In general, visualisations in Topologiq were developed using AI as coding partner (see CONTRIBUTING.md for details on category).
    model: Gemini, 2.5 Flash.

"""

import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredOffsetbox, AnchoredText, HPacker, TextArea
from matplotlib.widgets import Button, TextBox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray

from topologiq.core.beams import CubeBeams
from topologiq.core.blocks import ZXBlock
from topologiq.core.pathfinder.symbolic import rotate_pipe
from topologiq.utils.classes import StandardCoord
from topologiq.utils.vis import node_hex_map
from topologiq.utils.zx import ZXColors

#########
# PATHS #
#########
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
MEDIA_DIR = REPO_ROOT / "output/media"
_ORIGINAL_BACKEND = matplotlib.get_backend()


##################################
# BLOCKGRAPH VISUALISATION STATE #
##################################
@dataclass(frozen=True)
class VisualiserState:
    """Public data API for the visualiser.

    Attributes:
        bgraph: The primary NetworkX graph containing the blockgraph being built.
        beams: The beams for all the cubes in blockgraph that need beams.
        beams_short: The short beams for all the cubes in blockgraph that need beams.
        curr_edge_ids: A list of cube IDs forming part of the last edge processed, empty if final visualisation.
        is_cross_edge: True if the current edge is a cross edge.
        in_zx: The data of the input ZX packed as a dictionary.
        base_graph: The data of the base graph (input ZX after mandatory LS transformations) packed as a dictionary.
        twin_trace: A dictionary containing spiders that have twins.
        is_final_vis (optional): True if the visualisation shows an blockgraph that has been completed successfully.
        iter_fail (optional): True if the visualisation is called to diagnose an unsuccessful build.
        block_style (optional): True if the visualiser should show only blocks without pipes.
        base_graph_draw_style (optional): Style flag to determine 2D overlays' style and positioning strategy.
        stats (optional): Statistics about the blockgraph state.
        vis_mode (optional): Settings defining whether to show 3D visualisation live and/or save as snapshot.

    AI disclaimer:
        category: Coding partner (see CONTRIBUTING.md for details).
        model: Gemini, 2.5 Flash.

    """

    bgraph: nx.Graph
    beams: dict[int, CubeBeams]
    beams_short: dict[int, CubeBeams]
    tent_coords: list[StandardCoord]
    curr_edge_ids: list[int]
    is_cross_edge: bool
    in_zx: dict[str, Any]
    base_graph: dict[str, Any]
    twin_trace: dict[int, list[int]]
    is_final_vis: bool = True
    iter_fail: bool = False
    block_style: str = "pipe"
    base_graph_draw_style: str = "zx"
    stats: dict[str, Any] = field(default_factory=dict)
    vis_mode: tuple[bool, bool] = (True, False)


#########################
# BLOCKGRAPH VISUALISER #
#########################
class BlockGraphVisualiser:
    """Orchestrates visualiser lifecycle and interaction tracking.

    Consumes an immutable VisualiserState data snapshot to synchronise display assets,
    coordinate search tracking, and delegate cross-panel presentation layouts.

    AI disclaimer:
        category: Coding partner (see CONTRIBUTING.md for details).
        model: Gemini, 3.5 Flash.

    """

    def __init__(self, state: Any):
        """Initialise the orchestration controller with standard runtime variables."""
        self.state = state

        # Cancel pop up if visualisation is headless
        self.is_headless = not self.state.vis_mode[0]
        if self.is_headless:
            # Drop cleanly into non-interactive rendering for file dumps
            plt.switch_backend("Agg")
        else:
            # If a prior step turned on Agg, pop back out to the environment's true default
            if plt.get_backend().lower() == "agg":
                plt.switch_backend(_ORIGINAL_BACKEND)

        # Global UX layer visibility tracking flags
        self.current_view_mode: str = "BGRAPH"  # Options: "BGRAPH" | "ZX"
        self.show_long_beams, self.show_short_beams, self.show_tent_coords = (False, False, False)

        # System-wide interaction state memory tracking variables
        self.active_highlights: set[str] = set()
        self._last_node_click_coord: tuple[float, float] | None = None
        self.edge_box_expanded, self.hud_box_expanded = (False, False)

        # Cache structural geometry properties
        self.cube_size: list[float] = [0.33 if self.state.block_style == "pipe" else 1.0] * 3
        self.max_range: float = 5.0

        # Initialise explicit drawing engines
        self.view_3d = View3D(self)
        self.view_2d = View2D(self)

    def build_layout(self):
        """Execute structural visualization pipelines and register event listener loops."""
        # Render structural 3D components
        self.view_3d.render_spatial_scene()

        # Initialise control systems and companion layout interfaces
        self.view_3d.initialise_hud()
        self.view_2d.initialise_panels()

        # Synchronise bounds layout scaling aesthetics
        self.view_3d.apply_plot_scaling_themes()

        # Register event loop triggers
        self.view_3d.bind_interaction_events()

    def show(self):
        """Export high-fidelity canvas frames or display interactive UI windows."""
        # Save animation frame if animation flag is present
        if self.state.vis_mode[1]:
            self.view_3d.export_snapshot()

        # Open visualisation if not headless
        if not self.is_headless:
            self.view_3d.display_window()
        else:  # Clean up resources if absolutely no visual window layout was requested
            self.view_3d.close_resources()

    def request_overlay_toggle(self, overlay_key: str, graph_data: dict[str, Any]):
        """Request the 2D panel view to handle layout updates."""
        self.view_2d.toggle_overlay_view(overlay_key, graph_data)

    def handle_search_submit(self, text_input: str):
        """Pipes text inputs into the unified tracking coordinator."""
        target_ids = set()
        if text_input:
            for item in text_input.split(","):
                cleaned = item.strip()
                if cleaned:
                    target_ids.add(cleaned)
                    # Unpack twins based on search selection keys
                    t_key = int(cleaned) if cleaned.isdigit() else cleaned
                    twin_trace = getattr(self.state, "twin_trace", {})
                    for twin in twin_trace.get(t_key, []):
                        target_ids.add(str(twin))
        self.update_global_highlights(target_ids)

    def update_global_highlights(self, target_ids: set[str]):
        """Unify state coordination updating selection tracking sets.

        Note. Acts as the application's single source of truth for highlights, syncing
        both the 3D spatial scene and the active 2D circuit overlay.
        """
        self.active_highlights = target_ids  # Update active highligths
        self.view_3d.synchronise_highlights(target_ids)  # Update 3D scene and text overlays
        self.view_2d.refresh_overlay_highlights(target_ids)  # Update 2D projection subplots

    def _handle_hud_ui_clicks(self, label: str, artist: Any):
        """Process HUD collapsibles and rendering layer visibility updates."""
        match label:
            case "COLLAPSE_EDGE_BOX":
                self.edge_box_expanded = not self.edge_box_expanded
                self.view_3d.toggle_edge_info_box(self.edge_box_expanded)
            case "COLLAPSE_HUD_BOX":
                self.hud_box_expanded = not self.hud_box_expanded
                self.view_3d.toggle_stats_info_box(self.hud_box_expanded)
            case "VIEW_TOGGLE_BUTTON":
                self.current_view_mode = "ZX" if self.current_view_mode == "BGRAPH" else "BGRAPH"
                self.view_3d.switch_projection_style(self.current_view_mode, artist)
            case "LONG_BEAMS_BUTTON":
                self.show_long_beams = not self.show_long_beams
                self.view_3d.toggle_beam_visibility("long", self.show_long_beams, artist)
            case "SHORT_BEAMS_BUTTON":
                self.show_short_beams = not self.show_short_beams
                self.view_3d.toggle_beam_visibility("short", self.show_short_beams, artist)
            case "TENT_COORDS_BUTTON":
                self.show_tent_coords = not self.show_tent_coords
                self.view_3d.toggle_tentative_coordinates(self.show_tent_coords, artist)
            case _:  # Optional catch-all fallback for unhandled UI clicks
                pass

    def route_canvas_click_clear(self, event: Any):
        """Process canvas mouse hits to clean up UI selection state."""
        if event.inaxes is None or event.button != 1:
            return

        # Delegate geometric collision verification to the active 2D view
        if self.view_2d.is_axis_member(event.inaxes):
            hit_detected = self.view_2d.check_geometry_collisions(event)
            # If user clicked open canvas space, wipe all highlights cleanly
            if not hit_detected:
                self.view_3d.reset_search_input()
                self.update_global_highlights(set())

    def route_pick_event(self, event: Any):
        """Decode raw canvas item clicks and dispatch them to specific handlers."""
        artist = event.artist
        label, gid = artist.get_label(), artist.get_gid()

        # Handle empty background immediately
        if label == "EMPTY_CANVAS_BACKGROUND" or gid == "EMPTY_CANVAS_BACKGROUND":
            self.view_3d.reset_search_input()
            self.update_global_highlights(set())
            return

        # Check for our new string-based identifiers or custom node list attributes
        is_string_gid = isinstance(gid, str)
        is_node_layer = (
            isinstance(gid, list)
            or hasattr(artist, "node_ids")
            or (is_string_gid and gid.startswith("nodes_"))
            or (is_string_gid and gid == "overlay_nodes_layer")
        )

        is_edge_layer = isinstance(gid, tuple) or (is_string_gid and gid.startswith("edge_"))

        # Structural pattern match based on (label, is_node, is_edge)
        match (label, is_node_layer, is_edge_layer):
            case (_, True, False):
                # Ensure a list-like object fallback gets routed safely down to the picker
                node_list = getattr(artist, "node_ids", gid if isinstance(gid, list) else [])
                self._handle_node_pick(event, node_list)
            case (_, False, True):
                # Normalize edge key strings back to tuples format if needed by downstream unpackers
                edge_tuple = gid
                if is_string_gid and gid.startswith("edge_"):
                    # Converts "edge_u_v" -> ("edge", "u", "v")
                    parts = gid.split("_")
                    if len(parts) >= 3:
                        edge_tuple = ("edge", parts[1], parts[2])
                self._handle_edge_pick(event, edge_tuple)
            case (l, False, False) if (
                isinstance(l, str)
                and l
                and not l.endswith("_BUTTON")
                and not l.startswith("COLLAPSE_")
            ):
                self._handle_text_label_pick(l)
            case _:
                self._handle_hud_ui_clicks(label, artist)

    def _unpack_twins(self, keys: set[str]) -> set[str]:
        """Discover and append linked twin nodes to target IDs."""
        target_ids = set(keys)
        twin_trace = getattr(self.state, "twin_trace", {})
        for k in keys:
            t_key = int(k) if k.isdigit() else k
            for twin in twin_trace.get(t_key, []):
                target_ids.add(str(twin))
        return target_ids

    def _finalize_selection(self, target_ids: set[str]):
        """Commit highlights and synchronize text field across views."""
        if target_ids:
            self.view_3d.update_search_field_text(",".join(sorted(target_ids)))
            self.update_global_highlights(target_ids)

    def _handle_node_pick(self, event: Any, gid: list):
        """Process click events falling inside scatter collection node clusters."""
        # Get the artist object responsible for catching the mouse click
        artist = getattr(event, "artist", None)

        # Use custom node_ids attached to artist or fall back to the original gid parameter.
        active_node_sequence = getattr(artist, "node_ids", gid)

        if (
            hasattr(event, "ind")
            and len(event.ind) > 0
            and event.ind[0] < len(active_node_sequence)
        ):
            # Extract using the data index from the chosen array sequence
            clicked_node = str(active_node_sequence[event.ind[0]])

            if event.mouseevent is not None:
                self._last_node_click_coord = (event.mouseevent.x, event.mouseevent.y)

            target_ids = self._unpack_twins({clicked_node})
            self._finalize_selection(target_ids)

    def _handle_edge_pick(self, event: Any, gid: tuple):
        """Process interactions targeting logical links or physical pipe lines."""
        if event.mouseevent is not None and self._last_node_click_coord is not None:
            if (event.mouseevent.x, event.mouseevent.y) == self._last_node_click_coord:
                return  # Relinquish event priority to high-density node picker

        _, u, v = gid
        u_key = int(u) if str(u).isdigit() else u
        v_key = int(v) if str(v).isdigit() else v

        in_zx_edges = self.state.in_zx.get("completed_edges", {})
        base_edges = self.state.base_graph.get("completed_edges", {})

        all_edge_nodes = (
            in_zx_edges.get((u_key, v_key))
            or in_zx_edges.get((v_key, u_key))
            or base_edges.get((u_key, v_key))
            or base_edges.get((v_key, u_key))
            or [u_key, v_key]
        )

        edge_node_strings = {str(node) for node in all_edge_nodes}
        target_ids = self._unpack_twins(edge_node_strings)
        self._finalize_selection(target_ids)

    def _handle_text_label_pick(self, label: str):
        """Process fallback selections matching pure text artist labels."""
        target_ids = self._unpack_twins({label})
        self._finalize_selection(target_ids)


class View3D:
    """Encapsulates Matplotlib 3D projections, text box inputs, and window frames."""

    def __init__(self, controller: BlockGraphVisualiser):
        """Initialise 3D view."""
        # Controller
        self.ctx = controller

        # Primary Figure and Ax
        self.fig: Figure = plt.figure(figsize=(12, 8))
        self.ax: matplotlib.axes.Axes = self.fig.add_subplot(projection="3d")
        self.ax.set_proj_type("persp")
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

        # Graphic item component trackers
        self.spiders, self.phantom_edges = [], []
        self.long_beam_artists, self.short_beam_artists, self.tent_coords_artists = [], [], []
        self.show_tent_coords = False

        # Form HUD Container Pointers
        self._toggle_area: TextArea | None = None
        self.toggle_btn: matplotlib.text.Text | None = None
        self.long_beam_btn: matplotlib.text.Text | None = None
        self.short_beam_btn: matplotlib.text.Text | None = None
        self.tent_coords_btn: matplotlib.text.Text | None = None
        self.search_ax: plt.Axes | None = None
        self.search_box: TextBox | None = None
        self.edge_box: AnchoredText | None = None
        self.hud_box: AnchoredText | None = None

        # State tracking strings
        self._full_edge_text = ""
        self._full_stats_text = ""

    def render_spatial_scene(self):
        """Execute decoupled underlying pipeline render passes."""
        # Calls legacy drawing modules attached to the controller instance safely
        self._render_all_blocks()
        self._render_all_pipes()
        self._render_beams()
        self._render_tentative_coords()

    def apply_plot_scaling_themes(self):
        """Compute bounding constraints, loop layout aesthetics, and background validation."""
        # Get all coords
        state = self.ctx.state
        coords = np.array([c for c in nx.get_node_attributes(state.bgraph, "coords").values() if c])

        # Calculate bounds using NumPy matrix axes directly
        if coords.size > 0:
            c_min, c_max = coords.min(axis=0), coords.max(axis=0)
            mid = (c_max + c_min) / 2.0
            max_range = float((c_max - c_min).max() / 2.0)
            for i, set_lim in enumerate([self.ax.set_xlim, self.ax.set_ylim, self.ax.set_zlim]):
                set_lim(mid[i] - max_range - 1, mid[i] + max_range + 1)
        else:
            max_range = 5.0
            for set_lim in [self.ax.set_xlim, self.ax.set_ylim, self.ax.set_zlim]:
                set_lim(-max_range, max_range)

        # Loop through axes components to assign labels, locators, and formatters
        for name, axis in zip(["X", "Y", "Z"], [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]):
            getattr(self.ax, f"set_{name.lower()}label")(name)
            axis.set_major_locator(ticker.MultipleLocator(1))
            axis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        # Determine color layout from true runtime states
        bg_col = "#fcbbb8" if state.iter_fail else ("#5bffa2" if state.is_final_vis else "#befbcc")
        self.fig.patch.set_facecolor(bg_col)
        self.ax.patch.set_facecolor(bg_col)
        self.ctx.max_range = max_range

    def initialise_hud(self):
        """Construct control buttons, validation fields, and collapsible boxes."""
        # Setup text area control widgets
        areas = []
        t_props = dict(fontfamily="monospace", fontsize="small", weight="bold", color="white")
        btn_defs = [
            ("toggle_btn", " to ZX ", "VIEW_TOGGLE_BUTTON", "#303030"),
            ("long_beam_btn", "Beams", "LONG_BEAMS_BUTTON", "#7f8c8d"),
            ("short_beam_btn", "Short Beams", "SHORT_BEAMS_BUTTON", "#7f8c8d"),
            ("tent_coords_btn", "Tent", "TENT_COORDS_BUTTON", "#7f8c8d"),
        ]

        for attr, text, label, color in btn_defs:
            area = TextArea(text, textprops=t_props)
            btn = area.get_children()[0]
            btn.set_bbox(dict(facecolor=color, boxstyle="square,pad=0.5"))
            btn.set_picker(True)
            btn.set_label(label)
            setattr(self, attr, btn)
            if attr == "toggle_btn":
                self._toggle_area = area
            areas.append(area)

        control_row_anchor = AnchoredOffsetbox(
            loc="upper right",
            child=HPacker(children=areas, align="baseline", pad=0.1, sep=10),
            pad=0,
            bbox_to_anchor=(0.98, 0.98),
            bbox_transform=self.fig.transFigure,
            frameon=False,
        )
        self.ax.add_artist(control_row_anchor)

        # Setup interactive search text box & overflow
        self.search_ax = self.fig.add_axes([0.08, 0.945, 0.06, 0.035])
        sb = self.search_box = TextBox(self.search_ax, label="ID search: ", initial="")

        for text_obj in [sb.label, sb.text_disp]:
            text_obj.set_fontfamily("monospace")
            text_obj.set_fontsize("small")
        sb.label.set_weight("bold")
        sb.text_disp.set_clip_on(True)

        for patch in [
            p
            for p in self.search_ax.get_children()
            if isinstance(p, matplotlib.patches.FancyBboxPatch)
        ]:
            patch.set_boxstyle("square,pad=0.0")

        def _handle_overflow(txt):
            is_long = len(txt) >= 5
            sb.text_disp.set_horizontalalignment("right" if is_long else "left")
            sb.text_disp.set_x(0.95 if is_long else 0.05)

        sb.on_text_change(_handle_overflow)
        sb.on_submit(self.ctx.handle_search_submit)

        # Build dynamic edge strings
        edges = self.ctx.state.curr_edge_ids or []
        edge_fmt = (
            f"{edges[0]} -> {edges[-1]}" if len(edges) >= 2 else (edges[0] if edges else "None")
        )
        self._full_edge_text = f"Current Edge:\n{edge_fmt}\n\nActive Labels:\nNone"

        # Build statistical strings
        st = getattr(self.ctx.state, "stats", {}) or {}
        ft_str = "".join(
            f"  - d={d}: ~{v[0]}x{v[1]}.\n"
            for d, v in st.get("bgraph_surface_footprint", {}).items()
        )
        self._full_stats_text = (
            "STATS\n\nInput ZX graph\n"
            f"* Num spiders: {st.get('in_zx_spiders', 'N/A')}.\n"
            f"* Num edges: {st.get('in_zx_edges', 'N/A')}.\n"
            f"* Density: {st.get('in_zx_density', 0.0):.2f}.\n\n"
            "BlockGraph:\n"
            f"* Volume: {st.get('bgraph_volume', 'N/A')}.\n"
            f"* Overhead: {st.get('bgraph_overhead', 0.0):.2f}x.\n"
            f"* Req. qubit surface:\n{ft_str}"
        )

        # Build collapsible box anchors loop
        box_defs = [
            ("edge_box", "[+ Edge Info]", (0.02, 0.93), "COLLAPSE_EDGE_BOX"),
            ("hud_box", "[+ Stats]", (0.98, 0.93), "COLLAPSE_HUD_BOX"),
        ]

        for attr, text, anchor, label in box_defs:
            box = AnchoredText(
                text,
                loc="upper left" if anchor[0] < 0.5 else "upper right",
                prop=dict(fontfamily="monospace", fontsize="small", color="#ffffff"),
                bbox_to_anchor=anchor,
                bbox_transform=self.fig.transFigure,
                frameon=True,
            )
            for item in [box.patch, box.txt]:
                item.set_picker(5)
                item.set_label(label)
            box.patch.set_facecolor("#2c2c2c")
            box.patch.set_alpha(0.4)
            box.patch.set_boxstyle("square,pad=0.5")
            self.ax.add_artist(box)
            setattr(self, attr, box)

    def bind_interaction_events(self):
        """Attach standard backend canvas action callbacks."""

        def _on_canvas_click_clear(event):
            self.ctx.route_canvas_click_clear(event)

        def _onpick_handler(event):
            self.ctx.route_pick_event(event)

        # Connect the local wrapper functions to the canvas
        self.fig.canvas.mpl_connect("pick_event", _onpick_handler)
        self.fig.canvas.mpl_connect("button_press_event", _on_canvas_click_clear)

    def export_snapshot(self):
        """Output active canvas frame configurations safely into system directories."""
        temp_dir = MEDIA_DIR / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        frame_id = int(time.time_ns())
        self.fig.savefig(temp_dir / f"frame_{frame_id}.png", bbox_inches="tight", dpi=150)

    def display_window(self):
        """Trigger native interactive platform display engines."""
        plt.show()

    def close_resources(self):
        """Immediately release memory structures if processing headlessly."""
        plt.close(self.fig)

    def update_search_field_text(self, text_value: str):
        """Mutate the input display string without triggering text evaluation loops."""
        if self.search_box is not None:
            events_muted = self.search_box.eventson
            self.search_box.eventson = False
            self.search_box.set_val(text_value)
            self.search_box.eventson = events_muted

    def reset_search_input(self):
        """Clear the active text query field safely."""
        self.update_search_field_text("")

    def toggle_edge_info_box(self, is_expanded: bool):
        """Expand or collapse the left-hand text readout dashboard."""
        if is_expanded:
            self.edge_box.txt.set_text(self._full_edge_text)
            self.edge_box.patch.set_alpha(0.8)
        else:
            self._full_edge_text = self.edge_box.txt.get_text()
            self.edge_box.txt.set_text("[+ Edge Info]")
            self.edge_box.patch.set_alpha(0.4)
        self.fig.canvas.draw_idle()

    def toggle_stats_info_box(self, is_expanded: bool):
        """Expand or collapse the right-hand analytical dashboard."""
        if is_expanded:
            self.hud_box.txt.set_text(self._full_stats_text)
            self.hud_box.patch.set_alpha(0.8)
        else:
            self.hud_box.txt.set_text("[+ Stats]")
            self.hud_box.patch.set_alpha(0.4)
        self.fig.canvas.draw_idle()

    def switch_projection_style(self, mode: str, button_artist: Any):
        """Alter alpha settings across collections to emphasise structural properties or raw ports."""
        if mode == "ZX":
            if self._toggle_area is not None:
                self._toggle_area.set_text(" to BLOCKGRAPH ")
            button_artist.get_bbox_patch().set_facecolor("#27ae60")
            block_alpha, pipe_alpha, show_zx = 0.1, 0.0, True
        else:
            if self._toggle_area is not None:
                self._toggle_area.set_text(" to ZX ")
            button_artist.get_bbox_patch().set_facecolor("#2c3e50")
            block_alpha, pipe_alpha, show_zx = 1.0, 1.0, False

        for child in self.ax.collections:
            if hasattr(child, "_original_alpha"):
                if show_zx:
                    child.set_linewidths([0])
                else:
                    child.set_linewidths([getattr(child, "_original_linewidth", 0.3)])

                if hasattr(child, "_associated_text"):
                    child.set_alpha(block_alpha)
                    spider_dot = getattr(child, "_associated_spider", None)
                    if spider_dot is not None:
                        spider_dot.set_visible(show_zx)
                        spider_dot.set_alpha(0.8 if show_zx else 0.0)
                else:
                    child.set_alpha(pipe_alpha)

        for wire in self.phantom_edges:
            wire.set_visible(show_zx)
        self.fig.canvas.draw_idle()

    def toggle_beam_visibility(self, kind: str, is_visible: bool, button_artist: Any):
        """Toggle geometric occlusion settings for laser beam markers."""
        color_hex = "#c5d300" if is_visible else "#7f8c8d"
        button_artist.get_bbox_patch().set_facecolor(color_hex)
        target_artists = self.long_beam_artists if kind == "long" else self.short_beam_artists
        for beam in target_artists:
            beam.set_visible(is_visible)
        self.fig.canvas.draw_idle()

    def toggle_tentative_coordinates(self, is_visible: bool, button_artist: Any):
        """Toggle drawing visibility bounds for upcoming topological nodes."""
        color_hex = "#c5d300" if is_visible else "#7f8c8d"
        button_artist.get_bbox_patch().set_facecolor(color_hex)
        for scatter_artist in self.tent_coords_artists:
            scatter_artist.set_visible(is_visible)
        self.fig.canvas.draw_idle()

    def synchronise_highlights(self, target_ids: set[str]):
        """Redraw borders, sets line sizing parameters, and manages text readouts in the 3D scene."""
        target_ints = set(int(x) for x in target_ids if x.isdigit())
        highlighted_bgraph_edges = set()

        if hasattr(self.ctx.state, "bgraph") and self.ctx.state.bgraph is not None:
            for u, v in self.ctx.state.bgraph.edges:
                if u in target_ints and v in target_ints:
                    highlighted_bgraph_edges.add((u, v))
                    highlighted_bgraph_edges.add((v, u))

        active_entries = []
        for child in self.ax.collections:
            cube_label = child.get_label()
            edge_key = getattr(child, "_edge_key", None)

            is_target_cube = cube_label in target_ids
            is_target_pipe = edge_key is not None and edge_key in highlighted_bgraph_edges

            if is_target_cube or is_target_pipe:
                if not hasattr(child, "_prior_edgecolor"):
                    child._prior_edgecolor = child.get_edgecolor()
                if not hasattr(child, "_prior_linewidth"):
                    child._prior_linewidth = child.get_linewidths()

                child.set_edgecolor("red")
                child.set_linewidths([2.5])

                if is_target_cube and self.ctx.current_view_mode == "BGRAPH":
                    child.set_alpha(1.0)
                elif self.ctx.current_view_mode == "BGRAPH":
                    child.set_alpha(0.4)
            else:
                if hasattr(child, "_prior_edgecolor"):
                    child.set_edgecolor(child._prior_edgecolor)
                    del child._prior_edgecolor
                if hasattr(child, "_prior_linewidth"):
                    child.set_linewidths(child._prior_linewidth)
                    del child._prior_linewidth
                else:
                    orig_width = getattr(child, "_original_linewidth", 0.3)
                    child.set_linewidths([orig_width])

                if hasattr(child, "_original_alpha"):
                    child.set_alpha(child._original_alpha)

            txt = getattr(child, "_associated_text", None)
            spider = getattr(child, "_associated_spider", None)

            if txt is not None:
                txt.set_visible(is_target_cube)
                if is_target_cube:
                    c_type = getattr(child, "_zx_type", "Unknown")
                    coord_str = ""
                    if cube_label.isdigit() and int(cube_label) in self.ctx.state.bgraph.nodes:
                        c_coord = self.ctx.state.bgraph.nodes[int(cube_label)].get("coords")
                        coord_str = f"{c_coord}" if c_coord is not None else ""
                    active_entries.append(f"{cube_label}: {c_type} @ {coord_str}")

                if spider is not None:
                    if self.ctx.current_view_mode == "BGRAPH":
                        spider.set_visible(False)
                        spider.set_alpha(0.0)
                    else:
                        spider.set_visible(is_target_cube)
                        spider.set_alpha(0.8 if is_target_cube else 0.0)

        # Synchronise textual data frames back to containers
        if hasattr(self, "edge_box"):
            edge_ids = self.ctx.state.curr_edge_ids
            base_heading = (
                f"Current Edge:\n{edge_ids[0]} -> {edge_ids[-1]}"
                if len(edge_ids) >= 2
                else f"Current Edge:\n{edge_ids[0]}"
                if len(edge_ids) == 1
                else "Current Edge:\nNone"
            )

            labels_body = "\n".join(active_entries) if active_entries else "None"
            self._full_edge_text = f"{base_heading}\n\nActive Labels:\n{labels_body}"
            self.edge_box.txt.set_text(
                self._full_edge_text if self.ctx.edge_box_expanded else "[+ Edge Info]"
            )

        self.fig.canvas.draw_idle()

    def _render_all_blocks(self):
        """Draw 3D cubes (nodes) geometries."""
        for cube_id in self.ctx.state.bgraph.nodes():
            cube_coords = self.ctx.state.bgraph.nodes[cube_id]["coords"]
            if cube_coords:
                zx_block = self.ctx.state.bgraph.nodes[cube_id]["zx_block"]
                if zx_block:
                    render_block(
                        self.ax,
                        cube_id,
                        cube_coords,
                        self.ctx.cube_size,
                        zx_block,
                        in_curr_edge=cube_id in self.ctx.state.curr_edge_ids,
                    )

    def _render_all_pipes(self):
        """Draw 3D pipe (edges) geometries."""
        for u_id, v_id in self.ctx.state.bgraph.edges():
            u_coords = self.ctx.state.bgraph.edges[(u_id, v_id)]["start_coords"]
            v_coords = self.ctx.state.bgraph.edges[(u_id, v_id)]["end_coords"]
            pipe_kind = self.ctx.state.bgraph.edges[(u_id, v_id)]["kind"]

            if u_coords is not None and v_coords is not None:
                is_active_wire = (
                    u_id in self.ctx.state.curr_edge_ids and v_id in self.ctx.state.curr_edge_ids
                )

                pre_count = len(self.ax.collections)
                render_pipe(self.ax, u_coords, v_coords, pipe_kind, in_curr_edge=is_active_wire)

                # Tag newly appended visual elements with their corresponding edge tracking keys
                post_count = len(self.ax.collections)
                for i in range(pre_count, post_count):
                    self.ax.collections[i]._edge_key = (u_id, v_id)

                # Construct an invisible wide line trace for simplified click tracking intersections
                wire = render_phantom_edge(
                    self.ax, u_coords, v_coords, pipe_kind, alpha=0.7, in_curr_edge=is_active_wire
                )
                wire._edge_key = (u_id, v_id)
                wire.set_visible(False)
                self.phantom_edges.append(wire)

        self.ax._all_phantom_edges = self.phantom_edges

    def _render_tentative_coords(self):
        """Draw small cubes at all positions contained in tentative coords."""

        # Clear out previous scatter collections to prevent memory leaks across frame re-renders
        for old_scatter in self.tent_coords_artists:
            try:
                old_scatter.remove()
            except ValueError:
                pass
        self.tent_coords_artists.clear()

        if not hasattr(self.ctx.state, "tent_coords") or not self.ctx.state.tent_coords:
            return

        xs = [coord[0] for coord in self.ctx.state.tent_coords]
        ys = [coord[1] for coord in self.ctx.state.tent_coords]
        zs = [coord[2] for coord in self.ctx.state.tent_coords]

        tgt_color = "#000000"
        if self.ctx.state.curr_edge_ids:
            last_node = self.ctx.state.curr_edge_ids[-1]
            if last_node in self.ctx.state.bgraph.nodes:
                block_ref = self.ctx.state.bgraph.nodes[last_node].get("zx_block")
                if block_ref:
                    tgt_color = block_ref.get_zx_color

        scatter = self.ax.scatter(
            xs,
            ys,
            zs,
            marker="s",
            color=tgt_color,
            s=60,
            edgecolor="#000000",
            linewidths=1.5,
            zorder=100,
        )
        scatter.set_visible(self.show_tent_coords)
        self.tent_coords_artists.append(scatter)

    def _render_beams(self):
        """Draw long and short beams as per the start and end points contained in the respective objects."""
        gold_color = "#000000"

        def build_beam_lines(beam_dict: dict[int, Any], target_list: list[Any], is_infinite: bool):
            for cube_id, cube_beams in beam_dict.items():
                for single_beam in cube_beams:
                    if cube_id in self.ctx.state.curr_edge_ids:
                        other_edge_cube_ids = [
                            cid for cid in self.ctx.state.curr_edge_ids if cid != cube_id
                        ]

                        intersects_other_cube = False
                        for other_id in other_edge_cube_ids:
                            other_coords = self.ctx.state.bgraph.nodes[other_id].get("coords")
                            if other_coords and single_beam.contains(other_coords):
                                intersects_other_cube = True
                                break

                        if intersects_other_cube:
                            continue

                    start_x, start_y, start_z = single_beam.coords()

                    if single_beam.x.direction != 0:
                        end_x = (
                            start_x + (self.ctx.max_range * 3 * single_beam.x.direction)
                            if is_infinite
                            else single_beam.x.end
                        )
                        end_y, end_z = start_y, start_z
                    elif single_beam.y.direction != 0:
                        end_y = (
                            start_y + (self.ctx.max_range * 3 * single_beam.y.direction)
                            if is_infinite
                            else single_beam.y.end
                        )
                        end_x, end_z = start_x, start_z
                    else:
                        end_z = (
                            start_z + (self.ctx.max_range * 3 * single_beam.z.direction)
                            if is_infinite
                            else single_beam.z.end
                        )
                        end_x, end_y = start_x, start_y

                    (line_artist,) = self.ax.plot(
                        [start_x, end_x],
                        [start_y, end_y],
                        [start_z, end_z],
                        color=gold_color,
                        linewidth=1.8,
                        alpha=0.6 if is_infinite else 0.8,
                        zorder=130,
                    )
                    line_artist.set_visible(False)
                    target_list.append(line_artist)

        build_beam_lines(self.ctx.state.beams, self.long_beam_artists, is_infinite=True)
        build_beam_lines(self.ctx.state.beams_short, self.short_beam_artists, is_infinite=False)


class View2D:
    """Manages secondary multi-axis layouts for flat mathematical circuit representations."""

    def __init__(self, controller: BlockGraphVisualiser):
        """Initialise 2D View."""
        self.ctx = controller
        self._overlay_axes: dict[str, plt.Axes | None] = {"in_zx": None, "base_graph": None}
        self._buttons: list[Button] = []
        self._hl_scatters: dict[str, matplotlib.collections.PathCollection | None] = {
            "in_zx": None,
            "base_graph": None,
        }

    def initialise_panels(self):
        """Allocate split view layout controls below the primary 3D render box."""
        # Top level positioning
        fig = self.ctx.view_3d.fig
        ax_btn_zx = fig.add_axes([0.0, 0.0, 0.5, 0.08])
        ax_btn_base = fig.add_axes([0.5, 0.0, 0.5, 0.08])

        # Button definitions
        btn_zx = Button(ax_btn_zx, "RAW ZX INPUT", color="#f2f3fb", hovercolor="#b2b2b2")
        btn_base = Button(ax_btn_base, "BASE ZX GRAPH", color="#f2f3fb", hovercolor="#b2b2b2")

        # Map component interface switches directly through controller actions
        btn_zx.on_clicked(lambda e: self.ctx.request_overlay_toggle("in_zx", self.ctx.state.in_zx))
        btn_base.on_clicked(
            lambda e: self.ctx.request_overlay_toggle("base_graph", self.ctx.state.base_graph)
        )
        self._buttons.extend([btn_zx, btn_base])

        # Register backwards compatibility properties on figure structures
        fig._overlay_axes = self._overlay_axes
        fig._buttons = self._buttons

        # Run primary default baseline configuration path
        self.toggle_overlay_view("base_graph", self.ctx.state.base_graph)

    def toggle_overlay_view(self, overlay_key: str, graph_data: dict[str, Any]):
        """Manage active graph drawing canvases below the principal 3D scene."""

        # Extract key objects for facilitated usage
        fig = self.ctx.view_3d.fig
        other_key = "base_graph" if overlay_key == "in_zx" else "in_zx"

        # Wipe out the alternative view axis if open to prevent panels from stacking in lower canvas region
        if self._overlay_axes[other_key] is not None:
            if self._hl_scatters.get(other_key) is not None:
                self._hl_scatters[other_key] = None
            self._overlay_axes[other_key].remove()
            self._overlay_axes[other_key] = None

        # Toggle closure, clean state tracking, and remove axis if user clicks an active view tab
        if self._overlay_axes[overlay_key] is not None:
            if self._hl_scatters.get(overlay_key) is not None:
                self._hl_scatters[overlay_key] = None
            self._overlay_axes[overlay_key].remove()
            self._overlay_axes[overlay_key] = None
        else:  # Or build a fresh 2D viewport axis below the main 3D canvas boundary box and execute its layout
            overlay_ax = fig.add_axes([0.0, 0.08, 1.0, 0.30])
            self._overlay_axes[overlay_key] = overlay_ax
            active_highlights = set()

            draw_style = "zx" if overlay_key == "in_zx" else self.ctx.state.base_graph_draw_style
            self.render_overlay_graph_panel(
                overlay_ax, graph_data, draw_style=draw_style, active_highlights=active_highlights
            )

        # Enqueue an idle draw event to efficiently repainting layout structure variations
        fig.canvas.draw_idle()

    def render_overlay_graph_panel(
        self, ax: plt.Axes, graph_data: dict[str, Any], draw_style: str, active_highlights: set[str]
    ):
        """Execute layout rendering routines mapping logical elements into localized axes."""
        # Extract key objects
        subgraph_nodes = graph_data.get("ids", [])
        types_map = graph_data.get("types", {})
        twin_trace = getattr(self.ctx.state, "twin_trace", {})

        # Initialise and dress the canvas bounding box container
        ax.patch.set_facecolor("#f2f3fb")
        ax.patch.set_alpha(0.7)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Compute layout / positions
        positions = self._compute_overlay_layout(graph_data, subgraph_nodes, types_map, draw_style)
        if not positions:
            return

        ax._spider_positions = positions
        node_ids = list(positions.keys())
        curr_edge_ids_set = (
            set(self.ctx.state.curr_edge_ids) if self.ctx.state.curr_edge_ids else set()
        )

        # Layered painter execution
        self._render_overlay_edges(ax, graph_data, positions, twin_trace, curr_edge_ids_set)
        self._render_overlay_twin_halos(ax, node_ids, positions, twin_trace, curr_edge_ids_set)
        self._render_overlay_nodes(
            ax, node_ids, positions, types_map, active_highlights, curr_edge_ids_set
        )
        self._render_overlay_labels(ax, node_ids, positions, graph_data, twin_trace)

        # Scale limits based on spider footprints
        x_nodes = [positions[n][0] for n in node_ids]
        y_nodes = [positions[n][1] for n in node_ids]
        ax.set_xlim(min(x_nodes) - 1, max(x_nodes) + 1)
        ax.set_ylim(min(y_nodes) - 1, max(y_nodes) + 1)

    def is_axis_member(self, axis_obj: Any) -> bool:
        """Determine if a target axis belongs to the active 2D panel overlay."""
        return axis_obj in self._overlay_axes.values()

    def check_geometry_collisions(self, event) -> bool:
        """Run point-in-polygon calculations across flat 2D scatter and line layers."""
        # Inspect graph scatter nodes
        for collection in event.inaxes.collections:
            if collection == getattr(self, "_hl_scatter", None):
                continue
            if hasattr(collection, "contains"):
                inside, _ = collection.contains(event)
                if inside:
                    return True
        # Inspect invisible thick edge tracking backings
        for line in event.inaxes.lines:
            if hasattr(line, "contains"):
                inside, _ = line.contains(event)
                if inside:
                    return True
        return False

    def refresh_overlay_highlights(self, target_ids: set[str]):
        """Clean out old tracking markers and draws a red halo ring around highlighted 2D ports."""
        for panel_key, ax_obj in list(self._overlay_axes.items()):
            if ax_obj is not None:
                # Remove stale overlay references tied to this panel key to prevent highlight leakage
                if self._hl_scatters.get(panel_key) is not None:
                    try:
                        self._hl_scatters[panel_key].remove()
                    except (ValueError, KeyError):
                        pass
                    self._hl_scatters[panel_key] = None

                # Fetch coordinate mapping of this specific layout axis structure in last draw cycle
                positions_map = getattr(ax_obj, "_spider_positions", None)
                if positions_map and target_ids:
                    x_hl, y_hl = [], []

                    # Check string IDs against integer map to ensure cross-compatible identity matching
                    for n_str in target_ids:
                        n_id = int(n_str) if n_str.isdigit() else n_str
                        if n_id in positions_map:
                            x_hl.append(positions_map[n_id][0])
                            y_hl.append(positions_map[n_id][1])

                    # Draw high-visibility red halo ring behind matched spiders
                    if x_hl and y_hl:
                        self._hl_scatters[panel_key] = ax_obj.scatter(
                            x_hl,
                            y_hl,
                            s=550,
                            facecolors="none",
                            edgecolors="#FF0000",
                            linewidths=2.5,
                            zorder=5,
                        )

    def _compute_overlay_layout(
        self,
        graph_data: dict[str, Any],
        subgraph_nodes: list[str],
        types_map: dict[str, str],
        draw_style: str,
    ) -> dict[str, list[float]]:
        """Calculate spatial arrangements using geometric constraints or NetworkX force models."""
        # Fall back to algorithmic force-directed layout if specified
        positions = {}

        # Fall back to algorithmic force-directed layout if specified
        if "nx" in draw_style and hasattr(self.ctx.state, "bgraph") and self.ctx.state.bgraph:
            layout_ref_g = nx.Graph()
            layout_ref_g.add_nodes_from(subgraph_nodes)

            # FIX: Explicitly unpack the 2-tuple key using inner parentheses
            for u, v in graph_data.get("edge_types", {}).keys():
                layout_ref_g.add_edge(u, v)

            if draw_style == "nx-atlas":
                return nx.forceatlas2_layout(layout_ref_g)
            if draw_style == "nx-kamada_kawai":
                return nx.kamada_kawai_layout(layout_ref_g)
            if draw_style == "nx-fruchterman_reingold":
                return nx.fruchterman_reingold_layout(layout_ref_g, seed=19)
            return nx.spring_layout(layout_ref_g, seed=1, iterations=50)

        # Build grid coordinates using row/qubit indexing
        coord_groups = {}
        for spider_id in subgraph_nodes:
            r = graph_data["rows"].get(spider_id, -1)
            q = graph_data["qubits"].get(spider_id, -1)
            spider_type = types_map.get(spider_id, "Z")

            if q < 0 and spider_type == "Z":  # Filter out tracking paths above base rows
                coord_groups.setdefault((r, q), []).append(spider_id)
            else:
                positions[spider_id] = [r, -q]

        # Resolve grid collisions on overlapping baseline Z-anchors
        z_displacements = {}
        for (r, q), spider_ids in coord_groups.items():
            has_collision = len(spider_ids) > 1
            for idx, spider_id in enumerate(spider_ids):
                orig_x, orig_y = r, -q
                if not has_collision or idx == 0:
                    positions[spider_id] = [orig_x, orig_y]
                else:
                    y_baseline = -q + 1.0
                    offset = 0.35 * (idx - 1)
                    new_x = r + 0.35 + offset
                    new_y = y_baseline + offset

                    positions[spider_id] = [new_x, new_y]
                    z_displacements[spider_id] = (new_x - orig_x, new_y - orig_y)

        # Propagate displacement offsets to downstream peripheral gadgets
        if z_displacements:
            adjacency = {}
            for u, v in graph_data.get("edge_types", {}).keys():
                if u in subgraph_nodes and v in subgraph_nodes:
                    if graph_data["qubits"].get(u, -1) < 0 and graph_data["qubits"].get(v, -1) < 0:
                        adjacency.setdefault(u, set()).add(v)
                        adjacency.setdefault(v, set()).add(u)

            gadget_assignments = {}
            for z_id in z_displacements.keys():
                direct_neighbors = adjacency.get(z_id, set())
                secondary_neighbors = set()
                for neighbor in direct_neighbors:
                    secondary_neighbors.update(adjacency.get(neighbor, set()))

                all_pattern_nodes = direct_neighbors.union(secondary_neighbors)
                if z_id in all_pattern_nodes:
                    all_pattern_nodes.remove(z_id)

                for node in all_pattern_nodes:
                    if node not in coord_groups:
                        gadget_assignments.setdefault(node, []).append(z_id)

            for node, controlling_zs in gadget_assignments.items():
                avg_dx = sum(z_displacements[z_id][0] for z_id in controlling_zs) / len(
                    controlling_zs
                )
                avg_dy = sum(z_displacements[z_id][1] for z_id in controlling_zs) / len(
                    controlling_zs
                )
                positions[node][0] += avg_dx
                positions[node][1] += avg_dy

        return positions

    def _render_overlay_edges(
        self,
        ax: plt.Axes,
        graph_data: dict[str, Any],
        positions: dict,
        twin_trace: dict,
        curr_edge_ids_set: set,
    ):
        """Paint connecting graph lines alongside invisible thick pick-reactive hitboxes."""
        for (u, v), edge_type in graph_data.get("edge_types", {}).items():
            if u in positions and v in positions:
                x_coords = [positions[u][0], positions[v][0]]
                y_coords = [positions[u][1], positions[v][1]]

                try:
                    col = ZXColors.lookup(edge_type)
                except KeyError:
                    col = "black"

                # Thick invisible lines for easier mouse-click picking
                backing_lines = ax.plot(
                    x_coords, y_coords, color="none", linewidth=8.0, picker=True, zorder=1
                )

                # Stringify the tuple so vector rendering backends can escape it cleanly
                backing_lines[0].set_gid(f"edge_{u}_{v}")

                # Identify active execution pipeline tracing states
                is_edge_current = curr_edge_ids_set and (
                    (
                        u in curr_edge_ids_set
                        or any(i in curr_edge_ids_set for i in twin_trace.get(u, []))
                    )
                    and (
                        v in curr_edge_ids_set
                        or any(i in curr_edge_ids_set for i in twin_trace.get(v, []))
                    )
                )
                edge_line_style = "--" if is_edge_current else "-"

                # Determine completion state visibility weight adjustments
                is_edge_completed = False
                if hasattr(self.ctx.state, "in_zx") and hasattr(self.ctx.state, "base_graph"):
                    is_edge_completed = (
                        (u, v) in self.ctx.state.in_zx["completed_edges"]
                        or (v, u) in self.ctx.state.in_zx["completed_edges"]
                        or (u, v) in self.ctx.state.base_graph["completed_edges"]
                        or (v, u) in self.ctx.state.base_graph["completed_edges"]
                    )
                edge_alpha = 1.0 if is_edge_completed else 0.3

                ax.plot(
                    x_coords,
                    y_coords,
                    color=col,
                    alpha=edge_alpha,
                    linestyle=edge_line_style,
                    zorder=1,
                )

    def _render_overlay_twin_halos(
        self,
        ax: plt.Axes,
        node_ids: list,
        positions: dict,
        twin_trace: dict,
        curr_edge_ids_set: set,
    ):
        """Draw nested background concentric safety tracking rings for multi-qubit twin tracking."""
        if not twin_trace:
            return

        for spider_id in node_ids:
            if twin_trace.get(spider_id):
                all_twins = twin_trace[spider_id][1:]
                num_twins = len(all_twins)
                x, y = positions[spider_id]

                if curr_edge_ids_set.intersection(all_twins):
                    halo_color = "#FF01FB"
                    halo_style = "--"
                else:
                    halo_color = "#7f8c8d"
                    halo_style = "-"

                for ring_idx in range(1, num_twins + 1):
                    ring_size = 300 + (ring_idx * 160)
                    ax.scatter(
                        x,
                        y,
                        s=ring_size,
                        facecolors="none",
                        edgecolors=halo_color,
                        linewidths=1.2,
                        linestyle=halo_style,
                        alpha=0.6,
                        zorder=2,
                    )

    def _render_overlay_nodes(
        self,
        ax: plt.Axes,
        node_ids: list,
        positions: dict,
        types_map: dict,
        active_highlights: set,
        curr_edge_ids_set: set,
    ):
        """Paint localized primary node points and overlays the context selection layer."""
        node_colors, edge_colors, line_widths, line_styles, alpha_list = [], [], [], [], []
        has_bgraph = hasattr(self.ctx.state, "bgraph") and self.ctx.state.bgraph is not None

        completed_cubes_set = set()
        if has_bgraph:
            completed_cubes_set = {
                i for i, coords in self.ctx.state.bgraph.nodes(data="coords") if coords
            }

        for spider_id in node_ids:
            zx_type = types_map.get(spider_id)
            try:
                node_colors.append(ZXColors.lookup(zx_type))
            except KeyError:
                node_colors.append("#FD1818")

            if has_bgraph:
                is_completed = spider_id in completed_cubes_set
                is_current = spider_id in (self.ctx.state.curr_edge_ids or [])
                line_widths.append(1.5 if is_completed else 1)
                edge_colors.append("#000000" if is_completed else "#f2f3fb")
                line_styles.append("dashed" if is_current else "solid")
                alpha_list.append(1.0 if is_completed else 0.7)
            else:
                edge_colors.append("#f2f3fb")
                line_widths.append(1)
                line_styles.append("solid")
                alpha_list.append(0.7)

        # Primary interactive node scatter layout paint
        x_nodes = [positions[n][0] for n in node_ids]
        y_nodes = [positions[n][1] for n in node_ids]
        nodes_scatter = ax.scatter(
            x_nodes,
            y_nodes,
            c=node_colors,
            edgecolors=edge_colors,
            s=300,
            linewidths=line_widths,
            linestyles=line_styles,
            alpha=alpha_list,
            picker=True,
            zorder=3,
        )

        # Keep a safe, static generic label string for the SVG vector backend
        nodes_scatter.set_gid("overlay_nodes_layer")

        # Bind the actual node ID sequence directly to the object data dictionary
        nodes_scatter.node_ids = list(node_ids)

        # Safely clean out pre-existing local panel selection markers
        current_panel_key = "in_zx" if ax == self._overlay_axes.get("in_zx") else "base_graph"
        if self._hl_scatters.get(current_panel_key) is not None:
            try:
                self._hl_scatters[current_panel_key].remove()
            except (ValueError, KeyError):
                pass
            self._hl_scatters[current_panel_key] = None

        # Repaint active persistent red halo selections if found
        x_hl, y_hl = [], []
        for n_id in node_ids:
            if str(n_id) in active_highlights:
                x_hl.append(positions[n_id][0])
                y_hl.append(positions[n_id][1])
        if x_hl and y_hl:
            self._hl_scatters[current_panel_key] = ax.scatter(
                x_hl,
                y_hl,
                s=550,
                facecolors="none",
                edgecolors="#FF0000",
                linewidths=2.5,
                zorder=5,
            )

    def _render_overlay_labels(
        self, ax: plt.Axes, node_ids: list, positions: dict, graph_data: dict, twin_trace: dict
    ):
        """Annotate core identification indexing strings and continuous mathematical fraction phase terms."""
        has_bgraph = hasattr(self.ctx.state, "bgraph") and self.ctx.state.bgraph is not None
        completed_cubes_set = set()
        if has_bgraph:
            completed_cubes_set = {
                i for i, coords in self.ctx.state.bgraph.nodes(data="coords") if coords
            }

        for n_id in node_ids:
            x, y = positions[n_id]
            label_alpha = 1.0 if n_id in completed_cubes_set else 0.5

            # Print identity value over the center node mask
            ax.text(
                x,
                y,
                str(n_id),
                alpha=label_alpha,
                fontsize=8,
                ha="center",
                va="center",
                zorder=4,
                color="black",
            )

            # Evaluate phase representation string configurations
            phase = graph_data.get("phases", {}).get(n_id, 0)
            if phase != 0 and phase != Fraction(1, 1):
                if hasattr(phase, "numerator"):
                    num, den = phase.numerator, phase.denominator
                    phase_str = f"π/{den}" if num == 1 else f"{num}π/{den}" if den != 1 else "π"
                else:
                    phase_str = f"{phase}"

                # Offset phase labels lower if twin halos occupy immediate outer bounds
                y_offset = 0.38 if twin_trace.get(n_id) else 0.25
                ax.text(
                    x,
                    y - y_offset,
                    phase_str,
                    fontsize=8,
                    ha="center",
                    va="top",
                    zorder=4,
                    color="black",
                    weight="bold",
                )


#############
# RENDERERS #
#############
def render_block(
    ax: matplotlib.axes.Axes,
    node_id: int,
    coords: tuple[int, int, int],
    size: list[float],
    zx_block: ZXBlock,
    alpha: float = 1.0,
    edge_col: str = "#000000",
    border_width: float = 0.3,
    in_curr_edge: bool = False,
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
        in_curr_edge: True if block is part of edge placed when rendering the parent visualisation.

    AI disclaimer:
        category: Coding partner (see CONTRIBUTING.md for details).
        model: Gemini, 2.5 Flash.

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

    # Color as applicable
    cols = zx_block.get_face_colors
    if cols[0] == ZXColors.Y:
        cols = tuple([cols[0]] * 6)
    face_cols = [cols[2]] * 2 + [cols[1]] * 2 + [cols[0]] * 2
    edge_style = "--" if in_curr_edge else "-"
    border_width = 0.7 if in_curr_edge else border_width if zx_block.zx_type != "T" else 3
    edge_col = "#ffffff" if in_curr_edge else edge_col

    # Join into Poly collection
    poly_collection = Poly3DCollection(
        faces,
        facecolors=face_cols,
        linewidths=border_width,
        edgecolors=edge_col,
        linestyle=edge_style,
        alpha=alpha,
        picker=True,
        label=str(node_id),
    )
    poly_collection._associated_text = None
    poly_collection._original_alpha = alpha
    poly_collection._zx_type = zx_block.zx_type
    poly_collection._original_linewidth = border_width

    # Preflight spider as invisible
    spider_dot = render_phantom_spider(ax, coords, zx_block, alpha=0.8, in_curr_edge=in_curr_edge)
    spider_dot.set_visible(False)
    poly_collection._associated_spider = spider_dot

    # Attach labels if node has ID
    if node_id != "TBD":
        txt_obj = ax.text(
            coords[0],
            coords[1],
            coords[2],
            s=str(node_id),
            color="black",
            visible=False,
            fontsize="small",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            zorder=100,  # Forces the text to render visibly over the transparent block faces
        )
        poly_collection._associated_text = txt_obj

    # Add to plot
    ax.add_collection3d(poly_collection)

    # Return for usage in show/hide toggle features
    return [poly_collection]


def render_pipe(
    ax: matplotlib.axes.Axes,
    u_coords: StandardCoord,
    v_coords: StandardCoord,
    kind: str,
    edge_col: str = "#000000",
    in_curr_edge: bool = False,
    border_width: float = 0.3,
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
        in_curr_edge: True if block is part of edge placed when rendering the parent visualisation.
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
    artists = []

    # Process pipe
    if adjusted_length > 0:
        orientation = np.argmax(np.abs(v_coords - u_coords))
        size = [0.33, 0.33, 0.33]
        size[orientation] = float(adjusted_length)
        face_cols = ["gray"] * 6

        col = node_hex_map.get(kind.replace("*", "").lower(), ["gray"] * 3)
        face_cols = [col[2]] * 2 + [col[1]] * 2 + [col[0]] * 2

        edge_style = "--" if in_curr_edge else "-"
        border_width = 0.7 if in_curr_edge else border_width
        edge_col = "#ffffff" if in_curr_edge else edge_col

        # Regular pipes
        if "H" not in kind:
            artists = render_pipe_section(
                ax,
                midpoint,
                size,
                face_cols,
                edge_col,
                alpha,
                edge_style=edge_style,
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
                    edge_style=edge_style,
                    border_width=border_width,
                )
                artists_2 = render_pipe_section(
                    ax,
                    centre2,
                    size_yellow,
                    face_cols_yellow,
                    edge_col,
                    alpha,
                    edge_style=edge_style,
                    border_width=border_width,
                )
                artists_3 = render_pipe_section(
                    ax,
                    centre3,
                    size_col,
                    face_cols_2,
                    edge_col,
                    alpha,
                    edge_style=edge_style,
                    border_width=border_width,
                )

                artists = artists_1 + artists_2 + artists_3

        # Add tracking metadata
        for collection in artists:
            collection._original_alpha = alpha
            collection._original_linewidth = border_width

        # Return for usage in show/hide toggle features
        return artists


def render_pipe_section(
    ax: matplotlib.axes.Axes,
    centre: NDArray[np.float64],
    size: list[float],
    face_cols: list[str],
    edge_col: str,
    alpha: float | int,
    edge_style: str = "-",
    border_width: float = 0.3,
) -> Poly3DCollection:
    """Render edges/pipes.

    This function takes care of rendering a section of a 3D pipe/edge. It takes the coordinates
    and size of the pipe alongside other visual formatting parameters and calculates all
    geometric objects needed to render it in 3D.

    Args:
        ax: Matplotlib's 3D subplot object.
        centre: The (x, y, z) coordinates of the edge's centre (midpoint between connecting nodes).
        size: The (size_x, size_y, size_z) of the edge/pipe.
        face_cols: The colour pattern for the edge/pipe.
        edge_col: The color of the edges for the edge/pipe.
        alpha: Any desired value for alpha (transparency)
        edge_style: The style of the border.
        border_width: The width for borders of edge.

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
        linestyle=edge_style,
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


def render_phantom_spider(
    ax: matplotlib.axes.Axes,
    coords: tuple[int, int, int],
    zx_block: ZXBlock,
    alpha: float = 0.5,
    in_curr_edge: bool = False,
) -> Any:
    """Render a clean, lightweight 3D scatter dot representing a ZX-calculus spider."""
    x, y, z = coords
    z_center = z - 1.33 if zx_block.zx_type == "T" else z

    # Map the node type to standard ZX colors
    spider_color = zx_block.get_zx_color

    # Apply golden borders if the spider is part of the current active path
    edge_color = "#ffffff" if in_curr_edge else "black"
    edge_width = 1.5 if in_curr_edge else 0.75

    (spider_dot,) = ax.plot(
        [x],
        [y],
        [z_center],
        marker="o",
        markersize=12,  # Fixed pixel/coordinate boundary balance
        color=spider_color,
        alpha=alpha,
        markeredgecolor=edge_color,
        markeredgewidth=edge_width,
        zorder=150,
    )
    return spider_dot


def render_phantom_edge(
    ax: matplotlib.axes.Axes,
    u_coords: tuple[float, float, float],
    v_coords: tuple[float, float, float],
    pipe_kind: str,
    alpha: float = 0.5,
    in_curr_edge: bool = False,
) -> Any:
    """Render a thin, internal wire line through a pipe to represent a logical ZX-calculus edge."""
    # Split the start and end coordinates into separate spatial vectors
    x_coords = [u_coords[0], v_coords[0]]
    y_coords = [u_coords[1], v_coords[1]]
    z_coords = [u_coords[2], v_coords[2]]

    # Color hadamards and current edges appropriately
    if "h" in str(pipe_kind).lower():
        edge_color = "#FFFF00"
        line_style = "--" if in_curr_edge else "-"
    else:
        edge_color = "#ffffff" if in_curr_edge else "#000000"
        line_style = "--" if in_curr_edge else "-"

    # Draw a crisp, lightweight line through the absolute center of the pipe channel
    (phantom_wire,) = ax.plot(
        x_coords,
        y_coords,
        z_coords,
        color=edge_color,
        linewidth=1.2,
        linestyle=line_style,
        alpha=alpha,
        zorder=140,  # Layered cleanly inside the transparent physical pipe walls
    )

    return phantom_wire


#####################
# OTHER VISUALISERS #
#####################
def draw_as_zx(
    bgraph: nx.Graph,
    in_qubits: dict[int, int],
    in_rows: dict[int, int],
    draw_style: str = "zx",
    first_spider: int | None = None,
):
    """Draw a quick-and-dirty 2D graph of the NX BlockGraph using ZX-calculus or NX layouts and styling.

    Args:
        bgraph: The primary NetworkX graph containing the blockgraph being built.
        in_qubits: A dictionary containing spiders with explicit qubit number.
        in_rows: A dictionary containing spiders with explicit row number.
        draw_style: The style of drawing:
            zx: Positions nodes in a PyZX-like manner
            nx: Positions nodes using NX algorithms.
        first_spider: The spiders used as starting point for the build (needed for some layouts)

    AI disclaimer:
        category: Coding partner (see CONTRIBUTING.md for details).
        model: Gemini, 2.5 Flash.

    """

    # Position nodes
    if draw_style == "zx":
        try:
            positions = {k: [in_rows[k], -q] for k, q in in_qubits.items()}
        except Exception as _:
            positions = nx.spring_layout(bgraph)
    else:
        if draw_style == "nx-atlas":
            positions = nx.forceatlas2_layout(bgraph)
        elif draw_style == "nx-kamada_kawai":
            positions = nx.kamada_kawai_layout(bgraph)
        elif draw_style == "nx-fruchterman_reingold":
            positions = nx.fruchterman_reingold_layout(bgraph)
        elif draw_style == "nx-bfs" and first_spider:
            positions = nx.bfs_layout(bgraph, first_spider)
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
