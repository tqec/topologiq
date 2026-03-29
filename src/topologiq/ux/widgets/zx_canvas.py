"""UX blockgraph canvas.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pyzx as zx
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from vispy import scene
from vispy.color import Color

from topologiq.input.pyzx_manager import AugmentedZXGraph
from topologiq.input.utils import ZXColors, ZXEdgeTypes, ZXTypes
from topologiq.ux.utils import styles
from topologiq.ux.utils.aux import create_split_controls


class ZXCanvas(QWidget):  # noqa: D101
    toggle_requested = Signal(str)

    def __init__(self, manager, parent=None):  # noqa: D107
        super().__init__(parent)
        self.manager = manager
        self.current_aug_zx = None
        self.items = []
        self._tasks = set()
        self.is_reduced_view = False
        self.last_points = []

        # 1. Main Container Style
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(0)
        self.setStyleSheet("ZXCanvas { background: #1a1a1a; }")

        # 2. The Styled Frame Wrapper
        self.canvas_frame = QFrame()
        self.canvas_frame.setObjectName("MainCanvasFrame")
        self.canvas_frame.setStyleSheet("""
            #MainCanvasFrame {
                border: 2px solid #999;
                background: #a3a3a3;
            }
        """)

        # 3. VisPy Integration
        self.canvas = scene.SceneCanvas(
            keys="interactive",
            show=True,
            bgcolor="#bcffc6",
            config={"depth_size": 24},
        )

        # Ensure the native widget itself doesn't have a background/border conflict
        self.canvas.native.setStyleSheet("border: none; background: transparent;")

        frame_layout = QVBoxLayout(self.canvas_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(self.canvas.native)

        self.layout.addWidget(self.canvas_frame)

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "panzoom"

        self._setup_hud_anchors()

    def _handle_load_request(self, graph_key: str = "input"):
        path, _ = QFileDialog.getOpenFileName(
            self, "Direct ZX Import (Raw QASM)", "", "OpenQASM (*.qasm)"
        )

        if path:
            try:
                # FIX: Pass the 'path' variable directly to path_to_qasm_file.
                # Do NOT use .read_text() here if using that parameter.
                new_aug_zx = self.manager.zx_manager.add_graph_from_qasm(
                    path_to_qasm_file=Path(path),
                    graph_key=graph_key,
                )
                self.manage_aug_zx(new_aug_zx)
                self.manager._data_store["augmented_zx_graph_in"] = new_aug_zx
                self.manager.status_changed.emit("ZX Graph imported directly.")

            except Exception as e:
                self.manager.status_changed.emit(f"Direct Load Error: {e}")

    def manage_aug_zx(self, aug_zx_graph: AugmentedZXGraph):  # noqa: D102
        try:
            self.current_aug_zx = aug_zx_graph
            self._refresh_render()
        except Exception as e:
            self.manager.status_changed.emit(
                f"ERROR. Crash while trying to manage augmented NX graph in ZX canvas: {e}"
            )

    def _setup_hud_anchors(self):
        """Restores the four-corner HUD layout from the original TransformPane."""

        # --- TOP LEFT: Title ---
        # Parented to canvas_frame to stay inside the border
        self.info_label = QLabel("INPUT ZX", self.canvas_frame)
        self.info_label.setStyleSheet(
            "color: #111; font-weight: bold; font-size: 11px; background: transparent;"
        )

        # --- TOP RIGHT: Unified File/View Bar ---
        self.top_right_hud = QFrame(self.canvas_frame)
        self.top_right_hud.setStyleSheet("""
            QFrame { background: transparent; border: none; }
            QPushButton {
                background: #f2f3fb; color: #333; font-size: 9px; font-weight: bold;
                padding: 5px 12px; border: 1px solid #666;
            }
            QPushButton:hover { background: rgba(0,0,0,15); }
            QPushButton:checked { background: #2d5a27; color: white;}
        """)
        tr_layout = QHBoxLayout(self.top_right_hud)
        tr_layout.setContentsMargins(5, 2, 5, 2)
        tr_layout.setSpacing(0)

        self.btn_load_qasm = QPushButton("LOAD QASM")
        self.btn_load_json = QPushButton("LOAD JSON")
        self.btn_save_json = QPushButton("SAVE JSON")

        for btn in [self.btn_load_qasm, self.btn_load_json, self.btn_save_json]:
            tr_layout.addWidget(btn)

        # Signal connections
        self.btn_load_qasm.clicked.connect(lambda: self._handle_load_request())
        self.btn_load_json.clicked.connect(lambda: self._handle_json_io("LOAD"))
        self.btn_save_json.clicked.connect(lambda: self._handle_json_io("SAVE"))

        # --- BOTTOM RIGHT: Transformation Controls ---
        self.bottom_right_hud = QFrame(self.canvas_frame)
        self.bottom_right_hud.setStyleSheet("background: transparent;")
        br_layout = QHBoxLayout(self.bottom_right_hud)
        br_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_full = self._make_action_btn("COMPILE FULL")
        self.btn_reduced_compile = self._make_action_btn("COMPILE REDUCED")
        self.btn_full.setStyleSheet(styles.PILL_BTN_PYZX + "background: #333;")
        self.btn_reduced_compile.setStyleSheet(styles.PILL_BTN_PYZX + "background: #333;")

        br_layout.addWidget(self.btn_full)
        br_layout.addWidget(self.btn_reduced_compile)

        # Connect to the async wrapper
        self.btn_full.clicked.connect(lambda: self._trigger_surgery(False))
        self.btn_reduced_compile.clicked.connect(lambda: self._trigger_surgery(True))

        # --- BOTTOM LEFT: Layout Toggles + Navigation ---
        # 1. Triplet Buttons Cluster via Utility
        self.toggle_buttons = create_split_controls(
            self.canvas_frame, ["CLOSE CANVAS", "40/60"], self.toggle_requested.emit
        )

        # 2. Navigation (Round Reset Button)
        self.btn_reset_cam = QPushButton("⟲", self.canvas_frame)
        self.btn_reset_cam.setFixedSize(30, 30)
        self.btn_reset_cam.setCursor(Qt.PointingHandCursor)
        self.btn_reset_cam.setStyleSheet(
            styles.STYLE_CLOSE_BTN
            + "QPushButton { background: #333; border: 1px solid black; border-radius: 15px; font-size: 16px; }"
            "QPushButton:hover { background: #777; }"
        )
        self.btn_reset_cam.clicked.connect(self._reset_camera_view)

        # Initial size adjustment to ensure tr_layout and others have valid widths for first resizeEvent
        self.top_right_hud.adjustSize()
        self.bottom_right_hud.adjustSize()

    def _make_action_btn(self, text):
        """Create the dark action buttons."""
        btn = QPushButton(text)
        btn.setFixedSize(130, 30)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #1a1a2e; color: #aaccff; border: 1px solid #3e3e5e;
                border-radius: 15px; font-size: 9px; font-weight: bold;
            }
            QPushButton:hover { background-color: #252545; }
        """)
        return btn

    def render_zx(self, g):
        """Debug-ready vectorized renderer."""
        if g is None or g.num_vertices() == 0:
            return

        self._clear_scene()

        # --- DATA EXTRACTION ---
        v_list = list(g.vertices())
        num_v = len(v_list)
        node_to_idx = {node_id: i for i, node_id in enumerate(v_list)}

        pos = np.zeros((num_v, 3), dtype=np.float32)
        colors_v = np.zeros((num_v, 4), dtype=np.float32)
        for i, v in enumerate(v_list):
            pos[i] = [g.row(v) * 2.0, -g.qubit(v) * 2.0, 0.0]

            # 1. Get the PyZX int type (0, 1, 2)
            pyzx_type = g.type(v)

            # 2. Map to your Enum for standardization
            # (Assuming the Enum values match PyZX: Z=1, X=2, B=0)
            try:
                standard_type = ZXTypes(pyzx_type).name  # Returns 'Z', 'X', or 'BOUNDARY'
                hex_v = ZXColors.lookup(standard_type)
            except ValueError:
                hex_v = ZXColors.SIMPLE
            colors_v[i] = Color(hex_v).rgba

        # --- THE VISUALS ---
        # 1. Edges
        edges = list(g.edges())
        num_e = len(edges)

        num_v = len(v_list)
        dynamic_node_size = np.clip(14 - (num_v / 120), 6, 14)
        dynamic_font_size = np.clip(11 - (num_v / 80), 0, 11)

        # 2 points per edge (Start, End) and 2 points per color
        edge_coords = np.zeros((num_e * 2, 3), dtype=np.float32)
        colors_e = np.zeros((num_e * 2, 4), dtype=np.float32)

        for i, edge in enumerate(edges):
            u, v = edge
            # 1. Map Coordinates
            edge_coords[i * 2] = pos[node_to_idx[u]]  # Start point
            edge_coords[i * 2 + 1] = pos[node_to_idx[v]]  # End point

            # 2. Map Edge Type to Color
            # pyzx_edge_type is usually 1 (Simple) or 2 (Hadamard)
            pyzx_edge_type = g.edge_type(edge)

            if pyzx_edge_type == ZXEdgeTypes.HADAMARD:
                hex_e = ZXColors.HADAMARD
                rgba_c = Color(hex_e).rgba
            else:
                hex_e = ZXColors.SIMPLE
                rgba_c = Color(hex_e).rgba

            # Assign the same color to both vertices of the segment
            colors_e[i * 2] = rgba_c
            colors_e[i * 2 + 1] = rgba_c

        self.edge_visual = scene.visuals.Line(
            pos=edge_coords,
            connect="segments",
            color=colors_e,  # Pass the (N*2, 4) color array
            width=1.5,
            parent=self.view.scene,
        )
        self.edge_visual.order = 1
        self.edge_visual.set_gl_state(depth_test=False)
        self.items.append(self.edge_visual)

        # 2. Nodes
        self.node_visual = scene.visuals.Markers(
            pos=pos,
            face_color=colors_v,
            edge_color="#f2f3fb",
            edge_width=1,
            size=dynamic_node_size,  # Slightly larger for boxes to feel "substantial"
            symbol="square",  # Changed from 'disc' to 'square'
            parent=self.view.scene,
        )

        self.node_visual.order = 2
        self.node_visual.set_gl_state(depth_test=False)
        self.node_visual.interactive = True
        self.items.append(self.node_visual)

        # Text
        node_labels = [str(v) for v in v_list]
        self.text_visual = scene.visuals.Text(
            text=node_labels,
            pos=pos,
            font_size=dynamic_font_size,
            anchor_x="left",
            anchor_y="top",
            bold=True,
            parent=self.view.scene,
        )
        # Ensure labels are on top of nodes (Order 3)
        self.text_visual.visible = dynamic_font_size > 1.0
        self.text_visual.order = 3
        self.text_visual.set_gl_state("translucent", depth_test=False)
        self.items.append(self.text_visual)

        # --- THE CAMERA FIX ---
        self.last_points = pos

        # Force the camera to look at the data immediately
        # We add a 2-unit buffer around the min/max coordinates
        x_min, x_max = pos[:, 0].min() - 2, pos[:, 0].max() + 2
        y_min, y_max = pos[:, 1].min() - 2, pos[:, 1].max() + 2

        self.view.camera.rect = (x_min, y_min, x_max - x_min, y_max - y_min)
        self.canvas.update()

    def _handle_json_io(self, mode: str):
        """Unified handler for loading and saving PyZX Graph JSON."""
        ext = "PyZX Graph as JSON (*.json)"

        if mode == "SAVE":
            if not self.current_aug_zx:
                self.manager.status_changed.emit("Nothing to save.")
                return

            path_str, _ = QFileDialog.getSaveFileName(self, "Save Graph JSON", "", ext)
            if path_str:
                try:
                    # Fix: Ensure .json extension is present
                    path = Path(path_str)
                    if path.suffix.lower() != ".json":
                        path = path.with_suffix(".json")

                    # Get the currently active graph based on toggle state
                    g = (
                        self.current_aug_zx.zx_graph_reduced
                        if self.is_reduced_view
                        else self.current_aug_zx.zx_graph
                    )

                    # Save PyZX JSON string
                    path.write_text(g.to_json())
                    self.manager.status_changed.emit(f"Graph saved: {path.name}")
                except Exception as e:
                    self.manager.status_changed.emit(f"Save Error: {e}")

        elif mode == "LOAD":
            path, _ = QFileDialog.getOpenFileName(self, "Open Graph JSON", "", ext)
            if path:
                try:
                    json_data = Path(path).read_text()
                    # PyZX creates a new graph object from the string
                    new_graph = zx.Graph.from_json(json_data)

                    # Wrap it in your AugmentedZXGraph structure
                    # We assume a fresh import is both the main and reduced graph for now

                    new_aug_zx = AugmentedZXGraph(zx_graph=new_graph)

                    self.manage_aug_zx(new_aug_zx)
                    self.manager.status_changed.emit(f"Graph loaded: {Path(path).name}")
                except Exception as e:
                    self.manager.status_changed.emit(f"Load Error: {e}")

    def _trigger_surgery(self, use_reduced: bool):
        """Initiate lattice surgery via the manager."""
        self._run_async(self.manager.handle_lattice_surgery(use_reduced=use_reduced))

    def _clear_scene(self):
        for item in self.items:
            item.parent = None
        self.items = []

    def _toggle_reduction(self):
        self.is_reduced_view = self.btn_reduce.isChecked()
        new_text = "REDUCED [ON]" if self.is_reduced_view else "REDUCED [OFF]"
        self.btn_reduce.setText(new_text)

        # --- Force Layout Recalculation ---
        self.btn_reduce.adjustSize()
        self.top_right_hud.adjustSize()

        w = self.width()
        self.top_right_hud.move(w - self.top_right_hud.width() - 15, 10)

        self._refresh_render()

    def _refresh_render(self):
        if not self.current_aug_zx:
            return
        g = (
            self.current_aug_zx.zx_graph_reduced
            if self.is_reduced_view
            else self.current_aug_zx.zx_graph
        )
        self.render_zx(g)

    def _run_async(self, coro):
        """Manage task references and avoid GC warnings."""
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _reset_camera_view(self):
        """Reset PanZoom camera."""
        # Use .size to check if the array is empty
        if (
            hasattr(self, "last_points")
            and self.last_points is not None
            and self.last_points.size > 0
        ):
            pts = self.last_points

            # Extract bounds
            x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
            y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

            # Calculate width/height with a 10% padding
            w = x_max - x_min
            h = y_max - y_min
            padding = max(w, h, 2) * 0.1

            # PanZoom camera.rect is (x, y, width, height)
            self.view.camera.rect = (
                x_min - padding,
                y_min - padding,
                w + (padding * 2),
                h + (padding * 2),
            )
        else:
            # Default fallback if no graph is loaded
            self.view.camera.rect = (-5, -5, 10, 10)

    def resizeEvent(self, event):  # noqa: N802
        """Keep HUD elements anchored during window resizing."""
        super().resizeEvent(event)
        # Use the frame's internal dimensions for HUD positioning
        fw = self.canvas_frame.width()
        fh = self.canvas_frame.height()
        m = 15  # Uniform margin constant

        # Top Left
        self.info_label.move(m, m)

        # Top Right
        tr_w = self.top_right_hud.width()
        self.top_right_hud.move(fw - tr_w - m, m)

        # Bottom Left Cluster
        tb_w = self.toggle_buttons.width()
        tb_h = self.toggle_buttons.height()
        # Align bottom-up: Frame Height - Widget Height - Margin
        self.toggle_buttons.move(m, fh - tb_h - m)

        # Reset Camera (relative to toggle cluster)
        reset_x = m + tb_w + 10
        # Centering the 30px reset button vertically against the 24px toggles
        self.btn_reset_cam.move(reset_x, fh - 30 - m + 3)

        # Bottom Right
        br_w = self.bottom_right_hud.width()
        br_h = self.bottom_right_hud.height()
        self.bottom_right_hud.move(fw - br_w - m, fh - br_h - m)
