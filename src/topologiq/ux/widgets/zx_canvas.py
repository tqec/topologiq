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
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMenu,
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
        self.experimental_zx = None
        self.items = []
        self._tasks = set()
        self.is_reduced_view = False
        self.last_points = []
        self.hovered_node_idx = None
        self.toggle_buttons = None

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

        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        self._setup_hud_anchors()

    def _handle_load_request(self, graph_key: str = "primary"):
        path, _ = QFileDialog.getOpenFileName(
            self, "Direct ZX Import (Raw QASM)", "", "OpenQASM (*.qasm)"
        )

        if not path:
            return  # Safety exit: Nothing is destroyed if user cancels

        # Now it is safe to reset
        self.manager.clear_session()
        self._update_registry_list()

        try:
            # We access the FRESH zx_manager created by clear_session()
            new_aug_zx = self.manager.zx_manager.add_graph_from_qasm(
                path_to_qasm_file=Path(path),
                graph_key=graph_key,
            )

            # Update UI and Sync Manager Data Store
            self._update_registry_list()
            self.manage_aug_zx(new_aug_zx)

            # Re-link the input for future surgery/optimization
            self.manager._data_store["augmented_zx_graph_in"] = new_aug_zx
            self.manager.status_changed.emit(f"Imported: {Path(path).name}")

        except Exception as e:
            self.manager.status_changed.emit(f"Direct Load Error: {e}")

    def manage_aug_zx(self, aug_zx_graph: AugmentedZXGraph):  # noqa: D102
        try:
            self.current_aug_zx = aug_zx_graph
            self._update_registry_list()

            self._refresh_render()
        except Exception as e:
            self.manager.status_changed.emit(f"ERROR: {e}")

    def _setup_hud_anchors(self):
        """Initialise all HUD components parented to the styled canvas_frame."""
        # Top Left: Registry + Lens
        self._setup_graph_controls()

        # Top Right: Actions & Optimization
        self._setup_action_bar()

        # Bottom Left: Navigation
        self._setup_navigation_controls()

        # Prep sizes for the first resizeEvent
        self.graph_control_hud.adjustSize()
        self.top_right_hud.adjustSize()

    def _setup_graph_controls(self):
        """Top-Left HUD: Styled to match the Consolidated Action Bar."""
        self.graph_control_hud = QFrame(self.canvas_frame)

        # Style variables matching the Top-Right HUD
        bg_color = "#2a2a2a"
        hover_color = "#3a3a3a"
        border_color = "#555"
        text_color = "#eee"

        self.graph_control_hud.setStyleSheet(f"""
            QFrame {{ background: rgba(35, 35, 35, 220); border-radius: 4px; border: 1px solid #444; }}
            QComboBox {{ background: {bg_color}; color: {text_color}; border: 1px solid {border_color}; padding: 4px; font-size: 10px; font-weight: bold; min-width: 140px; border-radius: 3px; }}
            QComboBox:hover {{ border-color: #777; background: {hover_color}; }}
            QComboBox::drop-down {{ border: none; width: 0px; }}
            #PillFrame {{ background: #181818; border: 1px solid #333; }}
            QPushButton {{ background: transparent; color: #777; font-size: 9px; font-weight: bold; border: none; padding: 4px; min-width: 45px; }}

            QPushButton:checked {{ color: #fff; background: {hover_color}; }}
            QPushButton#btn_red:checked {{ background: #2d5a27; color: #fff; }}
        """)

        layout = QHBoxLayout(self.graph_control_hud)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

        # 1. Registry Dropdown
        self.combo_registry = QComboBox()
        self._update_registry_list()

        # CRITICAL: Force the dropdown list (view) to match the QMenu style
        self.combo_registry.view().setStyleSheet(f"""
        QAbstractItemView {{
        background-color: {bg_color};
        border: 1px solid {border_color};
        selection-background-color: {hover_color};
        selection-color: #fff;
        outline: none;
        }}
        QAbstractItemView::item {{
        padding: 4px 10px;
        color: {text_color};
        font-size: 10px;
        }}
        """)

        self.combo_registry.currentTextChanged.connect(self._handle_registry_change)
        layout.addWidget(self.combo_registry)

        # 2. Dual-State Pill
        self.pill_frame = QFrame()
        self.pill_frame.setObjectName("PillFrame")
        pill_layout = QHBoxLayout(self.pill_frame)
        pill_layout.setContentsMargins(2, 2, 2, 2)
        pill_layout.setSpacing(2)

        self.btn_view_full = QPushButton("FULL")
        self.btn_view_red = QPushButton("REDUCED")
        self.btn_view_red.setObjectName("btn_red")

        for btn in [self.btn_view_full, self.btn_view_red]:
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            pill_layout.addWidget(btn)

        self.btn_view_full.setChecked(not self.is_reduced_view)
        self.btn_view_red.setChecked(self.is_reduced_view)

        self.btn_view_full.clicked.connect(lambda: self._set_view_mode(False))
        self.btn_view_red.clicked.connect(lambda: self._set_view_mode(True))

        layout.addWidget(self.pill_frame)
        self.graph_control_hud.adjustSize()

    def _update_registry_list(self):
        """Force the UI to match the current Manager state."""
        self.combo_registry.blockSignals(True)
        self.combo_registry.clear()

        # 1. Get keys from the current zx_manager
        try:
            keys = list(self.manager.zx_manager._collection.keys())
        except (AttributeError, TypeError):
            keys = []

        # 2. Populate and set selection
        if not keys:
            self.combo_registry.addItem("No Graphs Loaded")
            self.combo_registry.setEnabled(False)
        else:
            self.combo_registry.setEnabled(True)
            self.combo_registry.addItems(keys)

            # Auto-select the latest one (the one just added/loaded)
            self.combo_registry.setCurrentIndex(len(keys) - 1)

        self.combo_registry.blockSignals(False)

    def _handle_registry_change(self, text):
        """Switch current graph when dropdown changes."""
        # Guard against empty states or initialization noise
        if not text or text == "No Graphs Loaded" or not self.combo_registry.isEnabled():
            return

        # Sanitize key (just in case of whitespace)
        key = text.strip()

        try:
            # 1. Retrieve from Manager
            registry_graph = self.manager.zx_manager.get_graph(graph_key=key)

            # 2. Optimization: Don't re-render if it's already the active graph
            if self.current_aug_zx == registry_graph:
                return

            # 3. Update State and View
            self.current_aug_zx = registry_graph
            self._refresh_render()

            self.manager.status_changed.emit(f"Switched to Registry: {key}")

        except Exception as e:
            self.manager.status_changed.emit(f"Registry Access Error: {e}")

    def _set_view_mode(self, reduced: bool):
        """Toggle between full and reduced versions of the CURRENT graph."""
        self.is_reduced_view = reduced
        self.manager.status_changed.emit(f"Mode: {'Reduced' if reduced else 'Full'}")
        self._refresh_render()

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

    #################
    def render_zx(self, g):
        """Proxy-Vectorized renderer: High performance with individual node addressability."""

        if g is None or g.num_vertices() == 0:
            self._clear_scene()
            return

        self._clear_scene()

        # --- 1. INITIALIZE BUFFERS & REGISTRY ---
        v_list = list(g.vertices())
        num_v = len(v_list)

        # Permanent buffers for real-time manipulation
        self.node_positions = np.zeros((num_v, 3), dtype=np.float32)
        self.node_colors = np.zeros((num_v, 4), dtype=np.float32)
        self.base_node_colors = self.node_colors.copy()
        self.node_registry = {}  # v_id -> buffer_index
        self.idx_to_node = {}  # buffer_index -> v_id

        for i, v in enumerate(v_list):
            # Map PyZX coords to VisPy space
            self.node_positions[i] = [g.row(v) * 2.0, -g.qubit(v) * 2.0, 0.0]
            self.node_registry[v] = i
            self.idx_to_node[i] = v

            # Color Mapping
            try:
                standard_type = ZXTypes(g.type(v)).name
                hex_v = ZXColors.lookup(standard_type)
            except Exception:
                hex_v = ZXColors.SIMPLE
            self.node_colors[i] = Color(hex_v).rgba
            self.base_node_colors = self.node_colors.copy()

        # --- 2. EDGE RENDERING (Order 1) ---
        edges = list(g.edges())
        if edges:
            num_e = len(edges)
            self.edge_coords = np.zeros((num_e * 2, 3), dtype=np.float32)
            self.edge_colors = np.zeros((num_e * 2, 4), dtype=np.float32)
            self.edge_to_indices = {}  # edge -> (idx_start, idx_end)

            for i, (u, v) in enumerate(edges):
                idx_u, idx_v = self.node_registry[u], self.node_registry[v]
                self.edge_coords[i * 2] = self.node_positions[idx_u]
                self.edge_coords[i * 2 + 1] = self.node_positions[idx_v]

                # Track edge indices for rubber-banding logic later
                self.edge_to_indices[(u, v)] = (i * 2, i * 2 + 1)

                hex_e = (
                    ZXColors.HADAMARD
                    if g.edge_type((u, v)) == ZXEdgeTypes.HADAMARD
                    else ZXColors.SIMPLE
                )
                rgba_e = Color(hex_e).rgba
                self.edge_colors[i * 2] = self.edge_colors[i * 2 + 1] = rgba_e

            self.edge_visual = scene.visuals.Line(
                pos=self.edge_coords,
                connect="segments",
                color=self.edge_colors,
                width=1.5,
                parent=self.view.scene,
            )
            self.edge_visual.set_gl_state(depth_test=False)
            self.edge_visual.order = 1
            self.items.append(self.edge_visual)

        # --- 3. NODE RENDERING (Order 2) ---
        self.current_node_size = np.clip(14 - (num_v / 120), 6, 14)
        self.node_visual = scene.visuals.Markers(
            pos=self.node_positions,
            face_color=self.node_colors,
            edge_color="#f2f3fb",
            edge_width=1,
            size=self.current_node_size,
            symbol="square",
            parent=self.view.scene,
        )
        self.node_visual.set_gl_state(depth_test=False)
        self.node_visual.order = 2
        # Enable picking for mouse interactions
        self.node_visual.interactive = True
        self.items.append(self.node_visual)

        # --- 4. TEXT RENDERING (Order 3) ---
        dynamic_font_size = np.clip(11 - (num_v / 80), 0, 11)
        if dynamic_font_size > 1.5:
            self.text_visual = scene.visuals.Text(
                text=[str(v) for v in v_list],
                pos=self.node_positions,
                font_size=dynamic_font_size,
                anchor_x="left",
                anchor_y="top",
                bold=True,
                parent=self.view.scene,
            )
        self.text_visual.set_gl_state("translucent", depth_test=False)
        self.text_visual.order = 3
        self.items.append(self.text_visual)

        # Finalize view
        self.last_points = self.node_positions
        self._reset_camera_view()
        self.node_visual.interactive = True
        self.canvas.update()

    ###############################
    def _handle_json_io(self, mode: str):
        """Handle save and load of JSON ZX graphs."""

        ext = "PyZX Graph as JSON (*.json)"
        if mode == "SAVE":
            if not self.current_aug_zx:
                self.manager.status_changed.emit("Nothing to save.")
                return

            path_str, _ = QFileDialog.getSaveFileName(self, "Save Graph JSON", "", ext)

            if not path_str:
                return

            try:
                path = Path(path_str).with_suffix(".json")
                g = (
                    self.current_aug_zx.zx_graph_reduced
                    if self.is_reduced_view
                    else self.current_aug_zx.zx_graph
                )
                path.write_text(g.to_json())
                self.manager.status_changed.emit(f"Graph saved: {path.name}")
            except Exception as e:
                self.manager.status_changed.emit(f"Save Error: {e}")

        elif mode == "LOAD":
            path, _ = QFileDialog.getOpenFileName(self, "Open Graph JSON", "", ext)
            if not path:
                return

            self.manager.clear_session()
            self._update_registry_list()

            try:
                # 1. Parse raw PyZX Graph
                new_graph = zx.Graph.from_json(Path(path).read_text())

                # 2. Wrap and Name
                new_aug_zx = AugmentedZXGraph(zx_graph=new_graph)
                graph_name = Path(path).stem

                # 3. Register in the NEW zx_manager
                self.manager.zx_manager.add_graph(new_aug_zx, graph_key=graph_name)

                # 4. Sync View
                self._update_registry_list()
                self.manage_aug_zx(new_aug_zx)

                # Critical: Ensure the Manager knows this is the new 'In' graph
                self.manager._data_store["augmented_zx_graph_in"] = new_aug_zx
                self.manager.status_changed.emit(f"Graph loaded: {graph_name}")

            except Exception as e:
                self.manager.status_changed.emit(f"Load Error: {e}")

    def _trigger_surgery(self, use_reduced: bool):
        """Initiate lattice surgery via the manager."""
        self._run_async(self.manager.handle_lattice_surgery(use_reduced=use_reduced))

    def _clear_scene(self):
        """Vaporise all existing VisPy visual items to prevent layering."""
        for item in self.items:
            try:
                item.parent = None
            except Exception as e:
                self.manager.status_changed.emit(f"Canvas Clear Warning: {e}")
                self.items.clear()

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

        if not hasattr(self, "top_right_hud"):
            return

        super().resizeEvent(event)
        fw, fh = self.canvas_frame.width(), self.canvas_frame.height()
        m = 15

        # Top Corners
        self.graph_control_hud.move(m, m)
        self.top_right_hud.move(fw - self.top_right_hud.width() - m, m)

        # Bottom Left
        tb_h = self.toggle_buttons.height()
        self.toggle_buttons.move(m, fh - tb_h - m)

        # Reset Cam stays relative to toggles
        self.btn_reset_cam.move(m + self.toggle_buttons.width() + 10, fh - 30 - m + 3)

    ######################

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

    def _setup_action_bar(self):
        """Top-Right HUD: Consolidated Categorized Menus with forced styling."""
        self.top_right_hud = QFrame(self.canvas_frame)

        # 1. Define the shared Style Variable
        menu_style = """
            QMenu { background-color: #2a2a2a; border: 1px solid #555; border-radius: 3px; padding: 2px; }
            QMenu::item { background-color: transparent; color: #eee; font-size: 10px; padding: 4px 20px 4px 10px; border-radius: 2px; }
            QMenu::item:selected { background-color: #3a3a3a; color: #fff; }
            QMenu::separator { height: 1px; background: #444; margin: 4px 8px; }
        """
        self.top_right_hud.setStyleSheet("""
            QFrame { background: rgba(35, 35, 35, 220); border-radius: 4px; border: 1px solid #444; }
            QPushButton {
                background: #2a2a2a; color: #eee; font-size: 10px; font-weight: bold;
                padding: 4px 12px; border: 1px solid #555; border-radius: 3px;
                text-align: left; min-width: 80px;
            }
            QPushButton::menu-indicator { image: none; }
            QPushButton:hover { background: #3a3a3a; border-color: #777; }
        """)

        layout = QHBoxLayout(self.top_right_hud)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # --- 1. PROJECT MENU ---
        self.btn_project = QPushButton("▾ File actions")
        project_menu = QMenu(self)
        project_menu.setStyleSheet(menu_style)  # <--- FORCING STYLE HERE
        project_menu.addAction("Load QASM", lambda: self._handle_load_request())
        project_menu.addAction("Load JSON", lambda: self._handle_json_io("LOAD"))
        project_menu.addSeparator()
        project_menu.addAction("Save JSON", lambda: self._handle_json_io("SAVE"))
        self.btn_project.setMenu(project_menu)

        # --- 2. OPTIMIZE MENU ---
        self.btn_optimize = QPushButton("▾ Optimise")
        opt_menu = QMenu(self)
        opt_menu.setStyleSheet(menu_style)  # <--- FORCING STYLE HERE
        opt_menu.addAction(
            "Spider Fusion", lambda: self._handle_optimization_request("Spider Fusion")
        )
        opt_menu.addAction("To RG Form", lambda: self._handle_optimization_request("To RG"))
        self.btn_optimize.setMenu(opt_menu)

        # --- 3. REGISTRY MENU ---
        self.btn_registry_actions = QPushButton("▾ Graph registry")
        reg_menu = QMenu(self)
        reg_menu.setStyleSheet(menu_style)  # <--- FORCING STYLE HERE
        reg_menu.addAction("Commit to registry", self._commit_to_registry)
        self.btn_registry_actions.setMenu(reg_menu)

        for btn in [self.btn_project, self.btn_optimize, self.btn_registry_actions]:
            layout.addWidget(btn)

    def _handle_optimization_request(self, opt_name: str):
        """Apply a PyZX reduction to a COPY of the current graph."""

        if not self.current_aug_zx:
            return

        temp_zx = self.current_aug_zx.zx_graph.copy()
        self.manager.status_changed.emit(f"Applying {opt_name}...")

        if opt_name == "Spider Fusion":
            zx.spider_simp(temp_zx)
        elif opt_name == "To RG":
            zx.to_rg(temp_zx)

        # Update Sandbox view
        self.current_aug_zx = AugmentedZXGraph(temp_zx)
        self._refresh_render()
        self.manager.status_changed.emit(f"Sandbox: Applied {opt_name}")

    def _commit_to_registry(self):
        """Take a snapshot of the current canvas state and sync the UI selector."""
        if not self.current_aug_zx:
            self.manager.status_changed.emit("Nothing to commit.")
            return

        # 1. Generate a unique name (e.g., "Snapshot 1", "Snapshot 2")
        # You could also use a QInputDialog here to ask the user for a name
        count = len(self.manager.zx_manager._collection)
        new_name = f"Snapshot {count}"

        try:
            # 2. Save to the manager
            self.manager.zx_manager.add_graph(self.current_aug_zx, graph_key=new_name)

            # 3. Refresh the list
            self._update_registry_list()

            # 4. FORCED RELABELLING: Set the dropdown to the new name
            index = self.combo_registry.findText(new_name)
            if index >= 0:
                self.combo_registry.blockSignals(True)  # Prevent triggering a re-render loop
                self.combo_registry.setCurrentIndex(index)
                self.combo_registry.blockSignals(False)

            self.manager.status_changed.emit(f"Committed to registry as: {new_name}")

        except Exception as e:
            self.manager.status_changed.emit(f"Commit Error: {e}")

    def _setup_navigation_controls(self):
        """Bottom-Left HUD: Layout toggles and camera resets."""

        # 1. Triplet Buttons Cluster via Utility
        self.toggle_buttons = create_split_controls(
            self.canvas_frame, ["CLOSE CANVAS", "40/60"], self.toggle_requested.emit
        )

        # 2. Camera Reset (The Round Button)
        self.btn_reset_cam = QPushButton("⟲", self.canvas_frame)
        self.btn_reset_cam.setFixedSize(30, 30)
        self.btn_reset_cam.setStyleSheet(
            styles.STYLE_CLOSE_BTN
            + "QPushButton { background: #333; border: 1px solid #000; border-radius: 15px; font-size: 16px; }"
            "QPushButton:hover { background: #555; }"
        )
        self.btn_reset_cam.clicked.connect(self._reset_camera_view)

    def on_mouse_press(self, event):  # noqa: D102
        # GUARD: If no nodes are rendered yet, ignore the click
        if not hasattr(self, "node_visual") or self.node_visual is None:
            return

        tr = self.view.scene.node_transform(self.node_visual)
        pos = tr.map(event.pos)[:2]

        button_map = {1: "Left", 2: "Right", 3: "Middle"}
        btn = button_map.get(event.button, "Unknown")

        self.manager.status_changed.emit(f"Mouse {btn} Click at: x={pos[0]:.2f}, y={pos[1]:.2f}")

    def on_mouse_move(self, event):  # noqa: D102
        if not hasattr(self, "node_positions") or not hasattr(self, "node_visual"):
            return

        # Map pixel (event.pos) -> World Coord (pos)
        # imap = Inverse Map (Screen to Scene)
        pos = self.view.camera.transform.imap(event.pos)[:2]

        # Vectorized Distance
        dist = np.linalg.norm(self.node_positions[:, :2] - pos, axis=1)
        closest_idx = np.argmin(dist)

        # 0.8 is a safe radius for your 2.0-unit spacing
        if dist[closest_idx] < 0.8:
            if self.hovered_node_idx != closest_idx:
                self._apply_hover_effect(closest_idx)
        else:
            if self.hovered_node_idx is not None:
                self._clear_hover_effect()

    def _apply_hover_effect(self, idx):
        """Highlight a node without wiping its style properties."""
        self.hovered_node_idx = idx
        v_id = self.idx_to_node.get(idx, "???")

        edge_colors = np.full(
            (len(self.node_positions), 4), [0.95, 0.95, 0.98, 1.0], dtype=np.float32
        )

        # 2. Apply Yellow Highlight [R, G, B, A]
        edge_colors[idx] = self.node_colors[idx]

        # 3. Update the Visual with explicit property enforcement
        try:
            self.node_visual.set_data(
                pos=self.node_positions.astype(np.float32),
                face_color=self.node_colors,
                edge_color=edge_colors,
                symbol="square",
                size=self.current_node_size,
            )
            self.node_visual.update()
            self.canvas.update()
            self.manager.status_changed.emit(f"Hovering Node ID: {v_id}")
        except Exception as e:
            self.manager.status_changed.emit(f"Hover Error: {e}")

    def _clear_hover_effect(self):
        """Restore original colors and styles."""
        if self.hovered_node_idx is None:
            return

        self.hovered_node_idx = None

        try:
            self.node_visual.set_data(
                pos=self.node_positions.astype(np.float32),
                face_color=self.base_node_colors.astype(np.float32),
                edge_color="#f2f3fb",
                symbol="square",
                size=self.current_node_size,
            )
            self.node_visual.update()
            self.canvas.update()
        except Exception as e:
            self.manager.status_changed.emit(f"Hover Clear Error: {e}")
