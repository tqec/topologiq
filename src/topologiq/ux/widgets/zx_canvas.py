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
from topologiq.ux.utils import styles  # noqa: F401


class ZXCanvas(QWidget):  # noqa: D101
    def __init__(self, manager, parent=None):  # noqa: D107
        super().__init__(parent)
        self.manager = manager

        self.current_aug_zx = None
        self.items = []
        self._tasks = set()  # Added: Task tracking
        self.is_reduced_view = False  # Added: View state
        self.last_points = []

        # Main Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 1. VISPY CANVAS
        self.canvas_color = "#d9d9d9"
        self.canvas = scene.SceneCanvas(
            keys="interactive", show=True, bgcolor=self.canvas_color, config={"depth_size": 24}
        )
        self.layout.addWidget(self.canvas.native)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "panzoom"  # Changed to panzoom for better 2D ZX interaction

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
        self.info_label = QLabel("INPUT ZX", self.canvas.native)
        self.info_label.setStyleSheet(
            "color: #111; font-weight: bold; font-size: 11px; background: transparent;"
        )

        # --- TOP RIGHT: Unified File/View Bar ---
        self.top_right_hud = QFrame(self.canvas.native)
        self.top_right_hud.setStyleSheet("""
            QFrame { background: rgba(210, 210, 210, 220); border-radius: 15px; border: 1px solid #bbb; }
            QPushButton {
                background: transparent; color: #111; font-size: 9px; font-weight: bold;
                padding: 5px 12px; border: none;
            }
            QPushButton:hover { background: rgba(0,0,0,15); border-radius: 13px; }
            QPushButton:checked { background: #2d5a27; color: white; border-radius: 13px; }
        """)
        tr_layout = QHBoxLayout(self.top_right_hud)
        tr_layout.setContentsMargins(5, 2, 5, 2)
        tr_layout.setSpacing(0)

        self.btn_load = QPushButton("LOAD QASM")
        self.btn_save = QPushButton("SAVE QASM")
        self.btn_reduce = QPushButton("REDUCED")
        self.btn_reduce.setCheckable(True)

        for btn in [self.btn_load, self.btn_save, self.btn_reduce]:
            tr_layout.addWidget(btn)

        # Signal connections using the async task runner to prevent GC warnings
        self.btn_load.clicked.connect(lambda: self._handle_load_request())
        self.btn_reduce.toggled.connect(self._toggle_reduction)

        # --- BOTTOM RIGHT: Transformation Controls ---
        self.bottom_right_hud = QFrame(self.canvas.native)
        self.bottom_right_hud.setStyleSheet("background: transparent;")
        br_layout = QHBoxLayout(self.bottom_right_hud)
        br_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_full = self._make_action_btn("COMPILE FULL")
        self.btn_reduced_compile = self._make_action_btn("COMPILE REDUCED")

        br_layout.addWidget(self.btn_full)
        br_layout.addWidget(self.btn_reduced_compile)

        # Connect to the async wrapper
        self.btn_full.clicked.connect(lambda: self._trigger_surgery(False))
        self.btn_reduced_compile.clicked.connect(lambda: self._trigger_surgery(True))

        # --- BOTTOM LEFT: Navigation (Round Reset Button) ---
        self.btn_reset_cam = QPushButton("⟲", self.canvas.native)
        self.btn_reset_cam.setFixedSize(36, 36)
        self.btn_reset_cam.setStyleSheet("""
            QPushButton {
                background: #111; color: #eee; border-radius: 18px;
                font-size: 16px; font-weight: bold; border: 2px solid #444;
            }
            QPushButton:hover { background: #333; }
        """)
        self.btn_reset_cam.clicked.connect(self._reset_camera_view)

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

    def render_zx(self, nx_graph):
        if nx_graph is None or nx_graph.number_of_nodes() == 0:
            return

        self._clear_scene()

        # Define explicit Z-Order (Higher = Top)
        ORDER_EDGES = 1
        ORDER_NODES = 2
        ORDER_TEXT = 3

        sq_v = np.array(
            [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=np.float32
        )
        sq_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        node_pos_map = {}
        is_output = any(data.get("qubit") == -1 for _, data in nx_graph.nodes(data=True))

        # --- 1. EDGES (Painter's order: Bottom) ---
        for u, v, data in nx_graph.edges(data=True):

            def get_coords(n_id):
                nd = nx_graph.nodes[n_id]
                if is_output:
                    return np.array(nd.get("pos", (0, 0, 0)), dtype=float)
                return np.array([nd.get("row", 0) * 2.0, -nd.get("qubit", 0) * 2.0, 0.0])

            p1, p2 = get_coords(u), get_coords(v)
            node_pos_map[u], node_pos_map[v] = p1, p2

            raw_c = data.get("color")
            e_hex = raw_c.value if hasattr(raw_c, "value") else "#000000"

            line = scene.visuals.Line(
                pos=np.array([p1, p2], dtype=np.float32), color=e_hex, parent=self.view.scene
            )
            # Disable depth test to ensure it stays behind the nodes
            line.set_gl_state(depth_test=False)
            line.order = ORDER_EDGES
            self.items.append(line)

        # --- 2. NODES & TEXT ---
        for node_id, data in nx_graph.nodes(data=True):
            pos = node_pos_map.get(node_id)
            if pos is None:
                pos = (
                    np.array(data.get("pos", (0, 0, 0)), dtype=float)
                    if is_output
                    else np.array([data.get("row", 0) * 2.0, -data.get("qubit", 0) * 2.0, 0.0])
                )

            hex_val = data.get("color").value if hasattr(data.get("color"), "value") else "#888888"
            size = 0.5 if data.get("type") == "BOUNDARY" else 0.8

            face = scene.visuals.Mesh(
                vertices=sq_v * size, faces=sq_f, color=hex_val, parent=self.view.scene
            )
            face.transform = scene.transforms.STTransform(translate=pos)
            face.set_gl_state(depth_test=False)  # Disable to avoid Z-fighting with Text
            face.order = ORDER_NODES

            rgb = Color(hex_val).rgb
            txt_color = (
                "white" if (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) < 0.5 else "black"
            )

            # --- THE TEXT FIX ---
            txt = scene.visuals.Text(
                str(node_id), color=txt_color, bold=True, font_size=8, parent=self.view.scene
            )
            txt.transform = scene.transforms.STTransform(translate=pos)
            # 'translucent' ensures smooth anti-aliasing without needing the depth buffer
            txt.set_gl_state("translucent", depth_test=False)
            txt.order = ORDER_TEXT

            self.items.extend([face, txt])

        self.last_points = list(node_pos_map.values())
        self._reset_camera_view()
        self.canvas.update()

    def _trigger_surgery(self, use_reduced: bool):
        """Initiate lattice surgery via the manager."""
        self._run_async(self.manager.handle_lattice_surgery(use_reduced=use_reduced))

    def _clear_scene(self):
        for item in self.items:
            item.parent = None
        self.items = []

    def _toggle_reduction(self):
        self.is_reduced_view = self.btn_reduce.isChecked()
        self.btn_reduce.setText("REDUCED [ON]" if self.is_reduced_view else "REDUCED [OFF]")
        self._refresh_render()

    def _refresh_render(self):
        if not self.current_aug_zx:
            return
        nx_data = self.current_aug_zx.get_visual_data(use_reduced=self.is_reduced_view)
        self.render_zx(nx_data)

    def _run_async(self, coro):
        """Manage task references and avoid GC warnings."""
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _reset_camera_view(self):
        """PanZoom specific reset."""
        if hasattr(self, "last_points") and self.last_points:
            pts = np.array(self.last_points)
            self.view.camera.rect = (
                pts[:, 0].min() - 2,
                pts[:, 1].min() - 2,
                pts[:, 0].max() - pts[:, 0].min() + 4,
                pts[:, 1].max() - pts[:, 1].min() + 4,
            )
        else:
            self.view.camera.rect = (-10, -10, 20, 20)

    def resizeEvent(self, event):  # noqa: N802
        """Keep HUD elements anchored during window resizing."""
        super().resizeEvent(event)
        w, h = self.width(), self.height()

        # Re-anchor Top Right
        self.info_label.move(15, 12)
        self.top_right_hud.adjustSize()
        self.top_right_hud.move(w - self.top_right_hud.width() - 15, 10)

        # Re-anchor Bottom Right
        self.bottom_right_hud.adjustSize()
        self.bottom_right_hud.move(w - self.bottom_right_hud.width() - 15, h - 45)

        # Re-anchor Bottom Left
        self.btn_reset_cam.move(15, h - 51)
