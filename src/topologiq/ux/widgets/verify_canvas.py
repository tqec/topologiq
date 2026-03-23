"""UX blockgraph canvas.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from vispy import scene
from vispy.color import Color

from topologiq.input.pyzx_manager import AugmentedZXGraph


class VerifyCanvas(QFrame):
    """Widget for verification of compilation."""

    def __init__(self, parent=None):  # noqa: D107
        super().__init__(parent)
        self.setFixedSize(280, 200)
        self.setObjectName("VerifyPiP")
        self.setAttribute(Qt.WA_StyledBackground, True)

        # 1. OUTER FRAME STYLE
        self.setStyleSheet("""
            QWidget#VerifyPiP {
                background-color: #eef2f6;
                border: 2px solid #444444;
                border-top-left-radius: 21px;
                border-bottom-right-radius: 7px;
            }
        """)

        # 2. OUTER LAYOUT (The "Gasket" for rounded corners)
        self.outer_layout = QVBoxLayout(self)
        self.outer_layout.setContentsMargins(5, 5, 5, 5)

        # 3. THE 3D CONTAINER (Prevents OpenGL bleed)
        self.canvas_container = QWidget()
        self.canvas_container.setStyleSheet("border: none; background: transparent;")
        self.outer_layout.addWidget(self.canvas_container)

        self.inner_layout = QVBoxLayout(self.canvas_container)
        self.inner_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = scene.SceneCanvas(keys="interactive", show=False, bgcolor="#eef2f6")
        self.inner_layout.addWidget(self.canvas.native)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"

        # --- 4. FLOATING BUTTONS (Top-Right) ---
        pill_style = """
            QPushButton {
                background: #e4e4e4;
                color: #111;
                border: 1px solid #ccc;
                border-radius: 12px;
                font-size: 8pt;
                font-weight: bold;
            }
            QPushButton:hover { background: #e0e4f5; }
        """

        # Parented to 'self' so they aren't pushed by the 5px container margin
        self.btn_save_zx = QPushButton("💾", self)
        self.btn_save_zx.setFixedSize(32, 24)
        self.btn_save_zx.setStyleSheet(pill_style)

        self.btn_toggle_red = QPushButton("REDUCE", self)
        self.btn_toggle_red.setCheckable(True)
        self.btn_toggle_red.setChecked(True)
        self.btn_toggle_red.setFixedSize(65, 24)
        self.btn_toggle_red.setStyleSheet(pill_style)

        # --- 5. STATUS PILL (Bottom-Left) ---
        self.status_pill = QFrame(self)  # Parented to self
        self.status_pill.setObjectName("StatusPill")
        self.status_pill.setFixedSize(100, 20)

        # Ensure the label styling is still applied
        self.status_pill.setStyleSheet("""
            QFrame#StatusPill {
                background: rgba(255, 255, 255, 200);
                border-radius: 10px;
                border: 1px solid #ccc;
            }
        """)

        pill_layout = QHBoxLayout(self.status_pill)
        pill_layout.setContentsMargins(0, 0, 0, 0)

        self.verify_badge = QLabel("UNVERIFIED")
        self.verify_badge.setObjectName("StatusLabel")
        self.verify_badge.setProperty("status", "unknown")
        self.verify_badge.setAlignment(Qt.AlignCenter)
        self.verify_badge.setStyleSheet("font-size: 7pt; font-weight: bold;")
        pill_layout.addWidget(self.verify_badge)

    def manage_aug_zx(self, aug_zx_graph: AugmentedZXGraph):
        """Standardised entry point for the Global Drawer."""
        self.current_aug_zx = aug_zx_graph
        self._refresh_render()

    def _toggle_reduction(self):
        self.is_reduced_view = self.btn_toggle_red.isChecked()
        self._refresh_render()

    def _refresh_render(self):
        if not self.current_aug_zx:
            return
        # Logic to pull reduced or full data
        nx_data = self.current_aug_zx.get_visual_data(use_reduced=self.is_reduced_view)
        self.render_zx(nx_data)

    def render_zx(self, nx_graph):  # noqa: D102
        if nx_graph is None or nx_graph.number_of_nodes() == 0:
            return

        self._clear_scene()
        sq_v = np.array(
            [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=np.float32
        )
        sq_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        node_pos_map = {}
        is_output = any(data.get("qubit") == -1 for _, data in nx_graph.nodes(data=True))

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
            line.transform = scene.transforms.STTransform(translate=(0, 0, -0.05))
            self.items.append(line)

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

            rgb = Color(hex_val).rgb
            txt_color = (
                "white" if (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) < 0.5 else "black"
            )
            txt = scene.visuals.Text(
                str(node_id), color=txt_color, bold=True, font_size=8, parent=self.view.scene
            )
            txt.transform = scene.transforms.STTransform(translate=pos + [0, 0, 0.02])  # noqa: RUF005

            self.items.extend([face, txt])

        self.last_points = list(node_pos_map.values())
        self._reset_camera(self.last_points)
        self.canvas.update()

    def _reset_camera(self, points):
        if not points:
            return
        pts = np.array(points)
        self.view.camera.center = pts.mean(axis=0)
        self.view.camera.distance = max(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) * 2, 25)
        self.view.camera.elevation, self.view.camera.azimuth = 90, 0
        self.view.camera.interactive = False

    def _clear_scene(self):
        for item in self.items:
            item.parent = None
        self.items = []

    def update_verification_badge(self, success: bool):
        """Toggle between Green (Verified) and Gray (Unknown)."""
        if success:
            self.verify_badge.setText("VERIFIED")
            self.verify_badge.setProperty("status", "verified")
        else:
            # Fallback for False or None
            self.verify_badge.setText("UNVERIFIED")
            self.verify_badge.setProperty("status", "unknown")

        # Force the style engine to refresh the background color
        self.verify_badge.style().unpolish(self.verify_badge)
        self.verify_badge.style().polish(self.verify_badge)
        self.verify_badge.update()

    def resizeEvent(self, event):  # noqa: N802
        """Anchor all floating UI elements to their respective corners."""
        super().resizeEvent(event)

        # 1. Geometry Constants
        margin = 10
        spacing = 6

        # 2. Position REDUCE (Top-Right)
        # X: Width - ButtonWidth - Margin
        # Y: Margin
        self.btn_toggle_red.move(self.width() - self.btn_toggle_red.width() - margin, margin)

        # 3. Position SAVE (Top-Right, left of REDUCE)
        # X: Width - BothButtonsWidth - Margin - Spacing
        # Y: Margin
        self.btn_save_zx.move(
            self.width()
            - self.btn_toggle_red.width()
            - self.btn_save_zx.width()
            - margin
            - spacing,
            margin,
        )

        # 4. Position STATUS PILL (Bottom-Left)
        # X: Margin (Left side)
        # Y: Total Height - Pill Height - Margin (Bottom side)
        self.status_pill.move(margin, self.height() - self.status_pill.height() - margin)
