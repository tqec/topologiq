"""UX blockgraph canvas.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

import numpy as np
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QVBoxLayout, QWidget
from vispy import scene
from vispy.color import Color

from topologiq.input.pyzx_manager import AugmentedZXGraph
from topologiq.ux.utils import styles


class VerifyCanvas(QWidget):
    """Widget for verification of compilation."""

    def __init__(self, parent=None):
        """Initialise verification widget."""
        super().__init__(parent)
        self.current_aug_zx = None
        self.is_reduced_view = True  # Default to Reduced
        self.items = []

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Soft white / Light blue background
        self.canvas = scene.SceneCanvas(keys="interactive", show=False, bgcolor="#eef2f6")
        self.layout.addWidget(self.canvas.native)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"

        # Simplified HUD: Only Toggle and Save
        self.hud = QFrame(self.canvas.native)
        self.hud.setStyleSheet(
            "background: rgba(200, 210, 230, 200); border-radius: 10px; border: 1px solid #ccc;"
        )
        self.hud.setFixedHeight(32)

        hud_layout = QHBoxLayout(self.hud)
        hud_layout.setContentsMargins(5, 0, 5, 0)

        self.btn_toggle_red = QPushButton("REDUCED")
        self.btn_toggle_red.setCheckable(True)
        self.btn_toggle_red.setChecked(True)
        self.btn_toggle_red.setStyleSheet(styles.TOGGLE_BUTTON_STYLE)
        self.btn_toggle_red.clicked.connect(self._toggle_reduction)

        self.btn_save_zx = QPushButton("SAVE")
        self.btn_save_zx.setStyleSheet(styles.TOGGLE_BUTTON_STYLE)

        hud_layout.addWidget(self.btn_toggle_red)
        hud_layout.addWidget(self.btn_save_zx)

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

    def resizeEvent(self, event):  # noqa: D102, N802
        super().resizeEvent(event)
        # Keep HUD anchored to top-right
        self.hud.move(self.width() - self.hud.width() - 10, 10)
