"""UX blockgraph verification canvas.

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


class VerifyCanvas(QFrame):
    """I/O ZX equality verification visualiser."""

    def __init__(self, parent=None):
        """Initialise verification canvas."""

        # Init
        super().__init__(parent)
        self.setFixedSize(280, 200)
        self.setObjectName("VerifyPiP")
        self.setAttribute(Qt.WA_StyledBackground, True)

        # Internal state
        self.items = []
        self.graph_in = None
        self.graph_out = None
        self.is_showing_out = False  # Toggle state: False = In, True = Out

        # Styles
        self.setStyleSheet("""
            QWidget#VerifyPiP {
                background-color: #f8f9fa;
                border: 2px solid #444;
                border-top-left-radius: 21px;
                border-bottom-right-radius: 7px;
            }
        """)

        # Layout
        self.outer_layout = QVBoxLayout(self)
        self.outer_layout.setContentsMargins(2, 2, 2, 2)
        self.canvas_container = QWidget()
        self.outer_layout.addWidget(self.canvas_container)
        self.inner_layout = QVBoxLayout(self.canvas_container)
        self.inner_layout.setContentsMargins(0, 0, 0, 0)

        # VisPy Canvas
        self.canvas = scene.SceneCanvas(keys="interactive", show=False, bgcolor="#909090")
        self.inner_layout.addWidget(self.canvas.native)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"

        # Internal controls
        pill_style = """
            QPushButton {
                background: #ffffff;
                border: 1px solid #ccc;
                border-radius: 10px;
                font-size: 7pt;
                font-weight: bold;
            }
            QPushButton:checked { background: #d1d9ff; border: 1px solid #4a90e2; }
        """

        self.btn_toggle_io = QPushButton("VIEW: IN", self)
        self.btn_toggle_io.setCheckable(True)
        self.btn_toggle_io.setFixedSize(70, 22)
        self.btn_toggle_io.setStyleSheet(pill_style)
        self.btn_toggle_io.clicked.connect(self._handle_io_toggle)

        # Status
        self.status_pill = QFrame(self)
        self.status_pill.setFixedSize(110, 24)

        # Add to layout
        pill_layout = QHBoxLayout(self.status_pill)
        self.verify_badge = QLabel("WAITING...")
        self.verify_badge.setAlignment(Qt.AlignCenter)
        self.verify_badge.setStyleSheet("color: white; font-size: 8pt; font-weight: 900;")
        pill_layout.addWidget(self.verify_badge)
        self._set_pill_color("unknown")

    def set_verification_state(self, graph_in, graph_out, match):
        """Manage initial call."""
        self.graph_in = graph_in
        self.graph_out = graph_out
        self.update_verification_badge(match)
        self._refresh_render()

    def update_verification_badge(self, success: bool):
        """Update visual status indicator based on match result."""
        if success is True:
            self.verify_badge.setText("VALID MATCH")
            self._set_pill_color("verified")
        elif success is False:
            self.verify_badge.setText("LOGIC ERROR")
            self._set_pill_color("failed")
        else:
            self.verify_badge.setText("WAITING...")
            self._set_pill_color("unknown")

    def _set_pill_color(self, status):
        """Update pill color based on match result."""
        colors = {"verified": "#2ecc71", "failed": "#e74c3c", "unknown": "#95a5a6"}
        self.status_pill.setStyleSheet(f"background-color: {colors[status]}; border-radius: 12px;")

    def _handle_io_toggle(self):
        """Handle toggle between input and output visuals."""
        self.is_showing_out = self.btn_toggle_io.isChecked()
        self.btn_toggle_io.setText("VIEW: OUT" if self.is_showing_out else "VIEW: IN")
        self._refresh_render()

    def _refresh_render(self):
        """Refresh canvas."""
        target = self.graph_out if self.is_showing_out else self.graph_in
        if target:
            # Strictly use reduced view for verification clarity
            nx_data = target.get_visual_data(use_reduced=True)
            self.render_zx(nx_data)

    def render_zx(self, nx_graph):
        """Render a single NX representation of a ZX graph."""

        # Clear scene on every run to start fresh.
        self._clear_scene()

        # Return if no graph
        if not nx_graph:
            return

        # Render loops
        node_pos_map = {}

        # Edges
        for u, v, data in nx_graph.edges(data=True):
            p1, p2 = np.array(nx_graph.nodes[u].get("pos")), np.array(nx_graph.nodes[v].get("pos"))
            node_pos_map[u], node_pos_map[v] = p1, p2
            line = scene.visuals.Line(
                pos=np.array([p1, p2]), color=data.get("color", "#000000"), parent=self.view.scene
            )
            self.items.append(line)

        # Nodes
        for node_id, data in nx_graph.nodes(data=True):
            pos = node_pos_map.get(node_id, np.array(data.get("pos", (0, 0, 0))))
            node_vis = scene.visuals.Markers()
            node_vis.set_data(np.array([pos]), face_color=data.get("color", "#888888"), size=10)
            node_vis.parent = self.view.scene
            self.items.append(node_vis)

        # Reset camera
        self._reset_camera(list(node_pos_map.values()))

    def _reset_camera(self, points):
        """Reset camera to match contents."""
        if not points:
            return
        pts = np.array(points)
        self.view.camera.center = pts.mean(axis=0)
        self.view.camera.distance = max(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) * 1.5, 10)
        self.view.camera.elevation, self.view.camera.azimuth = 90, 0

    def _clear_scene(self):
        """Clear scene on request."""
        for item in self.items:
            item.parent = None
        self.items = []

    def resizeEvent(self, event):  # noqa: N802 (native method)
        """Handle resize events."""
        super().resizeEvent(event)
        self.btn_toggle_io.move(self.width() - 80, 10)
        self.status_pill.move(10, self.height() - 34)
