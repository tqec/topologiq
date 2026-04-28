"""UX blockgraph canvas.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from vispy import app, scene

from topologiq.ux.utils.aux import create_split_controls
from topologiq.ux.utils.vis_blocks import (
    BlockMesh,
    create_infinite_axes,
    generate_block_data,
    rotate_pipe_kind,
)
from topologiq.ux.widgets.verify_canvas import VerifyCanvas


class BGraphCanvas(QWidget):
    """Blockgraph 3D primary canvas visualiser."""

    toggle_requested = Signal(str)

    def __init__(self, parent=None):
        """Initialise BGRAPH canvas."""

        # Init
        super().__init__(parent)
        self.setMinimumWidth(0)
        self.vispy_app = app.use_app("pyside6")

        # Wireframe
        self.setup_ui()

        # Trackers
        self.block_items = []
        self.label_items = []

    def setup_ui(self):
        """Apply layout."""

        # Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Header
        self.header_bar = QFrame()
        self.header_bar.setFixedHeight(28)
        self.header_bar.setStyleSheet("background: #222; border-bottom: 1px solid #333;")
        h_layout = QHBoxLayout(self.header_bar)
        h_layout.setContentsMargins(10, 0, 0, 0)

        self.status_label = QLabel("3D LATTICE SURGERY")
        self.status_label.setStyleSheet("color: #888; font-size: 10px; font-weight: bold;")

        # Layout controls
        self.layout_controls = create_split_controls(
            self, ["◫", "□", "✕"], self.toggle_requested.emit
        )
        h_layout.addWidget(self.status_label)
        h_layout.addStretch()
        h_layout.addWidget(self.layout_controls)

        # VisPy canvas
        self.canvas = scene.SceneCanvas(
            keys="interactive", show=False, bgcolor="#121212", config={"depth_size": 24}
        )

        # Viewbox and camera setup
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"
        self.view.camera.distance = 30

        # Add to layout
        self.layout.addWidget(self.header_bar)
        self.layout.addWidget(self.canvas.native)

    def render_blockgraph(self, cubes, pipes):
        """Render the blockgraph in 3D."""

        # Clear scene on any run to start fresh
        self._clear_scene()

        # Add axes (red=X, green=Y, blue=Z)
        self.inf_axes = create_infinite_axes(self.view.scene)
        self.block_items.append(self.inf_axes)

        # Trackers
        all_v, all_f, all_c, all_e = [], [], [], []
        all_centers = []
        v_offset = 0

        # Process cubes (nodes)
        for c_id, (pos, kind) in cubes.items():
            all_centers.append(pos)
            v, f, c, e = generate_block_data(pos, [1.0, 1.0, 1.0], kind)
            all_v.append(v)
            all_f.append(f + v_offset)
            all_c.append(c)
            all_e.append(e + v_offset)
            v_offset += len(v)

            # Labels
            txt = scene.visuals.Text(
                str(c_id),
                pos=pos,
                color="white",
                bold=True,
                font_size=10,
                anchor_x="center",
                anchor_y="center",
                parent=self.view.scene,
            )
            # Lift text slightly so it isn't inside the cube
            txt.transform = scene.transforms.STTransform(translate=(0, 0, 0.7))
            self.block_items.append(txt)

        # Process Pipes (Edges)
        for (u_id, v_id), pipe_data in pipes.items():

            # Extract information
            kind = pipe_data[0]
            u_pos = np.array(cubes[u_id][0])
            v_pos = np.array(cubes[v_id][0])
            midpoint = (u_pos + v_pos) / 2.0
            dist = np.linalg.norm(v_pos - u_pos)
            adj_len = dist - 1.0

            if adj_len <= 0:
                continue

            # STEP A: Determine orientation (0=x, 1=y, 2=z)
            clean_kind = kind.replace("h", "")
            data_orient = clean_kind.index("o")

            # STEP B: Map to render orientation
            render_orient = data_orient
            if data_orient == 1:
                render_orient = 2
            elif data_orient == 2:
                render_orient = 1

            # Regular (non-Hadamard) pipes
            if "h" not in kind:
                # Sizing
                size = [1.0, 1.0, 1.0]
                size[render_orient] = float(adj_len)

                # Generation
                v, f, c, e = generate_block_data(midpoint, size, kind)
                all_v.append(v)
                all_f.append(f + v_offset)
                all_c.append(c)
                all_e.append(e + v_offset)
                v_offset += len(v)

            # Hadamard Pipe: 3-section stretch logic
            else:
                # Triple-section stretch factors
                yell_len = 0.1 * adj_len
                col_len = 0.45 * adj_len

                # Normalised direction vector
                direction = (v_pos - u_pos) / dist

                # Sizing
                offsets = [-(yell_len / 2 + col_len / 2), 0, (yell_len / 2 + col_len / 2)]
                sizes = [col_len, yell_len, col_len]

                # Rotation (if applicable)
                rotated_kind = rotate_pipe_kind(kind)

                # Apply to triple-sections
                start_kind = kind if sum(direction) > 0 else rotated_kind
                end_kind = rotated_kind if sum(direction) > 0 else kind
                kinds = [start_kind, "hadamard", end_kind]

                # Render
                for off, s, k in zip(offsets, sizes, kinds):
                    p_size = [1.0, 1.0, 1.0]
                    p_size[render_orient] = float(s)
                    p_pos = midpoint + (direction * off)

                    lookup_k = "xxz" if k == "hadamard" else k
                    v, f, c, e = generate_block_data(p_pos, p_size, lookup_k)

                    if k == "hadamard":
                        c = np.tile([1.0, 1.0, 0.0, 1.0], (24, 1))

                    all_v.append(v)
                    all_f.append(f + v_offset)
                    all_c.append(c)
                    all_e.append(e + v_offset)
                    v_offset += len(v)

        # Assembly
        if not all_v:
            return

        # Stacks
        flat_v = np.vstack(all_v).astype(np.float32)
        flat_f = np.vstack(all_f).astype(np.int32)
        flat_c = np.vstack(all_c).astype(np.float32)
        flat_e = np.vstack(all_e).astype(np.int32)

        # Mesh
        mesh = BlockMesh(parent=self.view.scene)
        mesh.set_data(vertices=flat_v, faces=flat_f, vertex_colors=flat_c)

        # Append to mesh
        self.block_items.append(mesh)

        borders = scene.visuals.Line(
            pos=flat_v, connect=flat_e, color="black", width=1, parent=self.view.scene
        )
        borders.set_gl_state(depth_test=True)
        self.block_items.append(borders)

        self._reset_camera(all_centers)
        self.canvas.update()

    def _reset_camera(self, points):
        """Reset camera."""
        pts = np.array(points)
        self.view.camera.center = pts.mean(axis=0)
        span = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        self.view.camera.distance = max(span * 1.5, 50)
        self.view.camera.interactive = True

    def _clear_scene(self):
        """Remove blocks and labels from the 3D world."""
        # Clean axes if present
        if hasattr(self, "inf_axes"):
            self.inf_axes.parent = None

        # Clean blocks
        for item in self.block_items + self.label_items:
            item.parent = None
        self.block_items = []
        self.label_items = []

    def set_opacity(self, value):
        """Update alpha channel for the entire block mesh."""
        alpha = value / 100.0
        for item in self.block_items:
            if isinstance(item, scene.visuals.Mesh):
                # Get existing colors, modify alpha column (index 3), and re-upload
                v_colors = item.mesh_data.get_vertex_colors()
                if v_colors is not None:
                    v_colors[:, 3] = alpha
                    item.set_data(vertex_colors=v_colors)

    def resizeEvent(self, event):  # noqa: N802 (native method)
        """Reposition children when main canvas resizes."""

        # Init
        super().resizeEvent(event)

        # Look for VerifyCanvas child
        verify_widget = self.findChild(VerifyCanvas)
        if verify_widget:
            # Anchor to Bottom-Right:
            margin = 15
            w_width = 320  # Fixed width for PiP window
            w_height = 240  # Fixed height for PiP window

            # Resize
            verify_widget.setFixedSize(w_width, w_height)
            verify_widget.move(self.width() - w_width - margin, self.height() - w_height - margin)
