"""UX blockgraph canvas.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget
from vispy import app, scene

from topologiq.ux.utils.vis_blocks import (
    BlockMesh,
    create_infinite_axes,
    generate_block_data,
    rotate_pipe_kind,
)
from topologiq.ux.widgets.verify_canvas import VerifyCanvas


class BGraphCanvas(QWidget):  # noqa: D101
    def __init__(self, parent=None):  # noqa: D107
        super().__init__(parent)

        # Explicitly ensure the app backend is set
        self.vispy_app = app.use_app("pyside6")

        # Introduce main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Canvas setutp
        self.canvas = scene.SceneCanvas(keys="interactive", show=False, bgcolor="#d4edda")
        self.layout.addWidget(self.canvas.native)

        # Viewbox and Camera
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"
        self.view.camera.distance = 30

        # Trackers
        self.block_items = []
        self.label_items = []

    def render_blockgraph(self, cubes, pipes):
        """Render the blockgraph."""
        self._clear_scene()

        # Add Infinite Axes (Red=X, Green=Y, Blue=Z)
        self.inf_axes = create_infinite_axes(self.view.scene)
        self.block_items.append(self.inf_axes)

        all_v, all_f, all_c, all_e = [], [], [], []
        all_centers = []
        v_offset = 0

        # --- 1. Process Cubes (Nodes) ---
        for c_id, (pos, kind) in cubes.items():
            all_centers.append(pos)
            v, f, c, e = generate_block_data(pos, [1.0, 1.0, 1.0], kind)
            all_v.append(v)
            all_f.append(f + v_offset)
            all_c.append(c)
            all_e.append(e + v_offset)
            v_offset += len(v)

            # Labels (Billboarding enabled by default in SceneCanvas)
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

        # --- 2. Process Pipes (Edges) ---
        for (u_id, v_id), pipe_data in pipes.items():
            kind = pipe_data[0]
            u_pos = np.array(cubes[u_id][0])
            v_pos = np.array(cubes[v_id][0])

            midpoint = (u_pos + v_pos) / 2.0
            dist = np.linalg.norm(v_pos - u_pos)
            adj_len = dist - 1.0

            if adj_len <= 0:
                continue

            # STEP A: Determine Data Orientation (0=x, 1=y, 2=z)
            clean_kind = kind.replace("h", "")
            data_orient = clean_kind.index("o")

            # STEP B: Map to Render Orientation (Swap 1 and 2)
            # This aligns 'z' pipes with the Blue Vertical Axis
            render_orient = data_orient
            if data_orient == 1:
                render_orient = 2  # Data Y -> VisPy Depth
            elif data_orient == 2:
                render_orient = 1  # Data Z -> VisPy Height (Up)

            if "h" not in kind:
                # Regular Pipe
                size = [1.0, 1.0, 1.0]
                size[render_orient] = float(adj_len)

                v, f, c, e = generate_block_data(midpoint, size, kind)
                all_v.append(v)
                all_f.append(f + v_offset)
                all_c.append(c)
                all_e.append(e + v_offset)
                v_offset += len(v)
            else:
                # Hadamard Pipe: 3-section stretch logic
                yell_len = 0.1 * adj_len
                col_len = 0.45 * adj_len

                offsets = [-(yell_len / 2 + col_len / 2), 0, (yell_len / 2 + col_len / 2)]
                sizes = [col_len, yell_len, col_len]
                kinds = [kind, "hadamard", rotate_pipe_kind(kind)]

                for off, s, k in zip(offsets, sizes, kinds):
                    p_size = [1.0, 1.0, 1.0]
                    p_size[render_orient] = float(s)

                    p_pos = midpoint.copy()
                    p_pos[render_orient] += off

                    lookup_k = "xxz" if k == "hadamard" else k
                    v, f, c, e = generate_block_data(p_pos, p_size, lookup_k)

                    if k == "hadamard":
                        c = np.tile([1.0, 1.0, 0.0, 1.0], (24, 1))

                    all_v.append(v)
                    all_f.append(f + v_offset)
                    all_c.append(c)
                    all_e.append(e + v_offset)
                    v_offset += len(v)

        # --- 3. Final Assembly ---
        if not all_v:
            return

        flat_v = np.vstack(all_v).astype(np.float32)
        flat_f = np.vstack(all_f).astype(np.int32)
        flat_c = np.vstack(all_c).astype(np.float32)
        flat_e = np.vstack(all_e).astype(np.int32)

        mesh = BlockMesh(parent=self.view.scene)
        mesh.set_data(vertices=flat_v, faces=flat_f, vertex_colors=flat_c)
        self.block_items.append(mesh)

        borders = scene.visuals.Line(
            pos=flat_v, connect=flat_e, color="black", width=1, parent=self.view.scene
        )
        borders.set_gl_state(depth_test=True)
        self.block_items.append(borders)

        self._reset_camera(all_centers)
        self.canvas.update()

    def _reset_camera(self, points):
        pts = np.array(points)
        self.view.camera.center = pts.mean(axis=0)
        span = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        self.view.camera.distance = max(span * 1.5, 50)
        self.view.camera.interactive = True

    def _clear_scene(self):
        """Remove blocks and labels from the 3D world."""

        # Clean up the infinite axes if they exist
        if hasattr(self, "inf_axes"):
            self.inf_axes.parent = None

        for item in self.block_items + self.label_items:
            item.parent = None
        self.block_items = []
        self.label_items = []

    def set_opacity(self, value):
        """Update the alpha channel for the entire block mesh."""
        alpha = value / 100.0
        for item in self.block_items:
            if isinstance(item, scene.visuals.Mesh):
                # Get existing colors, modify alpha column (index 3), and re-upload
                v_colors = item.mesh_data.get_vertex_colors()
                if v_colors is not None:
                    v_colors[:, 3] = alpha
                    item.set_data(vertex_colors=v_colors)

    def resizeEvent(self, event):  # noqa: N802
        """Position floating children whenever the main canvas resizes."""
        super().resizeEvent(event)

        # Look for the VerifyCanvas child
        verify_widget = self.findChild(VerifyCanvas)
        if verify_widget:
            # Anchor to Bottom-Right:
            # width - child_width - margin, height - child_height - margin
            margin = 15
            w_width = 320  # Fixed width for the PiP window
            w_height = 240  # Fixed height for the PiP window

            verify_widget.setFixedSize(w_width, w_height)
            verify_widget.move(self.width() - w_width - margin, self.height() - w_height - margin)
