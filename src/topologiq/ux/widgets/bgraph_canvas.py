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

from topologiq.ux.utils.vis_blocks import BlockMesh, create_infinite_axes, generate_block_data


class BGraphCanvas(QWidget):  # noqa: D101
    def __init__(self, parent=None):  # noqa: D107
        super().__init__(parent)
        self.vispy_app = app.use_app("pyside6")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = scene.SceneCanvas(keys="interactive", show=False, bgcolor="#242424")
        self.layout.addWidget(self.canvas.native)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"
        self.items = []

    def render_blockgraph(self, cubes, pipes):  # noqa: D102
        self._clear_scene()
        self.items.append(create_infinite_axes(self.view.scene))
        all_v, all_f, all_c = [], [], []
        v_offset = 0

        for (pos, kind) in cubes.values():
            v, f, c, _ = generate_block_data(pos, [1.0, 1.0, 1.0], kind)
            all_v.append(v)
            all_f.append(f + v_offset)
            all_c.append(c)
            v_offset += len(v)

        if not all_v:
            return
        mesh = BlockMesh(parent=self.view.scene)
        mesh.set_data(
            vertices=np.vstack(all_v), faces=np.vstack(all_f), vertex_colors=np.vstack(all_c)
        )
        self.items.append(mesh)

        pts = [np.array(d[0]) for d in cubes.values()]
        self._reset_camera(pts)
        self.canvas.update()

    def _reset_camera(self, points):
        pts = np.array(points)
        self.view.camera.center = pts.mean(axis=0)
        span = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        self.view.camera.distance = max(span * 1.5, 50)
        self.view.camera.interactive = True

    def _clear_scene(self):
        for item in self.items:
            item.parent = None
        self.items = []
