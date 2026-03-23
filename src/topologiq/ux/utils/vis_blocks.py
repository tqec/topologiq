"""Visual cube and pipe objects for blockgraph visualisations in UX.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

import numpy as np
from vispy import scene
from vispy.color import Color
from vispy.geometry import create_box

# Your hex map converted to VisPy-friendly colors
NODE_HEX_MAP = {
    "xxz": ["#d7a4a1", "#d7a4a1", "#b9cdff"],
    "xzz": ["#d7a4a1", "#b9cdff", "#b9cdff"],
    "xzx": ["#d7a4a1", "#b9cdff", "#d7a4a1"],
    "zzx": ["#b9cdff", "#b9cdff", "#d7a4a1"],
    "zxx": ["#b9cdff", "#d7a4a1", "#d7a4a1"],
    "zxz": ["#b9cdff", "#d7a4a1", "#b9cdff"],
    "zxo": ["#b9cdff", "#d7a4a1", "gray"],
    "xzo": ["#d7a4a1", "#b9cdff", "gray"],
    "oxz": ["gray", "#d7a4a1", "#b9cdff"],
    "ozx": ["gray", "#b9cdff", "#d7a4a1"],
    "xoz": ["#d7a4a1", "gray", "#b9cdff"],
    "zox": ["#b9cdff", "gray", "#d7a4a1"],
    # ... any others will default to gray in the getter
}


def rotate_pipe_kind(k):
    """Rotate."""
    clean_k = k.replace("h", "")
    if "o" not in clean_k:
        return k
    o_idx = clean_k.index("o")
    idxs = [0, 1, 2]
    idxs.remove(o_idx)

    # Swap the non-'o' characters
    res = list(clean_k)
    res[idxs[0]], res[idxs[1]] = clean_k[idxs[1]], clean_k[idxs[0]]
    return "".join(res) + ("h" if "h" in k else "")


class BlockMesh(scene.visuals.Mesh):
    """A single Mesh containing ALL cubes or ALL pipes for performance."""

    def __init__(self, parent=None):  # noqa: D107
        # We use 'flat' shading because blockgraphs are geometric,
        # not organic. This gives that clean 'Lego' look.
        super().__init__(shading=None, parent=parent)
        self.unfreeze()
        self.set_gl_state(
            blend=False,
            depth_test=True,
            cull_face=False,
            polygon_offset_fill=True,  # Enable offset
            polygon_offset=(1.0, 1.0),  # Push faces slightly back
        )
        self.freeze()

    def update_data(self, vertices, faces, vertex_colors):  # noqa: D102
        if len(vertices) == 0:
            self.set_data(vertices=None)
            return
        self.set_data(
            vertices=vertices.astype(np.float32),
            faces=faces.astype(np.int32),
            vertex_colors=vertex_colors,
        )

    def on_draw(self, event):
        """Update light direction to match camera view."""
        if hasattr(self, "canvas") and self.canvas:
            view = self.canvas.central_widget.children[0]
            # Set light direction to match camera look vector
            self.shading_filter.light_dir = view.camera.transform.matrix[:3, 2]


def generate_block_data(coords, size, kind):  # noqa: D103
    x, y, z = coords
    sx, sy, sz = size  # Now sx=X, sy=Y(Depth), sz=Z(Up)

    vertices_data, faces, outline_indices = create_box(width=sx, height=sy, depth=sz)
    vertices = vertices_data["position"] + np.array([x, y, z])

    clean_kind = kind.replace("h", "")
    cols = NODE_HEX_MAP.get(clean_kind, ["gray", "gray", "gray"])

    v_cols = np.zeros((24, 4))

    # Data Map: cols[0]=X, cols[1]=Y, cols[2]=Z
    c_x = list(Color(cols[0]).rgba)
    c_x[3] = 1.0
    c_y = list(Color(cols[1]).rgba)
    c_y[3] = 1.0
    c_z = list(Color(cols[2]).rgba)
    c_z[3] = 1.0

    # VisPy Vertex Groups:
    # 0:8   = Depth/Vertical (SZ) -> Map to your Data's Z (c_z)
    # 8:15  = Height/Front-Back (SY) -> Map to your Data's Y (c_y)
    # 16:23 = Width/Left-Right (SX) -> Map to your Data's X (c_x)
    v_cols[0:8] = c_z
    v_cols[8:16] = c_y
    v_cols[16:24] = c_x

    return vertices, faces, v_cols, outline_indices


def create_infinite_axes(parent, length=10):  # noqa: D103
    pts = np.array(
        [
            [-length, 0, 0],
            [length, 0, 0],  # X (Red)
            [0, -length, 0],
            [0, length, 0],  # Y (Green)
            [0, 0, -length],
            [0, 0, length],  # Z (Blue)
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [[1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
    )

    axis_lines = scene.visuals.Line(
        pos=pts,
        color=colors,
        connect="segments",
        width=2,  # Slightly thicker for visibility
        parent=parent,
    )

    # This prevents the blockgraph from 'hiding' the lines
    return axis_lines
