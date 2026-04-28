"""UX ZX graphs visual registry.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection

# MATPLOTLIB IMPORTS
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from topologiq.input.utils import ZXColors, ZXEdgeTypes, ZXTypes
from topologiq.ux.utils.aux import create_split_controls


class ZXVisualSelect(QWidget):
    """ZX visual sidebar with registry sync."""

    graph_selected = Signal(str)
    toggle_requested = Signal(str)

    def __init__(self, manager, parent=None):
        """Initialise ZX sidebar."""

        # Init & manager
        super().__init__(parent)
        self.manager = manager

        # Wireframe
        self.setMinimumWidth(0)
        self.setup_ui()

        # Internal connections
        self.combo_registry.currentTextChanged.connect(self._on_combo_changed)

    def setup_ui(self):
        """Apply layout."""

        # Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(5)

        # Header
        self.header_bar = QFrame()
        self.header_bar.setFixedHeight(28)
        self.header_bar.setStyleSheet("background: #222; border-bottom: 1px solid #333;")
        h_layout = QHBoxLayout(self.header_bar)
        h_layout.setContentsMargins(8, 0, 0, 0)

        self.title_label = QLabel("REGISTRY")
        self.title_label.setStyleSheet(
            "color: #888; font-size: 10px; font-weight: bold; letter-spacing: 1px;"
        )

        # Layout controls
        self.layout_controls = create_split_controls(
            self, ["◫", "□", "✕"], self.toggle_requested.emit
        )
        h_layout.addWidget(self.title_label)
        h_layout.addStretch()
        h_layout.addWidget(self.layout_controls)

        # Registry selector
        self.combo_registry = QComboBox()
        self.combo_registry.setStyleSheet("""
            QComboBox { background: #0c0c0c; color: #ccc; border: 1px solid #444; padding: 4px; font-size: 11px; }
            QComboBox::drop-down { border: none; }
        """)

        # Visualisation of selected ZX graph
        self.preview_container = QFrame()
        self.preview_container.setStyleSheet(
            "background: #000; border: 1px solid #222; border-radius: 4px;"
        )
        p_layout = QVBoxLayout(self.preview_container)
        p_layout.setContentsMargins(2, 2, 2, 2)

        self.static_viewer = ZXStaticViewer()
        p_layout.addWidget(self.static_viewer)

        # Assembly
        self.main_layout.addWidget(self.header_bar)
        self.main_layout.addWidget(self.combo_registry)
        self.main_layout.addWidget(self.preview_container, stretch=1)

    def _on_combo_changed(self, key):
        """Update preview and notify parent."""

        # Reject invalid keys
        if not key or key == "No Graphs Available":
            return

        # Update valid key
        try:
            # Update 2D sidebar preview immediately
            aug_zx = self.manager.zx_manager_in.get_graph(graph_key=key)
            self.static_viewer.render_graph(aug_zx)

            # Emit to trigger 3D blockgraph update
            self.graph_selected.emit(key)
        except Exception as e:
            print(f"VisualSelect Switch Error: {e}")

    def sync_registry(self):
        """Sync dropdown with Manager."""

        # Clear registry
        self.combo_registry.blockSignals(True)
        current = self.combo_registry.currentText()
        self.combo_registry.clear()

        # Get all keys
        keys = list(self.manager.zx_manager_in._collection.keys())

        # Warning if no graph available
        if not keys:
            self.combo_registry.addItem("No Graphs Available")
            self.combo_registry.setEnabled(False)

        # Set appropriate graph as applicable
        else:
            # Set
            self.combo_registry.setEnabled(True)
            self.combo_registry.addItems(keys)
            idx = self.combo_registry.findText(current)
            self.combo_registry.setCurrentIndex(idx if idx >= 0 else len(keys) - 1)

            # Delay trigger to ensure widget is ready
            QTimer.singleShot(50, lambda: self._on_combo_changed(self.combo_registry.currentText()))

        # Release
        self.combo_registry.blockSignals(False)


class ZXStaticViewer(QWidget):
    """Simplified ZX graph display."""

    def __init__(self, parent=None):
        """Initialise ZX visual registry."""

        # Layout
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # Matplotlib Figure
        self.fig = Figure(figsize=(2, 2), dpi=100, facecolor="#909090")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#909090")
        self.ax.axis("off")

        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._layout.addWidget(self.canvas)

    def render_graph(self, aug_zx):
        """Manage graph rendering."""

        # Clear on every usage
        self.ax.clear()
        self.ax.axis("off")

        # Get ZX graph
        g = aug_zx.zx_graph
        if g is None or g.num_vertices() == 0:
            self.canvas.draw()
            return

        # Get vertices and position
        v_list = list(g.vertices())
        node_registry = {v: i for i, v in enumerate(v_list)}
        pos = np.array([[g.row(v) * 2.0, -g.qubit(v) * 2.0] for v in v_list], dtype=np.float32)

        # Render edges
        edges = list(g.edges())
        if edges:
            segments = []
            edge_colors = []

            for e in edges:
                u, v = e
                segments.append((pos[node_registry[u]], pos[node_registry[v]]))

                # Direct logic match from your VisPy snippet
                hex_e = (
                    ZXColors.HADAMARD if g.edge_type(e) == ZXEdgeTypes.HADAMARD else ZXColors.SIMPLE
                )
                edge_colors.append(hex_e)

            # Render as a single collection for performance
            lc = LineCollection(segments, colors=edge_colors, linewidths=2.0, zorder=1)
            self.ax.add_collection(lc)

        # Render nodes & labels
        for i, v in enumerate(v_list):
            try:
                t_name = ZXTypes(g.type(v)).name
                hex_val = ZXColors.lookup(t_name)
            except Exception:
                hex_val = "#888888"

            # Draw spiders
            self.ax.scatter(
                pos[i, 0],
                pos[i, 1],
                color=hex_val,
                s=120,
                marker="s",
                edgecolors="#333333",
                linewidths=0.5,
                zorder=2,
            )

            # Draw labels
            self.ax.text(
                pos[i, 0],
                pos[i, 1],
                str(v),
                color="black",
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=3,
            )

        # 3. Auto-fit
        if len(pos) > 0:
            x_min, y_min = pos.min(axis=0)
            x_max, y_max = pos.max(axis=0)
            pad = max(x_max - x_min, y_max - y_min, 2) * 0.15
            self.ax.set_xlim(x_min - pad, x_max + pad)
            self.ax.set_ylim(y_min - pad, y_max + pad)
        self.ax.set_aspect("equal")
        self.canvas.draw_idle()
