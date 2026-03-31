"""UX static simplified viewer for ZX graphs.

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
from PySide6.QtWidgets import QComboBox, QFrame, QLabel, QVBoxLayout, QWidget

from topologiq.input.utils import ZXColors, ZXTypes


class ZXStaticViewer(QWidget):
    """Pure Matplotlib-based widget for read-only ZX graph display."""

    def __init__(self, parent=None):  # noqa: D107
        super().__init__(parent)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # Setup Matplotlib Figure
        self.fig = Figure(figsize=(2, 2), dpi=100, facecolor="#eef2f6")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#eef2f6")
        self.ax.axis("off")

        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._layout.addWidget(self.canvas)

    def render_graph(self, aug_zx):
        """Clean, black-label rendering for the sidebar preview."""
        self.ax.clear()
        self.ax.axis("off")

        g = aug_zx.zx_graph
        if g is None or g.num_vertices() == 0:
            self.canvas.draw()
            return

        v_list = list(g.vertices())
        node_registry = {v: i for i, v in enumerate(v_list)}
        pos = np.array([[g.row(v) * 2.0, -g.qubit(v) * 2.0] for v in v_list], dtype=np.float32)

        # 1. Edges
        edges = list(g.edges())
        if edges:
            segments = [(pos[node_registry[u]], pos[node_registry[v]]) for u, v in edges]
            lc = LineCollection(segments, colors="#444444", linewidths=1.0, zorder=1)
            self.ax.add_collection(lc)

        # 2. Nodes & Labels
        for i, v in enumerate(v_list):
            try:
                t_name = ZXTypes(g.type(v)).name
                hex_val = ZXColors.lookup(t_name)
            except Exception:
                hex_val = "#888888"

            # Draw Spider
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

            # Draw Label (Always Black)
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

        # 3. Auto-Fit with DESIGN-friendly padding
        if len(pos) > 0:
            x_min, y_min = pos.min(axis=0)
            x_max, y_max = pos.max(axis=0)
            pad = max(x_max - x_min, y_max - y_min, 2) * 0.15
            self.ax.set_xlim(x_min - pad, x_max + pad)
            self.ax.set_ylim(y_min - pad, y_max + pad)

        self.ax.set_aspect("equal")
        self.canvas.draw_idle()


class ZXVisualSelect(QWidget):
    """Consolidated Sidebar with Registry Sync and Matplotlib Preview."""

    graph_selected = Signal(str)

    def __init__(self, manager, parent=None):  # noqa: D107
        super().__init__(parent)
        self.manager = manager

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(12)

        # --- Section 1: Selector Header ---
        self.header = QFrame()
        self.header.setStyleSheet("""
            QFrame { background: #1e1e1e; border-radius: 4px; border: 1px solid #333; }
            QLabel { color: #888; font-size: 9px; font-weight: bold; border: none; letter-spacing: 1px; }
        """)
        h_layout = QVBoxLayout(self.header)
        h_layout.setContentsMargins(8, 8, 8, 8)

        label = QLabel("INPUT REGISTRY")
        h_layout.addWidget(label)

        self.combo_registry = QComboBox()
        self.combo_registry.setStyleSheet("""
            QComboBox { background: #0c0c0c; color: #ccc; border: 1px solid #444; padding: 4px; font-size: 11px; }
            QComboBox::drop-down { border: none; }
        """)
        h_layout.addWidget(self.combo_registry)
        self.main_layout.addWidget(self.header)

        # --- Section 2: Visual Preview ---
        self.preview_container = QFrame()
        self.preview_container.setStyleSheet(
            "background: #000; border: 1px solid #222; border-radius: 4px;"
        )
        p_layout = QVBoxLayout(self.preview_container)
        p_layout.setContentsMargins(2, 2, 2, 2)

        self.static_viewer = ZXStaticViewer()
        p_layout.addWidget(self.static_viewer)

        self.main_layout.addWidget(self.preview_container, stretch=1)

        # Internal Connections
        self.combo_registry.currentTextChanged.connect(self._on_combo_changed)

    def _on_combo_changed(self, key):
        """Update preview and notify parent."""
        if not key or key == "No Graphs Available":
            return

        try:
            # Update the 2D sidebar preview immediately
            aug_zx = self.manager.zx_manager_in.get_graph(graph_key=key)
            self.static_viewer.render_graph(aug_zx)

            # Emit to trigger the 3D Blockgraph update in CompilePane
            self.graph_selected.emit(key)
        except Exception as e:
            print(f"VisualSelect Switch Error: {e}")

    def sync_registry(self):
        """Synchronize dropdown with the manager's state."""
        self.combo_registry.blockSignals(True)
        current = self.combo_registry.currentText()
        self.combo_registry.clear()

        keys = list(self.manager.zx_manager_in._collection.keys())

        if not keys:
            self.combo_registry.addItem("No Graphs Available")
            self.combo_registry.setEnabled(False)
        else:
            self.combo_registry.setEnabled(True)
            self.combo_registry.addItems(keys)

            idx = self.combo_registry.findText(current)
            self.combo_registry.setCurrentIndex(idx if idx >= 0 else len(keys) - 1)

            # Delay trigger to ensure widget is ready
            QTimer.singleShot(50, lambda: self._on_combo_changed(self.combo_registry.currentText()))

        self.combo_registry.blockSignals(False)
