"""Layout for the STATS section.

TO BE IMPLEMENTED. Displays summary statistics for the different circuits, graphs,
and blockgraphs across panes.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from fractions import Fraction

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyzx.graph.base import EdgeType, VertexType

from topologiq.ux.base_pane import BasePane


class StatsPane(BasePane):
    """Stats gathering and reporting."""

    def __init__(self, manager, parent=None):
        """Initialise STATS pane."""
        self.active_key = None
        super().__init__(manager, "STATS", parent)
        self.manager.blockgraph_ready.connect(self._on_data_updated)

    def setup_ui(self):
        """Build a dual-panel dashboard inside the inherited self.layout."""
        # Action Control HUD
        self.hud_frame = QFrame()
        self.hud_frame.setFixedHeight(45)
        self.hud_frame.setStyleSheet(
            "background: #1a1a1a; border: 1px solid #333; border-radius: 4px;"
        )
        hud_layout = QHBoxLayout(self.hud_frame)
        hud_layout.setContentsMargins(10, 0, 10, 0)
        hud_layout.setSpacing(15)

        self.combo_label = QLabel("TARGET:")
        self.combo_label.setStyleSheet(
            "background: #333; color: #fff; font-size: 10px; font-weight: bold; margin: 5px 0; padding: 0px 7px; border: 1px solid #fff"
        )

        self.combo_registry = QComboBox()
        self.combo_registry.setFixedWidth(160)
        self.combo_registry.setStyleSheet("""
            QComboBox { background: #0c0c0c; color: #fff; border: 1px solid #fff; padding: 4px; font-size: 11px; }
            QComboBox::drop-down { border: none; }
        """)
        self.combo_registry.currentTextChanged.connect(self._on_target_changed)

        hud_layout.addWidget(self.combo_label)
        hud_layout.addWidget(self.combo_registry)
        hud_layout.addStretch()
        self.layout.addWidget(self.hud_frame)

        # Main dashboard matrix container with split view columns
        self.dashboard_container = QWidget()
        dashboard_layout = QHBoxLayout(self.dashboard_container)
        dashboard_layout.setContentsMargins(0, 5, 0, 0)
        dashboard_layout.setSpacing(15)

        # Left column with metrics about the input ZX graph
        self.left_frame = QFrame()
        self.left_frame.setStyleSheet(
            "background: #050505; border: 1px solid #222; border-radius: 4px;"
        )
        left_layout = QVBoxLayout(self.left_frame)

        left_title = QLabel("ZX-GRAPH")
        left_title.setStyleSheet(
            "color: #00ffcc; font-size: 11px; font-weight: bold; letter-spacing: 0.5px;"
        )
        left_layout.addWidget(left_title)

        self.table_logical = self._create_styled_table(["Property", "Value"])
        left_layout.addWidget(self.table_logical)
        dashboard_layout.addWidget(self.left_frame)

        # Right column with metrics about the ouput BlockGraph
        self.right_frame = QFrame()
        self.right_frame.setStyleSheet(
            "background: #050505; border: 1px solid #222; border-radius: 4px;"
        )
        right_layout = QVBoxLayout(self.right_frame)

        right_title = QLabel("BLOCKGRAPH")
        right_title.setStyleSheet(
            "color: #fbff00; font-size: 11px; font-weight: bold; letter-spacing: 0.5px;"
        )
        right_layout.addWidget(right_title)

        self.table_physical = self._create_styled_table(["Physical Metric", "Value"])
        right_layout.addWidget(self.table_physical)
        dashboard_layout.addWidget(self.right_frame)

        # Fallback empty overlay layout label
        self.empty_msg = QLabel(
            "No compilation metrics available. Run compilation to populate stats."
        )
        self.empty_msg.setAlignment(Qt.AlignCenter)
        self.empty_msg.setStyleSheet("color: #555; font-size: 13px; font-weight: bold;")

        self.layout.addWidget(self.empty_msg)
        self.layout.addWidget(self.dashboard_container, stretch=1)
        self.dashboard_container.hide()

    def _create_styled_table(self, headers: list) -> QTableWidget:
        """Centralised stylesheet factory helper for rendering identical clean dashboard views."""
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setStyleSheet("""
            QTableWidget {
                background-color: #0c0c0c;
                color: #e0e0e0;
                gridline-color: #222;
                border: 1px solid #232323;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #141414;
                color: #aaa;
                padding: 6px;
                font-weight: bold;
                border: 1px solid #222;
                font-size: 11px;
            }
        """)
        return table

    def showEvent(self, event):  # noqa: N802
        """Show."""
        super().showEvent(event)
        self.sync_registry()

    def sync_registry(self):
        """Align active compilation ledger structures against target listings selection."""
        # Block signals to prevent firing recursively during sync
        self.combo_registry.blockSignals(True)

        # Prioritise active key
        current = self.active_key or self.combo_registry.currentText()
        self.combo_registry.clear()

        # Extract the compiled lattice surgery structures directly from the data store cache
        collection = self.manager.get_data("lattice_surgery") or {}
        keys = list(collection.keys())

        # Handle potential lack of metrics
        if not keys:
            self.combo_registry.addItem("No Metrics Available")
            self.combo_registry.setEnabled(False)
            self.dashboard_container.hide()
            self.empty_msg.show()
            self.active_key = None

        # Re-enable dropdown interactions if compiled outputs found
        else:
            self.combo_registry.setEnabled(True)
            self.combo_registry.addItems(keys)

            # Match last tracked focus key back into the dropdown list
            idx = self.combo_registry.findText(current)
            if idx >= 0:
                self.combo_registry.setCurrentIndex(idx)
                self.active_key = current
            else:
                # Fall back to most recently compiled variant if index missing
                self.combo_registry.setCurrentIndex(len(keys) - 1)
                self.active_key = keys[-1]

            # Shift layout visibility to present data grids
            self.empty_msg.hide()
            self.dashboard_container.show()
            self._render_metrics()

        # Unblock signals to resume standard usage
        self.combo_registry.blockSignals(False)

    def _on_target_changed(self, key: str):
        # Safeguard against uninitialized dropdown text items or generic empty states
        if not key or key == "No Metrics Available":
            return
        # Update tracker to focus on newly selected graph
        self.active_key = key

        # Flush active visualisation and paint newly selected analytics
        self._render_metrics()

    @Slot(object)
    def _on_data_updated(self, ledger_dict: dict):
        """Catches global compile updates to reload structural metrics maps dynamically."""
        # Query active layout window tree states to track current active window contexts
        main_win = self.window()
        if hasattr(main_win, "panes") and "COMPILE" in main_win.panes:
            self.active_key = main_win.panes["COMPILE"].active_key

        # Enforce execution boundaries to run exclusively on the GUI window thread loop
        QTimer.singleShot(0, self.sync_registry)

    def _render_metrics(self):
        """Extract and format stats parameters from targeted backend tracking elements."""
        if not self.active_key:
            return

        # Retrieve BGRAPH manager
        bgraph_manager = self.manager.get_data("lattice_surgery").get(self.active_key)
        if not bgraph_manager:
            return

        # Populates Left Panel
        self._render_logical_metrics()

        # Populates Right Panel
        self._render_physical_metrics(bgraph_manager)

    def _render_logical_metrics(self):
        """Parse abstract graph properties natively from PyZX core engines."""
        # Init table
        self.table_logical.setRowCount(0)

        # Get input Aug ZX graph
        aug_zx_obj = self.manager.get_data("augmented_zx_graph_in").get(self.active_key)
        if not (aug_zx_obj and hasattr(aug_zx_obj, "zx_graph")):
            return
        g = aug_zx_obj.zx_graph

        # Calculate key metrics
        v_counts = {VertexType.Z: 0, VertexType.X: 0, VertexType.BOUNDARY: 0}
        e_counts = {EdgeType.SIMPLE: 0, EdgeType.HADAMARD: 0}
        deg_counts = {1: 0, 2: 0, 3: 0, 4: 0, "5+ (Incompatible)": 0}
        s_count = 0
        t_count = 0
        v_names = {
            VertexType.Z: "Z-Spiders (Green)",
            VertexType.X: "X-Spiders (Red)",
            VertexType.BOUNDARY: "Boundary Ports",
        }
        e_names = {EdgeType.SIMPLE: "Simple Links", EdgeType.HADAMARD: "Hadamard Links"}

        for v in g.vertices():
            vt = g.type(v)
            if vt in v_counts:
                v_counts[vt] += 1

            deg = len(list(g.neighbors(v)))
            if deg in (1, 2, 3, 4):
                deg_counts[deg] += 1
            elif deg >= 5:
                deg_counts["5+ (Incompatible)"] += 1

            if vt == VertexType.Z:
                phase = g.phase(v)
                if phase:
                    frac = Fraction(phase)
                    if frac % 1 == Fraction(1, 2):
                        s_count += 1
                    elif frac.denominator == 4:
                        t_count += 1

        for e in g.edges():
            et = g.edge_type(e)
            if et in e_counts:
                e_counts[et] += 1

        num_v = g.num_vertices()
        num_e = g.num_edges()
        density = round(num_e / num_v, 3) if num_v > 0 else 0.0

        # Arrange metrics in an array to facilitate writing
        zx_presentation = [
            ("Total Spiders (Vertices)", num_v, False, False),
            (v_names[VertexType.Z], v_counts[VertexType.Z], True, False),
            (v_names[VertexType.X], v_counts[VertexType.X], True, False),
            (v_names[VertexType.BOUNDARY], v_counts[VertexType.BOUNDARY], True, False),
            ("Total Wires (Edges)", num_e, False, False),
            (e_names[EdgeType.SIMPLE], e_counts[EdgeType.SIMPLE], True, False),
            (e_names[EdgeType.HADAMARD], e_counts[EdgeType.HADAMARD], True, False),
            ("Graph Density (Edges/Spiders)", density, False, False),
            ("Spiders with 1 Edge", deg_counts[1], True, False),
            ("Spiders with 2 Edges", deg_counts[2], True, False),
            ("Spiders with 3 Edges", deg_counts[3], True, False),
            ("Spiders with 4 Edges", deg_counts[4], True, False),
            (
                "Spiders with 5+ Edges",
                deg_counts["5+ (Incompatible)"],
                True,
                deg_counts["5+ (Incompatible)"] > 0,
            ),
            ("Input Ports", g.num_inputs(), False, False),
            ("Output Ports", g.num_outputs(), False, False),
            ("Clifford S-Count (Z 𝛑/2)", s_count, False, False),
            ("Non-Clifford T-Count", t_count, False, False),
        ]

        # Loop write metrics
        for row_idx, (prop_name, prop_val, is_subitem, is_critical_error) in enumerate(
            zx_presentation
        ):
            self.table_logical.insertRow(row_idx)
            display_name = f"    ↳  {prop_name}" if is_subitem else prop_name
            item_name = QTableWidgetItem(display_name)
            item_val = QTableWidgetItem(str(prop_val))
            item_val.setTextAlignment(Qt.AlignCenter)

            if is_critical_error:
                item_name.setForeground(QColor("#ff3333"))
                item_val.setForeground(QColor("#ff3333"))
                bold_font = QFont()
                bold_font.setBold(True)
                item_name.setFont(bold_font)
                item_val.setFont(bold_font)
            elif is_subitem:
                item_name.setForeground(QColor("#777777"))
                item_val.setForeground(QColor("#888888"))
            else:
                item_val.setForeground(Qt.white)

            self.table_logical.setItem(row_idx, 0, item_name)
            self.table_logical.setItem(row_idx, 1, item_val)

    def _render_physical_metrics(self, bgraph_manager):
        """Parse lattice surgery footprint and unpack dynamic distance-based surface summaries."""
        # Init table
        self.table_physical.setRowCount(0)

        # Get stats
        stats_data = getattr(bgraph_manager, "stats", {})
        if not stats_data:
            bgraph_manager.get_stats()
            stats_data = getattr(bgraph_manager, "stats", {})

        # Pick data to show and rename for clarity
        blacklist = {"in_zx_spiders", "in_zx_edges", "in_zx_density"}
        clean_names = {
            "bgraph_volume": "Volume",
            "bgraph_overhead": "Overhead",
            "bgraph_surface_footprint": "Req. Chip Surface",
        }

        # Loop write metrics
        current_row = 0
        for metric_name, metric_value in stats_data.items():
            if metric_name in blacklist:
                continue

            # Unpack the dictionary values of surface footprint into sub-rows
            if metric_name == "bgraph_surface_footprint" and isinstance(metric_value, dict):
                self.table_physical.insertRow(current_row)
                root_item = QTableWidgetItem(clean_names[metric_name])
                empty_val = QTableWidgetItem("")  # Empty top-level value cell
                self.table_physical.setItem(current_row, 0, root_item)
                self.table_physical.setItem(current_row, 1, empty_val)
                current_row += 1

                for sub_k, sub_v in metric_value.items():
                    self.table_physical.insertRow(current_row)
                    sub_item_name = QTableWidgetItem(f"    ↳  k={sub_k}")
                    sub_item_name.setForeground(QColor("#777777"))

                    sub_item_val = QTableWidgetItem(str(tuple([int(i) for i in sub_v])))
                    sub_item_val.setTextAlignment(Qt.AlignCenter)
                    sub_item_val.setForeground(QColor("#888888"))

                    self.table_physical.setItem(current_row, 0, sub_item_name)
                    self.table_physical.setItem(current_row, 1, sub_item_val)
                    current_row += 1
                continue

            # Standard metric formatting
            self.table_physical.insertRow(current_row)

            item_name = QTableWidgetItem(clean_names[metric_name])
            item_val = QTableWidgetItem(str(metric_value))
            item_val.setTextAlignment(Qt.AlignCenter)
            item_val.setForeground(Qt.white)

            self.table_physical.setItem(current_row, 0, item_name)
            self.table_physical.setItem(current_row, 1, item_val)
            current_row += 1
