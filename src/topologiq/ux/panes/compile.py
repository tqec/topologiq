"""COMPILE pane.

Manages parallel visualisation of a variety of ZX graphs and their corresponding
lattice surgeries, as well as verification of input/output equivalence via a bridge
to the corresponding ZX manager.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

import asyncio

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QRegularExpression, QSize, Qt, QTimer, Slot
from PySide6.QtGui import QIntValidator, QRegularExpressionValidator, QResizeEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from topologiq.ux.utils import styles


class MutedNavigationToolbar(NavigationToolbar2QT):
    """Matplotlib Navigation toolbar with coordinate readout text stripped out."""

    def set_message(self, s):
        """Override to completely ignore cursor coordinate string telemetry."""
        pass


class CompilePane(QWidget):
    """ZX graph -> BlockGraph compile actions and unified Matplotlib rendering workspace."""

    def __init__(self, manager, parent=None):
        """Initialise COMPILE pane."""
        # Init init, init
        super().__init__(parent)
        self.manager = manager

        # Trackers
        self.active_key = None
        self.visualiser_instance = None
        self.canvas_widget = None
        self.canvas_toolbar = None

        # Call layout
        self.setup_ui()

        # Wire the global background processing signal to toggle our absolute loading frame
        self.manager.processing_state_changed.connect(self._set_loading_state)

    def setup_ui(self):
        """Define layout for configuration adjustments and embedded visualiser panels."""
        # Create layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # Action controls
        self.hud_frame = QFrame()
        self.hud_frame.setFixedHeight(45)
        self.hud_frame.setStyleSheet(
            "background: #1a1a1a; border: 1px solid #333; border-radius: 4px;"
        )
        hud_layout = QHBoxLayout(self.hud_frame)
        hud_layout.setContentsMargins(10, 0, 10, 0)
        hud_layout.setSpacing(15)

        # Target selector
        self.combo_label = QLabel("TARGET:")
        self.combo_label.setStyleSheet(
            "background: #333; color: #fff; font-size: 10px; font-weight: bold; margin: 5px 0; padding: 0px 7px; border: 1px solid #fff"
        )

        self.combo_registry = QComboBox()
        self.combo_registry.setFixedWidth(130)
        self.combo_registry.setStyleSheet(self._get_combo_style())
        self.combo_registry.currentTextChanged.connect(self._on_target_changed)

        # Drawer toggle action button
        self.btn_toggle_drawer = QPushButton("⚙️ OPTIONS")
        self.btn_toggle_drawer.setFixedSize(100, 28)
        self.btn_toggle_drawer.setStyleSheet("""
            QPushButton { background: #fbff00; color: #000; font-weight: bold; border-radius: 2px; font-size: 11px; }
            QPushButton:hover { background: #e0e500; }
            QPushButton:checked { background: #e0e500; border: 1px solid #fbff00; }
            QPushButton:disabled { background: #333; color: #666; }
        """)
        self.btn_toggle_drawer.setCheckable(True)
        self.btn_toggle_drawer.clicked.connect(self._toggle_config_drawer)

        # Compile trigger
        self.btn_execute_surgery = QPushButton("COMPILE")
        self.btn_execute_surgery.setFixedSize(160, 28)
        self.btn_execute_surgery.setStyleSheet(styles.PRIMARY_ACTION_STYLE)
        self.btn_execute_surgery.clicked.connect(self._handle_run_surgery_clicked)

        # Top HUD
        hud_layout.addWidget(self.combo_label)
        hud_layout.addWidget(self.combo_registry)
        hud_layout.addStretch()
        hud_layout.addWidget(self.btn_toggle_drawer)
        hud_layout.addWidget(self.btn_execute_surgery)
        self.main_layout.addWidget(self.hud_frame)

        # Main visual workspace stack area
        self.visual_stack = QStackedWidget()
        self.visual_stack.setStyleSheet("background: #050505; border: 1px solid #222;")
        self.empty_overlay = QLabel("Stage a graph from the canvas view to begin configuration.")
        self.empty_overlay.setAlignment(Qt.AlignCenter)
        self.empty_overlay.setStyleSheet("color: #555; font-size: 13px; font-weight: bold;")

        # Absolute overlay container
        self.visualiser_frame = QFrame()
        self.vis_layout = QVBoxLayout(self.visualiser_frame)
        self.vis_layout.setContentsMargins(0, 0, 0, 0)
        self.vis_layout.setSpacing(0)  # Flush toolbar tight against the canvas view

        # Build the actual floating drawer panel layer
        self._setup_floating_drawer()

        # Construct the floating translucent loading overlay matrix
        self.loading_overlay = QFrame(self.visualiser_frame)
        self.loading_overlay.setStyleSheet("""
            QFrame {
                background: rgba(5, 5, 5, 0.85);
                border: none;
                border-radius: 4px;
            }
            QLabel {
                color: #fbff00;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
                background: transparent;
            }
        """)
        overlay_layout = QVBoxLayout(self.loading_overlay)
        overlay_layout.setAlignment(Qt.AlignCenter)

        self.loading_label = QLabel("⚡⚡⚡ COMPILING ⚡⚡⚡")
        overlay_layout.addWidget(self.loading_label)
        self.loading_overlay.hide()

        # Assemble global layer stack
        self.visual_stack.addWidget(self.empty_overlay)
        self.visual_stack.addWidget(self.visualiser_frame)
        self.main_layout.addWidget(self.visual_stack, stretch=1)

    def _setup_floating_drawer(self):
        """Construct the floating KWARG adjustment box overlayed on the visualisation."""
        # Initialise drawer container frame
        self.config_drawer = QFrame(self.visualiser_frame)
        self.config_drawer.setFixedWidth(240)
        self.config_drawer.setStyleSheet("""
            QFrame {
                background: rgba(20, 20, 20, 0.95);
                border: 1px solid #444;
                border-radius: 4px;
            }
            QLabel { color: #aaa; font-size: 10px; font-weight: bold; border: none; background: transparent; }
            QCheckBox { color: #ccc; font-size: 11px; background: transparent; border: none; }
            QCheckBox::indicator { width: 13px; height: 13px; }
            QLineEdit { background: #0c0c0c; color: #ccc; border: 1px solid #444; padding: 4px; font-size: 11px; }
        """)

        # Tight vertical stack layout to arrange user options cleanly
        drawer_layout = QVBoxLayout(self.config_drawer)
        drawer_layout.setContentsMargins(15, 15, 15, 15)
        drawer_layout.setSpacing(10)

        # Section header
        title = QLabel("COMPILATION SETTINGS")
        title.setStyleSheet(
            "color: #fbff00; font-size: 11px; font-weight: bold; letter-spacing: 1px;"
        )
        drawer_layout.addWidget(title)
        drawer_layout.addSpacing(5)

        # Graph traverse config
        self.bfs_label = QLabel("TRAVERSE MODE")
        self.combo_bfs = QComboBox()
        self.combo_bfs.setStyleSheet(self._get_combo_style())
        self.combo_bfs.addItems(
            [
                "bfs",
                "bfs-cross",
                "bfs-cross-boundaries-last",
                "bfs-cycles",
                "bfs-rows",
                "bfs-cnots",
                "bfs-cnot-cycles",
                "tfs-cnots",
                "tfs",
            ]
        )
        drawer_layout.addWidget(self.bfs_label)
        drawer_layout.addWidget(self.combo_bfs)

        # First ID strategy
        self.strat_label = QLabel("FIRST ID STRATEGY")
        self.combo_strat = QComboBox()
        self.combo_strat.setStyleSheet(self._get_combo_style())
        self.combo_strat.addItems(
            [
                "first-spider",
                "centrality-majority",
                "central-in-first-cycle",
                "centrality-random",
                "random",
                "central-qubit",
            ]
        )
        drawer_layout.addWidget(self.strat_label)
        drawer_layout.addWidget(self.combo_strat)

        # Z-stretch factor
        self.stretch_label = QLabel("Z-STRETCH")
        self.combo_stretch = QComboBox()
        self.combo_stretch.setStyleSheet(self._get_combo_style())
        self.combo_stretch.addItems(["0", "1", "2", "3"])
        drawer_layout.addWidget(self.stretch_label)
        drawer_layout.addWidget(self.combo_stretch)

        # Gravity factor
        self.gravity_label = QLabel("GRAVITY")
        self.combo_gravity = QComboBox()
        self.combo_gravity.setStyleSheet(self._get_combo_style())
        self.combo_gravity.addItems([str(i) for i in range(11)])
        drawer_layout.addWidget(self.gravity_label)
        drawer_layout.addWidget(self.combo_gravity)

        # Distance (k) of foundational patch
        self.k_label = QLabel("Distance (k)")
        self.edit_k = QLineEdit()
        self.edit_k.setPlaceholderText("None (Default)")
        self.edit_k.setValidator(QIntValidator())
        drawer_layout.addWidget(self.k_label)
        drawer_layout.addWidget(self.edit_k)

        # Size of target chip
        self.chip_label = QLabel("SIZE OF CHIP (X, Y)")
        self.edit_chip = QLineEdit()
        self.edit_chip.setPlaceholderText("e.g. 15, 15")
        self.edit_chip.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^\d+\s*,\s*\d+$"))
        )
        drawer_layout.addWidget(self.chip_label)
        drawer_layout.addWidget(self.edit_chip)

        # Seed
        self.seed_label = QLabel("RANDOM SEED")
        self.edit_seed = QLineEdit()
        self.edit_seed.setPlaceholderText("None (Default)")
        self.edit_seed.setValidator(QIntValidator())
        drawer_layout.addWidget(self.seed_label)
        drawer_layout.addWidget(self.edit_seed)
        drawer_layout.addSpacing(5)

        # Boolean configs triggers
        self.chk_animate = QCheckBox("Animate Execution Pass")
        self.chk_write_bgraph = QCheckBox("Write to BGRAPH")
        self.chk_write_bgraph.setChecked(False)
        self.chk_twins = QCheckBox("Enable twins")
        self.chk_twins.setChecked(True)
        drawer_layout.addWidget(self.chk_animate)
        drawer_layout.addWidget(self.chk_write_bgraph)
        drawer_layout.addWidget(self.chk_twins)
        drawer_layout.addSpacing(5)

        # Initialise hidden
        self.config_drawer.hide()

    def _get_combo_style(self) -> str:
        """Centralised stylesheet for dark UI dropdown inputs."""
        return """
            QComboBox { background: #0c0c0c; color: #fff; border: 1px solid #fff; padding: 4px; font-size: 11px; }
            QComboBox::drop-down { border: none; }
        """

    def showEvent(self, event):  # noqa: N802
        """Sync internal registry data entries whenever this pane displays."""
        super().showEvent(event)
        self.sync_registry()

    def resizeEvent(self, event):  # noqa: N802
        """Dynamically reposition the floating options panel to pin to the bottom-right corner."""
        super().resizeEvent(event)
        if hasattr(self, "config_drawer") and self.config_drawer.isVisible():
            # Standardised layout tracking margins
            margin_x = 20
            margin_y = 20

            frame_w = self.visualiser_frame.width()
            frame_h = self.visualiser_frame.height()

            drawer_w = self.config_drawer.width()
            drawer_h = self.config_drawer.height()

            # Dynamically shift relative boundaries based on whether a toolbar widget is rendered
            toolbar_height = self.canvas_toolbar.height() if self.canvas_toolbar else 0

            new_x = frame_w - drawer_w - margin_x
            new_y = frame_h - drawer_h - margin_y - toolbar_height

            self.config_drawer.move(max(0, new_x), max(0, new_y))

    def ready_graph_for_configuration(self, graph_key: str, aug_zx: object):
        """Load item context on staging from load pane."""
        self.active_key = graph_key
        self.sync_registry()
        self.visual_stack.setCurrentIndex(1)
        self._refresh_active_workspace()

    def set_focus_key(self, key: str):
        """Orientation pass-through mapping supporting fallback navigation sweeps."""
        if not key or key == "No graphs available":
            return
        self.active_key = key
        self.sync_registry()

    def sync_registry(self):
        """Align dropdown target items against tracking data dictionaries."""
        # Block signals to prevent currentTextChanged from firing recursively during sync
        self.combo_registry.blockSignals(True)

        # Prioritise target key tracked by manager over old combo box text
        current = self.active_key or self.combo_registry.currentText()
        self.combo_registry.clear()

        # Extract latest collection mapping
        collection = self.manager.get_data("augmented_zx_graph_in") or {}
        keys = list(collection.keys())

        # Disable interactive compilation elements if no data frames exist
        if not keys:
            self.combo_registry.addItem("No graphs available")
            self.combo_registry.setEnabled(False)
            self.btn_execute_surgery.setEnabled(False)
            self.visual_stack.setCurrentIndex(0)
        else:
            # Re-enable controls if active graph targets are available
            self.combo_registry.setEnabled(True)
            self.btn_execute_surgery.setEnabled(True)
            self.combo_registry.addItems(keys)

            # Verification sweep: Match last active selection key back into new drop-down index
            idx = self.combo_registry.findText(current)
            if idx >= 0:
                self.combo_registry.setCurrentIndex(idx)
                self.active_key = current
            else:
                # If prev key was deleted or is missing focus on newest entry
                self.combo_registry.setCurrentIndex(len(keys) - 1)
                self.active_key = keys[-1]

            # Shift main view stack to display the active Matplotlib visualisation panel frame
            self.visual_stack.setCurrentIndex(1)

        # Unblock signals to restore reactive drop-down selection monitoring mechanics
        self.combo_registry.blockSignals(False)

    def _on_target_changed(self, key: str):
        """Triggered when the user swaps targets manually within the dropdown layout."""
        if not key or key == "No graphs available":
            return
        self.active_key = key
        self._refresh_active_workspace()

    def _handle_run_surgery_clicked(self):
        """Assembles user-selected configuration overrides and executes backend surgery."""
        # Prevent compilation passes if no valid data keys are registered
        if not self.active_key or self.active_key == "No graphs available":
            return

        # Init option map dictionary to capture user interface configurations
        override_configs = {}

        # Configs that can be fed directly irrespective of value
        override_configs["graph_traverse_mode"] = self.combo_bfs.currentText()
        override_configs["first_id_strategy"] = self.combo_strat.currentText()

        # Configs fed depending on value
        z_val = int(self.combo_stretch.currentText())
        if z_val != 0:
            override_configs["z_stretch"] = z_val

        grav_val = int(self.combo_gravity.currentText())
        if grav_val != 0:
            override_configs["gravity"] = grav_val

        k_text = self.edit_k.text().strip()
        if k_text:
            override_configs["k"] = int(k_text)

        chip_text = self.edit_chip.text().strip()
        if chip_text:
            try:
                parts = chip_text.split(",")
                override_configs["size_of_chip"] = (int(parts[0].strip()), int(parts[1].strip()))
            except (ValueError, IndexError):
                pass

        seed_text = self.edit_seed.text().strip()
        if seed_text:
            override_configs["seed"] = int(seed_text)

        if self.chk_animate.isChecked():
            override_configs["animate"] = True

        if not self.chk_twins.isChecked():
            override_configs["twins"] = False

        # Pull value of write toggle pass-through separate from build configurations
        should_write = self.chk_write_bgraph.isChecked()

        # Asynchronously dispatch lattice surgery layout tracking tasks
        task = asyncio.create_task(
            self.manager.handle_compile(
                self.active_key, options=override_configs, write_bgraph=should_write
            )
        )

        # Core async registration tracking update to protect backend linter bounds
        if hasattr(self.manager, "_tasks"):
            self.manager._tasks.add(task)
            # Route discard operations thread-safely via singleShot loopbacks
            task.add_done_callback(
                lambda t: QTimer.singleShot(0, lambda: self.manager._tasks.discard(t))
            )

    @Slot(object)
    def update_blockgraph(self, lattice_surgery_ledger: dict):
        """Catches signal updates indicating calculation completion."""
        if self.active_key in lattice_surgery_ledger:
            self._refresh_active_workspace()

    @Slot(str)
    def update_output(self, key: str):
        """Catches updates from verification state changes."""
        if key == self.active_key:
            self._refresh_active_workspace()

    @Slot(str, bool)
    def show_verification_result(self, key: str, success: bool):
        """Catches identity matching validation confirmations."""
        if key == self.active_key:
            self._refresh_active_workspace()

    @Slot(bool)
    def _set_loading_state(self, is_processing: bool):
        """Toggle the visibility of the absolute loading frame based on engine activity."""
        try:
            if not self.visualiser_frame or not self.visual_stack:
                return
        except RuntimeError:
            return

        if is_processing:
            self.loading_overlay.show()
            self.loading_overlay.raise_()

            try:
                # Force mask geometry sync using geometry parameters from the parent container
                # Avoids unpainted frames resulting in partial layout masks.
                self.loading_overlay.setGeometry(self.visualiser_frame.rect())
            except RuntimeError:
                pass
        else:
            try:
                self.loading_overlay.hide()
            except RuntimeError:
                pass

    def _refresh_active_workspace(self):
        """Build or synchronise the internal viewport using the figure from the data store."""
        if not self.active_key:
            return

        # Clean out the old canvas and toolbar widgets instantly
        self._clear_embedded_canvas()

        # Fetch the pre-compiled Figure object directly from the data store registry
        # Fallback to fetching it from the local manager instance if not pulled yet
        figures_registry = self.manager.get_data("compiled_figures") or {}
        fig_target = figures_registry.get(self.active_key)

        if not fig_target:
            # Fallback if your manager hasn't mapped it to the new registry layout yet
            bgraph_manager = self.manager.get_data("lattice_surgery").get(self.active_key)
            if not bgraph_manager:
                return
            visualiser = bgraph_manager.draw_blockgraph(is_final_vis=True, embedded=True)
            if not visualiser:
                return
            fig_target = visualiser.view_3d.fig

            # Cache it immediately
            if "compiled_figures" in self.manager._data_store:
                self.manager._data_store["compiled_figures"][self.active_key] = fig_target

        # Bind clean Figure directly to a fresh canvas widget viewport
        self.canvas_widget = FigureCanvas(fig_target)

        # Attach the navigation bar
        self.canvas_toolbar = MutedNavigationToolbar(self.canvas_widget, self)
        self.canvas_toolbar.setStyleSheet("""
            QToolBar { background: #141414; border: none; border-bottom: 1px solid #222; padding: 2px; }
            QToolButton { background: transparent; color: #fff; padding: 4px; border-radius: 3px; }
            QToolButton:hover { background: #252525; }
            QToolButton:checked { background: #fbff00; color: #000; }
        """)

        # Inject cleanly into layout
        self.vis_layout.addWidget(self.canvas_toolbar)
        self.vis_layout.addWidget(self.canvas_widget, stretch=1)

        # Paint viewport frame
        self.canvas_widget.draw_idle()

    def _clear_embedded_canvas(self):
        """Purge active visualisation UI widgets completely and safely."""
        if self.canvas_toolbar:
            self.canvas_toolbar.hide()
            self.canvas_toolbar.deleteLater()
            self.canvas_toolbar = None

        # Clean out only layout items to protect unmapped floating widgets
        while self.vis_layout.count() > 0:
            item = self.vis_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()
                widget.deleteLater()

        self.canvas_widget = None
        self.visualiser_instance = None

    def _toggle_config_drawer(self):
        """Toggles the visibility state of the floating parameter parameters module."""
        if self.btn_toggle_drawer.isChecked():
            self.config_drawer.show()
            self.config_drawer.raise_()
            self.config_drawer.adjustSize()

            self.updateGeometry()
            self.resizeEvent(QResizeEvent(self.size(), QSize()))
        else:
            self.config_drawer.hide()
