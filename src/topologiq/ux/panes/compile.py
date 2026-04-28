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

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import QFrame, QLabel, QSplitter, QStackedWidget, QVBoxLayout, QWidget

from topologiq.ux.utils.aux import handle_splitter_toggle
from topologiq.ux.widgets.bgraph_canvas import BGraphCanvas
from topologiq.ux.widgets.verify_canvas import VerifyCanvas
from topologiq.ux.widgets.zx_visual_select import ZXVisualSelect


class CompilePane(QWidget):
    """ZX graph -> BlockGraph compile actions and comparatives."""

    def __init__(self, manager, parent=None):
        """Initialise COMPILE pane."""
        super().__init__(parent)
        self.manager = manager
        self.active_key = None  # Currently rendered
        self.pending_key = None  # Requested by user/system
        self.setup_ui()

    def setup_ui(self):
        """Define the layout for COMPILE pane."""

        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Horizontal splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setObjectName("DesignMainSplitter")
        self.main_splitter.setHandleWidth(4)

        # ZX registry (sidebar)
        self.left_pane = ZXVisualSelect(self.manager)

        # 3D blockgraph (main workspace)
        self.right_workspace = QFrame()
        rw_layout = QVBoxLayout(self.right_workspace)
        rw_layout.setContentsMargins(0, 0, 0, 0)
        rw_layout.setSpacing(0)

        self.right_container = QStackedWidget()

        # Graph selector
        self.empty_overlay = QLabel("Select a graph to view the corresponding blockgraph.")
        self.empty_overlay.setAlignment(Qt.AlignCenter)
        self.empty_overlay.setStyleSheet(
            "background: #121212; color: #444; font-size: 13px; font-weight: bold;"
        )

        # 3D blockgraph visualiser (w. inset verification sub-section)
        self.block_canvas = BGraphCanvas()
        self.output_canvas = VerifyCanvas(parent=self.block_canvas)

        self.right_container.addWidget(self.empty_overlay)
        self.right_container.addWidget(self.block_canvas)

        rw_layout.addWidget(self.right_container)

        # Assembly
        self.main_splitter.addWidget(self.left_pane)
        self.main_splitter.addWidget(self.right_workspace)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 3)

        self.main_layout.addWidget(self.main_splitter)

        # Connections for layout Orchestration
        self.left_pane.toggle_requested.connect(
            lambda mode: self._trigger_layout_change("LEFT", mode)
        )
        self.block_canvas.toggle_requested.connect(
            lambda mode: self._trigger_layout_change("RIGHT", mode)
        )

        # Apply initial 40/60 split on next event loop cycle
        QTimer.singleShot(0, lambda: self._trigger_layout_change("LEFT", "40/60"))

    def showEvent(self, event):  # noqa: N802 (native method)
        """Refresh registry on tab switch."""
        super().showEvent(event)
        self.left_pane.sync_registry()
        if self.pending_key:
            self.retrieve_and_render(self.pending_key)
        elif not self.active_key:
            self.right_container.setCurrentIndex(0)  # Show "Select a Graph"

    def set_focus_key(self, key: str):
        """Set the key for the active source of truth."""
        if not key or key == "No Graphs Available":
            return
        self.pending_key = key
        if self.isVisible():
            self.retrieve_and_render(key)

    def retrieve_and_render(self, key: str):
        """Retrieve and render ZX/blockgraph associated with given key."""
        if not key or key == "No Graphs Available":
            return

        # Set active key to incoming parameter
        self.active_key = key

        # Fetch blockgraph
        surgery_data = self.manager.get_data("lattice_surgery").get(key)

        # Show blockgraph if blockgraph is available
        if surgery_data:
            cubes, pipes = surgery_data
            self.block_canvas.render_blockgraph(cubes, pipes)
            self.right_container.setCurrentIndex(1)  # Show 3D Canvas
            self.manager.status_changed.emit(f"Rendering: {key}")
        # Update pending if blockgraph is not yet available in store
        else:
            self.right_container.setCurrentIndex(0)
            self.empty_overlay.setText(f"Compiling '{key}'...\n(Lattice Surgery in progress)")

        # Fetch verification results
        aug_zx_in = self.manager.get_data("augmented_zx_graph_in").get(key)
        aug_zx_out = self.manager.get_data("augmented_zx_graph_out").get(key)
        match_result = self.manager.get_data("graphs_match").get(key)

        # Update the canvas with the full triplet of information
        if aug_zx_in and aug_zx_out:
            self.output_canvas.set_verification_state(aug_zx_in, aug_zx_out, match_result)

    def _trigger_layout_change(self, side, mode):
        """Bridge toggle event with external handler."""
        handle_splitter_toggle(
            splitter=self.main_splitter, total_width=self.width(), side=side, mode=mode
        )

    @Slot(str)
    def update_blockgraph(self, key: str):
        """Notify surgery ready for a specific key."""
        if key == self.active_key or (not self.active_key and key == self.pending_key):
            self.retrieve_and_render(key)

    @Slot(str)
    def update_output(self, key: str):
        """Update outputs for a given key."""
        if key == self.active_key:
            self.retrieve_and_render(key)

    @Slot(str, bool)
    def show_verification_result(self, key: str, success: bool):
        """Update verification results for a given key."""
        if key == self.active_key:
            # Fetch graphs again to ensure the canvas has the objects to render
            aug_zx_in = self.manager.get_data("augmented_zx_graph_in").get(key)
            aug_zx_out = self.manager.get_data("augmented_zx_graph_out").get(key)

            if aug_zx_in and aug_zx_out:
                self.output_canvas.set_verification_state(aug_zx_in, aug_zx_out, success)
            else:
                # Fallback if graphs aren't ready but boolean is
                self.output_canvas.update_verification_badge(success)
