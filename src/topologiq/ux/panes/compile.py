"""Layout for the TRANSFORM section.

Manages a 3-way split view using VTK/Vedo to visualize ZX-graphs,
lattice surgery blocks, and 3D network layouts of completed lattice surgeries.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

    Visual environment for loading, inspecting, and compiling ZX graphs into blockgraphs.

"""

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import QFrame, QLabel, QSplitter, QStackedWidget, QVBoxLayout, QWidget

from topologiq.ux.utils.aux import handle_splitter_toggle
from topologiq.ux.widgets.bgraph_canvas import BGraphCanvas
from topologiq.ux.widgets.verify_canvas import VerifyCanvas
from topologiq.ux.widgets.zx_visual_select import ZXVisualSelect


class CompilePane(QWidget):  # Changed from BasePane for layout flexibility  # noqa: D101
    def __init__(self, manager, parent=None):  # noqa: D107
        super().__init__(parent)
        self.manager = manager
        self.active_key = None  # What is currently rendered
        self.pending_key = None  # What was requested by the user/system

        self.setup_ui()

    def setup_ui(self):  # noqa: D102
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 1. Main Horizontal Splitter (Styled as DesignMainSplitter)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setObjectName("DesignMainSplitter")
        self.main_splitter.setHandleWidth(4)
        # Note: The actual stylesheet logic for this ID is in DESIGN/Global styles

        # 2. LEFT: Registry Sidebar
        self.left_pane = ZXVisualSelect(self.manager)

        # 3. RIGHT: 3D Workspace Frame
        self.right_workspace = QFrame()
        rw_layout = QVBoxLayout(self.right_workspace)
        rw_layout.setContentsMargins(0, 0, 0, 0)
        rw_layout.setSpacing(0)

        self.right_container = QStackedWidget()

        # Overlay 1: The "Select a Graph" prompt
        self.empty_overlay = QLabel("Select a graph from the registry to view 3D Lattice Surgery.")
        self.empty_overlay.setAlignment(Qt.AlignCenter)
        self.empty_overlay.setStyleSheet(
            "background: #121212; color: #444; font-size: 13px; font-weight: bold;"
        )

        # Overlay 2: The actual 3D Visualizer
        self.block_canvas = BGraphCanvas()
        self.output_canvas = VerifyCanvas(parent=self.block_canvas)

        self.right_container.addWidget(self.empty_overlay)
        self.right_container.addWidget(self.block_canvas)

        rw_layout.addWidget(self.right_container)

        # 4. Assembly
        self.main_splitter.addWidget(self.left_pane)
        self.main_splitter.addWidget(self.right_workspace)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 3)

        self.main_layout.addWidget(self.main_splitter)

        # 5. Connections for Layout Orchestration
        self.left_pane.toggle_requested.connect(
            lambda mode: self._trigger_layout_change("LEFT", mode)
        )
        self.block_canvas.toggle_requested.connect(
            lambda mode: self._trigger_layout_change("RIGHT", mode)
        )

        # Force initial 40/60 split on next event loop cycle
        QTimer.singleShot(0, lambda: self._trigger_layout_change("LEFT", "40/60"))

    def showEvent(self, event):  # noqa: N802
        """Refresh registry and check for pending data on tab switch."""
        super().showEvent(event)
        self.left_pane.sync_registry()

        if self.pending_key:
            self.retrieve_and_render(self.pending_key)
        elif not self.active_key:
            self.right_container.setCurrentIndex(0)  # Show "Select a Graph"

    def set_focus_key(self, key: str):
        """External and Internal trigger to set the 'Source of Truth' key."""
        if not key or key == "No Graphs Available":
            return

        self.pending_key = key
        if self.isVisible():
            self.retrieve_and_render(key)

    def retrieve_and_render(self, key: str):
        """Retrieve: One Input Key -> Multiple Results."""
        if not key or key == "No Graphs Available":
            return

        self.active_key = key

        # 1. Fetch Physical Lattice (The primary goal of COMPILE)
        surgery_data = self.manager.get_data("lattice_surgery").get(key)

        if surgery_data:
            cubes, pipes = surgery_data
            self.block_canvas.render_blockgraph(cubes, pipes)
            self.right_container.setCurrentIndex(1)  # Show 3D Canvas
            self.manager.status_changed.emit(f"Rendering: {key}")
        else:
            # If the input exists but surgery isn't in the store yet
            self.right_container.setCurrentIndex(0)
            self.empty_overlay.setText(f"Compiling '{key}'...\n(Lattice Surgery in progress)")

        # 2. Fetch Verification Output (The result of the transpile)
        aug_zx_in = self.manager.get_data("augmented_zx_graph_in").get(key)
        aug_zx_out = self.manager.get_data("augmented_zx_graph_out").get(key)

        # 3. Fetch Match Status
        match_result = self.manager.get_data("graphs_match").get(key)
        print(f"DEBUG: match_result is: {match_result}")

        # Update the canvas with the full triplet of information
        if aug_zx_in and aug_zx_out:
            # If you rename the method in VerifyCanvas to set_verification_state:
            self.output_canvas.set_verification_state(aug_zx_in, aug_zx_out, match_result)

    def _trigger_layout_change(self, side, mode):
        """Bridge to the existing aux utility."""
        handle_splitter_toggle(
            splitter=self.main_splitter, total_width=self.width(), side=side, mode=mode
        )

    # --- Signal Handlers from Manager ---
    @Slot(str)
    def update_blockgraph(self, key: str):
        """Notify that surgery is done for a specific key."""
        if key == self.active_key or (not self.active_key and key == self.pending_key):
            self.retrieve_and_render(key)

    @Slot(str)
    def update_output(self, key: str):  # noqa: D102
        if key == self.active_key:
            self.retrieve_and_render(key)

    @Slot(str, bool)
    def show_verification_result(self, key: str, success: bool):  # noqa: D102
        if key == self.active_key:
            # Fetch graphs again to ensure the canvas has the objects to render
            aug_zx_in = self.manager.get_data("augmented_zx_graph_in").get(key)
            aug_zx_out = self.manager.get_data("augmented_zx_graph_out").get(key)

            if aug_zx_in and aug_zx_out:
                self.output_canvas.set_verification_state(aug_zx_in, aug_zx_out, success)
            else:
                # Fallback if graphs aren't ready but boolean is
                self.output_canvas.update_verification_badge(success)
