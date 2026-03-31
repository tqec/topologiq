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

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QLabel, QSplitter, QStackedWidget, QVBoxLayout, QWidget

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

        # 1. The Splitter (Multi-Canvas | Blockgraph)
        self.splitter = QSplitter(Qt.Horizontal)

        # LEFT: The Selector/Multi-Canvas
        self.left_pane = ZXVisualSelect(self.manager)
        self.left_pane.graph_selected.connect(self.set_focus_key)

        # RIGHT: The 3D Workspace with Stacked Overlay
        self.right_container = QStackedWidget()

        # Overlay 1: The "Select a Graph" prompt
        self.empty_overlay = QLabel("Select graph from the registry.")
        self.empty_overlay.setAlignment(Qt.AlignCenter)
        self.empty_overlay.setStyleSheet("background: #1a1a1a; color: #666; font-size: 14px;")

        # Overlay 2: The actual 3D Visualizer
        self.block_canvas = BGraphCanvas()
        self.output_canvas = VerifyCanvas(parent=self.block_canvas)

        self.right_container.addWidget(self.empty_overlay)
        self.right_container.addWidget(self.block_canvas)

        self.splitter.addWidget(self.left_pane)
        self.splitter.addWidget(self.right_container)
        self.splitter.setStretchFactor(0, 1)  # ZX Multi
        self.splitter.setStretchFactor(1, 2)  # Blockgraph

        self.main_layout.addWidget(self.splitter)

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
