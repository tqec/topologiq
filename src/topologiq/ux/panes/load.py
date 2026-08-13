"""LOAD pane.

Manages circuit consumption from traditional circuit design frameworks like Qiskit,
pytket, and Qrisp, as well as direct consumptino of PyZX circuits with a two-section vertical
split: IDE and ZX canvas. The IDE section is used for consumption of programmatic circuits:
a circuit given in Python that is not a PyZX circuit will be converted to a ZX graph using qBraid.
The ZX canvas is used to visualise ZX graphs being uploaded. Both IDE and ZX canvas have buttons to
upload files. The IDE supports QASM uploads, which are treated as plain circuits and converted
into ZX graphs using qBraid. The ZX canvas supports QASM and JSON uploads, but assumes any file
uploaded directly via its load button is already PyZX compatible (i.e. was produced with PyZX) and
therefore skips any conversion and interprets is directly as a PyZX graph.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter

from topologiq.ux.base_pane import BasePane
from topologiq.ux.utils import styles
from topologiq.ux.utils.aux import handle_splitter_toggle
from topologiq.ux.widgets.ide_canvas import CircuitIDE
from topologiq.ux.widgets.zx_canvas import ZXCanvas


class LoadPane(BasePane):
    """Circuit load and ZX transpilation panel workspace view."""

    def __init__(self, manager, parent=None):
        """Initialise LOAD pane."""
        super().__init__(manager, "LOAD", parent)

    def setup_ui(self):
        """Define the layout for the LOAD pane."""
        # Enforce zero-edge boundaries for full bleed layout integration
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Horizontal splitter to manage flexible workspace distributions
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setObjectName("LoadMainSplitter")
        self.main_splitter.setHandleWidth(4)
        self.main_splitter.setStyleSheet(styles.MAIN_SPLITTER_STYLE)

        # Primary interactive visualiser sub-components
        self.ide = CircuitIDE(self.manager)
        self.zx_canvas = ZXCanvas(self.manager)

        # Structural distribution: Assign 40% width to IDE layout and 60% to canvas layout frame
        self.main_splitter.addWidget(self.ide)
        self.main_splitter.addWidget(self.zx_canvas)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 3)

        self.layout.addWidget(self.main_splitter)

        # Link interactive panel toggle signals up to layout handlers
        self.ide.toggle_requested.connect(lambda mode: self._trigger_layout_change("LEFT", mode))
        self.zx_canvas.toggle_requested.connect(
            lambda mode: self._trigger_layout_change("RIGHT", mode)
        )

    def _trigger_layout_change(self, side, mode):
        """Bridge sub-component toggle events with external layout utilities."""
        handle_splitter_toggle(
            splitter=self.main_splitter, total_width=self.width(), side=side, mode=mode
        )

    def handle_zx_input(self, aug_zx):
        """Pass the augmented ZX graph from the data manager to the active canvas."""
        self.zx_canvas.manage_aug_zx(aug_zx)

        # Minimize the IDE view layout step automatically to clear room for graph visualisations
        self._trigger_layout_change("LEFT", "MINIMIZE")

    def update_visuals(self, qasm_text: str, ascii_text: str):
        """Pass raw source text and ASCII diagrams into IDE viewer tabs."""
        if hasattr(self, "ide") and self.ide:
            self.ide.ascii_viewer.setPlainText(ascii_text)

            # Push telemetry status trace straight to internal panel logger console
            self.ide.terminal_output.appendPlainText(
                "[Ingestion Update] Received text compilation layouts."
            )
