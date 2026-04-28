"""DESIGN pane.

Manages circuit consumption from traditional circuit design frameworks (e.g., Qiskit,
pytket, and Qrisp, and their conversion into ZX graphs. Enables direct consumption
of ZX graphs (bypass circuit->ZX transpilation if ZX graph is given as Python code, or
upload ZX from QASM or JSON if ZX exists in a file).

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QSplitter

from topologiq.ux.base_pane import BasePane
from topologiq.ux.utils import styles
from topologiq.ux.utils.aux import handle_splitter_toggle
from topologiq.ux.widgets.ide_canvas import CircuitIDE
from topologiq.ux.widgets.zx_canvas import ZXCanvas


class DesignPane(BasePane):
    """Circuit design and --> ZX transpilation."""

    def __init__(self, manager, parent=None):
        """Initialise DESIGN pane."""
        super().__init__(manager, "DESIGN", parent)
        self._tasks = set()
        self.current_file_path = None
        self._initial_layout_done = False

    def setup_ui(self):
        """Define the layout for DESIGN pane."""

        # Margins
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Horizontal splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setObjectName("DesignMainSplitter")
        self.main_splitter.setHandleWidth(4)
        self.main_splitter.setStyleSheet(styles.MAIN_SPLITTER_STYLE)

        # Main components
        self.ide = CircuitIDE(self.manager)
        self.zx_canvas = ZXCanvas(self.manager)

        # Assembly
        self.main_splitter.addWidget(self.ide)
        self.main_splitter.addWidget(self.zx_canvas)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 3)

        self.layout.addWidget(self.main_splitter)

        # Internal toggle signals (emitted by child widgets)
        self.ide.toggle_requested.connect(lambda mode: self._trigger_layout_change("LEFT", mode))
        self.zx_canvas.toggle_requested.connect(
            lambda mode: self._trigger_layout_change("RIGHT", mode)
        )

    def showEvent(self, event):  # noqa: N802 (native method)
        """Handle show events."""
        super().showEvent(event)
        QTimer.singleShot(50, self._apply_initial_layout)

    def _apply_initial_layout(self):
        """Apply 40/60 split on startup."""
        self._trigger_layout_change("LEFT", "40/60")

    def _trigger_layout_change(self, side, mode):
        """Bridge toggle event with external handler."""
        handle_splitter_toggle(
            splitter=self.main_splitter, total_width=self.width(), side=side, mode=mode
        )

    def handle_zx_input(self, aug_zx):
        """Pass data from Manager to Canvas."""
        self.zx_canvas.manage_aug_zx(aug_zx)
