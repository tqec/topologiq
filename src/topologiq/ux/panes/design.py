"""Layout for the DESIGN section.

Integrates Monaco editor for circuit coding and buttons
for importing/exporting circuit diagrams into qBraid and exporting
QASM and PyZX-compatible formats.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import QSplitter

from topologiq.ux.base_pane import BasePane
from topologiq.ux.utils import styles
from topologiq.ux.utils.aux import handle_splitter_toggle
from topologiq.ux.widgets.ide_canvas import CircuitIDE
from topologiq.ux.widgets.zx_canvas import ZXCanvas


class DesignPane(BasePane):
    """Orchestrate circuit design section."""

    def __init__(self, manager, parent=None):
        """Initialise DESIGN pane."""
        super().__init__(manager, "DESIGN", parent)
        self._tasks = set()
        self.current_file_path = None
        self._initial_layout_done = False

    def setup_ui(self):
        """Define the layout for DESIGN pane."""
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 1. Simplified Horizontal Splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setObjectName("DesignMainSplitter")
        self.main_splitter.setHandleWidth(4)
        self.main_splitter.setStyleSheet(styles.MAIN_SPLITTER_STYLE)
        # 2. Instantiate components
        self.ide = CircuitIDE(self.manager)
        self.zx_canvas = ZXCanvas(self.manager)

        # 3. Assembly (No Rail)
        self.main_splitter.addWidget(self.ide)
        self.main_splitter.addWidget(self.zx_canvas)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 3)

        self.layout.addWidget(self.main_splitter)

        # 4. Connect the new "Internal" toggle signals
        # These will be emitted by the child widgets themselves
        self.ide.toggle_requested.connect(lambda mode: self._trigger_layout_change("LEFT", mode))
        self.zx_canvas.toggle_requested.connect(
            lambda mode: self._trigger_layout_change("RIGHT", mode)
        )

    def showEvent(self, event):  # noqa: D102, N802
        super().showEvent(event)
        # Use a very short delay to ensure the OS has painted the window
        # and reported a non-zero width.
        QTimer.singleShot(50, self._apply_initial_layout)

    def _apply_initial_layout(self):
        """Force a 40/60 split on startup."""
        self._trigger_layout_change("LEFT", "40/60")

    def _trigger_layout_change(self, side, mode):
        """Bridge to the external utility."""
        handle_splitter_toggle(
            splitter=self.main_splitter, total_width=self.width(), side=side, mode=mode
        )

    @Slot(str, str)
    def update_visuals(self, qasm, ascii_art):
        """Pass data from Manager to the IDE component."""
        self.ide.ascii_viewer.setPlainText(ascii_art)
        # Tab logic...

    def handle_zx_input(self, aug_zx):
        """Pass data from Manager to the Canvas component."""
        self.zx_canvas.manage_aug_zx(aug_zx)
