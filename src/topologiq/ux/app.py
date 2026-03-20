"""UX entry point.

Initialises the PySide6 server and handles top-level UX orchestration.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

import asyncio
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from topologiq.ux import styles
from topologiq.ux.manager import UXManager
from topologiq.ux.panes import DesignPane, RunPane, SimulatePane, TransformPane

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


class TopologiqApp(QMainWindow):
    """UX primary orchestrator."""

    def __init__(self, manager):
        """Initialise UX."""

        # Start
        super().__init__()
        self.manager = manager
        self.running = True

        # Window Setup
        self.setWindowTitle("TOPOLOGIQ | Algorithmic Lattice Surgery")
        self.resize(1280, 720)
        self._set_icon()

        # Main Container
        main_widget = QWidget()
        main_widget.setStyleSheet(styles.MAIN_WINDOW_STYLE)
        self.setCentralWidget(main_widget)

        self.root_layout = QHBoxLayout(main_widget)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

        # Content Stack
        self.content_stack = QStackedWidget()
        self.root_layout.addWidget(self.content_stack)

        # Sidebar Setup
        self._setup_sidebar()

        # Initialize Panes
        self.panes = {
            "DESIGN": DesignPane(),
            "TRANSFORM": TransformPane(),
            "SIMULATE": SimulatePane(),
            "RUN": RunPane(),
        }

        for pane in self.panes.values():
            self.content_stack.addWidget(pane)

        # 5. Status Bar
        self._setup_status_bar()

        self._switch_to_page("DESIGN")
        self._connect_signals()

    def _set_icon(self):
        icon_path = ASSETS_DIR / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _setup_sidebar(self):
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(28)
        self.sidebar.setStyleSheet(styles.SIDEBAR_STYLE)
        # Keeps buttons from hitting top/bottom edges
        self.sidebar.setContentsMargins(0, 14, 0, 14)

        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar_layout.setSpacing(0)

        self.nav_buttons = []
        sections = ["DESIGN", "TRANSFORM", "SIMULATE", "RUN"]

        for i, name in enumerate(sections):
            btn = QPushButton(name[0])  # Single letter
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn.setMinimumHeight(0)
            btn.setStyleSheet(styles.NAV_BUTTON_STYLE)

            if i == len(sections) - 1:
                btn.setProperty("isLast", True)

            btn.clicked.connect(lambda checked, n=name: self._switch_to_page(n))

            self.sidebar_layout.addWidget(btn, 1)
            self.nav_buttons.append(btn)

        self.root_layout.addWidget(self.sidebar)

    def _setup_status_bar(self):
        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.status_bar.setStyleSheet(styles.STATUS_BAR_STYLE)
        self.setStatusBar(self.status_bar)

        self.coords_label = QLabel("Num gates: 0 | Spiders: 0 | Volume: 0 | Equivalence: True")
        self.status_bar.addPermanentWidget(self.coords_label)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress)
        self.status_bar.showMessage("System Ready", 5000)

    def _switch_to_page(self, name):
        index = list(self.panes.keys()).index(name)
        self.content_stack.setCurrentIndex(index)
        for btn in self.nav_buttons:
            # Match first letter to name
            btn.setChecked(name.startswith(btn.text()))

    def _connect_signals(self):
        self.manager.status_changed.connect(self.status_bar.showMessage)
        # More connections as needed...

    def closeEvent(self, event):  # noqa: N802 (native Qt method)
        """Handle window close event.

        Args:
            event (QCloseEvent): Event object with details of closure request.

        """
        self.running = False
        event.accept()

    def _create_pane(self, name):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel(f"{name} View")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        widget.info_label = label
        return widget

    def _connect_signals(self):
        """Standardized Signal Routing."""
        # UI Orchestration
        self.manager.status_changed.connect(self.status_bar.showMessage)
        self.manager.section_changed.connect(self._switch_tab)

        # Data Flow
        # Direct connection: Manager -> Specific Pane Slot
        self.manager.qb_circuit_ready.connect(self.panes["DESIGN"].update_visuals)

        # Disable sidebar during processing
        self.manager.processing_state_changed.connect(self.sidebar.setDisabled)

    @Slot(str)
    def _switch_tab(self, section_name: str):
        if section_name in self.panes:
            index = list(self.panes.keys()).index(section_name)
            self.content_stack.setCurrentIndex(index)
            # Sync sidebar highlight
            for btn in self.nav_buttons:
                btn.setChecked(section_name.startswith(btn.text()))


async def main():
    """Primary QT execution logic."""
    app = QApplication(sys.argv)
    manager = UXManager()
    window = TopologiqApp(manager)
    window.show()

    # The loop now watches the window's internal state
    while window.running:
        app.processEvents()
        await asyncio.sleep(0.01)

    # Force the Qt app to exit and clean up
    app.quit()
    print("Process terminated.")


class AsyncBridge:
    """Helper to run asyncio coroutines from the Qt Event Loop."""

    @staticmethod
    def run(coro):
        """Create a task in the background asyncio loop."""
        return asyncio.ensure_future(coro)


if __name__ == "__main__":
    asyncio.run(main())
