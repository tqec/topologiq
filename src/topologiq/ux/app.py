"""UX App.

Initialises the PySide6 server and handles top-level UX orchestration.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

import asyncio
import os
import sys
from pathlib import Path

# Force software rasterization for stability in specific environments
os.environ["QT_QUICK_BACKEND"] = "software"
os.environ["QT_OPENGL"] = "software"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

from PySide6.QtCore import Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from topologiq.ux.manager import UXManager
from topologiq.ux.panes import CompilePane, DesignPane, SimulatePane, StatsPane
from topologiq.ux.utils import styles

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


class TopologiqApp(QMainWindow):
    """Top-level UX orchestrator."""

    def __init__(self, manager):
        """Initialise UX."""
        super().__init__()
        self.manager = manager
        self.running = True

        # Window
        self.setWindowTitle("TOPOLOGIQ: Algorithmic Lattice Surgery")
        self.resize(1400, 850)
        self._set_icon()

        # Primary widget & stack
        self.central_container = QWidget()
        self.setCentralWidget(self.central_container)
        self.main_layout = QVBoxLayout(self.central_container)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.content_stack = QStackedWidget()
        self.main_layout.addWidget(self.content_stack)

        # Initialise panes
        self.panes = {
            "DESIGN": DesignPane(self.manager),
            "COMPILE": CompilePane(self.manager),
            "SIMULATE": SimulatePane(self.manager),
            "STATS": StatsPane(self.manager),
        }
        for pane in self.panes.values():
            self.content_stack.addWidget(pane)

        # UX misc
        self._setup_status_bar()
        self._connect_signals()

        # Default pane
        self._switch_tab("DESIGN")

    @Slot(str)
    def _switch_tab(self, section_name: str):
        """Handle pane transitions."""
        if section_name not in self.panes:
            return

        # Update left section
        self.content_stack.setCurrentWidget(self.panes[section_name])

        # Update NAV styles
        for name, btn in self.nav_buttons.items():
            btn.blockSignals(True)
            is_active = name == section_name
            btn.setChecked(is_active)
            btn.setText(f"|  {name}  ⟩" if is_active else f"   {name}   ")
            btn.blockSignals(False)

    # Standard housekeeping methods
    def _set_icon(self):
        """Set the UX icon."""
        icon_path = ASSETS_DIR / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _setup_status_bar(self):
        """Set the main status bar."""
        # Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet(styles.STATUS_BAR_STYLE)

        # Container
        self.nav_container = QWidget()
        nav_layout = QHBoxLayout(self.nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)

        # Buttons
        self.nav_buttons = {}
        for name in ["DESIGN", "COMPILE", "SIMULATE", "STATS"]:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet(styles.NAV_BUTTON_STYLE)
            btn.clicked.connect(lambda _, n=name: self._switch_tab(n))
            nav_layout.addWidget(btn)
            self.nav_buttons[name] = btn
        self.status_bar.addWidget(self.nav_container)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)
        self.progress_bar.setFixedWidth(120)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_msg_label = QLabel("Ready")
        self.status_bar.addPermanentWidget(self.status_msg_label)

    def _connect_signals(self):
        """Manage top-level signals."""
        # UX state
        self.manager.status_changed.connect(self.status_msg_label.setText)
        self.manager.section_changed.connect(self._switch_tab)
        self.manager.processing_state_changed.connect(self._handle_processing_ui)

        # Data pipelines (retrievers)
        self.manager.zx_input_ready.connect(self.panes["DESIGN"].handle_zx_input)
        self.manager.qb_circuit_ready.connect(self.panes["DESIGN"].update_visuals)

        # Partial outputs (ready) notifications
        self.manager.blockgraph_ready.connect(self.panes["COMPILE"].update_blockgraph)
        self.manager.zx_output_ready.connect(self.panes["COMPILE"].update_output)
        self.manager.verification_ready.connect(self.panes["COMPILE"].show_verification_result)

        # Pane switchers
        design_pane = self.panes["DESIGN"]
        design_pane.zx_canvas.compile_requested.connect(self._bridge_to_compile)

    def _bridge_to_compile(self, graph_key: str):
        """Manage the transition between the Canvas and Compile panes."""
        # Switch
        self._switch_tab("COMPILE")

        # Force focus on destination
        self.panes["COMPILE"].set_focus_key(graph_key)

    def _handle_processing_ui(self, is_active: bool):
        """Unify UX response to background processing."""
        # Prevent tab-switching during heavy tasks
        for btn in self.nav_buttons.values():
            btn.setDisabled(is_active)

        # Toggle progress bar
        self.progress_bar.setVisible(is_active)

        # Update the status label when finished
        if not is_active:
            self.status_msg_label.setText("Ready")

    def closeEvent(self, event):  # noqa: N802 (native method)
        """Handle force close events."""
        self.running = False
        if hasattr(self, "manager"):
            self.manager.emergency_stop()
        event.accept()


async def main():
    """Run the UX app."""
    # Fire App
    app = QApplication(sys.argv)

    # Fire UX manager
    manager = UXManager()

    # Fire window
    window = TopologiqApp(manager)
    window.show()

    # Loopy McLoopy
    while window.running:
        app.processEvents()
        await asyncio.sleep(0.01)
    app.quit()


# ...
if __name__ == "__main__":
    asyncio.run(main())
