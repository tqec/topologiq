"""UX entry point.

Initialises the PySide6 server and handles top-level UX orchestration.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

import os

# Force Qt to use the software "Software" rasterizer instead of the GPU
os.environ["QT_QUICK_BACKEND"] = "software"
os.environ["QT_OPENGL"] = "software"
# For Linux/Mesa specifically:
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
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
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from topologiq.ux.manager import UXManager
from topologiq.ux.panes import CompilePane, DesignPane, RunPane, SimulatePane
from topologiq.ux.utils import styles
from topologiq.ux.widgets.zx_canvas import ZXCanvas

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


class TopologiqApp(QMainWindow):
    """UX primary orchestrator."""

    def __init__(self, manager):
        """Initialise UX with Global Drawer architecture."""
        super().__init__()
        self.manager = manager
        self.running = True

        self.setWindowTitle("TOPOLOGIQ: Algorithmic Lattice Surgery")
        self.resize(1280, 720)
        self._set_icon()

        # --- 1. GLOBAL DRAWER SETUP ---
        self.zx_drawer = QWidget()
        self.zx_drawer.setObjectName("ZXDrawer")

        # This layout holds the Vertical Tab and the Main Canvas
        self.zx_drawer_layout = QHBoxLayout(self.zx_drawer)
        self.zx_drawer_layout.setContentsMargins(0, 0, 0, 0)
        self.zx_drawer_layout.setSpacing(0)

        # A. THE VERTICAL TAB BAR (Left handle)
        self.zx_tab_bar = QFrame()
        self.zx_tab_bar.setFixedWidth(25)
        self.zx_tab_bar.setStyleSheet("background: transparent; border-right: 1px solid #222;")
        zx_tab_layout = QVBoxLayout(self.zx_tab_bar)
        zx_tab_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_toggle_ver = QPushButton("O\nP\nE\nN\n\nI\nD\nE")
        self.btn_toggle_ver.setCheckable(True)
        self.btn_toggle_ver.setFixedWidth(24)
        self.btn_toggle_ver.setStyleSheet(styles.ACTION_BTN + "border-right: 0; border-radius: 0; border-bottom-left-radius: 14px; border-top-right-radius: 7px; padding: 20px 0;")
        self.btn_toggle_ver.clicked.connect(self._toggle_zx_drawer)

        zx_tab_layout.addStretch()
        zx_tab_layout.addWidget(self.btn_toggle_ver)
        zx_tab_layout.addStretch()

        # B. THE CANVAS & HORIZONTAL TOGGLE
        # We need a vertical sub-container for the Canvas + the Bottom Toggle
        canvas_container = QWidget()
        canvas_v_layout = QVBoxLayout(canvas_container)
        canvas_v_layout.setContentsMargins(0, 0, 0, 0)
        canvas_v_layout.setSpacing(0)

        self.zx_canvas = ZXCanvas(self.manager)

        self.drawer_toggle = QPushButton("Collapse ZX")
        self.drawer_toggle.setCheckable(True)
        self.drawer_toggle.setStyleSheet(styles.TOGGLE_BUTTON_STYLE)
        self.drawer_toggle.clicked.connect(self._toggle_zx_drawer)

        self.toggle_bar_widget = QWidget()  # Wrapper to hide/show easily
        self.toggle_bar = QHBoxLayout(self.toggle_bar_widget)
        self.toggle_bar.addWidget(self.drawer_toggle)
        self.toggle_bar.addStretch()
        self.toggle_bar.setContentsMargins(10, 0, 0, 5)

        canvas_v_layout.addWidget(self.zx_canvas)
        canvas_v_layout.addWidget(self.toggle_bar_widget)

        # Assemble the Drawer: [Vertical Tab] | [Canvas + Bottom Toggle]
        self.zx_drawer_layout.addWidget(self.zx_tab_bar)
        self.zx_drawer_layout.addWidget(canvas_container)

        # --- 2. MAIN LAYOUT & STACK ---
        main_widget = QWidget()
        main_widget.setStyleSheet(styles.MAIN_WINDOW_STYLE)
        self.setCentralWidget(main_widget)

        self.root_layout = QVBoxLayout(main_widget)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.setHandleWidth(2)

        self.content_stack = QStackedWidget()
        self.main_splitter.addWidget(self.content_stack)
        self.main_splitter.addWidget(self.zx_drawer)

        self.root_layout.addWidget(self.main_splitter)

        # Initialize Panes
        self.panes = {
            "DESIGN": DesignPane(self.manager),
            "COMPILE": CompilePane(self.manager),
            "SIMULATE": SimulatePane(self.manager),
            "RUN": RunPane(self.manager),
        }

        for pane in self.panes.values():
            self.content_stack.addWidget(pane)

        self._setup_status_bar()
        self._connect_signals()
        self._switch_tab("DESIGN")

    def _set_icon(self):
        icon_path = ASSETS_DIR / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _setup_status_bar(self):
        """Render standard status bar with Nav buttons (logic remains similar)."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet(styles.STATUS_BAR_STYLE)

        self.nav_container = QWidget()
        nav_layout = QHBoxLayout(self.nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)

        self.nav_buttons = {}
        sections = ["DESIGN", "COMPILE", "SIMULATE", "RUN"]

        for name in sections:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet(styles.NAV_BUTTON_STYLE)
            btn.clicked.connect(lambda _, n=name: self._switch_tab(n))
            nav_layout.addWidget(btn)
            self.nav_buttons[name] = btn

        self.status_bar.addWidget(self.nav_container)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)
        self.progress_bar.setFixedWidth(120)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.status_msg_label = QLabel("Ready")
        self.status_bar.addPermanentWidget(self.status_msg_label)

    def closeEvent(self, event):  # noqa: N802 (native Qt method)
        """Handle window close event.

        Args:
            event (QCloseEvent): Event object with details of closure request.

        """
        self.running = False

        # Ensure the sandbox is killed immediately
        if hasattr(self, "manager"):
            self.manager.emergency_stop()

        event.accept()

    def _create_pane(self, name):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel(f"{name} View")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        widget.info_label = label
        return widget

    def _update_global_zx(self, data):
        """Update the shared ZX Visualizer widget."""
        # This will eventually call the update method on your VisPy canvas
        print("Global ZX Visualizer updated with new graph.")

    def _toggle_zx_drawer(self):
        # Determine current mode
        is_design = self.main_splitter.orientation() == Qt.Horizontal

        # Get state from whichever button is currently visible/active
        # (They will be synced, so either works, but we'll check both for safety)
        is_collapsed = (
            self.btn_toggle_ver.isChecked() if is_design else self.drawer_toggle.isChecked()
        )

        total_dim = self.width() if is_design else self.height()

        if is_collapsed:
            # Full Screen ZX
            self.main_splitter.setSizes([0, total_dim])
            ver_text = "O\nP\nE\nN\n\nI\nD\nE"
            hor_text = "CLOSE CANVAS"
        else:
            # Split View
            split = int(total_dim) - 24
            self.main_splitter.setSizes([split, total_dim - split])
            ver_text = "O\nP\nE\nN\n\nC\nA\nN\nV\nA\nS"
            hor_text = "OPEN CANVAS"

        # Update both buttons
        self.btn_toggle_ver.blockSignals(True)
        self.drawer_toggle.blockSignals(True)

        self.btn_toggle_ver.setChecked(is_collapsed)
        self.drawer_toggle.setChecked(is_collapsed)
        self.btn_toggle_ver.setText(ver_text)
        self.drawer_toggle.setText(hor_text)

        self.btn_toggle_ver.blockSignals(False)
        self.drawer_toggle.blockSignals(False)

    def _connect_signals(self):
        """Standardised Signal Routing for the Global Drawer Architecture."""

        def debug_bridge(data):
            print(f"DEBUG 3: app.py CAUGHT zx_input_ready signal. Data type: {type(data)}")

        # --- 1. GENERAL UI & NAVIGATION ---
        # Update status bar text
        self.manager.status_changed.connect(self.status_msg_label.setText)

        # Handle programmatic tab switches (e.g., auto-switching to COMPILE after load)
        self.manager.section_changed.connect(self._switch_tab)

        # Handle UI locking during heavy computations (Lattice Surgery/Transpilation)
        self.manager.processing_state_changed.connect(
            lambda proc: [
                [b.setDisabled(proc) for b in self.nav_buttons.values()],
                self.progress_bar.setVisible(proc),
            ]
        )

        # --- 2. GLOBAL COMPONENT BRIDGES ---

        # The ZX Input Graph is now global.
        # Both DESIGN and COMPILE draw from this single shared visualizer.
        self.manager.zx_input_ready.connect(self.zx_canvas.manage_aug_zx)

        # --- 3. PANE-SPECIFIC BRIDGES ---

        # DESIGN: Update the ASCII Inspector/Raw QASM view
        self.manager.qb_circuit_ready.connect(self.panes["DESIGN"].update_visuals)

        # COMPILE: Update the 3D VisPy Blockgraph (The result of Lattice Surgery)
        self.manager.blockgraph_ready.connect(self.panes["COMPILE"].update_blockgraph)

        # COMPILE: Update the Verification Graph
        # (This is the ZX graph derived BACK from the blocks to prove equality)
        self.manager.zx_output_ready.connect(self.panes["COMPILE"].update_output)

        # COMPILE: Verification Result (Boolean match/mismatch)
        # If we add a "Check Mark" or "X" to the UI, it hooks here.
        self.manager.equality_verification.connect(self.panes["COMPILE"].show_verification_result)

    @Slot(str)
    def _switch_tab(self, section_name: str):
        """Handle context-aware drawer heights and toggle button synchronization."""
        if section_name not in self.panes:
            return

        # 1. Update the Content Stack
        target_pane = self.panes[section_name]
        self.content_stack.setCurrentWidget(target_pane)

        # 2. Geometry calculations
        h = self.height()
        w = self.width()
        is_design = (section_name == "DESIGN")

        # 3. Apply Mode-Specific Layouts
        if is_design:
            # IDE MODE: Side-by-side
            self.main_splitter.setOrientation(Qt.Horizontal)
            # Default to 100% ZX Canvas (Drawer at index 1), 0% Editor (Pane at index 0)
            self.main_splitter.setSizes([0, w])

            # Sync Vertical Button State
            self.btn_toggle_ver.setChecked(True)
            self.btn_toggle_ver.setText("O\nP\nE\nN\n\nI\nD\nE")

            # Show the Tab Bar, hide the bottom toggle
            self.zx_tab_bar.show()
            self.toggle_bar_widget.hide()

        else:
            # CLASSIC MODE: Top-down split
            self.main_splitter.setOrientation(Qt.Vertical)

            # Show the bottom toggle, hide the vertical tab
            self.zx_tab_bar.hide()
            self.toggle_bar_widget.show()

            # Start open (30/70)
            self.main_splitter.setSizes([int(h * 0.9), int(h * 0.1)])
            self.drawer_toggle.setChecked(False)
            self.drawer_toggle.setText("OPEN CANVAS")

        # 4. Update Navigation Buttons Styling
        for name, btn in self.nav_buttons.items():
            btn.blockSignals(True)
            is_active = name == section_name
            btn.setChecked(is_active)
            btn.setText(f"|  {name}  ⟩" if is_active else f"   {name}   ")
            btn.blockSignals(False)

        self.status_bar.showMessage(f"Mode: {section_name}", 2000)


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
