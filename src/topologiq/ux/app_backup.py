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
import os
import sys
from pathlib import Path

# Force software rasterization for stability in specific environments
os.environ["QT_QUICK_BACKEND"] = "software"
os.environ["QT_OPENGL"] = "software"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

from PySide6.QtCore import Qt, QTimer, Slot
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
    """UX primary orchestrator with Permanent Vertical Split architecture."""

    def __init__(self, manager):  # noqa: D107
        super().__init__()
        self.manager = manager
        self.running = True

        self.setWindowTitle("TOPOLOGIQ: Algorithmic Lattice Surgery")
        self.resize(1400, 850)
        self._set_icon()

        # --- 1. THE TRIPLE SPLITTER ---
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(2)
        # Allow dragging to 0, but buttons handle the actual hide() for clarity
        self.main_splitter.setChildrenCollapsible(True)
        self.main_splitter.setStyleSheet("QSplitter::handle { background: #222; }")

        # --- 2. LEFT SECTION (IDE) ---
        self.content_stack = QStackedWidget()
        self.content_stack.setMinimumWidth(0)

        # --- 3. CENTER SECTION (CONTROL RAIL) ---
        self.center_rail = QFrame()
        self.center_rail.setFixedWidth(34)
        self.center_rail.setStyleSheet(
            "background-color: #1a1a1a; border-left: 1px solid #333; border-right: 1px solid #333;"
        )
        rail_layout = QVBoxLayout(self.center_rail)
        rail_layout.setContentsMargins(0, 20, 0, 20)

        self.btn_close_left = QPushButton("C\nL\nO\nS\nE\n\nI\nD\nE")
        self.btn_open_left = QPushButton("O\nP\nE\nN\n\nI\nD\nE")
        self.btn_open_left.hide()

        for btn in [self.btn_close_left, self.btn_open_left]:
            btn.setFixedSize(30, 160)
            btn.setStyleSheet(styles.IDE_PILL_STYLE)
            rail_layout.addWidget(btn)

        rail_layout.addStretch()

        self.btn_close_right = QPushButton("C\nL\nO\nS\nE\n\nC\nA\nN\nV\nA\nS")
        self.btn_open_right = QPushButton("O\nP\nE\nN\n\nC\nA\nN\nV\nA\nS")
        self.btn_open_right.hide()

        for btn in [self.btn_close_right, self.btn_open_right]:
            btn.setFixedSize(30, 200)
            btn.setStyleSheet(styles.CANVAS_PILL_STYLE)
            rail_layout.addWidget(btn)

        # --- 4. RIGHT SECTION (CANVAS) ---
        self.zx_canvas = ZXCanvas(self.manager)
        self.zx_canvas.setMinimumWidth(0)

        # --- 5. ASSEMBLY ---
        self.main_splitter.addWidget(self.content_stack)
        self.main_splitter.addWidget(self.center_rail)
        self.main_splitter.addWidget(self.zx_canvas)
        self.setCentralWidget(self.main_splitter)

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

        # Initial Pane setup
        self._switch_tab("DESIGN")

        # Apply initial 40/60 split after the window is rendered
        QTimer.singleShot(0, self._apply_initial_layout)

        # Exclusive Toggle Connections
        self.btn_close_left.clicked.connect(lambda: self._set_side_visibility("LEFT", False))
        self.btn_open_left.clicked.connect(lambda: self._set_side_visibility("LEFT", True))
        self.btn_close_right.clicked.connect(lambda: self._set_side_visibility("RIGHT", False))
        self.btn_open_right.clicked.connect(lambda: self._set_side_visibility("RIGHT", True))

    def _apply_initial_layout(self):
        """Set the first-load sizing once. No more resets on tab switches."""
        w = self.width()
        RAIL_W = 34  # noqa: N806
        avail = w - RAIL_W
        self.main_splitter.setSizes([int(avail * 0.4), RAIL_W, int(avail * 0.6)])

    def _set_side_visibility(self, side: str, visible: bool):
        sizes = self.main_splitter.sizes()
        total_w = sum(sizes)
        RAIL_W = 34  # noqa: N806

        if side == "LEFT":
            if not visible:
                # CLOSE IDE: Canvas takes everything
                self._prev_left_w = sizes[0]
                self.main_splitter.setSizes([0, RAIL_W, total_w - RAIL_W])
                self.btn_close_left.hide()
                self.btn_open_left.show()
                self.btn_close_right.show()
                self.btn_open_right.hide()
            else:
                # OPEN IDE: Restore to previous or a default split
                self.content_stack.show()
                # If no previous width, default to a fair split
                self.main_splitter.setSizes([total_w - RAIL_W, RAIL_W, 0])
                self.btn_open_left.hide()
                self.btn_close_left.show()
                self.btn_open_right.show()
                self.btn_close_right.hide()

        elif side == "RIGHT":
            if not visible:
                # CLOSE CANVAS: IDE takes everything
                self._prev_right_w = sizes[2]
                self.main_splitter.setSizes([total_w - RAIL_W, RAIL_W, 0])
                self.btn_close_right.hide()
                self.btn_open_right.show()
                self.btn_close_left.show()
                self.btn_open_left.hide()
            else:
                # OPEN CANVAS: Restore
                self.zx_canvas.show()
                self.main_splitter.setSizes([0, RAIL_W, total_w - RAIL_W])
                self.btn_open_right.hide()
                self.btn_close_right.show()
                self.btn_open_left.show()
                self.btn_close_left.hide()

    @Slot(str)
    def _switch_tab(self, section_name: str):
        """Handle layout transitions for the Triple-Splitter architecture."""
        if section_name not in self.panes:
            return

        # 1. Update the Left-side Content Stack
        self.content_stack.setCurrentWidget(self.panes[section_name])

        # 2. Reset visibility to a clean state for the new section
        # This ensures both the IDE and Canvas/Visuals are visible upon entry
        self.content_stack.show()
        self.zx_canvas.show()
        self.btn_open_left.hide()
        self.btn_close_left.show()
        self.btn_open_right.hide()
        self.btn_close_right.show()

        w = self.width()
        RAIL_W = 34  # The fixed width of our center control column  # noqa: N806

        if section_name == "DESIGN":
            # DESIGN: 40% IDE | RAIL | 60% ZX Canvas (approx)
            available_w = w - RAIL_W
            self.main_splitter.setSizes([int(available_w * 0.4), RAIL_W, int(available_w * 0.6)])

        elif section_name == "COMPILE":
            # COMPILE: 30% IDE (Slider) | RAIL | 70% Blockgraph (Visuals)
            available_w = w - RAIL_W
            self.main_splitter.setSizes([int(available_w * 0.3), RAIL_W, int(available_w * 0.7)])

        # 3. Update Navigation Buttons Styling
        for name, btn in self.nav_buttons.items():
            btn.blockSignals(True)
            is_active = name == section_name
            btn.setChecked(is_active)
            btn.setText(f"|  {name}  ⟩" if is_active else f"   {name}   ")
            btn.blockSignals(False)

    # --- Standard Housekeeping Methods ---
    def _set_icon(self):
        icon_path = ASSETS_DIR / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _setup_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet(styles.STATUS_BAR_STYLE)
        self.nav_container = QWidget()
        nav_layout = QHBoxLayout(self.nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        self.nav_buttons = {}
        for name in ["DESIGN", "COMPILE", "SIMULATE", "RUN"]:
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

    def _connect_signals(self):
        self.manager.status_changed.connect(self.status_msg_label.setText)
        self.manager.section_changed.connect(self._switch_tab)
        self.manager.processing_state_changed.connect(
            lambda proc: [
                [b.setDisabled(proc) for b in self.nav_buttons.values()],
                self.progress_bar.setVisible(proc),
            ]
        )
        # Global Bridges
        self.manager.zx_input_ready.connect(self.zx_canvas.manage_aug_zx)
        self.manager.qb_circuit_ready.connect(self.panes["DESIGN"].update_visuals)
        self.manager.blockgraph_ready.connect(self.panes["COMPILE"].update_blockgraph)
        self.manager.zx_output_ready.connect(self.panes["COMPILE"].update_output)
        self.manager.equality_verification.connect(self.panes["COMPILE"].show_verification_result)

    def closeEvent(self, event):  # noqa: D102, N802
        self.running = False
        if hasattr(self, "manager"):
            self.manager.emergency_stop()
        event.accept()


async def main():  # noqa: D103
    app = QApplication(sys.argv)
    manager = UXManager()
    window = TopologiqApp(manager)
    window.show()
    while window.running:
        app.processEvents()
        await asyncio.sleep(0.01)
    app.quit()


if __name__ == "__main__":
    asyncio.run(main())
