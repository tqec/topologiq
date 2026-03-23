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

import asyncio

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
)

from topologiq.ux.base_pane import BasePane
from topologiq.ux.utils import styles
from topologiq.ux.widgets.bgraph_canvas import BGraphCanvas
from topologiq.ux.widgets.verify_canvas import VerifyCanvas


class CompilePane(BasePane):
    """Laboratory for 3D Lattice Surgery and Verification."""

    def __init__(self, manager, parent=None):
        """Initialise with manager and setup specialized canvases."""
        super().__init__(manager, "COMPILE", parent)
        self._running_tasks = set()

    def setup_ui(self):
        """Build the workstation layout."""
        # 1. TOP ACTION BAR
        self.action_bar = QFrame()
        self.action_bar.setFixedHeight(60)
        bar_layout = QHBoxLayout(self.action_bar)

        self.btn_compile_full = QPushButton("COMPILE FULL")
        self.btn_compile_red = QPushButton("COMPILE REDUCED")

        for btn in [self.btn_compile_full, self.btn_compile_red]:
            btn.setStyleSheet(styles.ACTION_BTN)
            btn.setFixedSize(160, 36)

        self.btn_compile_full.clicked.connect(lambda: self._run_surgery(use_reduced=False))
        self.btn_compile_red.clicked.connect(lambda: self._run_surgery(use_reduced=True))

        self.verify_badge = QLabel("UNVERIFIED")
        self.verify_badge.setStyleSheet(styles.STATUS_BADGE_UNVERIFIED)
        self.verify_badge.setFixedSize(180, 30)
        self.verify_badge.setAlignment(Qt.AlignCenter)

        bar_layout.addWidget(self.btn_compile_full)
        bar_layout.addWidget(self.btn_compile_red)
        bar_layout.addStretch()
        bar_layout.addWidget(self.verify_badge)

        self.layout.addWidget(self.action_bar)

        # 2. CENTRAL WORKSTATION (Horizontal Splitter)
        self.comp_splitter = QSplitter(Qt.Horizontal)
        self.comp_splitter.setStyleSheet("QSplitter::handle { background: #444; width: 2px; }")

        # LEFT: The 3D Blockgraph (The "Physical" Implementation)
        self.block_canvas = BGraphCanvas()
        # Applying the Pastel Green aesthetic for TQEC compatibility
        self.block_canvas.canvas.bgcolor = "#d4edda"
        self.comp_splitter.addWidget(self.block_canvas)

        # RIGHT: The Verification ZX (The Logical Proof)
        self.output_canvas = VerifyCanvas()
        self.comp_splitter.addWidget(self.output_canvas)

        # Set proportions: 70% Blockgraph, 30% Verification
        self.comp_splitter.setSizes([700, 300])
        self.layout.addWidget(self.comp_splitter)

    def _run_surgery(self, use_reduced: bool):
        """Trigger the heavy 3D transformation in the manager."""
        task = asyncio.create_task(self.manager.handle_lattice_surgery(use_reduced=use_reduced))
        self._running_tasks.add(task)
        task.add_done_callback(self._running_tasks.discard)

    @Slot(dict, dict)
    def update_blockgraph(self, cubes: dict, pipes: dict):
        """Update blockgraph."""
        if cubes:
            self.block_canvas.render_blockgraph(cubes, pipes)
            # Ensure the canvas repaints to show the new geometry
            self.block_canvas.update()

    @Slot(object)
    def update_output(self, aug_zx_out):
        """Update output."""
        self.output_canvas.manage_aug_zx(aug_zx_out)

    @Slot(bool)
    def show_verification_result(self, success: bool):
        """Update the HUD badge based on equality check."""
        if success:
            self.verify_badge.setText("VERIFIED EQUIVALENT")
            self.verify_badge.setStyleSheet(styles.STATUS_BADGE_VERIFIED)
        else:
            self.verify_badge.setText("VERIFICATION FAILED")
            self.verify_badge.setStyleSheet(styles.STATUS_BADGE_FAILED)
