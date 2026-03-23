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

from PySide6.QtCore import Slot

from topologiq.ux.base_pane import BasePane
from topologiq.ux.widgets.bgraph_canvas import BGraphCanvas
from topologiq.ux.widgets.verify_canvas import VerifyCanvas


class CompilePane(BasePane):
    """Laboratory for 3D Lattice Surgery and Verification."""

    def __init__(self, manager, parent=None):
        """Initialise with manager and setup specialized canvases."""
        super().__init__(manager, "COMPILE", parent)
        self._running_tasks = set()

    def setup_ui(self):
        """Build the workstation layout with floating verification."""
        self.layout.setContentsMargins(0, 0, 0, 3)
        self.layout.setSpacing(0)

        # 1. Main 3D Workspace
        self.block_canvas = BGraphCanvas()

        # 2. Floating Verification (Parented to block_canvas)
        self.output_canvas = VerifyCanvas(parent=self.block_canvas)

        # Add only the block_canvas to the layout
        self.layout.addWidget(self.block_canvas)

    def _run_surgery(self, use_reduced: bool):
        """Trigger the heavy 3D transformation in the manager."""
        task = asyncio.create_task(self.manager.handle_lattice_surgery(use_reduced=use_reduced))
        self._running_tasks.add(task)
        task.add_done_callback(self._running_tasks.discard)

    @Slot(dict, dict)
    def update_blockgraph(self, cubes: object, pipes: object):
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
        """Delegate the badge update to the verification canvas."""
        self.output_canvas.update_verification_badge(success)
