"""Layout for the TRANSFORM section.

Manages a 3-way split view using VTK/Vedo to visualize ZX-graphs,
lattice surgery blocks, and 3D network layouts of completed lattice surgeries.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from topologiq.ux.base_pane import BasePane


class TransformPane(BasePane):
    """Topologiq's core ZX -> blockgraph LS transformations."""

    def __init__(self, parent=None):
        """Initialise TRANSFORM section."""
        super().__init__("TRANSFORM", parent)
