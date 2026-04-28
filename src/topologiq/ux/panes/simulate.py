"""Layout for the SIMULATION section.

Provides a container for TQEC-generated blockgraph visualisations (JS/HTML)
and simulation results (Matplotlib).

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from topologiq.ux.base_pane import BasePane


class SimulatePane(BasePane):
    """Topologiq -> TQEC simulations."""

    def __init__(self, manager, parent=None):
        """Initialise SIMULATE pane."""
        super().__init__(manager, "SIMULATE", parent)
        self.manager = manager

