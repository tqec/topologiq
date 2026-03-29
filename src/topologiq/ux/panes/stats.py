"""Layout for the STATS section.

Displays summary statistics for the different circuits, graphs,
and blockgraphs across panes.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from topologiq.ux.base_pane import BasePane


class StatsPane(BasePane):
    """Final output and execution management."""

    def __init__(self, manager, parent=None):
        """Initialise STATS section."""
        super().__init__(manager, "STATS", parent)
        self.manager = manager
