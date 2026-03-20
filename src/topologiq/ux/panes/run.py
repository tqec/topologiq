"""Layout for the RUN (in hardware) section.

Displays future hardware compatibility status and, for the time being,
extracts circuit intended for running in actual QPU.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from topologiq.ux.base_pane import BasePane


class RunPane(BasePane):
    """Final output and execution management."""

    def __init__(self, parent=None):
        """Initialise RUN section."""
        super().__init__("RUN", parent)
