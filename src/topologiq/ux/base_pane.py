"""UX panes initialisation.

Initialises the different panes of the UX.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from PySide6.QtWidgets import QVBoxLayout, QWidget


class BasePane(QWidget):
    """Base widget for all UX panes.

    Provides a standardized layout and access to the UXManager
    for signal/slot synchronization across the app.
    """

    def __init__(self, manager, name: str, parent=None):
        """Initialise pane with manager access."""
        super().__init__(parent)
        self.manager = manager
        self.pane_id = name
        self.setMinimumHeight(0)

        # Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Trigger UX build (to be overridden by children)
        self.setup_ui()

    def setup_ui(self):
        """Override this in child classes to add widgets."""
        pass

    def update_visuals(self, *args, **kwargs):
        """Standardise interface for the Manager to push data updates."""
        pass
