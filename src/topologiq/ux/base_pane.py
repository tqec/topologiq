"""UX panes initialisation.

Initialises the different panes of the UX.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from topologiq.ux.styles import PANE_HEADER_STYLE


class BasePane(QWidget):
    """Base class for all UX sections.

    Provides a standardised layout and info label for child classes
    to inherit, ensuring a consistent 'look and feel' across the stack.
    """

    def __init__(self, name: str, parent=None):
        """Initialise base pane.

        Args:
            name (str): The display name of the section.
            parent (QWidget, optional): The parent widget. Defaults to None.

        """

        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.info_label = QLabel(f"{name}")
        self.info_label.setAlignment(Qt.AlignRight)
        self.info_label.setStyleSheet(PANE_HEADER_STYLE)
        self.layout.addWidget(self.info_label)
