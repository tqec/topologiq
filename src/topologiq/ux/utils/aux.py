"""UX misc. utils.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton


def create_split_controls(parent, labels, signal_to_emit):
    """Create a cluster of three buttons for layout control.

    Args:
        parent: The widget to parent the frame to.
        labels: List of 3 strings (e.g., ["CLOSE", "40/60", "OPEN"]).
        signal_to_emit: The Signal.emit method to connect to.

    """
    container = QFrame(parent)
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(1)

    for txt in labels:
        btn = QPushButton(txt)
        btn.setFixedHeight(24)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton {
                padding-left: 10px;
                padding-right: 10px;
            }
        """
        )
        # Connect to the passed signal
        btn.clicked.connect(lambda checked=False, t=txt: signal_to_emit(t))
        layout.addWidget(btn)

    container.adjustSize()
    return container


def handle_splitter_toggle(splitter, total_width, side, mode):
    """Redefine layout split.

    Args:
        splitter: The QSplitter instance.
        total_width: Current width of the parent container.
        side: "LEFT" or "RIGHT" (which widget emitted the signal).
        mode: The button text ("CLOSE IDE", "40/60", etc).

    """
    if total_width <= 0:
        return

    # 1. The 40/60 Split (Middle Button)
    if mode == "◫" or "40/60" in mode:
        ratio = 0.4
        left_w = int(total_width * ratio)
        splitter.setSizes([left_w, total_width - left_w])
        return

    # 2. Minimize / Collapse (Left Button "✕")
    if mode == "✕":
        if side == "LEFT":
            # Collapse IDE -> [0, Total]
            splitter.setSizes([0, total_width])
        else:
            # Collapse Canvas -> [Total, 0]
            splitter.setSizes([total_width, 0])
        return

    # 3. Maximize / Full Width (Right Button "□")
    if mode == "□":  # Keeping 'X' logic for backward compatibility
        if side == "LEFT":
            # Maximize IDE -> [Total, 0]
            splitter.setSizes([total_width, 0])
        else:
            # Maximize Canvas -> [0, Total]
            splitter.setSizes([0, total_width])
        return
