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

from topologiq.ux.utils import styles


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
        btn.setStyleSheet(
            styles.ACTION_BTN
            + """
            QPushButton {
                padding-left: 10px;
                padding-right: 10px;
                min-width: 40px;
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

    if "/" in mode or "RESET" in mode:
        ratio = int(mode.split("/")[0]) / 100
        left_w = int(total_width * ratio)
        splitter.setSizes([left_w, total_width - left_w])
        return

    if side == "LEFT":
        # "CLOSE IDE" -> Hide Left (0, W) | "OPEN IDE" -> Hide Right (W, 0)
        new_sizes = [0, total_width] if "CLOSE" in mode else [total_width, 0]
        splitter.setSizes(new_sizes)

    elif side == "RIGHT":
        # "CLOSE CANVAS" -> Hide Right (W, 0) | "OPEN CANVAS" -> Hide Left (0, W)
        new_sizes = [total_width, 0] if "CLOSE" in mode else [0, total_width]
        splitter.setSizes(new_sizes)
