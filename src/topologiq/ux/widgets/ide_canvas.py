"""UX circuit IDE canvas.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

import asyncio
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from topologiq.ux.utils import styles
from topologiq.ux.utils.aux import create_split_controls
from topologiq.ux.utils.highlighter import PygmentsHighlighter


class CircuitIDE(QWidget):
    """The self-contained Editor, Inspector, and Footer unit."""

    toggle_requested = Signal(str)

    def __init__(self, manager, parent=None):
        """Initialise Circuit IDE pane."""
        super().__init__(parent)
        self.manager = manager
        self.current_file_path = None
        self.highlighter = None
        self._tasks = set()
        self.setup_ui()

    def setup_ui(self):
        """Define the layout for the circuit IDE."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(5)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

        # 1. Vertical Splitter: Editor / Inspector
        self.v_splitter = QSplitter(Qt.Vertical)
        self.v_splitter.setStyleSheet("QSplitter::handle { background: #222; height: 1px; }")

        # --- TOP: Editor ---
        self.editor_container = QFrame()
        ed_layout = QVBoxLayout(self.editor_container)
        ed_layout.setContentsMargins(0, 0, 0, 5)

        self.header_bar = self._create_header_bar()
        self.code_editor = QPlainTextEdit()
        self.code_editor.setPlaceholderText("Write Python or QASM...")
        self.code_editor.setStyleSheet(styles.TEXT_STYLE_CODE)
        self.code_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.code_editor.selectionChanged.connect(self._handle_selection_sync)

        self.btn_draw_only = QPushButton("↓↓↓ DRAW ASCII ↓↓↓")
        self.btn_draw_only.setStyleSheet(
            styles.PILL_BTN_PYZX + "border-radius: 0px; background: #333;"
        )
        self.btn_draw_only.clicked.connect(lambda: self._process_and_emit(switch_pane=False))

        ed_layout.addWidget(self.header_bar)
        ed_layout.addWidget(self.code_editor)
        ed_layout.addWidget(self.btn_draw_only)

        # --- BOTTOM: Inspector ---
        self.inspector_tabs = QTabWidget()
        self.inspector_tabs.setStyleSheet(
            "QTabBar::tab { height: 25px; font-size: 10px; background: #1a1a1a; color: #666; padding: 0 15px; } "
            "QTabBar::tab:selected { background: #2a2a2a; color: #f2f3fb; border-bottom: 1px dotted #ec0202; }"
            "QTabWidget::pane { border: 1px solid #222; background: #050505; }"
        )

        self.ascii_viewer = QPlainTextEdit()
        self.ascii_viewer.setReadOnly(True)
        self.ascii_viewer.setStyleSheet(styles.TEXT_STYLE_CODE)

        self.terminal_output = QPlainTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setStyleSheet(styles.TEXT_STYLE_CODE)

        self.inspector_tabs.addTab(self.ascii_viewer, "ASCII DIAGRAM")
        self.inspector_tabs.addTab(self.terminal_output, "TERMINAL / LOGS")

        self.v_splitter.addWidget(self.editor_container)
        self.v_splitter.addWidget(self.inspector_tabs)
        self.v_splitter.setSizes([650, 350])

        # 2. Footer
        self.footer_bar = self._create_footer_bar()

        self.layout.addWidget(self.v_splitter)
        self.layout.addWidget(self.footer_bar)
        self.setMinimumWidth(0)  # Critical for the "Crush" logic

    def _create_header_bar(self):
        bar = QFrame()
        bar.setFixedHeight(30)
        bar.setStyleSheet("background: #1a1a1a; border-bottom: 1px solid #333;")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 5, 0)
        layout.setSpacing(1)  # Tight spacing for the red button cluster

        # 1. The Layout Toggle Triplet
        self.toggle_buttons = create_split_controls(
            self, ["CLOSE IDE", "40/60"], self.toggle_requested.emit
        )
        layout.addWidget(self.toggle_buttons)

        # 2. Spacer and File Actions
        layout.addSpacing(10)  # Gap between toggles and file buttons
        layout.addStretch()

        self.btn_load_py = QPushButton("LOAD .PY")
        self.btn_load_qasm = QPushButton("LOAD .QASM")
        self.btn_save = QPushButton("SAVE")

        self.btn_load_py.clicked.connect(lambda: self._handle_open_file("python"))
        self.btn_load_qasm.clicked.connect(lambda: self._handle_open_file("qasm"))
        self.btn_save.clicked.connect(self._handle_save_file)

        for btn in [self.btn_load_py, self.btn_load_qasm, self.btn_save]:
            btn.setFixedSize(85, 22)
            btn.setStyleSheet(styles.PILL_BTN_PYZX + "background: #333;")
            layout.addWidget(btn)

        return bar

    def _create_footer_bar(self):
        bar = QFrame()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(3, 0, 0, 0)

        self.mode_label = QLabel("SOURCE: TEXT")

        self.var_input = QLineEdit("__circuit__")
        self.var_input.setFixedWidth(120)
        self.var_input.setStyleSheet("border: 1px solid #666;")
        self.btn_to_zx = QPushButton("GENERATE ZX GRAPH →")
        self.btn_to_zx.setStyleSheet(styles.PILL_BTN_PYZX + "background: #333; font-size: 14px;")
        self.btn_to_zx.clicked.connect(lambda: self._process_and_emit(switch_pane=True))

        layout.addStretch()
        layout.addWidget(QLabel("Circuit name:"))
        layout.addWidget(self.var_input)
        layout.addWidget(self.btn_to_zx)
        return bar

    def _handle_open_file(self, mode):
        ext = "Python (*.py)" if mode == "python" else "OpenQASM (*.qasm)"
        path, _ = QFileDialog.getOpenFileName(self, f"Open {mode.upper()}", "", ext)
        if path:
            self.current_file_path = Path(path)
            self.code_editor.setPlainText(self.current_file_path.read_text())
            self.mode_label.setText(f"SOURCE: {mode.upper()}")

            if self.highlighter:
                self.highlighter.setDocument(None)
            self.highlighter = PygmentsHighlighter(self.code_editor.document(), mode)

    def _handle_save_file(self):
        """Save current editor content to disk."""
        path = self.current_file_path

        # If no file is open, or if user wants a new location, trigger Save As
        if not path:
            mode = "python" if "PYTHON" in self.mode_label.text() else "qasm"
            ext = "Python (*.py)" if mode == "python" else "OpenQASM (*.qasm)"
            path_str, _ = QFileDialog.getSaveFileName(self, "Save File", "", ext)
            if not path_str:
                return
            path = Path(path_str)
            self.current_file_path = path

        try:
            path.write_text(self.code_editor.toPlainText())
            self.window().status_bar.showMessage(f"Saved: {path.name}", 3000)
        except Exception as e:
            self.window().status_bar.showMessage(f"Save failed: {e}", 5000)

    def _handle_selection_sync(self):
        cursor = self.code_editor.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText().strip()
            if text.isidentifier():
                self.var_input.setText(text)

    def _connect_internal_signals(self):
        """Link the local IDE buttons to the Manager logic."""
        # The 'Draw' button (No pane switch)
        self.btn_draw_only.clicked.connect(lambda: self._process_and_emit(switch_pane=False))
        # The 'Generate ZX' button (Pane switch)
        self.btn_to_zx.clicked.connect(lambda: self._process_and_emit(switch_pane=True))

    def _process_and_emit(self, switch_pane: bool):
        code = self.code_editor.toPlainText()
        # Determine mode based on your mode_label text
        mode = "python" if "PYTHON" in self.mode_label.text().upper() else "qasm"

        # Use the manager method we see in manager.py
        task = asyncio.ensure_future(
            self.manager.handle_load_source_circuit(
                code, mode, var_name=self.var_input.text(), switch_to_transform=switch_pane
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
