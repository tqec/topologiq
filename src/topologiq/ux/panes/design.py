"""Layout for the DESIGN section.

Integrates Monaco editor for circuit coding and buttons
for importing/exporting circuit diagrams into qBraid and exporting
QASM and PyZX-compatible formats.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

import asyncio
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
)

from topologiq.ux.base_pane import BasePane
from topologiq.ux.highlighter import PygmentsHighlighter
from topologiq.ux.styles import TEXT_STYLE_CODE, TEXT_STYLE_DRAW_BTN, TEXT_STYLE_TRANSPILE_BTN


class DesignPane(BasePane):
    """Circuit Design IDE for loading, editing, and transpiling quantum circuits."""

    def __init__(self, parent=None):
        """Initialise DESIGN pane."""
        super().__init__("DESIGN", parent)
        self.current_file_path = None
        self._tasks = set()
        self._init_ui()

    def _init_ui(self):
        self.top_splitter = QSplitter(Qt.Horizontal)

        # Left Column: The Code Editor
        self.editor_container = QFrame()
        self.editor_title = QLabel("SOURCE CODE: TEXT")
        editor_layout = QVBoxLayout(self.editor_container)
        editor_layout.addWidget(self.editor_title)

        self.code_editor = QPlainTextEdit()
        self.code_editor.setPlaceholderText("Write code here or load a script or QASM file...")
        self.code_editor.setStyleSheet(TEXT_STYLE_CODE)
        self.code_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.code_editor.selectionChanged.connect(self._handle_selection_sync)
        editor_layout.addWidget(self.code_editor)

        # Right Column: The ASCII Inspector
        self.viewer_container = QFrame()
        viewer_layout = QVBoxLayout(self.viewer_container)
        viewer_layout.addWidget(QLabel("ASCII INSPECTOR"))

        self.ascii_viewer = QPlainTextEdit()
        self.ascii_viewer.setReadOnly(True)
        self.ascii_viewer.setPlaceholderText("If circuit validates, ASCII diagram will appear here...")
        self.ascii_viewer.setStyleSheet(TEXT_STYLE_CODE)
        self.ascii_viewer.setLineWrapMode(QPlainTextEdit.NoWrap)
        viewer_layout.addWidget(self.ascii_viewer)

        self.top_splitter.addWidget(self.editor_container)
        self.top_splitter.addWidget(self.viewer_container)
        self.layout.addWidget(self.top_splitter, 9)

        # Bottom Bar
        self.bottom_bar = QFrame()
        self.bottom_bar.setFixedHeight(50)
        bottom_layout = QHBoxLayout(self.bottom_bar)

        self.btn_load_py = QPushButton("Load PYTHON")
        self.btn_load_py.clicked.connect(lambda: self._handle_open_file("python"))

        self.btn_load_qasm = QPushButton("Load QASM")
        self.btn_load_qasm.clicked.connect(lambda: self._handle_open_file("qasm"))

        self.btn_save = QPushButton("Save")
        save_menu = QMenu(self)
        save_menu.addAction("Save", self._handle_save_file)
        save_menu.addAction("Save As...", self._handle_save_as)
        self.btn_save.setMenu(save_menu)

        self.btn_close = QPushButton("Close / Clear")
        self.btn_close.setStyleSheet("color: #ff5555;")
        self.btn_close.clicked.connect(self._handle_close_clear)

        self.btn_draw = QPushButton("VALIDATE / DRAW CIRCUIT")
        self.btn_draw.setStyleSheet(TEXT_STYLE_DRAW_BTN)
        self.btn_draw.clicked.connect(lambda: self._process_and_emit(switch_pane=False))

        self.var_input = QLineEdit("__circuit__")
        self.var_input.setFixedWidth(140)
        self.var_input.setToolTip("Highlight a variable in code to auto-populate this.")

        self.btn_ingest = QPushButton("TRANSPILE CIRCUIT TO ZX")
        self.btn_ingest.setStyleSheet(TEXT_STYLE_TRANSPILE_BTN)
        self.btn_ingest.clicked.connect(lambda: self._process_and_emit(switch_pane=True))

        bottom_layout.addWidget(self.btn_load_py)
        bottom_layout.addWidget(self.btn_load_qasm)
        bottom_layout.addWidget(self.btn_save)
        bottom_layout.addWidget(self.btn_close)
        bottom_layout.addStretch()
        bottom_layout.addWidget(QLabel("Variable to transpile:"))
        bottom_layout.addWidget(self.var_input)
        bottom_layout.addWidget(self.btn_draw)
        bottom_layout.addSpacing(10)
        bottom_layout.addWidget(self.btn_ingest)
        self.layout.addWidget(self.bottom_bar, 1)

    def _process_and_emit(self, switch_pane: bool):
        """Unified method to send code and variable to the manager."""
        code = self.code_editor.toPlainText()
        title_text = self.editor_title.text().upper()

        mode = "python" if "PYTHON" in title_text else "qasm" if "QASM" in title_text else "text"
        target_var = self.var_input.text().strip() if mode == "python" else None

        app = self.window()
        if hasattr(app, "manager"):
            # We pass switch_pane to the manager so it knows if we want to move to TRANSFORM
            task = asyncio.ensure_future(
                app.manager.handle_load_source_circuit(
                    code, mode, var_name=target_var, switch_to_transform=switch_pane
                )
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

            msg = "Transpiling..." if switch_pane else "Drawing ASCII..."
            self.window().statusBar().showMessage(msg, 2000)

    def _handle_open_file(self, mode):
        ext_filter = "Python (*.py)" if mode == "python" else "OpenQASM (*.qasm)"
        path, _ = QFileDialog.getOpenFileName(self, f"Open {mode.upper()}", "", ext_filter)

        if path:
            self.code_editor.clear()
            if hasattr(self, "highlighter"):
                self.highlighter.setDocument(None)

            self.highlighter = PygmentsHighlighter(self.code_editor.document(), mode)
            self.current_file_path = Path(path)
            self.code_editor.setPlainText(self.current_file_path.read_text())
            self.editor_title.setText(f"SOURCE CODE: {mode.upper()}")

    def _handle_save_file(self):
        if self.current_file_path:
            try:
                self.current_file_path.write_text(self.code_editor.toPlainText())
                self.window().statusBar().showMessage(f"Saved: {self.current_file_path.name}", 3000)
            except Exception as e:
                self.window().statusBar().showMessage(f"Save Failed: {e}", 5000)
        else:
            self._handle_save_as()

    def _handle_save_as(self):
        """Save current editor content to a new file and update IDE mode."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Circuit As", "", "Python Script (*.py);;OpenQASM (*.qasm)"
        )
        if path:
            self.current_file_path = Path(path)
            # Write the file content
            self.current_file_path.write_text(self.code_editor.toPlainText())

            # Use the helper to sync the title and highlighter automatically
            self._update_mode_from_path(self.current_file_path)
            self.window().statusBar().showMessage(f"Saved As: {self.current_file_path.name}", 3000)

    def _handle_selection_sync(self):
        cursor = self.code_editor.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText().strip()
            if text.isidentifier():
                self.var_input.setText(text)

    def _handle_close_clear(self):
        self.code_editor.clear()
        if hasattr(self, "highlighter"):
            self.highlighter.setDocument(None)
        self.current_file_path = None
        self.editor_title.setText("SOURCE CODE: TEXT")
        self.var_input.setText("__circuit__")
        self.ascii_viewer.clear()
        self.window().statusBar().showMessage("Session cleared", 2000)

    def _update_mode_from_path(self, path: Path):
        """Standardizes how we switch the UI between Python, QASM, and Text."""
        mode = "python" if path.suffix == ".py" else "qasm"
        self.editor_title.setText(f"SOURCE CODE: {mode.upper()}")

        if hasattr(self, "highlighter"):
            self.highlighter.setDocument(None)
        self.highlighter = PygmentsHighlighter(self.code_editor.document(), mode)

    @Slot(str, str)
    def update_visuals(self, qasm, ascii_art):
        """Update the ASCII viewer with output from the manager."""
        self.ascii_viewer.setPlainText(ascii_art)
