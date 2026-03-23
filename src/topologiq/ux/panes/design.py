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
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
)

from topologiq.ux.base_pane import BasePane
from topologiq.ux.utils import styles
from topologiq.ux.utils.highlighter import PygmentsHighlighter


class DesignPane(BasePane):
    """Circuit Design IDE with Code/Visual toggle and ASCII validation."""

    def __init__(self, manager, parent=None):  # noqa: D107
        super().__init__(manager, "DESIGN", parent)
        self.current_file_path = None
        self.highlighter = None
        self._tasks = set()
        # We don't call setup_ui here because BasePane calls it.

    def setup_ui(self):
        """Build the clean IDE layout."""
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Main Container (Full width, no sidebar)
        self.main_container = QFrame()
        self.main_layout = QVBoxLayout(self.main_container)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(5)

        # 1. Vertical Splitter: [Editor Section] over [Inspector Tabs]
        self.v_splitter = QSplitter(Qt.Vertical)
        self.v_splitter.setStyleSheet("QSplitter::handle { background: #222; height: 1px; }")

        # --- TOP: Editor Section ---
        self.editor_container = QFrame()
        ed_layout = QVBoxLayout(self.editor_container)
        ed_layout.setContentsMargins(0, 0, 0, 5)

        self.editor_container = QFrame()
        ed_layout = QVBoxLayout(self.editor_container)
        ed_layout.setContentsMargins(0, 0, 0, 5)

        # NEW: Header Bar instead of simple Label
        self.header_bar = QFrame()
        header_layout = QHBoxLayout(self.header_bar)
        header_layout.setContentsMargins(0, 0, 0, 2)

        # Load buttons moved here
        btn_py = QPushButton("LOAD .PY")
        btn_py.setStyleSheet(styles.PILL_BTN_PYZX + "border-radius: 1px; background: #333333;")
        btn_py.clicked.connect(lambda: self._handle_open_file("python"))

        btn_qasm = QPushButton("LOAD .QASM")
        btn_qasm.setStyleSheet(styles.PILL_BTN_PYZX + "border-radius: 1px; background: #333333;")
        btn_qasm.clicked.connect(lambda: self._handle_open_file("qasm"))

        btn_save = QPushButton("SAVE")
        btn_save.setStyleSheet(styles.PILL_BTN_PYZX + "border-radius: 1px;")
        btn_save.clicked.connect(self._handle_save_file)

        header_layout.addStretch()
        header_layout.addWidget(btn_py)
        header_layout.addWidget(btn_qasm)
        header_layout.addWidget(btn_save)

        # IDE / code editor
        self.code_editor = QPlainTextEdit()
        self.code_editor.setPlaceholderText("Write Python or QASM...")
        self.code_editor.setStyleSheet(styles.TEXT_STYLE_CODE)
        self.code_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.code_editor.selectionChanged.connect(self._handle_selection_sync)

        # The DRAW Button (Waterfall Bridge)
        self.btn_draw_only = QPushButton("↓↓↓ DRAW ASCII ↓↓↓")
        self.btn_draw_only.setStyleSheet(styles.ACTION_BTN + "margin: 5px 0; padding: 8px;")
        self.btn_draw_only.clicked.connect(lambda: self._process_and_emit(switch_pane=False))

        ed_layout.addWidget(self.header_bar)
        ed_layout.addWidget(self.code_editor)
        ed_layout.addWidget(self.btn_draw_only)

        # --- BOTTOM: Inspector Tabs ---
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

        # Assemble Splitter
        self.v_splitter.addWidget(self.editor_container)
        self.v_splitter.addWidget(self.inspector_tabs)
        self.v_splitter.setSizes([650, 350])

        # 2. Bottom Nav Bar (Loaders and Generator)
        self.footer_bar = self._create_footer_bar()

        # Final Assembly
        self.main_layout.addWidget(self.v_splitter)
        self.main_layout.addWidget(self.footer_bar)

        self.layout.addWidget(self.main_container)

    def _create_footer_bar(self):
        bar = QFrame()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(3, 0, 0, 0)

        self.mode_label = QLabel("SOURCE: TEXT")
        layout.addWidget(self.mode_label)

        self.var_input = QLineEdit("__circuit__")
        self.var_input.setFixedWidth(120)
        self.var_input.setStyleSheet(
            "background: #111; color: #aaa; border: 1px solid #333; padding: 4px;"
        )

        self.btn_to_zx = QPushButton("GENERATE ZX GRAPH →")
        self.btn_to_zx.setStyleSheet(styles.ACTION_BTN)
        self.btn_to_zx.clicked.connect(lambda: self._process_and_emit(switch_pane=True))

        layout.addStretch()
        layout.addWidget(QLabel("Variable:"))
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

    def _handle_selection_sync(self):
        cursor = self.code_editor.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText().strip()
            if text.isidentifier():
                self.var_input.setText(text)

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

    def _process_and_emit(self, switch_pane: bool):
        code = self.code_editor.toPlainText()
        mode = "python" if "PYTHON" in self.mode_label.text() else "qasm"
        task = asyncio.ensure_future(
            self.manager.handle_load_source_circuit(
                code, mode, var_name=self.var_input.text(), switch_to_transform=switch_pane
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @Slot(str, str)
    def update_visuals(self, qasm, ascii_art):  # noqa: D102
        self.ascii_viewer.setPlainText(ascii_art)
        # If qasm contains "ERROR" or "STDOUT", maybe auto-switch to Terminal tab
        if "--- STDOUT ---" in ascii_art or "Error" in ascii_art:
            self.terminal_output.setPlainText(ascii_art)
            self.inspector_tabs.setCurrentIndex(1)  # Auto-focus Terminal on error
        else:
            self.inspector_tabs.setCurrentIndex(0)
