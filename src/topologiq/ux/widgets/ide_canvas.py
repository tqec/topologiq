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
    """Self-contained IDE canvas."""

    toggle_requested = Signal(str)

    def __init__(self, manager, parent=None):
        """Initialise IDE canvas."""
        super().__init__(parent)
        self.manager = manager
        self.current_file_path = None
        self.highlighter = None
        self._tasks = set()
        self.setup_ui()

    def setup_ui(self):
        """Define the IDE layout."""

        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(5)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

        # Vertical splitter: Editor / Inspector
        self.v_splitter = QSplitter(Qt.Vertical)
        self.v_splitter.setObjectName("IDEVerticalSplitter")
        self.v_splitter.setStyleSheet("""
            QSplitter#IDEVerticalSplitter::handle {
                height: 4px;
                border-top: 1px solid #333;
                padding-bottom: 3px;
            }
            QSplitter#IDEVerticalSplitter::handle:hover {
                height: 1px;
                border-top: 4px solid #4d8dc1;
            }
            QSplitter#IDEVerticalSplitter::handle:pressed {
                height: 1px;
                border-top: 4px solid #1e92df;
            }
        """)

        # Editor (Top)
        self.editor_container = QFrame()
        ed_layout = QVBoxLayout(self.editor_container)
        ed_layout.setContentsMargins(0, 0, 0, 5)

        self.header_bar = self._create_header_bar()
        self.code_editor = QPlainTextEdit()
        self.code_editor.setPlaceholderText("Load Python or QASM file...")
        self.code_editor.setStyleSheet(styles.TEXT_STYLE_CODE)
        self.code_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.code_editor.selectionChanged.connect(self._handle_selection_sync)

        self.var_row = QFrame()
        var_layout = QHBoxLayout(self.var_row)

        var_label = QLabel("Target circuit:")
        var_label.setStyleSheet("color: #999; font-size: 10px; font-weight: bold;")

        self.var_input = QLineEdit("circuit")
        self.var_input.setFixedWidth(150)
        self.var_input.setStyleSheet(
            "background: #121212; color: #fbff00; border: 1px solid #444; font-weight: bold;"
        )

        self.btn_draw_only = QPushButton("↓↓↓ DRAW ASCII ↓↓↓")
        self.btn_draw_only.clicked.connect(lambda: self._process_and_emit(switch_pane=False))

        var_layout.addWidget(var_label)
        var_layout.addWidget(self.var_input)
        var_layout.addStretch()
        var_layout.addWidget(self.btn_draw_only)

        ed_layout.addWidget(self.header_bar)
        ed_layout.addWidget(self.code_editor)
        ed_layout.addWidget(self.var_row)

        # Inspector (Bottom)
        self.inspector_tabs = QTabWidget()
        self.inspector_tabs.setStyleSheet(
            "QTabBar::tab { height: 25px; font-size: 10px; background: #1a1a1a; color: #999; padding: 0 15px; } "
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

        # Footer
        self.footer_bar = self._create_footer_bar()

        self.layout.addWidget(self.v_splitter)
        self.layout.addWidget(self.footer_bar)
        self.setMinimumWidth(0)

    def _create_header_bar(self):
        """Create IDE's editor top menu bar."""

        # Layout
        bar = QFrame()
        bar.setStyleSheet("background: #222;")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)  # Left margin for title/file buttons
        layout.setSpacing(5)

        # File Actions
        self.btn_load = QPushButton("📁")
        self.btn_save = QPushButton("💾")
        for btn in [self.btn_load, self.btn_save]:
            btn.setStyleSheet(styles.ACTION_BTN + "font-size: 21px;")
            layout.addWidget(btn)
        self.btn_load.clicked.connect(self._handle_open_file)
        self.btn_save.clicked.connect(self._handle_save_file)

        layout.addStretch()

        # Layout controls
        self.toggle_buttons = create_split_controls(
            self, ["◫", "□", "✕"], self.toggle_requested.emit
        )
        layout.addWidget(self.toggle_buttons)

        return bar

    def _create_footer_bar(self):
        """Create IDE's bottom action bar."""

        # Layout
        bar = QFrame()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(3, 0, 0, 0)

        # Label (not shown but functionally required: buttons get content-type from it)
        self.mode_label = QLabel("SOURCE: TEXT")

        # ZX graph generation
        self.btn_to_zx = QPushButton("GENERATE ZX GRAPH →")
        self.btn_to_zx.setStyleSheet(styles.PRIMARY_ACTION_STYLE)
        self.btn_to_zx.clicked.connect(lambda: self._process_and_emit(switch_pane=True))
        layout.addStretch()

        # Add to layout
        layout.addWidget(self.btn_to_zx)

        return bar

    def _handle_open_file(self):
        """Load file by extension (.py or .qasm)."""

        # Open dialogue
        file_filter = "Quantum Source (*.py *.qasm);;Python (*.py);;OpenQASM (*.qasm)"
        path, _ = QFileDialog.getOpenFileName(self, "Open Circuit Source", "", file_filter)

        # Handle file
        if path:
            # Map mode
            self.current_file_path = Path(path)
            ext = self.current_file_path.suffix.lower()
            mode = "python" if ext == ".py" else "qasm"

            # Set mode
            self.code_editor.setPlainText(self.current_file_path.read_text())
            self.mode_label.setText(f"SOURCE: {mode.upper()}")

            # Code highlights
            if self.highlighter:
                self.highlighter.setDocument(None)
            self.highlighter = PygmentsHighlighter(self.code_editor.document(), mode)

    def _handle_save_file(self):
        """Save current editor content to disk."""

        # Path
        path = self.current_file_path

        # If no file is open, Save As
        if not path:
            mode = "python" if "PYTHON" in self.mode_label.text() else "qasm"
            ext = "Python (*.py)" if mode == "python" else "OpenQASM (*.qasm)"
            path_str, _ = QFileDialog.getSaveFileName(self, "Save File", "", ext)
            if not path_str:
                return
            path = Path(path_str)
            self.current_file_path = path

        # Try write
        try:
            path.write_text(self.code_editor.toPlainText())
            self.window().status_bar.showMessage(f"Saved: {path.name}", 3000)
        except Exception as e:
            self.window().status_bar.showMessage(f"Save failed: {e}", 5000)

    def _handle_selection_sync(self):
        """Sync variable name highligth/selection."""
        cursor = self.code_editor.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText().strip()
            if text.isidentifier():
                self.var_input.setText(text)

    def _connect_internal_signals(self):
        """Link local IDE buttons to Manager."""
        # Draw button (no pane switch)
        self.btn_draw_only.clicked.connect(lambda: self._process_and_emit(switch_pane=False))
        # Generate ZX (switch pane)
        self.btn_to_zx.clicked.connect(lambda: self._process_and_emit(switch_pane=True))

    def _process_and_emit(self, switch_pane: bool):
        """Process circuit and emit results."""
        code = self.code_editor.toPlainText()
        mode = "python" if "PYTHON" in self.mode_label.text().upper() else "qasm"

        # Get key (circuit name)
        circuit_key = self.var_input.text().strip()
        if not circuit_key:
            circuit_key = "untitled_circuit"
            self.var_input.setText(circuit_key)

        # Trigger Manager
        task = asyncio.ensure_future(
            self.manager.handle_load_source_circuit(
                source_design=code, mode=mode, var_name=circuit_key, switch_to_transform=switch_pane
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
