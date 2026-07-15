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

import pyzx as zx
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

        # Init init, init
        super().__init__(parent)

        # Set manager
        self.manager = manager

        # Trackers
        self.current_file_path = None
        self.highlighter = None
        self._tasks = set()

        # Call layout
        self.setup_ui()

    def setup_ui(self):
        """Define the IDE layout."""

        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(5)
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
        self.btn_draw_only.clicked.connect(lambda: self._process_and_emit(draw_only=True))

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

        self.main_layout.addWidget(self.v_splitter)
        self.main_layout.addWidget(self.footer_bar)
        self.setMinimumWidth(0)

    def _create_header_bar(self):
        """Create IDE's editor top menu bar."""

        # Layout
        bar = QFrame()
        bar.setStyleSheet("background: #222;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
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
        layout.setContentsMargins(3, 0, 3, 0)

        # PyZX & qBraid attribution
        self.lbl_attribution = QLabel(
            'IDE canvas powered by <a href="https://github.com/zxcalc/pyzx" style="color: #bee0ff; text-decoration: none;">PyZX</a> '
            '&amp; <a href="https://github.com/qBraid/qBraid" style="color: #bee0ff; text-decoration: none;">qBraid</a>'
        )
        self.lbl_attribution.setOpenExternalLinks(True)
        self.lbl_attribution.setStyleSheet("""
            QLabel {
                color: #999;
                font-size: 11px;
                padding-left: 2px;
            }
        """)

        # Label (not shown but functionally required: buttons get content-type from it)
        self.mode_label = QLabel("SOURCE: TEXT")

        # ZX graph generation
        self.btn_to_zx = QPushButton("STAGE TO CANVAS →")
        self.btn_to_zx.setStyleSheet(styles.PRIMARY_ACTION_STYLE)
        self.btn_to_zx.clicked.connect(lambda: self._process_and_emit(draw_only=False))

        # Add to layout
        layout.addWidget(self.lbl_attribution)
        layout.addStretch()
        layout.addWidget(self.btn_to_zx)

        return bar

    def _connect_internal_signals(self):
        """Link local IDE buttons to Manager."""
        # Draw button (no pane switch)
        self.btn_draw_only.clicked.connect(lambda: self._process_and_emit(draw_only=True))
        # Generate ZX (switch pane)
        self.btn_to_zx.clicked.connect(lambda: self._process_and_emit(draw_only=False))

    def _handle_selection_sync(self):
        """Sync variable name highligth/selection."""
        cursor = self.code_editor.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText().strip()
            if text.isidentifier():
                self.var_input.setText(text)

    def _handle_open_file(self):
        """Load file by extension (.py, .qasm, .json, or .zxg)."""
        # Define OS file system selection filters covering standard formats
        file_filter = "Quantum Source (*.py *.qasm *.json *.zxg);;Python (*.py);;OpenQASM (*.qasm);;PyZX Graph (*.json *.zxg)"
        path, _ = QFileDialog.getOpenFileName(self, "Open Circuit Source", "", file_filter)

        if path:
            self.current_file_path = Path(path)
            ext = self.current_file_path.suffix.lower()

            # Native PyZX formats bypass qBraid transpilation
            if ext in (".json", ".zxg"):
                try:
                    # Ingest raw string
                    raw_content = self.current_file_path.read_text(encoding="utf-8")
                    g = zx.Graph.from_json(raw_content)
                    target_key = self.current_file_path.stem

                    # Query application layout hierarchy for ZXCanvas
                    top_window = self.window()
                    canvas = top_window.findChild(QWidget, "ZXCanvas") if top_window else None
                    if not canvas and top_window:
                        for child in top_window.findChildren(QWidget):
                            if child.__class__.__name__ == "ZXCanvas":
                                canvas = child
                                break

                    # Inject graph directly into canvas if engine layers are online
                    if canvas and hasattr(canvas, "zxlive_app"):
                        canvas.zxlive_app.edit_graph(g, target_key)
                        canvas.current_graph_key = target_key
                        canvas.centre_graph_view()

                        # Evict initial generic "Workspace" tab if present
                        if hasattr(canvas.raw_window, "tab_widget"):
                            tw = canvas.raw_window.tab_widget
                            for i in range(tw.count()):
                                if tw.tabText(i) == "Workspace":
                                    tw.removeTab(i)
                                    break

                        # Update IDE editor metadata to show static JSON graph layout
                        self.code_editor.setPlainText(
                            f"// Successfully loaded graph tab: {target_key}"
                        )
                        self.mode_label.setText("SOURCE: JSON")
                        self.var_input.setText(target_key)

                        # Strip syntax highlighter (editing is disabled for static JSON)
                        if self.highlighter:
                            self.highlighter.setDocument(None)
                    else:
                        raise RuntimeError(
                            "Global application widget scan failed to locate ZXCanvas engine instance."
                        )

                except Exception as e:
                    # Reset UI indicators and print traceback natively into the code panel
                    self.code_editor.setPlainText(f"// GRAPH STRUCTURAL LOAD FAILURE:\n// {e!s}")
                    self.mode_label.setText("SOURCE: ERROR")
                    if self.highlighter:
                        self.highlighter.setDocument(None)
                    print(f"Direct Ingestion Error: {e}")

            # Programmatic scripts or input files (Python/OpenQASM)
            else:
                # Get raw content
                raw_content = self.current_file_path.read_text(encoding="utf-8")
                self.code_editor.setPlainText(raw_content)

                # Determine context mode
                mode = "python" if ext == ".py" else "qasm"

                # Drop old highlighter instance and attach new tokeniser parser
                self.mode_label.setText(f"SOURCE: {mode.upper()}")
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

    def _handle_direct_json_load(self):
        """Execute a direct PyZX JSON parse bypassing the circuit compilation layers."""
        # Get JSON data
        json_data = self.code_editor.toPlainText().strip()
        graph_key = self.var_input.text().strip()
        if not graph_key:
            graph_key = "imported_json_graph"
            self.var_input.setText(graph_key)

        # Trigger native execution loop asynchronously inside the Manager layer
        task = asyncio.ensure_future(
            self.manager.handle_load_json_graph(json_str=json_data, graph_key=graph_key)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _process_and_emit(self, draw_only: bool):
        """Process circuit or raw JSON input and emit results."""
        # Query active header state tracking strings to determine underlying format types
        mode_text = self.mode_label.text().upper()

        # JSON files bypass text extraction loops completely.
        if "JSON" in mode_text:
            # ASCII layout strings are text-circuit specific
            if draw_only:
                self.manager.status_changed.emit(
                    "ASCII visualisation is unavailable for raw JSON structures."
                )
                return
            # Pass directly to JSON ingest workflows
            self._handle_direct_json_load()
            return

        # Fetch contents of the main viewport text buffer frame
        code = self.code_editor.toPlainText()
        mode = "python" if "PYTHON" in mode_text else "qasm"

        # Sanitise text tokens inside target circuit input variables
        circuit_key = self.var_input.text().strip()
        if not circuit_key:
            circuit_key = "untitled_circuit"
            self.var_input.setText(circuit_key)

        # Offload text conversions to async engine loop
        task = asyncio.ensure_future(
            self.manager.handle_load_source_circuit(
                source_circuit=code,
                mode=mode,
                var_name=circuit_key,
                draw_only=draw_only,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
