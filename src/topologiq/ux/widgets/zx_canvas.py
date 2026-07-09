"""UX blockgraph canvas.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from __future__ import annotations

import asyncio

import pyzx as zx
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QVBoxLayout, QWidget
from zxlive.app import get_embedded_app

from topologiq.input.zx_manager import AugmentedZXGraph
from topologiq.ux.utils import styles


class ZXCanvas(QWidget):
    """Interactive ZXLive canvas with integrated compilation and branch tracking management."""

    toggle_requested = Signal(str)

    def __init__(self, manager, parent=None):
        """Initialise the unified ZX canvas wrapper."""

        # Init init, init
        super().__init__(parent)
        self.manager = manager
        self.current_aug_zx = None
        self.current_graph_key = "circuit"

        # Match the IDE's main structural margins and spacing
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(5)
        self.setStyleSheet("ZXCanvas { background: #1a1a1a; }")

        # Establish the bounded visual container frame
        self.canvas_frame = QFrame()
        self.canvas_frame.setObjectName("MainCanvasFrame")
        self.canvas_frame.setStyleSheet("""
            #MainCanvasFrame {
                border: 1px solid #222;
                background: #050505;
            }
        """)
        self.frame_layout = QVBoxLayout(self.canvas_frame)
        self.frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_layout.setSpacing(0)

        # Inject core engine components (Populates menus and visual engine)
        self._embed_zxlive_engine()
        self._ghost_parent_window()

        # Add the framed canvas container to the main view
        self.main_layout.addWidget(self.canvas_frame)

        # Append the lower control HUD outside the border frame matching the IDE's footer
        self._setup_control_hud()

        # Focus monitor hook
        self._connect_tab_change_listener()

    def _embed_zxlive_engine(self):
        """Instantiate ZXLive, capture its central core, and dock it natively with localised menus."""

        # Get ZXLive as an embedded app
        self.zxlive_app = get_embedded_app()
        self.zxlive_app.edit_graph(zx.Graph(), "Workspace")

        self.raw_window = self.zxlive_app.main_window
        if self.raw_window is None:
            raise RuntimeError("ZXLive failed to initialise its main_window component.")

        # Force spider labels/IDs to show up by default via QSettings integration
        if hasattr(self.raw_window, "settings"):
            self.raw_window.settings.setValue("show-vertex-indices", True)
            self.raw_window.settings.sync()

        # Extract and localise the native Menu Bar
        self.menu_bar = self.raw_window.menuBar()
        if self.menu_bar:
            self.menu_bar.setParent(self)
            self.menu_bar.setStyleSheet("""
                QMenuBar {
                    background-color: #222222;
                    color: #999999;
                    border-bottom: 1px solid #333333;
                    font-size: 11px;
                }
                QMenuBar::item {
                    background-color: transparent;
                    padding: 5px 12px;
                }
                QMenuBar::item:selected {
                    background-color: #2a2a2a;
                    color: #fbff00;
                }
                QMenu {
                    background-color: #222222;
                    color: #e0e0e0;
                    border: 1px solid #444444;
                }
                QMenu::item {
                    padding: 6px 20px;
                }
                QMenu::item:selected {
                    background-color: #2a2a2a;
                    color: #fbff00;
                }
            """)
            self.frame_layout.addWidget(self.menu_bar)

        # Capture and dock the main canvas view panels directly beneath the local menu bar
        self.zxlive_core_layout = self.raw_window.centralWidget()
        if self.zxlive_core_layout is None:
            raise RuntimeError(
                "Could not resolve the central content layout from the ZXLive frame."
            )

        self.zxlive_core_layout.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #999; background: #222;}
            QTabBar::tab {
                height: 25px;
                margin-top: 10px;
                font-size: 12px;
                font-variant: small-caps;
                background: #3a3a3a;
                color: #f2f3fb;
                padding: 0 14px;
                border: 1px dashed #999;
                border-bottom: 1px solid #333;
            }
            QTabBar::tab:selected {
                height: 28px;
                margin-top: 0;
                background: #222;
                padding: 3px 14px;
                border: 1px solid #999;
                border-bottom: 2px solid #666;
            }
        """)

        self.frame_layout.addWidget(self.zxlive_core_layout)
        self.zxlive_core_layout.show()

    def _ghost_parent_window(self):
        """Render the leftover floating QMainWindow shell invisible."""
        self.raw_window.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.raw_window.setGeometry(-1000, -1000, 1, 1)
        self.raw_window.show()

    def _setup_control_hud(self):
        """Dock the action layout beneath the canvas view frame matching the IDE footer alignment."""
        # Create Hud
        hud_layout = QHBoxLayout()
        hud_layout.setContentsMargins(3, 0, 3, 0)
        hud_layout.setSpacing(10)

        # Pastel-white restoration button overlay for closed IDE frame recovery
        self.btn_open_ide = QPushButton("OPEN IDE")
        self.btn_open_ide.setFixedHeight(32)
        self.btn_open_ide.setStyleSheet("""
            QPushButton {
                background: #f4f4f6;
                color: #222;
                font-weight: bold;
                border-radius: 2px;
                font-size: 11px;
                padding: 0px 15px;
                border: 1px solid #dcdcdf;
            }
            QPushButton:hover { background: #e8e8eb; }
            QPushButton:disabled { background: #333; color: #666; }
        """)
        self.btn_open_ide.clicked.connect(self._handle_open_ide_click)

        # Main direct-action compile switch button
        self.btn_compile_nav = QPushButton("COMPILE →")
        self.btn_compile_nav.setFixedHeight(32)
        self.btn_compile_nav.setStyleSheet(styles.PRIMARY_ACTION_STYLE)
        self.btn_compile_nav.clicked.connect(self._handle_compile_nav_click)

        # Assembled HUD layout without the old staging button asset
        hud_layout.addWidget(self.btn_open_ide)
        hud_layout.addStretch()
        hud_layout.addWidget(self.btn_compile_nav)

        self.main_layout.addLayout(hud_layout)

    def manage_aug_zx(self, aug_zx_graph: AugmentedZXGraph, key: str = "circuit"):
        """Ingest graph from Topologiq's core manager and project it onto ZXLive."""
        try:
            # Cache newly arrived AugZX
            self.current_aug_zx = aug_zx_graph
            self.current_graph_key = key

            # Ensure data is structurally intact and contains a valid PyZX graph
            if self.current_aug_zx and hasattr(self.current_aug_zx, "zx_graph"):
                # Clear and populate fresh layout panels matching active collection
                self._rebuild_canvas_tabs()
                QTimer.singleShot(50, self.centre_graph_view)

            # Communicate success to user
            self.manager.status_changed.emit(f"Viewing interactively using ZX live: {key}")
        except Exception as e:
            # Communicate error to user
            self.manager.status_changed.emit(f"Canvas embedding sync error: {e}")

    def _rebuild_canvas_tabs(self):
        """Clear out stale tabs and reconstruct the visual workspace mapping all collected variants."""
        if not hasattr(self.raw_window, "tab_widget"):
            return

        tw = self.raw_window.tab_widget
        collection = self.manager.get_data("augmented_zx_graph_in") or {}
        if not collection:
            return

        # Temporarily disable layout signals to prevent cascade rendering events
        # or focus shifts while erasing active widgets
        tw.blockSignals(True)
        while tw.count() > 0:
            tw.removeTab(0)
        tw.blockSignals(False)

        # Sorting layout subroutine
        def _get_sort_key(k: str):
            parts = k.split("_")
            if len(parts) > 1 and parts[-1].isdigit():
                return (parts[0], int(parts[-1]))
            return (k, 0)

        target_index = 0
        sorted_items = sorted(collection.items(), key=lambda item: _get_sort_key(item[0]))

        # Feed sorted graph structures back
        for graph_key, aug_graph in sorted_items:
            self.zxlive_app.edit_graph(aug_graph.zx_graph, graph_key)
            if graph_key == self.current_graph_key:
                target_index = tw.count() - 1

        # Restore selection state indices
        if tw.count() > 0:
            tw.blockSignals(True)
            tw.setCurrentIndex(target_index)
            tw.blockSignals(False)
            self.current_graph_key = tw.tabText(target_index)

    def _connect_tab_change_listener(self):
        """Attach a reactive event monitor to fit the viewport instantly whenever a tab gains focus."""
        if hasattr(self.raw_window, "tab_widget"):
            tw = self.raw_window.tab_widget
            tw.currentChanged.connect(lambda _: QTimer.singleShot(30, self.centre_graph_view))

    def resizeEvent(self, event):  # noqa: N802
        """Handle layout resising actions to maintain proper camera centering and toggle visibility."""
        super().resizeEvent(event)

        # Calculate width percentage dynamically within the local frame bounds
        if self.parentWidget():
            parent_w = self.parentWidget().width()
            if parent_w > 0:
                canvas_percentage = self.width() / parent_w

                # Show the button only if the canvas expands to occupy 95% or more of the view space
                if canvas_percentage >= 0.95:
                    self.btn_open_ide.show()
                else:
                    self.btn_open_ide.hide()

        QTimer.singleShot(10, self.centre_graph_view)

    def _handle_open_ide_click(self):
        """Signal the parent load container splitter shell to restore the workspace to 40/60 split ratios."""
        self.toggle_requested.emit("40/60")

    def _handle_stage_to_compile_click(self):
        """Extract the active graph layout and stage it into the compilation control pane."""
        # Query state management engine to pull down all input variant tracks
        collection = self.manager.get_data("augmented_zx_graph_in") or {}
        if not collection:
            self.manager.status_changed.emit("No graphs found in collection to stage.")
            return

        # Fallback default tracking marker
        active_key = self.current_graph_key

        # Pull down precise runtime layout indices from the embedded window instance properties
        if hasattr(self.raw_window, "tab_widget"):
            tw = self.raw_window.tab_widget
            if tw.count() > 0:
                # Extract structural identity tags straight out of tab text
                active_key = tw.tabText(tw.currentIndex())

        # Resolve the active baseline object mapping for verification
        target_aug_zx = collection.get(active_key)
        if not target_aug_zx:
            self.manager.status_changed.emit(
                f"Staging Aborted: Key '{active_key}' not found in data store."
            )
            return

        # Sync state frames and push focus down to the rewrite panels
        self.manager.handle_stage_to_compile(active_key, target_aug_zx)

    def _handle_snapshot_click(self):
        """Extract live layout mutations from the active view pane and commit them as a new circuit_n key."""
        # Fallback to last known tracking key
        active_key = self.current_graph_key

        # Resolve active tab string identity directly from the visual workspace
        if hasattr(self.raw_window, "tab_widget"):
            tw = self.raw_window.tab_widget
            if tw.count() > 0:
                active_key = tw.tabText(tw.currentIndex())

        # Guard against uninitialised workspace states
        if not active_key or active_key == "Workspace":
            self.manager.status_changed.emit("No active canvas tab found to snapshot.")
            return

        # Pull a clean snapshot copy from the underlying ZXLive canvas engine
        live_graph = self._pull_graph_from_zxlive(active_key)
        if not live_graph:
            self.manager.status_changed.emit(
                f"Could not extract a valid live graph instance for '{active_key}'."
            )
            return

        # Determine naming convention prefix based on existing data
        existing_inputs = self.manager.get_data("augmented_zx_graph_in") or {}
        if existing_inputs:
            first_key = sorted(existing_inputs.keys())[0]
            base_prefix = first_key.split("_")[0]
        else:
            base_prefix = "circuit"

        # Calculate next increment offset and update local tracking keys
        next_index = len(existing_inputs)
        next_key = f"{base_prefix}_{next_index}"
        self.current_graph_key = next_key

        # Core task dispatch
        task = asyncio.create_task(self.manager.handle_snapshot_zx_graph(live_graph, next_key))

        # Register task to guard thread safety boundaries
        if hasattr(self.manager, "_tasks"):
            self.manager._tasks.add(task)
            task.add_done_callback(
                lambda t: QTimer.singleShot(0, lambda: self.manager._tasks.discard(t))
            )

    def _handle_compile_nav_click(self):
        """Compare live graph against its tab's baseline. Updates focus or auto-increments the key."""
        # Resolve active canvas identifier tracking token
        active_key = self.current_graph_key
        if hasattr(self.raw_window, "tab_widget"):
            tw = self.raw_window.tab_widget
            if tw.count() > 0:
                active_key = tw.tabText(tw.currentIndex())

        # Fall back to compilation pane if workspace is uninitialised or generic
        if not active_key or active_key == "Workspace":
            self.manager.section_changed.emit("COMPILE")
            return

        # Retrieve a duplicate snapshot of live model
        live_graph = self._pull_graph_from_zxlive(active_key)
        if not live_graph:
            self.manager.section_changed.emit("COMPILE")
            return

        existing_inputs = self.manager.get_data("augmented_zx_graph_in") or {}
        baseline_obj = existing_inputs.get(active_key)

        # Isolate structural edits compared against baseline properties
        is_mutated = True
        if baseline_obj and hasattr(baseline_obj, "zx_graph"):
            g = baseline_obj.zx_graph
            if (
                g.num_vertices() == live_graph.num_vertices()
                and g.num_edges() == live_graph.num_edges()
            ):
                is_mutated = False

        if not is_mutated:
            # Advance view directly utilising current context reference tracking
            self._finalise_compile_navigation(active_key)
            return

        # Handle mutated state tracking increments safely
        base_prefix = active_key.split("_")[0] if "_" in active_key else "circuit"
        next_index = len(existing_inputs)
        new_version_key = f"{base_prefix}_{next_index}"

        # Dispatch snapshot compilation block via the standardised manager task layout
        task = asyncio.create_task(
            self.manager.handle_snapshot_zx_graph(live_graph, new_version_key)
        )

        # Use localised protective set variables to enforce linter compliance safely
        if hasattr(self.manager, "_tasks"):
            self.manager._tasks.add(task)
            task.add_done_callback(self.manager._tasks.discard)

        # Guard thread context boundaries during pane navigation switches
        task.add_done_callback(
            lambda t: QTimer.singleShot(
                0, lambda: self._finalise_compile_navigation(new_version_key)
            )
        )

    def _finalise_compile_navigation(self, target_key: str):
        """Force synchronise the data store registries and execute the COMPILE view setup."""
        data_map = self.manager.get_data("augmented_zx_graph_in") or {}
        target_obj = data_map.get(target_key)

        # Synchronise active frame keys directly across configuration layers safely
        main_win = self.window()
        if hasattr(main_win, "panes") and "COMPILE" in main_win.panes:
            main_win.panes["COMPILE"].active_key = target_key

        self.manager.handle_stage_to_compile(target_key, target_obj)
        self.manager.section_changed.emit("COMPILE")

    def _pull_graph_from_zxlive(self, current_key: str) -> zx.Graph:
        """Extract the current live state of a named graph from the editor space."""
        return self.zxlive_app.get_copy_of_graph(current_key)

    def centre_graph_view(self):
        """Force the embedded engine to recalculate its bounds and centre camera layout."""
        QTimer.singleShot(50, self._execute_safe_centre)

    def _execute_safe_centre(self):
        """Centre graph once layout parameters are calculated."""
        try:
            if not hasattr(self, "raw_window") or not self.raw_window:
                return
            if hasattr(self.raw_window, "tab_widget"):
                tw = self.raw_window.tab_widget
                current_panel = tw.currentWidget()
                if current_panel and hasattr(current_panel, "graph_view"):
                    gv = current_panel.graph_view
                    if hasattr(gv, "zoom_to_fit"):
                        gv.zoom_to_fit()
                    elif hasattr(gv, "fitInView") and gv.scene():
                        gv.fitInView(gv.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
        except Exception as e:
            print(f"Canvas Layout Realignment Alert: {e}")
