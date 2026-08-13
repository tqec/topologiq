"""Styling options for the UX.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from PySide6.QtGui import QColor, QPalette

# APP
MAIN_WINDOW_STYLE = "background-color: #1a1a1a;"
STATUS_BAR_STYLE = """
    QStatusBar {
        border-top: 1px solid #333;
        background-color: #0f0f0f; /* Slightly darker than the main window */
    }
    QStatusBar::item { border: none; }
"""
NAV_BUTTON_STYLE = """
    QPushButton {
        background-color: transparent;
        color: #666; /* Dim inactive text */
        border: none;
        padding: 0px 15px;
        font-family: 'Courier New', monospace;
        font-size: 18px;
        outline: none;
    }
    QPushButton:hover {
        color: #bbb; /* Subtle highlight on hover */
        font-weight: bold;
    }
    QPushButton:checked {
        color: #ff0000; /* High-visibility Quantum Red */
        background-color: #1e0000; /* Deep red subtle glow */
        /* Optional: add a tiny underline to ground the bracket */
    }
"""
TOGGLE_BUTTON_STYLE = """
    QPushButton {
        background-color: #f2f3fb;
        color: black;
        border: 1px solid white;
        border-top: 0;
        border-radius: 0 0 4px 4px;
        padding: 4px 12px;
        font-family: mono;
        font-size: 11px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: rgba(60, 60, 60, 255);
        border: 1px solid #666666;
    }
    QPushButton:pressed {
        background-color: #1a1a1a;
    }
    QPushButton:checked {
        background-color: #3d4933; /* Subtitles green to hint at 'Ready' */
        color: #9fe2bf;
        border: 1px solid #556644;
    }
"""
STATUS_LABEL_STYLE = "color: #007acc; font-family: monospace; font-size: 10px;"
TEXT_STYLE_TRANSPILE_COMPILE = "background-color: #007acc; font-weight: bold; color: white;"

BTN_BASE = (
    "border: 1px solid ridge inset; border-color: #999; padding: 3px 7px;  border-radius: 3px;"
)
CLOSE_BTN = f"{BTN_BASE} background-color: #ddd; color: #000;"
ACTION_BTN = f"{BTN_BASE} color: #fff; background-color: #2a2a2a;"


GHOST_COLOR = "#333"
x = "#ffff77"
COMMAND_RAIL_STYLE = "background-color: #1a1a1a; border-right: 1px solid #333;"
# The IDE tab (Left side) sticks to its right edge
LEFT_TAB_STYLE = """
    QPushButton {
        background-color: #f1c40f; color: #000; font-weight: bold; font-size: 11px;
        border: 2px solid #000;
        border-right: none; /* Stick to IDE */
        border-radius: 8px 0px 0px 8px;
        padding: 5px 0px;
    }
    QPushButton:hover { background-color: #f3d147; }
"""
STYLE_CLOSE_BTN = """
    QPushButton {
        border: 1px solid #666;
        border-radius: 0px;
        font-weight: bold;
        padding: 3px 7px;
    }
    QPushButton:hover { background: #ff3333; }
"""

# The Canvas tab (Right side) sticks to its left edge
RIGHT_TAB_STYLE = """
    QPushButton {
        background-color: #f1c40f; color: #000; font-weight: bold; font-size: 11px;
        border: 2px solid #000;
        border-left: none; /* Stick to Canvas */
        border-radius: 0px 8px 8px 0px;
        padding: 5px 0px;
    }
    QPushButton:hover { background-color: #f3d147; }
"""
CENTER_TAB_STYLE = """
    QPushButton {
        background-color: #f1c40f;
        color: #000;
        font-weight: bold;
        font-size: 11px;
        border: 2px solid #000;
        border-radius: 4px;
        padding: 5px 0px;
    }
    QPushButton:hover { background-color: #f3d147; }
"""
# IDE Button: Rounded on the left, flat on the right (points to IDE)
IDE_PILL_STYLE = """
    QPushButton {
        background-color: #f1c40f; color: #000; font-weight: bold; font-size: 11px;
        border: 2px solid #666;
        border-radius: 12px 0px 0px 12px;
    }
    QPushButton:hover { background-color: #f3d147; }
"""

# Canvas Button: Flat on the left, rounded on the right (points to Canvas)
CANVAS_PILL_STYLE = """
    QPushButton {
        background-color: #f1c40f; color: #000; font-weight: bold; font-size: 11px;
        border: 2px solid #666;
        border-radius: 0px 12px 12px 0px;
    }
    QPushButton:hover { background-color: #f3d147; }
"""

# DESIGN PANE
TEXT_STYLE_CODE = "font-family: 'Courier New', monospace; background-color: #0f0f0f; color: #dcdcdc; border: 1px solid #999; font-size: 12px;"
HUD_FRAME_STYLE = "background: rgba(30, 30, 30, 200); border-radius: 6px; border: 1px solid #444;"
HUD_BUTTON_STYLE = (
    "color: #bbb; border: none; padding: 4px 8px; font-size: 10px; font-weight: bold;"
)
HUD_ACTION_BUTTON_STYLE = (
    "background: #2ecc71; color: white; border-radius: 4px; padding: 4px 12px; font-weight: bold;"
)
HUD_ROUND_BUTTON_STYLE = "background: rgba(50, 50, 50, 200); color: white; border-radius: 17px; font-size: 16px; border: 1px solid #666;"

# TRANSFORM PANE
TRANSFORM_PANE_BG = "background-color: #0f0f0f;"
CANVAS_FRAME_STYLE = "background-color: #121212; border: 1px solid #333; border-radius: 4px;"
SECTION_LABEL_STYLE = "color: #ffffff; font-weight: bold; font-size: 11px; letter-spacing: 1.2px; background: transparent;"
STATUS_BADGE_UNVERIFIED = "padding: 2px 10px; background: #992222; border-radius: 10px; font-size: 10px; border: 1px solid black;"
STATUS_BADGE_VERIFIED = "background: #1a3d1a; color: #99ff99; border-radius: 10px; font-size: 10px;"
CONTROL_BAR_STYLE = "background-color: #1a1a1a; border-top: 1px solid #333;"
PRIMARY_ACTION_STYLE = "background-color: #1fffb4; color: black; border: 1px solid white; font-weight: bold; padding: 5px 15px;"
x = "#1fffb4"
PILL_BTN_BASE = (
    "padding: 4px 12px; font-size: 10px; font-weight: bold; border: 1px solid; border-radius: 0px;"
)

# Variants
PILL_BTN_PYZX = f"{PILL_BTN_BASE} background-color: #333; border-color: #666; color: #aaccff;"
PILL_BTN_REDUCE = f"{PILL_BTN_BASE} background-color: #222; border-color: #444; color: #aaa;"


MAIN_SPLITTER_STYLE = """ QSplitter#DesignMainSplitter::handle { border-left: 1px solid #333; margin: 7px 0; } QSplitter#DesignMainSplitter::handle:hover { background-color: #4d8dc1; } QSplitter#DesignMainSplitter::handle:pressed { background-color:#1e92df; } """


# ---------------------------------------------------------------------------
# GLOBAL DARK THEME
# ---------------------------------------------------------------------------
# A palette-driven theme applied at the QApplication level. It fills the gaps
# left by the per-widget stylesheets above (the main window background, default
# dialogs, combobox dropdowns, scrollbars, tooltips, and the embedded ZXLive
# window) so nothing falls back to the light system theme. Per-widget
# stylesheets still win by specificity, so existing visuals are preserved.
_DARK_PALETTE = {
    QPalette.Window: "#1a1a1a",
    QPalette.WindowText: "#dcdcdc",
    QPalette.Base: "#0f0f0f",
    QPalette.AlternateBase: "#1a1a1a",
    QPalette.ToolTipBase: "#2a2a2a",
    QPalette.ToolTipText: "#dcdcdc",
    QPalette.PlaceholderText: "#666666",
    QPalette.Text: "#dcdcdc",
    QPalette.Button: "#2a2a2a",
    QPalette.ButtonText: "#dcdcdc",
    QPalette.BrightText: "#ff0000",
    QPalette.Highlight: "#007acc",  # Accent shared with STATUS_LABEL_STYLE
    QPalette.HighlightedText: "#ffffff",
    QPalette.Link: "#007acc",
    QPalette.LinkVisited: "#007acc",
}

# Polish for the few controls Fusion renders with poor contrast at night.
# Selectors are limited to controls that no per-widget stylesheet targets, so
# they cannot collide with existing styles.
_DARK_GLOBAL_QSS = """
QToolTip {
    background-color: #2a2a2a;
    color: #dcdcdc;
    border: 1px solid #444;
    padding: 3px 5px;
}
QScrollBar:vertical {
    background: transparent;
    width: 12px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #3a3a3a;
    min-height: 24px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover { background: #007acc; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal {
    background: transparent;
    height: 12px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #3a3a3a;
    min-width: 24px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover { background: #007acc; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
QMenu {
    background-color: #1a1a1a;
    border: 1px solid #444;
    color: #dcdcdc;
    padding: 4px;
}
QMenu::item { padding: 4px 16px; }
QMenu::item:selected { background-color: #007acc; }
QMenu::separator { height: 1px; background: #333; margin: 4px 8px; }
QComboBox QAbstractItemView {
    background-color: #1a1a1a;
    border: 1px solid #444;
    color: #dcdcdc;
    selection-background-color: #007acc;
    selection-color: #ffffff;
    outline: 0;
}
"""


def apply_dark_theme(app):
    """Apply a consistent dark theme to the application.

    Uses the palette-driven ``Fusion`` style — uniform across platforms, unlike
    the native Windows/macOS styles which ignore much of the palette — paired
    with a dark :class:`QPalette` and a small global stylesheet. Call once on
    the :class:`QApplication` before any widgets are shown.
    """
    app.setStyle("Fusion")

    palette = QPalette()
    for role, hex_color in _DARK_PALETTE.items():
        palette.setColor(role, QColor(hex_color))

    # Dim disabled controls so they read as inactive
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#666666"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#666666"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#666666"))

    app.setPalette(palette)
    app.setStyleSheet(_DARK_GLOBAL_QSS)
