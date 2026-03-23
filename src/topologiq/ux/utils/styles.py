"""Styling options for the UX.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

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
        font-size: 14px;
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
ACTION_BTN = "color: #ff0000; background-color: #1e0000; border: 1px solid ridge inset; border-color: #343434; border-radius: 3px; padding: 3px 7px;"
GHOST_COLOR = "#333"

# DESIGN PANE
TEXT_STYLE_CODE = "font-family: 'Courier New', monospace; background-color: #0f0f0f; color: #dcdcdc; border: 1px solid #333;"
HUD_FRAME_STYLE = "background: rgba(30, 30, 30, 200); border-radius: 6px; border: 1px solid #444;"
HUD_BUTTON_STYLE = "color: #bbb; border: none; padding: 4px 8px; font-size: 10px; font-weight: bold;"
HUD_ACTION_BUTTON_STYLE = "background: #2ecc71; color: white; border-radius: 4px; padding: 4px 12px; font-weight: bold;"
HUD_ROUND_BUTTON_STYLE = "background: rgba(50, 50, 50, 200); color: white; border-radius: 17px; font-size: 16px; border: 1px solid #666;"

# TRANSFORM PANE
TRANSFORM_PANE_BG = "background-color: #0f0f0f;"
CANVAS_FRAME_STYLE = "background-color: #121212; border: 1px solid #333; border-radius: 4px;"
SECTION_LABEL_STYLE = "color: #ffffff; font-weight: bold; font-size: 11px; letter-spacing: 1.2px; background: transparent;"
STATUS_BADGE_UNVERIFIED = "padding: 2px 10px; background: #992222; border-radius: 10px; font-size: 10px; border: 1px solid black;"
STATUS_BADGE_VERIFIED = "background: #1a3d1a; color: #99ff99; border-radius: 10px; font-size: 10px;"
CONTROL_BAR_STYLE = "background-color: #1a1a1a; border-top: 1px solid #333;"
PRIMARY_ACTION_STYLE = (
    "background-color: #2e5a2e; color: white; font-weight: bold; padding: 5px 15px;"
)

PILL_BTN_BASE = (
    "padding: 4px 12px; border-radius: 12px; font-size: 10px; font-weight: bold; border: 1px solid;"
)

# Variants
PILL_BTN_PYZX = f"{PILL_BTN_BASE} background-color: #1a1a2e; border-color: #3e3e5e; color: #aaccff;"
PILL_BTN_REDUCE = f"{PILL_BTN_BASE} background-color: #222; border-color: #444; color: #aaa;"


