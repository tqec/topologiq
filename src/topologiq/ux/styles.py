"""Styling options for the UX.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""
# styles.py

MAIN_WINDOW_STYLE = "background-color: #1a1a1a;"

STATUS_BAR_STYLE = """
    QStatusBar {
        border-top: 1px dotted white;
        background-color: #1a1a1a;
    }
    QLabel {
        color: #007acc;
        font-family: monospace;
        border: none;
    }
"""

SIDEBAR_STYLE = "background-color: #1a1a1a;"

PANE_HEADER_STYLE = "color: #007acc; font-size: 21px; margin: 0px"

NAV_BUTTON_STYLE = """
    QPushButton {
        background-color: #222;
        color: #f2fbf0;
        border-top: 1px dotted white;
        border-left: 1px dotted white;
        font-weight: bold;
        outline: none;
    }
    QPushButton[isLast="true"] {
        border-bottom: 1px dotted white;
    }
    QPushButton:checked {
        background-color: #3d3d3d;
        color: #007acc;
        border-left: 4px solid #007acc;
    }
    QPushButton:hover:!checked {
        background-color: #454545;
    }
"""

TEXT_STYLE_CODE = "font-family: 'Courier New', monospace; background-color: #0f0f0f; color: #dcdcdc; border: 1px solid #333;"

TEXT_STYLE_DRAW_BTN = "font-weight: bold; color: #00ff00;"
TEXT_STYLE_TRANSPILE_BTN = "background-color: #007acc; font-weight: bold; color: white;"
