"""UX syntax highlighting utils.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

from pygments import lex, lexers, styles
from PySide6.QtGui import QColor, QSyntaxHighlighter, QTextCharFormat


class PygmentsHighlighter(QSyntaxHighlighter):
    """Syntax highlighter based on Pygments."""

    def __init__(self, parent, lang="python"):
        """Initialise syntax highlighter."""
        super().__init__(parent)
        self._mapping = {}

        # Safely get lexer; default to Text if lang is unknown
        try:
            self._lexer = lexers.get_lexer_by_name(lang)
        except Exception as _:
            self._lexer = lexers.get_lexer_by_name("text")

        self._style = styles.get_style_by_name("monokai")

    def highlightBlock(self, text):  # noqa: N802
        """Process text block and apply Pygments tokens."""
        if not hasattr(self, "_lexer") or not text:
            return

        tokens = lex(text, self._lexer)
        current_index = 0
        for ttype, value in tokens:
            length = len(value)
            fmt = self._get_format(ttype)
            if fmt:
                self.setFormat(current_index, length, fmt)
            current_index += length

    def _get_format(self, ttype):
        """Convert a Pygments token type to a QTextCharFormat."""
        if ttype in self._mapping:
            return self._mapping[ttype]

        style_def = self._style.style_for_token(ttype)
        if not style_def:
            return None

        fmt = QTextCharFormat()
        if style_def["color"]:
            fmt.setForeground(QColor(f"#{style_def['color']}"))
        if style_def["bold"]:
            fmt.setFontWeight(700)

        self._mapping[ttype] = fmt
        return fmt
