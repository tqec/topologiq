"""Quick utils to assist input."""

from enum import Enum


class ZXTypes(int, Enum):
    """Colour palette to standardise visualisations."""

    BOUNDARY = 0
    X = 2
    Z = 1
    SIMPLE = 1
    HADAMARD = 2


class ZXColors(str, Enum):
    """Colour palette to standardise visualisations."""

    X = "#d7a4a1"
    Y = "#7fff00"
    Z = "#b9cdff"
    P = "#333333"
    HADAMARD = "#ffff00"
    BOUNDARY = "#000000"
    SIMPLE = "#000000"

    @classmethod
    def lookup(cls, char: str) -> str:
        """Get standardised HEX colours for an arbitrary SpiderBlock.

        Args:
            char: A character, typically signifying a zx_type or basis.

        Returns:
            zx_color: A colour HEX corresponding to the character.

        """

        try:
            return cls[char.upper()]
        except (KeyError, AttributeError):
            return cls.SIMPLE

