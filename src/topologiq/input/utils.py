"""Quick utils to assist input."""

from enum import Enum


class ZXTypes(int, Enum):
    """ZX vertex type conversions to standardise visualisations."""

    BOUNDARY = 0
    X = 2
    Z = 1

    @classmethod
    def from_str(cls, name: str) -> int:
        """Convert a string like 'Z' or 'x' into the PyZX integer type."""
        try:
            # Normalize to uppercase to match Enum keys
            return cls[name.upper()].value
        except KeyError:
            return cls.BOUNDARY.value # Safe default

class ZXEdgeTypes(int, Enum):
    """ZX edge type conversions to standardise visualisations."""

    SIMPLE = 1
    HADAMARD = 2

    @classmethod
    def from_str(cls, name: str) -> int:
        """Convert a string like 'Z' or 'x' into the PyZX integer type."""
        try:
            # Normalize to uppercase to match Enum keys
            return cls[name.upper()].value
        except KeyError:
            return cls.SIMPLE.value # Safe default


class ZXColors(str, Enum):
    """Colour palette to standardise visualisations."""

    X = "#d7a4a1"
    Y = "#7fff00"
    Z = "#b9cdff"
    P = "#777777"
    HADAMARD = "#ffff00"
    BOUNDARY = "#777777"
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

