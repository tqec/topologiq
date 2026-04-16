"""Beams and related classes used across graph_manager and pathfinder.

Usage:
    Call any required class from a separate script.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topologiq.core.pathfinder.utils import get_manhattan
from topologiq.kwargs import BEAMS_SHORT_LEN
from topologiq.utils.classes import StandardCoord


@dataclass
class BeamAxisComponent:
    """Class representing the beam coordinates for any given axis.

    Attributes:
        start: The starting point for the segment.
        end: The end point for the segment (== start if segment is a point).
        direction: Whether segment grows towards the positive or negative end of its axis.

    """

    start: int | float = -np.inf
    end: int | float = np.inf
    direction: int = 0 if start == end else 1 if end > start else -1

    def __hash__(self) -> int:
        """Return start and end for hashing."""
        return hash((self.start, self.end, self.direction))

    def __eq__(self, other: object) -> bool:
        """Check equality against any other segments."""
        return (
            isinstance(other, BeamAxisComponent)
            and self.start == other.start
            and self.end == other.end
        )

    def __str__(self) -> str:
        """Return a readable representation."""
        return f"[{self.start} => {self.end})"

    def contains(self, point: int) -> bool:
        """Check if a given point is contained in the segment."""
        if self.direction == 0 and (self.start == point == self.end):
            return True
        if self.direction == 1 and (self.start < point <= self.end):
            return True
        if self.direction == -1 and (self.start > point >= self.end):
            return True

        return False

    def to_array(self, len_of_materialised_beam: int) -> list[int] | None:
        """Convert segment into an array of arbitrary length."""
        if self.direction != 0:
            return [self.start + i * self.direction for i in range(len_of_materialised_beam)]

    def get_length(self) -> int:
        """Get the length of the beam."""
        return abs(self.start - self.end)


@dataclass
class SingleBeam:
    """Class representing a single beam.

    Attributes:
        x: The beam for the x-axis (a point if x-axis has no beam).
        y: The beam for the y-axis (a point if y-axis has no beam).
        z: The beam for the y-axis (a point if z-axis has no beam).

    """

    x: BeamAxisComponent
    y: BeamAxisComponent
    z: BeamAxisComponent

    def __post_init__(self) -> None:
        """Ensure beam runs only along a single dimension."""
        if (abs(self.x.direction) + abs(self.y.direction) + abs(self.z.direction)) != 1:
            raise ValueError("Malformed beam. Beam must run only along a single dimension.")

    def __hash__(self) -> int:
        """Return start and end for hashing."""
        return hash(self.coords)

    def __eq__(self, other: object) -> bool:
        """Check equality against any other segments."""
        return isinstance(other, SingleBeam) and self.coords == other.coords

    def __str__(self) -> str:
        """Return a readable representation."""
        return f"({self.x!s}, {self.y!s}, {self.z!s})"

    def coords(self) -> StandardCoord:
        """Return the beam coordinates across all axes."""
        return self.x.start, self.y.start, self.z.start

    def direction(self) -> StandardCoord:
        """Return te beam direction as a coordinate tuple."""
        return self.x.direction, self.y.direction, self.z.direction

    def contains(self, coords_to_check: StandardCoord) -> bool:
        """Check if beam contains a given coordinate."""
        x, y, z = coords_to_check
        return self.x.contains(x) and self.y.contains(y) and self.z.contains(z)

    def intersects(self, other: SingleBeam, short_beams: bool = True) -> bool:
        """Check if two beams intersect one another."""

        # Get source coords for both beams
        p1 = self.coords()
        p2 = other.coords()

        # If checking on short mode,
        # exit if beams' sources are further than LEN_SHORT_BEAMS
        if short_beams and get_manhattan(p1, p2) > BEAMS_SHORT_LEN:
            return False

        # Check if beams are parallel or orthogonal
        # No clashes possible if beams are parallel
        d1 = self.direction()
        d2 = other.direction()
        orientation = np.dot(d1, d2)

        if orientation != 0:
            return False

        # Evaluate clash if beams are orthogonal
        # Source of the other beam must be in the positive quadrant of the span of {d1, -d2}
        # Sigma is the position of the source of the other beam relative to the source of this beam
        sigma = np.subtract(p2, p1)
        basis = np.subtract(d1, d2)

        return np.all((sigma == 0) | (np.sign(sigma) == np.sign(basis)))


CubeBeams = list[SingleBeam]
