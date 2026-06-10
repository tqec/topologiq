"""Classes for key objects used in Topologiq.

Usage:
    Call any required class from a separate script.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from topologiq.kwargs import BEAMS_SHORT_LEN
from topologiq.utils.misc import get_manhattan

# Types & class for input ZX graph
GraphNode = tuple[int, str]
GraphEdge = tuple[tuple[int, int], str]


class SimpleDictGraph(TypedDict):
    """A simple graph composed of nodes and edges."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]


# Type & classes needed to create, store, and manage beams
StandardCoord = tuple[int, int, int]
StandardBlock = tuple[StandardCoord, str]
StandardBeam = list[StandardCoord]


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
        if not self.z.contains(z):
            return False
        if not self.y.contains(y):
            return False
        if not self.x.contains(x):
            return False
        return True

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


# Edge path class with in-built value-function to enable path comparisons
@dataclass(order=True)
class PathBetweenNodes:
    """A 3D path between the cubes corresponding to two nodes/spiders in the input ZX graph."""

    tgt_coords: StandardCoord
    tgt_kind: str
    tgt_beams: CubeBeams
    tgt_beams_short: CubeBeams
    coords_in_path: list[StandardCoord]
    all_nodes_in_path: list[StandardBlock]
    beams_broken_by_path: int
    len_of_path: int
    tgt_unobstr_exit_n: int

    def weighed_value(self, **kwargs) -> int:
        """Return the weighed value of a given path.

        This function returns the weighed value of a given PathBetweenNodes,
        which can be used for comparing many paths.

        Args:
            **kwargs: Only relevant kwargs listed below.
                weights: A tuple (int, int) of weights used to pick the best of several paths when there are several valid alternatives.

        Returns:
            (int): The weighed value of a path

        """

        path_len_hp, beams_broken_hp = kwargs["weights"]

        return self.len_of_path * path_len_hp + self.beams_broken_by_path * beams_broken_hp


# Misc classes
class Colors:
    """Colours to use in printouts."""

    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"
