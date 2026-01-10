"""Classes for key objects used in Topologiq.

Usage:
    Call any required class from a separate script.

"""

from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

import numpy as np

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


class Axis(Enum):
    """Class representing a 3D axis."""

    X = (1, 0, 0)
    Y = (0, 1, 0)
    Z = (0, 0, 1)


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
        return hash((self.start, self.end))

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

    def origin(self) -> int:
        """Return the origin coordinates for the axis component."""
        return int(self.start)

    def contains(self, point: int) -> bool:
        """Check if a given point is contained in the segment."""

        if self.direction == 0 and (self.start == point == self.end):
            return True

        if self.direction == 1 and (self.start < point < self.end):
            return True

        if self.direction == -1 and (self.start > point > self.end):
            return True

        return False

    def is_point(self) -> bool:
        """Check if the segment takes only a single point/coordinate."""
        return self.direction


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

    def coords(self, c: int) -> tuple[BeamAxisComponent, BeamAxisComponent, BeamAxisComponent]:
        """Return the coords for any given point of the beam."""
        coords = (
            self.x.start + (c * self.x.direction),
            self.y.start + (c * self.y.direction),
            self.z.start + (c * self.z.direction),
        )
        return coords

    def contains(self, coords: StandardCoord) -> bool:
        """Check if beam contains a given coordinate."""
        x, y, z = coords
        return self.x.contains(x) and self.y.contains(y) and self.z.contains(z)

    def is_parallel(self, other: object) -> bool:
        """Check if two beams run parallel to one another (ignores collinearity)."""
        return (
            self.x.is_point() == other.x.is_point()
            and self.y.is_point() == other.y.is_point()
            and self.z.is_point() == other.z.is_point()
        )

    def intersects(self, other: object) -> bool:
        """Check if two beams intersect one another."""

        if self.is_parallel(other):
            return False

        c = 1
        prev_manhattan = np.sum(
            np.abs(
                np.array([float(c) for c in self.coords(0)])
                - np.array([float(c) for c in other.coords(0)])
            )
        )
        while True:
            coords_self, coords_other = (self.coords(c), other.coords(c))
            if coords_self == coords_other:
                return True
            curr_manhattan = np.sum(
                np.abs(
                    np.array([float(c) for c in coords_self])
                    - np.array([float(c) for c in coords_other])
                )
            )
            if curr_manhattan > prev_manhattan:
                return False
            prev_manhattan = curr_manhattan
            c += 1


CubeBeams = list[SingleBeam]


# Edge path class with in-built value-function to enable path comparisons
@dataclass(order=True)
class PathBetweenNodes:
    """A 3D path between the cubes corresponding to two nodes/spiders in the input ZX graph."""

    tgt_coords: StandardCoord
    tgt_kind: str
    tgt_beams: CubeBeams
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
            **kwargs:
                weights: weights for the value function to pick best of many paths.
                length_of_beams: length of each of the beams coming out of open nodes.
                deterministic: A boolean flag to tell the function if choice is deterministic or random.

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
    RESET = "\033[0m"
