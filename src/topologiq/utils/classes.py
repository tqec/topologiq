"""Classes for key objects used in Topologiq.

Usage:
    Call any required class from a separate script.

"""

from dataclasses import dataclass
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

    def is_parallel(self, other: object) -> bool:
        """Check if two beams run parallel to one another (ignores collinearity)."""
        return (
            self.x.is_point() == other.x.is_point()
            and self.y.is_point() == other.y.is_point()
            and self.z.is_point() == other.z.is_point()
        )

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
        return self.x.direction, self.y.direction, self.z.direction

    def contains(self, coords_to_check: StandardCoord) -> bool:
        """Check if beam contains a given coordinate."""
        x, y, z = coords_to_check
        return self.x.contains(x) and self.y.contains(y) and self.z.contains(z)

    def to_array(self, len_of_materialised_beam: int) -> list[StandardCoord]:
        """Convert beam into an array of 3D coordinates of arbitrary length."""

        if self.x.direction != 0:
            y_start = self.y.start
            z_start = self.z.start
            return [(i, y_start, z_start) for i in self.x.to_array(len_of_materialised_beam)]

        if self.y.direction != 0:
            x_start = self.x.start
            z_start = self.z.start
            return [(x_start, i, z_start) for i in self.y.to_array(len_of_materialised_beam)]

        if self.z.direction != 0:
            x_start = self.x.start
            y_start = self.y.start
            return [(x_start, y_start, i) for i in self.z.to_array(len_of_materialised_beam)]

    def check_co_planarity(self, other: object) -> tuple[bool, int | None]:
        """Check if two beams are co-planar."""
        co_planarity_checks = [
            self.x.direction == other.x.direction == 0,
            self.y.direction == other.y.direction == 0,
            self.z.direction == other.z.direction == 0,
        ]

        if sum(co_planarity_checks) == 1:
            return True, co_planarity_checks.index(True)

        return False, None

    INTERSECTION_CONSISTENCY_CHECKS = True
    INTERSECTION_BY_RAYS = False

    def intersects(self, other: 'SingleBeam', len_of_materialised_beam: int) -> bool:
        """Check if two beams intersect one another."""

        if SingleBeam.INTERSECTION_CONSISTENCY_CHECKS:
            intersecting_beams = self.intersects_co_planarity(other)
            other_as_array = other.to_array(50)
            self_as_array = self.to_array(50)
            intersecting_arrays = any(c in self_as_array for c in other_as_array)

            if intersecting_arrays != intersecting_beams:
                report  = f"INTERSECTION inconsistency. {self} vs {other} [A:{intersecting_arrays}/R:{intersecting_beams}]\n"
                report += f"> {self.to_array(len_of_materialised_beam)}\n> {other_as_array}."
                raise Exception(report)
        elif SingleBeam.INTERSECTION_BY_RAYS:
            intersecting_beams = self.intersects_co_planarity(other)
        else: # SingleBeam.INTERSECTION_BY_RAYS == False
            other_as_array = other.to_array(len_of_materialised_beam)
            intersecting_beams = any([self.contains(c) for c in other_as_array])

        return intersecting_beams

    def intersects_co_planarity(self, other: 'SingleBeam') -> bool:
        """Check if two beams intersect one another."""

        p1 = self.coords()
        d1 = self.direction()
        p2 = other.coords()
        d2 = other.direction()

        # The orientation will describe which case of directions we're dealing with; same, opposite or orthogonal
        orientation = np.dot(d1, d2)
        # sigma is the position of the source of the other beam relative to the source of this beam
        sigma = np.subtract(p2, p1)

        if orientation == 0:
            # Beams are orthogonal; source of the other beam must be in the positive quadrant of the span of {d1, -d2}
            basis = np.subtract(d1, d2)
            intersecting_rays = np.all( (sigma == 0) | ( np.sign(sigma) == np.sign(basis)) )
        else:
            # Beams are parallel; source of other beam must be located along the same line as direction of this beam
            cross = np.cross(d1, sigma)
            intersecting_rays = np.dot(cross, cross) == 0
            if orientation == -1:
                # Beams are in opposite directions; source of other beam cannot be located behind source of this beam
                intersecting_rays &= np.dot(d1, sigma) >= 0

        return intersecting_rays


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
