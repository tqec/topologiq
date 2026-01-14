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

    def contains(self, point: int) -> bool:
        """Check if a given point is contained in the segment."""
        if self.direction == 0 and (self.start == point == self.end):
            return True
        if self.direction == 1 and (self.start < point < self.end):
            return True
        if self.direction == -1 and (self.start > point > self.end):
            return True

        return False

    def to_array(self, array_length: int) -> list[int] | None:
        """Convert segment into an array of arbitrary length."""
        if self.direction != 0:
            return [self.start + i * self.direction for i in range(array_length)]


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

    def coords(self) -> tuple[BeamAxisComponent, BeamAxisComponent, BeamAxisComponent]:
        """Return the beam coordinates across all axes."""
        return (self.x, self.y, self.z)

    def contains(self, coords_to_check: StandardCoord) -> bool:
        """Check if beam contains a given coordinate."""
        x, y, z = coords_to_check
        return self.x.contains(x) and self.y.contains(y) and self.z.contains(z)

    def to_array(self, d: int) -> list[StandardCoord]:
        """Convert beam into an array of 3D coordinates of arbitrary length."""

        if self.x.direction != 0:
            y_start = self.y.start
            z_start = self.z.start
            return [(i, y_start, z_start) for i in self.x.to_array(d)]

        if self.y.direction != 0:
            x_start = self.x.start
            z_start = self.z.start
            return [(x_start, i, z_start) for i in self.y.to_array(d)]

        if self.z.direction != 0:
            x_start = self.x.start
            y_start = self.y.start
            return [(x_start, y_start, i) for i in self.z.to_array(d)]

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

    def intersects(self, other: object) -> bool:  # noqa: PLR0911, for readability
        """Check if two beams intersect one another."""

        #self_vector = np.array((self.x.direction, self.y.direction, self.z.direction))
        #other_vector = np.array((other.x.direction, other.y.direction, other.z.direction))
        #cos = dot(self_vector, other_vector) / norm(self_vector) / norm(other_vector)
        #angle = abs(int(np.degrees(arccos(clip(cos, -1, 1)))))
        #if angle != 90:
            #return False

        beams_are_co_planar, co_planarity_idx = self.check_co_planarity(other)
        if beams_are_co_planar:
            if co_planarity_idx == 0:
                if self.y.direction != 0:
                    return self.y.contains(other.y.start) and other.y.contains(self.y.start)
                elif self.z.direction !=0:
                    return self.z.contains(other.z.start) and other.z.contains(self.z.start)
            elif co_planarity_idx == 1:
                if self.x.direction != 0:
                    return self.x.contains(other.x.start) and other.x.contains(self.x.start)
                elif self.z.direction !=0:
                    return self.z.contains(other.z.start) and other.z.contains(self.z.start)
            elif co_planarity_idx == 2:
                if self.x.direction != 0:
                    return self.x.contains(other.x.start) and other.x.contains(self.x.start)
                elif self.y.direction !=0:
                    return self.y.contains(other.y.start) and other.y.contains(self.y.start)

        return False


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
            **kwargs: !
                weights: A tuple (int, int) of weights used to pick the best of several paths when there are several valid alternatives.
                deterministic: A boolean flag to tell the function if choice is deterministic or random.
                random_seed: Typically `None`, but can be used to pass a specific seed across the entire algorithm.

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
