"""Beams and related classes used across graph_manager and pathfinder.

Usage:
    Call any required class from a separate script.

"""

from dataclasses import dataclass

import numpy as np

from topologiq.core.pathfinder.utils import get_manhattan
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
        """Return te beam direction as a coordinate tuple."""
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

    INTERSECTION_CONSISTENCY_CHECKS = False

    def intersects(
        self, other: "SingleBeam", len_of_materialised_beam: int, by_rays: bool = False
    ) -> bool:
        """Check if two beams intersect one another."""

        if SingleBeam.INTERSECTION_CONSISTENCY_CHECKS:
            intersecting_beams = self.intersects_co_planarity(other)
            other_as_array = other.to_array(50)
            self_as_array = self.to_array(50)
            intersecting_arrays = any(c in self_as_array for c in other_as_array)

            if intersecting_arrays != intersecting_beams:
                report = f"INTERSECTION inconsistency. {self} vs {other} [A:{intersecting_arrays}/R:{intersecting_beams}]\n"
                report += f"> {self.to_array(len_of_materialised_beam)}\n> {other_as_array}."
                raise Exception(report)
        elif by_rays:
            intersecting_beams = self.intersects_co_planarity(other)
        else:  # SingleBeam.INTERSECTION_BY_RAYS == False
            other_as_array = other.to_array(len_of_materialised_beam)
            intersecting_beams = any([self.contains(c) for c in other_as_array])

        return intersecting_beams

    def intersects_co_planarity(self, other: "SingleBeam") -> bool:
        """Check if two beams intersect one another."""

        p1 = self.coords()
        p2 = other.coords()

        if get_manhattan(p1, p2) > 9:
            return False

        d1 = self.direction()
        d2 = other.direction()

        # The orientation will describe which case of directions we're dealing with; same, opposite or orthogonal
        orientation = np.dot(d1, d2)

        if orientation != 0:
            return False

        p1 = self.coords()
        p2 = other.coords()

        # Beams are orthogonal; source of the other beam must be in the positive quadrant of the span of {d1, -d2}
        # sigma is the position of the source of the other beam relative to the source of this beam
        sigma = np.subtract(p2, p1)

        basis = np.subtract(d1, d2)
        return np.all((sigma == 0) | (np.sign(sigma) == np.sign(basis)))


CubeBeams = list[SingleBeam]
