from math import sqrt
from functools import total_ordering
from dataclasses import dataclass

@dataclass(frozen = True)
class Coordinates:
    x: float
    y: float
    z: float

    @staticmethod
    def from_list(l: list[float]):
        if len(l) != 3:
            raise Exception(f"Provided list is too long. Needs to have 3 components.")
        return Coordinates(l[0], l[1], l[2])

    @staticmethod
    def from_tuple(t: tuple[float, float, float]):
        return Coordinates(t[0], t[1], t[2])

    def as_tuple(self):
        return self.x, self.y, self.z

    def __add__(self, other):
        return Coordinates(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Coordinates(self.x - other.x, self.y - other.y, self.z - other.z)

    def dmul(self, other):
        return Coordinates(self.x * other.x, self.y * other.y, self.z * other.z)

    def div(self, scalar: float):
        return Coordinates(self.x / scalar, self.y / scalar, self.z / scalar)

    def normalized(self):
        return self.div(sqrt(self.dot(self)))

    def dot(self, other) -> float:
        return sum([ s * o for s, o in zip(self, other) ])

    def get_manhattan_distance(self, other):
        return sum([ abs(s - o) for s, o in zip(self, other) ])

    def __different_components(self, other):
        different_x = 1 if self.x != other.x else 0
        different_y = 1 if self.y != other.y else 0
        different_z = 1 if self.z != other.z else 0
        return different_x + different_y + different_z

    # The coordinates are colinear if they share one identical components
    def colinear(self, other) -> bool:
        return self.__different_components(other) == 1

    # The coordinates are coplanar if they share two identical components
    def coplanar(self, other) -> bool:
        return self.__different_components(other) == 2

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.as_tuple())

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    @total_ordering
    def __lt__(self, other):
        return self.as_tuple().__lt__(other.as_tuple())

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"