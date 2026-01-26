from enum import Enum

# Essentially vectors with their basic operations
class Coordinates:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Coordinates(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Coordinates(self.x - other.x, self.y - other.y, self.z - other.z)

    def invert(self):
        return Coordinates(-self.x, -self.y, -self.z)

    def mul(self, scalar: int):
        return Coordinates(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other) -> int:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

class Step(Enum):
    # Represented by unit vectors
    XP = Coordinates(+1, 0, 0)
    XM = Coordinates(-1, 0, 0)
    YP = Coordinates(0, +1, 0)
    YM = Coordinates(0, -1, 0)
    ZP = Coordinates(0, 0, +1)
    ZM = Coordinates(0, 0, -1)

    def __str__(self):
        return f"Step-{self.name}"

class Plane(Enum):
    # Represented by vectors
    XY = Coordinates(+1, +1, 0)
    XZ = Coordinates(+1, 0, +1)
    YZ = Coordinates(0, +1, +1)

    def contains(self, step: Step) -> bool:
        # Dot product will tell us whether the step lies in this plane
        return self.value.dot(step.value) != 0

    def __str__(self):
        return f"Plane-{self.name}"

class BlockGraphSpace:
    ORIGIN = Coordinates(0, 0, 0)

    STEPS = [ Step.XP, Step.XM, Step.YP, Step.YM, Step.ZP, Step.ZM ]
    PLANES = [ Plane.XY, Plane.XZ, Plane.YZ ]

    @staticmethod
    def get_orthogonal_plane(plane: Plane, line_of_intersection: Step) -> Plane:
        if not plane.contains(line_of_intersection):
            raise ValueError(f"Line of intersection {line_of_intersection} does not lie in plane {plane}.")

        if abs(plane.value.x) != abs(line_of_intersection.value.x):
            return Plane.YZ
        elif abs(plane.value.y) != abs(line_of_intersection.value.y):
            return Plane.XZ
        else: # abs(plane.value.z) != abs(line_of_intersection.value.z)
            return Plane.XY

    @staticmethod
    def get_constellation(position: Coordinates, restriction: Plane = None) -> list[Coordinates]:
        constellation = []
        for step in BlockGraphSpace.STEPS:
            if restriction is None or restriction.contains(step):
                constellation.append(position + step.value)
        return constellation