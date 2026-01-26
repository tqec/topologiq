from enum import Enum

class Plane(Enum):
    NA = 0
    XY = 1
    XZ = 2
    YZ = 3

class Coordinates:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Coordinates(self.x + other.x, self.y + other.y, self.z + other.z)

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

class BlockGraphSpace:
    ORIGIN = Coordinates(0, 0, 0)

    XP = Coordinates(+1,  0,  0)
    XM = Coordinates(-1,  0,  0)
    YP = Coordinates( 0, +1,  0)
    YM = Coordinates( 0, -1,  0)
    ZP = Coordinates( 0,  0, +1)
    ZM = Coordinates( 0,  0, -1)

    STEPS = {
        Plane.NA : [XP, XM, YP, YM, ZP, ZM] ,
        Plane.XY : [XP, XM, YP, YM] ,
        Plane.XZ : [XP, XM, ZP, ZM] ,
        Plane.YZ : [YP, YM, ZP, ZM]
    }

    @staticmethod
    def constellation(position: Coordinates, plane: Plane) -> list[Coordinates]:
        constellation = []
        for step in BlockGraphSpace.STEPS[plane]:
            constellation.append(position + step)
        return constellation