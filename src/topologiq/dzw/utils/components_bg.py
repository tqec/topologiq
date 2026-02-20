from enum import Enum
from functools import total_ordering

from topologiq.dzw.helpers.spacetime import Spacetime
from topologiq.dzw.utils.coordinates import Coordinates

from topologiq.dzw.utils.components_zx import NodeType

CubeId = int

class CubeKind(Enum):
    OOO = 0
    XZZ = 1
    ZXZ = 2
    ZZX = 3
    ZXX = 4
    XZX = 5
    XXZ = 6
    YYY = 7

    @staticmethod
    def suitable_kinds(node_type: NodeType):
        if   node_type == NodeType.X:
            return [CubeKind.XZZ, CubeKind.ZXZ, CubeKind.ZZX]
        elif node_type == NodeType.Y:
            return [CubeKind.YYY]
        elif node_type == NodeType.Z:
            return [CubeKind.ZXX, CubeKind.XZX, CubeKind.XXZ]
        elif node_type == NodeType.O:
            return [CubeKind.OOO]
        else:
            raise Exception(f"{node_type} has no representation as a cube of any kind.")

    @staticmethod
    def convert(node_type: NodeType, node_reach: Coordinates):
        if node_type == NodeType.X:
            if node_reach == Spacetime.XY:
                return CubeKind.ZZX
            elif node_reach == Spacetime.XZ:
                return CubeKind.ZXZ
            else:
                return CubeKind.XZZ
        elif node_type == NodeType.Z:
            if node_reach == Spacetime.XY:
                return CubeKind.XXZ
            elif node_reach == Spacetime.XZ:
                return CubeKind.XZX
            else:
                return CubeKind.ZXX
        elif node_type == NodeType.Y:
            return CubeKind.YYY
        else: # node_type == NodeType.O:
            return CubeKind.OOO

    def get_type(self) -> NodeType:
        if   self == CubeKind.XZZ or self == CubeKind.ZXZ or self == CubeKind.ZZX:
            return NodeType.X
        elif self == CubeKind.ZXX or self == CubeKind.XZX or self == CubeKind.XXZ:
            return NodeType.Z
        elif self == CubeKind.YYY:
            return NodeType.Y
        else: # self == CubeKind.OOO
            return NodeType.O

    # TODO: a CubeKind.YYY has Spacetime.XYZ and single port ?
    def get_reach(self) -> Coordinates:
        if self == CubeKind.XZZ or self == CubeKind.ZXX:
            return Spacetime.YZ
        elif self == CubeKind.ZXZ or self == CubeKind.XZX:
            return Spacetime.XZ
        elif self == CubeKind.ZZX or self == CubeKind.XXZ:
            return Spacetime.XY
        elif self == CubeKind.OOO or self == CubeKind.YYY:
            return Spacetime.XYZ
        else:
            raise ValueError(f"Not applicable to cube kind {self.name}")

    @total_ordering
    def __lt__(self, other):
        return self.value.__lt__(other.value)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name