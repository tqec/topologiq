from enum import Enum
import pyzx as zx

from topologiq.dzw.BlockGraphSpace import Coordinates, Plane, BlockGraphSpace

SUPPORTED_VERTEX_TYPES = [zx.VertexType.BOUNDARY, zx.VertexType.Z, zx.VertexType.X]
class NodeType(Enum):
    O = 0 # Boundary
    X = 1 # X-Spider
    Y = 2 # Y-Spider
    Z = 3 # Z-Spider

    @staticmethod
    def convert(vertex_type: zx.VertexType):
        if vertex_type == zx.VertexType.Z:
            return NodeType.Z
        elif vertex_type == zx.VertexType.X:
            return NodeType.X
        elif vertex_type == zx.VertexType.BOUNDARY:
            return NodeType.O
        else:
            raise ValueError(f"Unsupported vertex type: {vertex_type}")

    def __str__(self):
        return self.name


SUPPORTED_EDGE_TYPES = [zx.EdgeType.SIMPLE, zx.EdgeType.HADAMARD]
class EdgeType(Enum):
    IDENTITY = 0
    HADAMARD = 1

    @staticmethod
    def convert(edge_type: zx.EdgeType):
        if edge_type == zx.EdgeType.SIMPLE:
            return EdgeType.IDENTITY
        elif edge_type == zx.EdgeType.HADAMARD:
            return EdgeType.HADAMARD
        else:
            raise ValueError(f"Unsupported edge type: {edge_type}")

    def __str__(self):
        return self.name

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

    def compatible_adjacent(self):
        compatibles = []

        for step in BlockGraphSpace.STEPS[self.name]:
            compatibles.append((step, self.name))
            if self.name == CubeKind.XZZ:
                compatibles.append( (step , CubeKind.XXZ) )

        return compatibles

    def get_plane(self) -> Plane:
        if self.name == CubeKind.XZZ or self.name == CubeKind.ZXX:
            return Plane.YZ
        elif self.name == CubeKind.ZXZ or self.name == CubeKind.XZX:
            return Plane.XZ
        elif self.name == CubeKind.ZZX or self.name == CubeKind.XXZ:
            return Plane.XY
        else: # self.name == CubeKind.OOO or self.name == CubeKind.YYY
            raise ValueError(f"Not applicable to cube kind {self.name}")

    def __str__(self):
        return self.name