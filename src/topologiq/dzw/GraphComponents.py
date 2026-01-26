from enum import Enum
import pyzx as zx

from topologiq.dzw.BlockGraphSpace import Plane, BlockGraphSpace

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

    def flip(self):
        if self == NodeType.X:
            return NodeType.Z
        elif self == NodeType.Z:
            return NodeType.X
        else:
            raise ValueError(f"Flipping color not supported for node type: {self}")

    def __str__(self):
        return self.name

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

    @staticmethod
    def convert(node_type: NodeType, node_plane: Plane):
        if node_type == NodeType.X:
            if node_plane == Plane.XY:
                return CubeKind.ZZX
            elif node_plane == Plane.XZ:
                return CubeKind.ZXZ
            else:
                return CubeKind.XZZ
        elif node_type == NodeType.Z:
            if node_plane == Plane.XY:
                return CubeKind.XXZ
            elif node_plane == Plane.XZ:
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


    def get_plane(self) -> Plane:
        if self == CubeKind.XZZ or self == CubeKind.ZXX:
            return Plane.YZ
        elif self == CubeKind.ZXZ or self == CubeKind.XZX:
            return Plane.XZ
        elif self == CubeKind.ZZX or self == CubeKind.XXZ:
            return Plane.XY
        else: # self.name == CubeKind.OOO or self.name == CubeKind.YYY
            raise ValueError(f"Not applicable to cube kind {self.value}")

    def __str__(self):
        return self.name