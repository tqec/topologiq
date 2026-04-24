from enum import Enum
import pyzx

NodeId = int
EdgeId = tuple[NodeId, NodeId]
QubitId = int
LayerId = int

class NodeType(Enum):
    O = 0 # Boundary
    X = 1 # X-Spider
    Y = 2 # Y-Spider
    Z = 3 # Z-Spider

    @staticmethod
    def convert_from_pyzx(vertex_type: pyzx.VertexType):
        if vertex_type == pyzx.VertexType.Z:
            return NodeType.Z
        elif vertex_type == pyzx.VertexType.X:
            return NodeType.X
        elif vertex_type == pyzx.VertexType.BOUNDARY:
            return NodeType.O
        else:
            raise ValueError(f"Unsupported vertex type: {vertex_type}")

    @staticmethod
    def convert_into_pyzx(self):
        if self == NodeType.Z:
            return pyzx.VertexType.Z
        elif self == NodeType.X:
            return pyzx.VertexType.X
        elif self == NodeType.O:
            return pyzx.VertexType.BOUNDARY
        else:
            raise ValueError(f"Unsupported conversion for node type: {self}")

    @staticmethod
    def convert_simple(vertex_type: str):
        if vertex_type == pyzx.VertexType.Z.name:
            return NodeType.Z
        elif vertex_type == pyzx.VertexType.X.name:
            return NodeType.X
        elif vertex_type == 'O': # zx.VertexType.BOUNDARY.name:
            return NodeType.O
        else:
            raise ValueError(f"Unsupported vertex type: {vertex_type}")

    def __str__(self):
        return self.name


class EdgeType(Enum):
    IDENTITY = 0
    HADAMARD = 1

    @staticmethod
    def convert_from_pyzx(edge_type: pyzx.EdgeType):
        if edge_type == pyzx.EdgeType.SIMPLE:
            return EdgeType.IDENTITY
        elif edge_type == pyzx.EdgeType.HADAMARD:
            return EdgeType.HADAMARD
        else:
            raise ValueError(f"Unsupported edge type: {edge_type}")

    @staticmethod
    def convert_into_pyzx(self):
        if self == EdgeType.IDENTITY:
            return pyzx.EdgeType.SIMPLE
        else: # self == EdgeType.HADAMARD:
            return pyzx.EdgeType.HADAMARD

    @staticmethod
    def convert_simple(edge_type: str):
        if edge_type == pyzx.EdgeType.SIMPLE.name:
            return EdgeType.IDENTITY
        elif edge_type == pyzx.EdgeType.HADAMARD.name:
            return EdgeType.HADAMARD
        else:
            raise ValueError(f"Unsupported edge type: {edge_type}")

    def __str__(self):
        return self.name