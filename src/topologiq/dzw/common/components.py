from recordclass import RecordClass  # type: ignore[import-untyped]

from topologiq.utils.classes import StandardCoord

from topologiq.dzw.common.attributes_zx import NodeId, NodeType, QubitId, LayerId, EdgeType
from topologiq.dzw.common.attributes_bg import CubeId, CubeKind, PipeId

class ZxNode(RecordClass):
    id: NodeId
    type: NodeType
    qubit: QubitId = -1
    layer: LayerId = -1
    realising_cube: CubeId = -1

    def __str__(self):
        return f"N{self.id}:{self.type}"

    def __repr__(self):
        return str(self)

class ZxEdge(RecordClass):
    source: ZxNode
    target: ZxNode
    type: EdgeType
    realisation: list[PipeId] = []

    def __str__(self):
        return f"N{self.source.id}-{self.type.name[0]}-N{self.target.id}"

    def __repr__(self):
        return str(self)

class BgCube(RecordClass):
    kind: CubeKind
    position: StandardCoord
    id: CubeId = -1
    realised_node: NodeId = -1

    def __str__(self):
        content = ""
        if self.id != -1:
            content += f"#{self.id}:"
        if self.realised_node != -1:
            content += f"N{self.realised_node}:"
        content += f"{self.kind}@{self.position}"
        return content

    def __repr__(self):
        return str(self)

class BgPipe(RecordClass):
    source: CubeId
    target: CubeId
    type: EdgeType

    def __str__(self):
        return f"#{self.source}-{self.type.name[0]}-#{self.target}"

    def __repr__(self):
        return str(self)
