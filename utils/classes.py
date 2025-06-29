from typing import List, Tuple, TypedDict
from dataclasses import dataclass
from typing import Tuple, List

# TYPES FOR INCOMING 2D GRAPH
GraphNode = Tuple[int, str]
GraphEdge = Tuple[Tuple[int, int], str]


class SimpleDictGraph(TypedDict):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


# COMMON TYPES IN ALGORITHM
StandardCoord = Tuple[int, int, int]
StandardBeam = List[StandardCoord]
NodeBeams = List[StandardBeam]

StandardBlock = Tuple[StandardCoord, str]


# MAIN DATA CLASS TO STORE PATHS AND ENABLE COMPARISONS
@dataclass(order=True)
class PathBetweenNodes:
    target_pos: StandardCoord
    target_kind: str
    target_beams: NodeBeams
    coords_in_path: List[StandardCoord]
    all_nodes_in_path: List[StandardBlock]
    beams_broken_by_path: int
    len_of_path: int
    target_unobstructed_exits_n: int

    def weighed_value(self, stage, **kwargs) -> int:
        path_len_hp, beams_broken_hp, target_exits_hp = kwargs["weights"]
        return (
            self.len_of_path * path_len_hp
            + self.beams_broken_by_path * beams_broken_hp
            + self.target_unobstructed_exits_n * target_exits_hp
        )


# THE TWO CLASSES BELOW ARE NOT YET USED
# THEY NEED TO BE USED TO SPECIFY EDGE AND EDGE_PATHS TYPES
@dataclass(order=True)
class Block:
    position: StandardCoord
    kind: str


EdgePath = List[Block]


# LET'S ADD SOME COLOUR TO PRINTS BECAUSE, WHY NOT
class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"
