from dataclasses import dataclass
from typing import Tuple, List
from two_stage_greedy_bfs_config import VALUE_HYPERPARAMS

from typing import List, Tuple, TypedDict

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
    target_pos: Tuple[int, int, int]
    target_kind: str
    target_beams: NodeBeams
    coords_in_path: List[Tuple[int, int, int]]
    all_nodes_in_path: List[Tuple[Tuple[int, int, int], str]]
    beams_broken_by_path: int
    len_of_path: int
    target_unobstructed_exits_n: int

    def weighed_value(self, stage) -> int:
        path_len_hp, beams_broken_hp, target_exits_hp = VALUE_HYPERPARAMS
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
    # n_unobstructed_exits: int
    # beams: List[Tuple[int, int, int]]


EdgePath = List[Block]
