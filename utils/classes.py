import numpy as np
from typing import List, Tuple, TypedDict
from dataclasses import dataclass
from typing import Tuple, List, cast

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
        """ Returns the weighed value of a given path, which can be used for comparing many paths
        
        Args:
            - stage (not in use): may eventually be used to determine if algorithm is at the start, middle, or end of a given circuit
        
        Keyword arguments (**kwargs):
            - weights: weights for the value function to pick best of many paths.
            - length_of_beams: length of each of the beams coming out of open nodes.
            - max_search_space: maximum size of 3D space to generate paths for.
        
        Returns:
            - Weighed value of a path
        
        """

        path_len_hp, beams_broken_hp = kwargs["weights"]

        return (
            self.len_of_path * path_len_hp
            + self.beams_broken_by_path * beams_broken_hp
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
