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
    tgt_pos: StandardCoord
    tgt_kind: str
    tgt_beams: NodeBeams
    coords_in_pth: List[StandardCoord]
    all_nodes_in_pth: List[StandardBlock]
    beams_broken_by_pth: int
    len_of_pth: int
    tgt_unobstr_exit_n: int

    def weighed_value(self, **kwargs) -> int:
        """ Returns the weighed value of a given path, which can be used for comparing many paths
        
        Args:
            - n/a.
        
        Keyword arguments (**kwargs):
            - weights: weights for the value function to pick best of many paths.
            - length_of_beams: length of each of the beams coming out of open nodes.
        
        Returns:
            - Weighed value of a path
        
        """

        pth_len_hp, beams_broken_hp = kwargs["weights"]

        return (
            self.len_of_pth * pth_len_hp
            + self.beams_broken_by_pth * beams_broken_hp
        )


# THE TWO CLASSES BELOW ARE NOT YET USED
# THEY NEED TO BE USED TO SPECIFY EDGE AND EDGE_pthS TYPES
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
