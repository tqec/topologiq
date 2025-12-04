"""Classes for key objects used in Topologiq.

Usage:
    Call any required class from a separate script.

"""


from dataclasses import dataclass
from typing import TypedDict

# Types for input ZX graph
GraphNode = tuple[int, str]
GraphEdge = tuple[tuple[int, int], str]

class SimpleDictGraph(TypedDict):
    """A simple graph composed of nodes and edges."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]


# Common types used across the algorithm
StandardCoord = tuple[int, int, int]
StandardBlock = tuple[StandardCoord, str]
StandardBeam = list[StandardCoord]
NodeBeams = list[StandardBeam]


# Main data class to store paths and enable comparisons between pathsMAIN DATA CLASS TO STORE PATHS AND ENABLE COMPARISONS
@dataclass(order=True)
class PathBetweenNodes:
    """A 3D path between the cubes corresponding to two nodes/spiders in the input ZX graph."""

    tgt_coords: StandardCoord
    tgt_kind: str
    tgt_beams: NodeBeams
    coords_in_path: list[StandardCoord]
    all_nodes_in_path: list[StandardBlock]
    beams_broken_by_path: int
    len_of_path: int
    tgt_unobstr_exit_n: int

    def weighed_value(self, **kwargs) -> int:
        """Return the weighed value of a given path.

        This function returns the weighed value of a given PathBetweenNodes,
        which can be used for comparing many paths.

        Args:
            **kwargs:
                weights: weights for the value function to pick best of many paths.
                length_of_beams: length of each of the beams coming out of open nodes.

        Returns:
            (int): The weighed value of a path

        """

        path_len_hp, beams_broken_hp = kwargs["weights"]

        return self.len_of_path * path_len_hp + self.beams_broken_by_path * beams_broken_hp


# Colour class to use colours in terminal printouts
class Colors:
    """Colours to use in printouts."""

    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"
