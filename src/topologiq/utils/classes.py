"""Classes for key objects used in Topologiq.

Usage:
    Call any required class from a separate script.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

###############
# QUICK TYPES #
###############
StandardCoord = tuple[int, int, int]
StandardBlock = tuple[StandardCoord, str]
StandardBeam = list[StandardCoord]


################
# SIMPLE GRAPH #
#################
GraphNode = tuple[int, str]
GraphEdge = tuple[tuple[int, int], str]


class SimpleDictGraph(TypedDict):
    """A simple graph composed of nodes and edges."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]


#######
# AUX #
#######
@dataclass
class GraphBounds:
    """Class to initialise and hold graph boundaries."""

    x: int | None = None
    y: int | None = None
    z: None = None


class Colors:
    """Colours to use in printouts."""

    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"
