import pyzx as zx
from typing import Union
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS


def cnot() -> Union[BaseGraph, GraphS]:

    c = zx.Circuit(2)
    c.add_gate("CNOT", 1, 0)

    g = c.to_graph()
    zx.draw(g)

    return g
