import pyzx as zx
from typing import Union

from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS
from pyzx.generate import CNOT_HAD_PHASE_circuit


def cnot(draw_graph: bool = False) -> Union[BaseGraph, GraphS]:

    c = zx.Circuit(2)
    c.add_gate("CNOT", 1, 0)
    g = c.to_graph()

    if draw_graph:
        zx.draw(g)

    return g


def cnots(draw_graph: bool = False) -> Union[BaseGraph, GraphS]:

    c = CNOT_HAD_PHASE_circuit(qubits=2, depth=4, clifford=True)
    g = c.to_graph()

    if draw_graph:
        zx.draw(g)

    return g


def random(draw_graph: bool = False) -> Union[BaseGraph, GraphS]:

    g = zx.generate.cliffordT(3, 7)

    if draw_graph:
        zx.draw(g)

    return g


def random_optimised(draw_graph: bool = False) -> Union[BaseGraph, GraphS]:

    g = zx.generate.cliffordT(2, 7)
    zx.simplify.phase_free_simp(g)

    if draw_graph:
        zx.draw(g)

    return g
