import pyzx as zx
import random
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS
from typing import Union


def cnot(draw_graph: bool = False) -> Union[BaseGraph, GraphS]:
    
    c = zx.Circuit(2)
    c.add_gate("CNOT", 1, 0)
    g = c.to_graph()

    if draw_graph:
        zx.draw(g, labels=True)

    return g


def cnots(draw_graph: bool = False) -> Union[BaseGraph, GraphS]:

    c = zx.Circuit(2)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("CNOT", 1, 0)
    c.add_gate("CNOT", 1, 0)

    g = c.to_graph()

    if draw_graph:
        zx.draw(g, labels=True)

    return g


def simple_mess(draw_graph: bool = False) -> Union[BaseGraph, GraphS]:

    c = zx.Circuit(3)
    c.add_gate("CNOT", 1, 2)
    c.add_gate("S", 2)
    c.add_gate("CNOT", 1, 0)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("S", 2)
    c.add_gate("CNOT", 0, 2)

    g = c.to_graph()

    if draw_graph:
        zx.draw(g, labels=True)

    return g


def random_graph(draw_graph: bool = False) -> Union[BaseGraph, GraphS]:

    qubits = random.randint(2, 5)
    depth = random.randint(7, 15)
    c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=qubits, depth=depth, clifford=True)
    g = c.to_graph()
    # zx.clifford_simp(g)
    # g.normalize()

    if draw_graph:
        zx.draw(g, labels=True)

    return g
