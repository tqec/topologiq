import matplotlib
import pyzx as zx
import random
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS
from typing import Optional, Tuple, Union
import matplotlib.figure

def cnot(draw_graph: bool = False) -> Tuple[Union[BaseGraph, GraphS], Optional[matplotlib.figure.Figure]]:

    c = zx.Circuit(2)
    c.add_gate("CNOT", 1, 0)
    g = c.to_graph()

    fig = None
    if draw_graph:
        fig = zx.draw_matplotlib(g, labels=True)

    return g, fig


def cnots(draw_graph: bool = False) -> Tuple[Union[BaseGraph, GraphS], Optional[matplotlib.figure.Figure]]:

    c = zx.Circuit(2)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("CNOT", 1, 0)
    c.add_gate("CNOT", 1, 0)

    g = c.to_graph()

    fig = None
    if draw_graph:
        fig = zx.draw(g, labels=True)

    return g, fig


def simple_mess(draw_graph: bool = False) -> Tuple[Union[BaseGraph, GraphS], Optional[matplotlib.figure.Figure]]:

    c = zx.Circuit(3)
    c.add_gate("CNOT", 1, 2)
    c.add_gate("S", 2)
    c.add_gate("CNOT", 1, 0)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("S", 2)
    c.add_gate("CNOT", 0, 2)

    g = c.to_graph()
    
    fig = None
    if draw_graph:
        fig = zx.draw(g, labels=True)

    return g, fig


def random_graph(
    qubit_range: Tuple[int, int],
    depth_range: Tuple[int, int],
    draw_graph: bool = False
) -> Tuple[Union[BaseGraph, GraphS] | None, matplotlib.figure.Figure] | None:

    # Determine size of graph
    min_qubits, max_qubits = qubit_range
    min_depth, max_depth = depth_range
    qubits = random.randrange(min_qubits, max_qubits)
    depth = random.randrange(min_depth, max_depth)

    # Check graph integrity
    # PyZX sometimes generates graphs of disconnected subgraphs,
    # which aren't compatible with Topologiq. This block emulates
    # Topologiq's core BFS logic to ensure the returned PyZX graph is
    # a single big interconnected graph.
    i = 0
    max_attempts = 100
    while i < max_attempts:
        # Increase counter from start to not forget
        i += 1

        # Generate a graph
        c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=qubits, depth=depth, clifford=True)
        g = c.to_graph()

        # Run a canonical BFS loop to confirm all spiders are hit by BFS
        queue = []
        visited = {}

        ids_original_spiders = list(g.vertices())
        queue.append(ids_original_spiders[0])
        visited[ids_original_spiders[0]] = True
        while queue:
            nxt = queue.pop(0)
            all_neighbours = list(g.neighbors(nxt))
            for neigh in all_neighbours:
                if neigh not in visited:
                    queue.append(neigh)
                    visited[neigh] = True

        # Check BFS visited IDs against original PyZX graph
        if ids_original_spiders == sorted(list(visited.keys())):
            # Return if all IDs are present
            if draw_graph:
                fig = zx.draw(g, labels=True)
            
            # Return graph and figure
            return g, fig

    return None, None
