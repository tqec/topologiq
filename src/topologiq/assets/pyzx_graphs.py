"""PyZX graphs to use in examples, demonstrations and testing.

Usage:
    Call any graph from a separate script.

"""

import matplotlib
import matplotlib.figure
import pyzx as zx
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS


def cnot(draw_graph: bool = False) -> tuple[BaseGraph | GraphS, matplotlib.figure.Figure | None]:
    """Produce a PyZX graph corresponding to a CNOT.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        g: The PyZX graph corresponding to the requested circuit.

    """

    c = zx.Circuit(2)
    c.add_gate("CNOT", 1, 0)
    g = c.to_graph()

    fig = None
    if draw_graph:
        fig = zx.draw_matplotlib(g, labels=True)

    return g, fig


def cnots(draw_graph: bool = False) -> tuple[BaseGraph | GraphS, matplotlib.figure.Figure | None]:
    """Produce a PyZX graph corresponding to three CNOTs.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        g: The PyZX graph corresponding to the requested circuit.

    """

    c = zx.Circuit(2)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("CNOT", 1, 0)
    c.add_gate("CNOT", 1, 0)

    g = c.to_graph()

    fig = None
    if draw_graph:
        fig = zx.draw(g, labels=True)

    return g, fig


def simple_mess(draw_graph: bool = False) -> tuple[BaseGraph | GraphS, matplotlib.figure.Figure | None]:
    """Produce a PyZX graph corresponding to a small CNOT-based circuit.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        g: The PyZX graph corresponding to the requested circuit.

    """

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
    qubit_n: int,
    depth: int,
    draw_graph: bool = False
) -> tuple[BaseGraph | GraphS | None, matplotlib.figure.Figure | None]:
    """Produce a random PyZX graph.

    Args:
        qubit_n: The number of qubit lines in the desired graph.
        depth: The depth of the desired graph.
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        g: The PyZX graph corresponding to the requested circuit.

    """

    # Generate inside loop to check graph integrity
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
        c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=qubit_n, depth=depth, clifford=False)
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
                fig = zx.draw(g)

            # Return graph and figure
            return g, fig

    return None, None
