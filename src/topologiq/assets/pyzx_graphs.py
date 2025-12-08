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
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    pyzx_circuit = zx.Circuit(2)
    pyzx_circuit.add_gate("CNOT", 1, 0)
    pyzx_graph = pyzx_circuit.to_graph()

    fig = None
    if draw_graph:
        fig = zx.draw_matplotlib(pyzx_graph, labels=True)

    return pyzx_graph, fig


def cnots(draw_graph: bool = False) -> tuple[BaseGraph | GraphS, matplotlib.figure.Figure | None]:
    """Produce a PyZX graph corresponding to three CNOTs.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    pyzx_circuit = zx.Circuit(2)
    pyzx_circuit.add_gate("CNOT", 0, 1)
    pyzx_circuit.add_gate("CNOT", 1, 0)
    pyzx_circuit.add_gate("CNOT", 1, 0)

    pyzx_graph = pyzx_circuit.to_graph()

    fig = None
    if draw_graph:
        fig = zx.draw(pyzx_graph, labels=True)

    return pyzx_graph, fig


def simple_mess(draw_graph: bool = False) -> tuple[BaseGraph | GraphS, matplotlib.figure.Figure | None]:
    """Produce a PyZX graph corresponding to a small CNOT-based circuit.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    pyzx_circuit = zx.Circuit(3)
    pyzx_circuit.add_gate("CNOT", 1, 2)
    pyzx_circuit.add_gate("S", 2)
    pyzx_circuit.add_gate("CNOT", 1, 0)
    pyzx_circuit.add_gate("CNOT", 0, 1)
    pyzx_circuit.add_gate("S", 2)
    pyzx_circuit.add_gate("CNOT", 0, 2)

    pyzx_graph = pyzx_circuit.to_graph()

    fig = None
    if draw_graph:
        fig = zx.draw(pyzx_graph, labels=True)

    return pyzx_graph, fig


def steane_pyzx(draw_graph: bool = False) -> tuple[BaseGraph | GraphS | None, matplotlib.figure.Figure | None]:
    """Return an full (unreduced) PyZX graph of a Steane encoding.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    pyzx_circuit = zx.Circuit(10)

    ancilla_qubits = [0, 1, 2]
    qubits= [[3, 4, 5, 6], [3, 4, 7, 8], [3, 5, 7, 9]]

    for i, ancilla_qubit in enumerate(ancilla_qubits):
        pyzx_circuit.add_gate("HAD", ancilla_qubit)
        for qubit in qubits[i]:
            pyzx_circuit.add_gate("CNOT", ancilla_qubit, qubit)
        pyzx_circuit.add_gate("HAD", ancilla_qubit)
        pyzx_circuit.add_gate("PostSelect", ancilla_qubit)

    pyzx_graph = pyzx_circuit.to_graph()
    pyzx_graph.apply_state('0'*10)

    return pyzx_graph


def random_graph(
    qubit_n: int,
    depth: int,
    graph_type: str = "cnot",
    draw_graph: bool = False
) -> tuple[BaseGraph | GraphS | None, matplotlib.figure.Figure | None]:
    """Produce a random PyZX graph.

    Args:
        qubit_n: The number of qubit lines in the desired graph.
        depth: The depth of the desired graph.
        graph_type: The type of graph to generate.
            "cnot": A graph composed of only CNOTs.
            "cnot_had_phase": A graph with CNOTs, Hadamards, and phases.
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

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
        if graph_type == "cnot_had_phase":
            pyzx_circuit = zx.generate.CNOT_HAD_PHASE_circuit(qubits=qubit_n, depth=depth, clifford=False)
            pyzx_graph = pyzx_circuit.to_graph()
        elif graph_type == "cnot":
            pyzx_graph = zx.generate.cnots(qubits=qubit_n, depth=depth)
        else:
            raise ValueError('ERROR generating random graph. Invalid graph type. Valid graph types are: "cnot", "cnot_had_phase".')

        # Run a canonical BFS loop to confirm all spiders are hit by BFS
        queue = []
        visited = {}

        ids_original_spiders = list(pyzx_graph.vertices())
        queue.append(ids_original_spiders[0])
        visited[ids_original_spiders[0]] = True
        while queue:
            nxt = queue.pop(0)
            all_neighbours = list(pyzx_graph.neighbors(nxt))
            for neigh in all_neighbours:
                if neigh not in visited:
                    queue.append(neigh)
                    visited[neigh] = True

        # Check BFS visited IDs against original PyZX graph
        if ids_original_spiders == sorted(list(visited.keys())):
            # Return if all IDs are present
            if draw_graph:
                fig = zx.draw_matplotlib(pyzx_graph, labels=True)

            # Return graph and figure
            return pyzx_graph, fig

    return None, None
