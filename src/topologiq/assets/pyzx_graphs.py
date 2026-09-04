"""PyZX graphs to use in examples, demonstrations and testing.

Usage:
    Call any graph from a separate script.

"""

import random

import pyzx as zx
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS
from pyzx.utils import EdgeType

from topologiq.utils.zx import apply_bialgebra, rm_unnecessary_phases


######################
# ENCODING FUNCTIONS #
######################
def xyi(draw_graph: bool = False) -> BaseGraph | GraphS:
    """Produce a PyZX graph with a single X and a single Z spider.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    pyzx_circuit = zx.Circuit(1)
    pyzx_circuit.add_gate("NOT", 0)
    pyzx_graph = pyzx_circuit.to_graph()
    rm_unnecessary_phases(pyzx_graph)

    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def memory(draw_graph: bool = False) -> BaseGraph | GraphS:
    """Produce a PyZX graph with a single X and a single Z spider.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    pyzx_circuit = zx.Circuit(1)
    pyzx_circuit.add_gate("Z", 0)
    pyzx_circuit.add_gate("Z", 0)
    pyzx_circuit.add_gate("Z", 0)
    pyzx_graph = pyzx_circuit.to_graph()
    rm_unnecessary_phases(pyzx_graph)

    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def cnot_cz(draw_graph: bool = False) -> BaseGraph | GraphS:
    """Produce a PyZX graph with a single X and a single Z spider.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    pyzx_circuit = zx.Circuit(1)
    pyzx_circuit.add_gate("NOT", 0)
    pyzx_circuit.add_gate("Z", 0)
    pyzx_graph = pyzx_circuit.to_graph()
    pyzx_graph.remove_vertices([0, 3])
    pyzx_graph.set_inputs(tuple([1]))
    pyzx_graph.set_outputs(tuple([2]))
    rm_unnecessary_phases(pyzx_graph)

    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def one_hadamard(
    draw_graph: bool = False,
) -> BaseGraph | GraphS:
    """Return an PyZX graph with a single Hadamard.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    # Foundational circuit
    pyzx_circuit = zx.Circuit(1)

    # GHZ encoding
    pyzx_circuit.add_gates("Z H", 0)
    pyzx_graph = pyzx_circuit.to_graph()
    pyzx_graph.set_edge_type((1, 2), EdgeType.HADAMARD)
    pyzx_graph.remove_vertices([0, 3])
    pyzx_graph.set_inputs(tuple([1]))
    pyzx_graph.set_outputs(tuple([2]))
    rm_unnecessary_phases(pyzx_graph)

    # Draw if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def cnot(draw_graph: bool = False) -> BaseGraph | GraphS:
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

    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def cnots(draw_graph: bool = False) -> BaseGraph | GraphS:
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

    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def simple_mess(
    draw_graph: bool = False,
) -> BaseGraph | GraphS:
    """Produce a PyZX graph corresponding to a small CNOT-based circuit.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    pyzx_circuit = zx.Circuit(3)
    pyzx_circuit.add_gate("CNOT", 1, 2)
    pyzx_circuit.add_gate("Z", 2)
    pyzx_circuit.add_gate("CNOT", 1, 0)
    pyzx_circuit.add_gate("CNOT", 0, 1)
    pyzx_circuit.add_gate("Z", 2)
    pyzx_circuit.add_gate("CNOT", 0, 2)

    pyzx_graph = pyzx_circuit.to_graph()

    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def split_loops(
    draw_graph: bool = False,
) -> BaseGraph | GraphS:
    """Return a PyZX graph with two separate cycles connected via a central bridge.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    spiders = {0: [14, 15, 16], 1: [1, 3, 5, 7, 9, 11, 13], 2: [2, 4, 6, 8, 10, 12]}
    edges = {
        1: [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 1),
            (4, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 8),
            (7, 13),
            (5, 14),
            (13, 15),
            (10, 16),
        ]
    }

    # Foundational graph
    pyzx_graph = zx.Graph()

    # Add spiders
    for k, spider_ids in spiders.items():
        for spider_id in spider_ids:
            qubit = (
                1
                if spider_id in [1, 2, 3, 12]
                else 3
                if spider_id in [13, 14, 16]
                else 2
                if spider_id != 15
                else 4
            )
            row = (
                1
                if spider_id in [5, 14]
                else 7
                if spider_id in [13, 15]
                else 9
                if spider_id in [12, 16]
                else spider_id
            )
            pyzx_graph.add_vertex(ty=k, index=spider_id, qubit=qubit, row=row)

    for k, edge_pairs in edges.items():
        for u, v in edge_pairs:
            pyzx_graph.add_edge((u, v), edgetype=k)

    pyzx_graph.set_outputs(spiders[0])

    # Draw if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def bialg(
    draw_graph: bool = False,
) -> BaseGraph | GraphS:
    """Return a PyZX graph of a Steane code.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    spiders = {0: [9, 10, 11, 12], 1: [1, 2, 3], 2: [5, 6]}
    edges = {
        1: [
            (1, 5),
            (2, 6),
            (3, 6),
            (2, 5),
            (1, 6),
            (9, 1),
            (10, 2),
            (11, 5),
            (12, 6),
        ]
    }

    rows = {0: [11, 12], 1: [3, 5, 6], 2: [1, 2], 3: [9, 10]}
    qubits = {0: [1, 5, 9, 11], 1: [2, 6, 10, 12], 2: [3]}

    # Foundational graph
    pyzx_graph = zx.Graph()

    # Add spiders
    for k, spider_ids in spiders.items():
        for i, spider_id in enumerate(spider_ids):
            pyzx_graph.add_vertex(ty=k, index=spider_id)

    for k, edge_pairs in edges.items():
        for u, v in edge_pairs:
            pyzx_graph.add_edge((u, v), edgetype=k)

    # Re-organise rows
    for r, spider_ids in rows.items():
        for spider_id in spider_ids:
            pyzx_graph.set_row(spider_id, r)
    for q, spider_ids in qubits.items():
        for spider_id in spider_ids:
            pyzx_graph.set_qubit(spider_id, q)

    # Draw if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    # Adjust spiders in pattern to meet transformation requirements
    pyzx_graph.add_vertex(ty=2, index=13, qubit=2, row=0)
    pyzx_graph.remove_edges([(12, 6), (3, 6)])
    pyzx_graph.add_edges([(12, 13), (6, 13), (13, 3)])

    # Draw again if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    # Simplify
    zx.simplify.bialg_op_simp.apply(pyzx_graph, [5, 6, 1, 2])

    # Draw again if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def steane(draw_graph: bool = False, use_bialgebra=True) -> BaseGraph | GraphS:
    """Return a PyZX graph of a Steane code.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.
        use_bialgebra: Whether to apply the bialgebra rule to graph or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    # Foundational circuit
    pyzx_circuit = zx.Circuit(10)

    ancilla_qubits = [0, 1, 2]
    qubits = [[3, 4, 5, 6], [3, 4, 7, 8], [3, 5, 7, 9]]
    for i, ancilla_qubit in enumerate(ancilla_qubits):
        pyzx_circuit.add_gate("HAD", ancilla_qubit)
        for qubit in qubits[i]:
            pyzx_circuit.add_gate("CNOT", ancilla_qubit, qubit)
        pyzx_circuit.add_gate("HAD", ancilla_qubit)
    pyzx_graph = pyzx_circuit.to_graph()

    # States & effects
    num_apply_state = pyzx_graph.num_inputs()
    pyzx_graph.apply_state("0" * num_apply_state)
    pyzx_graph.apply_effect("000///////")

    # Reduction
    zx.full_reduce(pyzx_graph)
    zx.to_rg(pyzx_graph)
    zx.phase_free_simp(pyzx_graph)

    # Re-organise rows
    rows = {0: [43, 44, 45, 47], 4: [11, 13, 15, 25], 8: [10, 20, 30], 12: [46, 48, 49]}
    qubits = {0: [10, 11, 43, 46], 2: [13, 20, 44, 48], 4: [15, 30, 45, 49], 6: [25, 47]}
    for r, spider_ids in rows.items():
        for spider_id in spider_ids:
            pyzx_graph.set_row(spider_id, r)
    for q, spider_ids in qubits.items():
        for spider_id in spider_ids:
            pyzx_graph.set_qubit(spider_id, q)

    if use_bialgebra:
        apply_bialgebra(pyzx_graph)
        zx.simplify.id_simp(pyzx_graph)

    # Draw if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def steane_obfuscated(
    draw_graph: bool = False,
) -> BaseGraph | GraphS:
    """Return a PyZX graph of a Steane code with one more spider than its fully reduced version.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    spiders = {0: [9, 10, 11, 12, 13, 14, 15], 1: [1, 5, 6, 7], 2: [4, 8, 3, 2]}
    edges = {
        1: [
            (4, 1),
            (4, 7),
            (8, 7),
            (8, 6),
            (8, 5),
            (3, 6),
            (3, 1),
            (2, 5),
            (2, 1),
            (1, 9),
            (2, 12),
            (3, 11),
            (4, 10),
            (5, 13),
            (6, 14),
            (7, 15),
        ]
    }

    # Foundational graph
    pyzx_graph = zx.Graph()

    # Add spiders
    for k, spider_ids in spiders.items():
        if k == 0:
            row = 0
        else:
            row = 2 if k == 1 else 1
        for i, spider_id in enumerate(spider_ids):
            if k == 0:
                row = 3 if spider_id in [9, 13, 14, 15] else 0
            pyzx_graph.add_vertex(ty=k, index=spider_id, qubit=i, row=row)

    for k, edge_pairs in edges.items():
        for u, v in edge_pairs:
            pyzx_graph.add_edge((u, v), edgetype=k)

    pyzx_graph.set_outputs(spiders[0])

    # Draw if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def hadamard_line(
    draw_graph: bool = False,
) -> BaseGraph | GraphS:
    """Return an PyZX graph of a line of Hadamards.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    # Foundational circuit
    pyzx_circuit = zx.Circuit(1)

    # GHZ encoding
    pyzx_circuit.add_gates("H H H H H H", 0)
    pyzx_graph = pyzx_circuit.to_graph()
    pyzx_graph.set_edge_type((6, 7), EdgeType.HADAMARD)
    for i in [2, 4, 6]:
        pyzx_graph.set_type(i, 2)

    # Draw if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def hadamard_bend(
    draw_graph: bool = False,
) -> BaseGraph | GraphS:
    """Return an PyZX graph of a 2-qubit Hadamard sequence.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    # Foundational circuit
    pyzx_circuit = zx.Circuit(2)

    # Encoding
    pyzx_circuit.add_gates("H H", 0)
    pyzx_circuit.add_gate("CNOT", 0, 1)
    pyzx_circuit.add_gates("H H", 0)
    pyzx_circuit.add_gates("H H", 1)
    pyzx_graph = pyzx_circuit.to_graph()
    pyzx_graph.remove_vertex(1)
    for i in [3, 6, 9]:
        pyzx_graph.set_type(i, 2)
    for u, v in [(3, 5), (5, 4), (7, 10), (9, 11)]:
        pyzx_graph.set_edge_type((u, v), EdgeType.HADAMARD)

    # Draw if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def hadamard_mess(
    draw_graph: bool = False,
) -> BaseGraph | GraphS:
    """Return a PyZX graph of a Steane code with Hadamards added for complexity.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    # Foundational circuit
    pyzx_circuit = zx.Circuit(10)

    ancilla_qubits = [0, 1, 2]
    qubits = [[3, 4, 5, 6], [3, 4, 7, 8], [3, 5, 7, 9]]
    for i, ancilla_qubit in enumerate(ancilla_qubits):
        pyzx_circuit.add_gate("HAD", ancilla_qubit)
        for qubit in qubits[i]:
            pyzx_circuit.add_gate("CNOT", ancilla_qubit, qubit)
        pyzx_circuit.add_gate("HAD", ancilla_qubit)
    pyzx_graph = pyzx_circuit.to_graph()

    # States & effects
    num_apply_state = pyzx_graph.num_inputs()
    pyzx_graph.apply_state("0" * num_apply_state)
    pyzx_graph.apply_effect("000///////")

    # Reduction
    zx.full_reduce(pyzx_graph)
    zx.to_rg(pyzx_graph)
    zx.phase_free_simp(pyzx_graph)

    # Exchange some edges for Hadamards
    for u, v in [(11, 20), (20, 25), (15, 30)]:
        pyzx_graph.set_edge_type((u, v), EdgeType.HADAMARD)

    # Draw if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def ghz(
    draw_graph: bool = False,
) -> BaseGraph | GraphS:
    """Return an PyZX graph of a GHZ.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    # Foundational circuit
    n_qubits = 16
    pyzx_circuit = zx.Circuit(n_qubits)

    # GHZ encoding
    pyzx_circuit.add_gate("HAD", 0)
    for i in range(n_qubits - 1):
        pyzx_circuit.add_gate("CNOT", i, i + 1)
    pyzx_graph = pyzx_circuit.to_graph()

    # States & effects
    num_apply_state = pyzx_graph.num_inputs()
    pyzx_graph.apply_state("0" * num_apply_state)

    # Reduction
    zx.full_reduce(pyzx_graph)
    zx.to_rg(pyzx_graph)
    zx.phase_free_simp(pyzx_graph)

    # Draw if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def random_graph(
    qubit_n: int,
    depth: int,
    p_t: int = 0.2,
    graph_type: str = "cnot",
    draw_graph: bool = False,
    **kwargs,
) -> BaseGraph | GraphS:
    """Produce a random PyZX graph.

    Args:
        qubit_n: The number of qubit lines in the desired graph.
        depth: The depth of the desired graph.
        p_t (optional): Probability of getting a T-gate.
        graph_type (optional): The type of graph to generate.
            "cnot": A graph composed of only CNOTs.
            "cnot_had_phase": A graph with CNOTs, Hadamards, and phases.
        draw_graph (optional): Whether to pop-up PyZX graph visualisation or not.
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    # Set seed if desired
    if kwargs.get("seed"):
        random.seed(kwargs["seed"])

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
            pyzx_circuit = zx.generate.CNOT_HAD_PHASE_circuit(
                qubits=qubit_n, depth=depth, clifford=False, p_t=p_t
            )
            pyzx_graph = pyzx_circuit.to_graph()
        elif graph_type == "cnot":
            pyzx_graph = zx.generate.cnots(qubits=qubit_n, depth=depth)
        else:
            raise ValueError(
                'ERROR generating random graph. Invalid graph type. Valid graph types are: "cnot", "cnot_had_phase".'
            )

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
            # Gadgetise if phases involved
            if graph_type == "cnot_had_phase":
                zx.simplify.gadgetize(pyzx_graph, graphlike=False)
                zx.id_simp(pyzx_graph)

            # Return if all IDs are present
            if draw_graph:
                zx.draw(pyzx_graph, labels=True)

            # Return graph and figure
            return pyzx_graph

    return None, None


def yi(draw_graph: bool = False) -> BaseGraph | GraphS:
    """Return an PyZX graph of a single Y-cube followed by a colour spider.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """
    # Foundational circuit
    pyzx_circuit = zx.Circuit(1)

    # GHZ encoding
    pyzx_circuit.add_gates("S NOT", 0)
    pyzx_graph = pyzx_circuit.to_graph()
    pyzx_graph.remove_vertices([0, 3])
    pyzx_graph.set_inputs(tuple([1]))
    pyzx_graph.set_outputs(tuple([2]))

    # Draw if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def s(
    draw_graph: bool = False,
) -> BaseGraph | GraphS:
    """Return an PyZX graph of a line of gates with an S in the middle.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """
    # Foundational circuit
    pyzx_circuit = zx.Circuit(1)

    # GHZ encoding
    pyzx_circuit.add_gates("S", 0)
    pyzx_graph = pyzx_circuit.to_graph()
    rm_unnecessary_phases(pyzx_graph)

    # Draw if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def msc(draw_graph: bool = False) -> BaseGraph | GraphS:
    """Return an PyZX graph of a single T-spider followed by a colour spider.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """
    # Foundational circuit
    pyzx_circuit = zx.Circuit(1)

    # GHZ encoding
    pyzx_circuit.add_gates("T NOT", 0)
    pyzx_graph = pyzx_circuit.to_graph()
    rm_unnecessary_phases(pyzx_graph)
    pyzx_graph.remove_vertices([0, 3])
    pyzx_graph.set_inputs(tuple([1]))
    pyzx_graph.set_outputs(tuple([2]))

    # Draw if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def t(
    draw_graph: bool = False,
) -> BaseGraph | GraphS:
    """Return an PyZX graph of a single Y-cube followed by a colour spider.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """
    # Foundational circuit
    pyzx_circuit = zx.Circuit(1)

    # GHZ encoding
    pyzx_circuit.add_gates("T", 0)
    pyzx_graph = pyzx_circuit.to_graph()

    # Gadgetise
    zx.simplify.gadgetize(pyzx_graph, graphlike=False)
    zx.id_simp(pyzx_graph)
    rm_unnecessary_phases(pyzx_graph)

    # Draw if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def ht(draw_graph: bool = False, num_t: int = 5) -> BaseGraph | GraphS:
    """Return an PyZX graph of a single Y-cube followed by a colour spider.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.
        num_t (optional): The number of T-gates in circuit.

    Returns:
        pyzx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """
    # Foundational circuit
    pyzx_circuit = zx.Circuit(1)

    # Add as many T-gates as desired
    for i in range(num_t):
        pyzx_circuit.add_gates("T", 0)

    # Convert to graph
    pyzx_graph = pyzx_circuit.to_graph()

    # Exchange all edges for Hadamards
    [pyzx_graph.set_edge_type(e, 2) for e in pyzx_graph.edges()]

    # Gadgetise
    zx.simplify.gadgetize(pyzx_graph, graphlike=False)
    zx.id_simp(pyzx_graph)
    # rm_unnecessary_phases(pyzx_graph)

    # Draw if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph


def disconnected_graph(draw_graph: bool = False) -> BaseGraph | GraphS:
    """Return a PyZX graph with disconnected subgraphs.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        pyzx_graph: The requested PyZX graph.

    """

    spiders = {0: [14, 15, 16], 1: [1, 3, 5, 7, 9, 11, 13], 2: [2, 4, 6, 8, 10, 12]}
    edges = {
        1: [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 1),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 8),
            (7, 13),
            (5, 14),
            (13, 15),
            (10, 16),
        ]
    }

    # Foundational graph
    pyzx_graph = zx.Graph()

    # Add spiders
    for k, spider_ids in spiders.items():
        for spider_id in spider_ids:
            qubit = (
                1
                if spider_id in [1, 2, 3, 12]
                else 3
                if spider_id in [13, 14, 16]
                else 2
                if spider_id != 15
                else 4
            )
            row = (
                1
                if spider_id in [5, 14]
                else 7
                if spider_id in [13, 15]
                else 9
                if spider_id in [12, 16]
                else spider_id
            )
            pyzx_graph.add_vertex(ty=k, index=spider_id, qubit=qubit, row=row)

    for k, edge_pairs in edges.items():
        for u, v in edge_pairs:
            pyzx_graph.add_edge((u, v), edgetype=k)

    pyzx_graph.set_outputs(spiders[0])

    # Draw if needed
    if draw_graph:
        zx.draw(pyzx_graph, labels=True)

    return pyzx_graph

##############
# PUBLIC DEF #
##############
__all__ = [  # noqa: RUF022  (do not sort: circuits organised in increasing order of difficulty)
    "xyi",
    "cnot_cz",
    "cnot",
    "cnots",
    "simple_mess",
    "hadamard_line",
    "hadamard_bend",
    "steane",
    "steane_obfuscated",
    "hadamard_mess",
    "ghz",
    "yi",
    "s",
    "msc",
    "t",
    "split_loops",
    "disconnected_graph"
]


if __name__ == "__main__":
    pyzx_graph, _ = ht(draw_graph=True)
