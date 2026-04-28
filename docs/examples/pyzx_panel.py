"""Example of how to use Topologiq to perform LS on predefined PyZX graphs.

Usage:
    Run script as given.

"""

import matplotlib
import matplotlib.figure
import pyzx as zx
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS

from topologiq.assets.pyzx_graphs import random_graph
from topologiq.input.pyzx_manager import ZXGraphManager

#################
# SHARED KWARGS #
#################
kwargs = {
    "debug": 0,
}


###############
# PyZX GRAPHS #
###############
def simple_cnot_sequence(
    draw_graph: bool = False,
) -> tuple[BaseGraph | GraphS, matplotlib.figure.Figure | None]:
    """Produce a PyZX graph corresponding to a small CNOT-based circuit.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        zx_graph: The PyZX graph corresponding to the requested circuit.
        fig: The Matplotlib figure of the graph.

    """

    # Circuit
    pyzx_circuit = zx.Circuit(3)
    pyzx_circuit.add_gate("CNOT", 1, 2)
    pyzx_circuit.add_gate("CNOT", 1, 0)
    pyzx_circuit.add_gate("CNOT", 0, 1)
    pyzx_circuit.add_gate("CNOT", 0, 2)

    # --> Graph
    zx_graph = pyzx_circuit.to_graph()

    # Draw if needed
    fig = None
    if draw_graph:
        fig = zx.draw_matplotlib(zx_graph, labels=True)

    return zx_graph, fig


def steane_no_effects(
    draw_graph: bool = False,
) -> tuple[BaseGraph | GraphS | None, matplotlib.figure.Figure | None]:
    """Return an full (unreduced) PyZX graph of a Steane encoding.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        zx_graph: The PyZX graph corresponding to the requested circuit.
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
    zx_graph = pyzx_circuit.to_graph()

    # Draw if needed
    fig = None
    if draw_graph:
        fig = zx.draw(zx_graph, labels=True)

    return zx_graph, fig


def steane_effects(
    draw_graph: bool = False,
) -> tuple[BaseGraph | GraphS | None, matplotlib.figure.Figure | None]:
    """Return an full (unreduced) PyZX graph of a Steane encoding.

    Args:
        draw_graph: Whether to pop-up PyZX graph visualisation or not.

    Returns:
        zx_graph: The PyZX graph corresponding to the requested circuit.
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
    zx_graph = pyzx_circuit.to_graph()

    # States & effects
    num_apply_state = zx_graph.num_inputs()
    zx_graph.apply_state("0" * num_apply_state)
    zx_graph.apply_effect("000///////")

    # Draw if needed
    fig = None
    if draw_graph:
        fig = zx.draw(zx_graph, labels=True)

    return zx_graph, fig


# ...
if __name__ == "__main__":
    # Create circuit or import it from somewhere

    circuit_name = "simple_cnot_sequence"
    circuit_name = "random"
    circuit_name = "steane"
    effects = True  # Only kicks in if using Steane

    if circuit_name == "simple_cnot_sequence":
        zx_graph, fig_data = simple_cnot_sequence(draw_graph=False)

    if circuit_name == "random":
        seed = None
        qubit_n, depth = (3, 5)
        circuit_name = f"random_{seed if seed else 'noseed'}_{qubit_n}_{depth}"
        zx_graph, fig_data = random_graph(
            qubit_n, depth, graph_type="cnot", draw_graph=False, **kwargs
        )

    if circuit_name == "steane":
        if effects:
            zx_graph, fig_data = steane_effects(draw_graph=False)
        else:
            zx_graph, fig_data = steane_no_effects(draw_graph=False)

    # --> Augmented ZX Graph
    zx_graph_manager = ZXGraphManager()
    aug_zx = zx_graph_manager.add_graph_from_pyzx(zx_graph=zx_graph, graph_key=circuit_name)
    zx.draw(aug_zx.zx_graph, labels=True)
    zx.draw(aug_zx.zx_graph_reduced, labels=True)

    # Check if tags were preserved
    print("REDUCED")
    for v in aug_zx.zx_graph_reduced.vertices():
        if aug_zx.zx_graph_reduced.vdata(v, "b"):
            print(v, ":", aug_zx.zx_graph_reduced.vdata(v, "b"))

    # Compare inputs
    for i, v in enumerate([aug_zx.zx_graph, aug_zx.zx_graph_reduced]):
        print(
            f"- {'Full' if i == 0 else 'Reduced'} => ",
            "I:",
            v.inputs(),
            "O:",
            v.outputs(),
        )

    # Run Topologiq
    lattice_nodes, lattice_edges = aug_zx.get_blockgraph(
        circuit_name=circuit_name, use_reduced=False, final_vis=False
    )

    # Check equality
    aug_zx_out = zx_graph_manager.add_graph_from_blockgraph(
        blockgraph_cubes=lattice_nodes,
        blockgraph_pipes=lattice_edges,
        graph_key=f"{circuit_name}_out",
        other=aug_zx,
    )
    zx.draw(aug_zx_out.zx_graph, labels=True)
    zx.draw(aug_zx_out.zx_graph_reduced, labels=True)

    # Compare inputs
    for ele in (aug_zx, aug_zx_out):
        print(f"\nBOUNDARIES {'Input' if ele == aug_zx else 'Output'} ZX:")
        for i, v in enumerate([ele.zx_graph, ele.zx_graph_reduced]):
            print(
                f"- {'Full' if i == 0 else 'Reduced'} => ",
                "I:",
                v.inputs(),
                "O:",
                v.outputs(),
            )

    equal = aug_zx.check_equality(aug_zx_out)
    print("\n I/O EQUALITY VERIFICATION:", equal)
