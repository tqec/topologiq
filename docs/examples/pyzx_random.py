"""Example of how to use Topologiq to perform LS on random PyZX graphs.

Notes:
    While we have identified improvements that might allow Topologiq to handle large graphs,
        this is not yet possible. You will start to see some attempts fail at around 50 spiders,
        which can be recovered by asking Topologiq to increase the number of attempts per graph.
        Graphs over 100 spiders might fail entirely irrespective of number of attempts.

"""

import random
from pathlib import Path

from pyinstrument import Profiler
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS

from topologiq.assets.pyzx_graphs import random_graph
from topologiq.input.pyzx_manager import ZXGraphManager

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT_DIR / "output/bgraph"


####################
# CIRCUIT RETRIEVE #
####################
def get_random_pyzx_circuit(
    qubit_n: int, depth: int, draw_graph: bool = False, **kwargs
) -> BaseGraph | GraphS:
    """Retrieve and return a random PyZX graph.

    Args:
        qubit_n: The number of qubits in the desired random circuit.
        depth: The depth of the the desired random circuit.
        draw_graph: Whether to pop-up PyZX graph visualisation or not.
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

    Returns:
        pyzx_graph: The requested random PyZX graph.
        circuit_name: The name for the circuit/graph.

    """
    if "seed" in kwargs:
        random.seed(kwargs["seed"])
    circuit_name = f"random_{kwargs['seed'] if kwargs.get('seed') else 'noseed'}_{qubit_n}_{depth}"
    pyzx_graph, _ = random_graph(qubit_n, depth, draw_graph=draw_graph, graph_type="cnot", **kwargs)
    return pyzx_graph, circuit_name


##########
# KWARGS #
##########
# If a specific kwarg is not explicitly declared here, it will be auto-generated
kwargs = {
    "first_id_strategy": "first_spider",
    "seed": 42,
    "debug": 1,
    "size_of_chip": (10, 10),
    "k": 3,
}


############
# MAIN RUN #
############
if __name__ == "__main__":

    # Start PyInstrument
    profiler = Profiler()

    # Retrieve circuit
    qubit_n, depth = (5, 50)
    pyzx_graph, circuit_name = get_random_pyzx_circuit(qubit_n, depth, draw_graph=True, **kwargs)

    # Convert ZX graph into AugmentedZXGraph
    profiler.start()
    zx_graph_manager = ZXGraphManager()
    aug_zx = zx_graph_manager.add_graph_from_pyzx(pyzx_graph, use_primary=True)

    # Run Topologiq
    bgraph_manager = aug_zx.get_blockgraph(**kwargs)
    profiler.stop()

    # Visualise results
    bgraph_manager.draw_blockgraph()

    # Write profiling results
    with open("profiled_run.html", "w") as f:
        f.write(profiler.output_html())
