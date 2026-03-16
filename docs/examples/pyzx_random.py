"""Example of how to use Topologiq to perform LS on random PyZX graphs.

Notes:
    While we have identified improvements that might allow Topologiq to handle large graphs,
        this is not yet possible. You will start to see some attempts fail at around 50 spiders,
        which can be recovered by asking Topologiq to increase the number of attempts per graph.
        Graphs over 100 spiders might fail entirely irrespective of number of attempts.

"""

import matplotlib.figure
import pyzx as zx
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS

from topologiq.assets.pyzx_graphs import random_graph
from topologiq.core.graph_manager.graph_manager import runner
from topologiq.input.pyzx_manager import ZXGraphManager, pyzx_g_to_simple_g


####################
# MAIN RUN MANAGER #
####################
def run_random(pyzx_graph: BaseGraph | GraphS, fig_data: matplotlib.figure.Figure, **kwargs):
    """Run a single random PyZX graph m_times.

    Args:
        pyzx_graph: The random PyZX graph.
        fig_data (optional): The visualisation of the input ZX graph (to overlay it over other visualisations).
        **kwargs: See `./kwargs.py` for a comprehensive breakdown.
            NB! If an arbitrary kwarg is not given explicitly, it is created against defaults on `./src/topologiq/kwargs.py`.
            NB! By extension, it only makes sense to give the specific kwargs where user wants to deviate from defaults.

    """

    # Convert ZX graph into AugmentedZXGraph
    zx_graph_manager = ZXGraphManager()
    aug_zx_graph = zx_graph_manager.add_graph_from_pyzx(pyzx_graph, use_primary=True)
    zx.draw(aug_zx_graph.zx_graph)

    # Call Topologiq on `simple_graph` of circuit
    simple_graph = pyzx_g_to_simple_g(aug_zx_graph.zx_graph)
    if circuit_name and simple_graph["nodes"] and simple_graph["edges"]:
        _, _, _, _ = runner(
            simple_graph,
            circuit_name,
            fig_data=fig_data,
            **kwargs,
        )


# ...
if __name__ == "__main__":
    # KWARGs (if not here, kwarg is auto-generated)
    kwargs = {
        "first_id_strategy": "first_spider",
        "seed": None,
        "vis_options": ("final", None),
        "max_attempts": 1,
        "stop_on_first_success": True,
        "debug": 1,
        "log_stats": False,
    }

    # Circuit
    qubit_n = 5
    depth = 5
    circuit_name = f"random_{kwargs['seed'] if kwargs.get('seed') else 'noseed'}_{qubit_n}_{depth}"
    pyzx_graph, fig_data = random_graph(qubit_n, depth, graph_type="cnot", **kwargs)

    # Run Topologiq
    run_random(pyzx_graph, fig_data, **kwargs)
