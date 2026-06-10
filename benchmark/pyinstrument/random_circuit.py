"""PyInstrument profiling using a random PyZX graph."""

import random
from pathlib import Path

from pyinstrument import Profiler

from topologiq.assets.pyzx_graphs import random_graph
from topologiq.input.zx_manager import ZXGraphManager

##########
# KWARGS #
##########
# If a specific kwarg is not explicitly declared here, it will be auto-generated
kwargs = {"first_id_strategy": "first_spider", "seed": 42, "debug": 0}

############
# MAIN RUN #
############
if __name__ == "__main__":
    # Set seed if in KWARGS
    if "seed" in kwargs:
        random.seed(kwargs["seed"])

    # Profile with and without Hadamards and phases
    for had_phase in [True]:
        # Start PyInstrument
        profiler = Profiler()
        profiler.start()

        # Retrieve circuit
        qubit_n, depth = (5, 10)
        circuit_name = (
            f"random_{kwargs['seed'] if kwargs.get('seed') else 'noseed'}_{qubit_n}_{depth}"
        )
        pyzx_graph, _ = random_graph(
            qubit_n,
            depth,
            draw_graph=False,
            graph_type="cnot_had_phase" if had_phase else "cnot",
            **kwargs,
        )

        # Convert ZX graph into AugmentedZXGraph
        zx_graph_manager = ZXGraphManager()
        aug_zx = zx_graph_manager.add_graph_from_pyzx(pyzx_graph, use_primary=True)

        # Run Topologiq
        bgraph_manager = aug_zx.get_blockgraph(**kwargs)
        bgraph_manager.draw_blockgraph()

        # Run Topologiq
        profiler.stop()

        # Write profiling results
        path_to_profiled_run = (
            Path(__file__).resolve().parent
            / f"pyinst_random{'_had_phase' if had_phase else ''}.html"
        )
        with open(path_to_profiled_run, "w") as f:
            f.write(profiler.output_html())
