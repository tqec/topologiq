"""Example of a script prepared for usage with Topologiq's UX.

IMPORTANT! Check OTHER scripts for examples of how to use Topologiq programmatically.
This script is exclusively for usage in the UX. At any given point in time, it
may not be in a runnable shape because the UX allows direct editing of the file
via an IDE, so the file sometimes changes for no other reason than showing it can be
rewritten arbitrarily from the UX.

"""

import random

from topologiq.assets.pyzx_graphs import (
    cnot_cz,  # noqa: F401
    memory,  # noqa: F401
    msc,  # noqa: F401
    one_hadamard,  # noqa: F401
    random_graph,
    s,  # noqa: F401
    t,  # noqa: F401
    xyi,  # noqa: F401
    yi,  # noqa: F401
)

# from topologiq.input.zx_manager import ZXGraphManager

###############
# PYZX KWARGS #
###############
# These KWARGs are only for the generation of the PyZX graph.
# They do NOT get used by Topologiq.
# This is also not how you feed KWARGs directly to Topologiq.
# By extension, you only need to give KWARGs that deviate from default parameters.
# All default parameters are available at the repository root: `src/topologiq/kwargs.py`.
pyzx_kwargs = {"seed": 5}
kwargs = {
    "debug": 3,  # Verbosity. Change to `3` for step by step visuals.
    "first_id_strategy": "first-spider",  # Strategy for choosing the first spider/cube ID.
    "graph_traverse_mode": "tfs",  # Graph traversing strategy
    "gravity": 7,  # Integer weight that pulls paths towards graph centre
    "z_stretch": 1,
    # "animate": "MP4",
}

#######
# RUN #
#######
if __name__ == "__main__":
    # Set seed if in KWARGS
    if "seed" in pyzx_kwargs:
        random.seed(pyzx_kwargs["seed"])

    # Retrieve circuit
    had_phase = True
    qubit_n, depth = (4, 10)
    graph_name = f"random_pyzx_{qubit_n}_{depth}"
    pyzx_graph = random_graph(
        qubit_n,
        depth,
        p_t=0.6,
        draw_graph=False,
        graph_type="cnot_had_phase" if had_phase else "cnot",
        **pyzx_kwargs,
    )

    # pyzx_graph = xyi(draw_graph=False)
    # graph_name = "pyzx_xyi"

    # QASM -> ZX manager
    # zx_graph_manager = ZXGraphManager()
    # aug_zx_in = zx_graph_manager.add_graph_from_pyzx(pyzx_graph, graph_key="input")

    # Run Topologiq
    # bgraph_manager = aug_zx_in.get_blockgraph(**kwargs)

    # Visualise results
    # bgraph_manager.draw_blockgraph()

    # Animate and clean up
    # if kwargs.get("animate"):
    # bgraph_manager.animate(filename_prefix=graph_name)
