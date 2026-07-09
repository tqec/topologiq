"""Example of a script prepared for usage with Topologiq's UX.

IMPORTANT! Check OTHER scripts for examples of how to use Topologiq programmatically.
This script is exclusively for usage in the UX. It aims to be as plain as possible to mimic
simply using a circuit available in an existing codebase.

"""

import random

from topologiq.assets.pyzx_graphs import ghz, random_graph, steane, t

###############
# PYZX KWARGS #
###############
# These KWARGs are only for the generation of the PyZX graph.
# They do NOT get used by Topologiq.
# This is also not how you feed KWARGs directly to Topologiq.
# By extension, you only need to give KWARGs that deviate from default parameters.
# All default parameters are available at the repository root: `src/topologiq/kwargs.py`.
pyzx_kwargs = {"seed": 5}


#######
# RUN #
#######
if __name__ == "__main__":
    # Set seed if in KWARGS
    if "seed" in pyzx_kwargs:
        random.seed(pyzx_kwargs["seed"])

    # Retrieve circuit
    had_phase = False
    qubit_n, depth = (4, 10)
    pyzx_random, _ = random_graph(
        qubit_n,
        depth,
        p_t=0.2,
        draw_graph=False,
        graph_type="cnot_had_phase" if had_phase else "cnot",
        **pyzx_kwargs,
    )

    pyzx_steane, _ = steane(draw_graph=False)

    pyzx_ghz, _ = ghz(draw_graph=False)

    pyzx_t, _ = t(draw_graph=False)


