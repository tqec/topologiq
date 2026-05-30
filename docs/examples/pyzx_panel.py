"""Example of how to use Topologiq to perform LS with predefined PyZX graphs."""

from topologiq.assets import pyzx_graphs
from topologiq.input.zx_manager import ZXGraphManager

##########
# KWARGS #
##########
# Note. Topologiq auto-completes any KWARGs not given explicitly.
# By extension, you only need to give KWARGs that deviate from default parameters.
# All default parameters are available at the repository root: `src/topologiq/kwargs.py`.
kwargs = {"debug": 1, "first_id_strategy": "centrality_random"}

#######
# RUN #
#######
if __name__ == "__main__":
    # Import all graphs available in PyZX graphs script and call LS on them
    # - 3 * CNOTs (single, 3 in a row, multiple)
    # - 2 * Steane (minimal spiders & slightly obfuscated version)
    # - 3 * graphs using Hadamards (line, bend, and a Steane-like graph)

    # (OPTIONAL) Exclude graphs by name
    include = [
        # "xyi",
        # "cnot_cz",
        # "one_hadamard",
        # "cnot",
        # "cnots",
        # "simple_mess",
        # "hadamard_line",
        # "hadamard_bend",
        # "hadamard_mess",
        "steane",
        # "steane_obfuscated",
        # "ghz",
        # "yi",
        # "s",
        # "msc",
        # "t",
    ]

    # Loop over available encoding functions
    for graph_name in include:
        # Update user
        print(f"\n==> Now processing: {graph_name}.")

        # Get PyZX graph
        encoding_fx = getattr(pyzx_graphs, graph_name)
        pyzx_graph, _ = encoding_fx(draw_graph=False)

        # QASM -> ZX manager
        zx_graph_manager_in = ZXGraphManager()
        aug_zx_in = zx_graph_manager_in.add_graph_from_pyzx(pyzx_graph, graph_key="input")

        # Run Topologiq
        bgraph_manager = aug_zx_in.get_blockgraph(**kwargs)

        # Visualise results
        bgraph_manager.draw_blockgraph()

        # Verification for standard graphs
        if graph_name not in ["yi", "s", "msc", "t"]:
            # Verify input/output logical equality
            out_zx_graph_manager = ZXGraphManager()
            aug_zx_out = out_zx_graph_manager.add_graph_from_blockgraph(
                bgraph_manager, graph_key="output"
            )
            equality = aug_zx_in.check_equality(aug_zx_out)
            print(f"Equality verification: {equality}")
            print("------------------------------------\n")
