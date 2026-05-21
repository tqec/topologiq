"""Example of how to use Topologiq  to perform LS on predefined PyZX graphs.

The script contains a loop that goes over several different PyZX graphs. It includes
automated input/output verification of results (currently only available for very small graphs).

Usage:
    Run script as given.

"""

from topologiq.assets import pyzx_graphs
from topologiq.input.pyzx_manager import ZXGraphManager

#################
# SHARED KWARGS #
#################
kwargs = {
    "first_id_strategy": "first_spider",
    "debug": 1,
    # "seed": 69,
    "graph_traversing_mode": "bfs-cross",
    "size_of_chip": (10, 10),
    "k": 3,
}

#############
# MAIN LOOP #
#############
if __name__ == "__main__":
    # Import all graphs available in PyZX graphs script and call LS on them
    # - 3 * CNOTs (single, 3 in a row, multiple)
    # - 2 * Steane (minimal spiders & slightly obfuscated version)
    # - 3 * graphs using Hadamards (line, bend, and a Steane-like graph)

    # (OPTIONAL) Exclude graphs by name
    include = ["line_with_t"]
    # [
        # "cnot", "cnots", "simple_mess",
        # "hadamard_line", "hadamard_bend", "hadamard_mess",
        # "steane", "steane_obfuscated",
        # "ghz",
        # "y_init", "line_with_s",
        # "t_init", "line_with_t",
    # ]

    # Loop over available encoding functions
    for graph_name in include:
        # Update user
        print(f"\n==> Now processing: {graph_name}.")

        # Get PyZX graph
        encoding_fx = getattr(pyzx_graphs, graph_name)
        pyzx_graph, _ = encoding_fx(draw_graph=True)

        # QASM -> ZX manager
        zx_graph_manager_in = ZXGraphManager()
        aug_zx_in = zx_graph_manager_in.add_graph_from_pyzx(pyzx_graph, graph_key="input")

        # Run Topologiq
        bgraph_manager = aug_zx_in.get_blockgraph(**kwargs)

        # Visualise results
        bgraph_manager.draw_blockgraph()

        # Standard graphs
        if graph_name not in ["y_init", "line_with_s", "t_init", "line_with_t"]:
            # Verify input/output logical equality
            out_zx_graph_manager = ZXGraphManager()
            aug_zx_out = out_zx_graph_manager.add_graph_from_blockgraph(
                bgraph_manager, graph_key="output"
            )
            equality = aug_zx_in.check_equality(aug_zx_out)
            print(f"Equality verification: {equality}")
            print("------------------------------------\n")
