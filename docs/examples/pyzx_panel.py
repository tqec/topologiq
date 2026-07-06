"""Example of how to use Topologiq to perform LS with predefined PyZX graphs."""

from topologiq.assets import pyzx_graphs
from topologiq.input.zx_manager import ZXGraphManager

##########
# KWARGS #
##########
# Note 1. Topologiq auto-completes KWARGs, so only give explicitly any KWARGs that deviate from defaults.
# All default parameters are available at the repository root: `src/topologiq/kwargs.py`.

# Note 2. This script uses a standard graph_traverse_mode but not necessarily the optimal for all graphs.
# Note 3. To get minimal volumes, change "twins" to `False`.
# Contributions geared to automatically choosing optimal KWARGs given graph type are extremely welcome.

# Note 4. Not all "first_id_strategy" and "graph_traverse_mode" are compatible with all graphs.
# Contributinos geared to create fallbacks for when specialised strategies fail are welcome.
kwargs = {
    "debug": 0,
    "first_id_strategy": "first-spider",
    "graph_traverse_mode": "bfs-cross",
    "gravity": 10,
    "z_stretch": 0,
    "twins": True,
}

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
        "xyi",
        "memory",
        "cnot_cz",
        "one_hadamard",
        "cnot",
        "cnots",
        "simple_mess",
        "split_loops",
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
    ]

    # Loop over available encoding functions
    for graph_name in include:
        # Update user
        print(f"\n==> Now processing: {graph_name}.")

        # Get PyZX graph
        encoding_fx = getattr(pyzx_graphs, graph_name)
        pyzx_graph, _ = encoding_fx(draw_graph=True)

        # QASM -> ZX manager
        zx_graph_manager = ZXGraphManager()
        aug_zx_in = zx_graph_manager.add_graph_from_pyzx(pyzx_graph, graph_key="input")

        # Run Topologiq
        bgraph_manager = aug_zx_in.get_blockgraph(**kwargs)

        # Visualise results
        bgraph_manager.draw_blockgraph()

        # Write results to file
        bgraph_manager.write_bgraph(circuit_name=graph_name)

        # Animate and clean up
        if kwargs.get("animate"):
            bgraph_manager.animate(filename_prefix=graph_name)

        # Verification for standard graphs
        if graph_name not in ["yi", "s", "msc", "t"]:
            # Verify input/output logical equality
            aug_zx_out = zx_graph_manager.add_graph_from_blockgraph(
                bgraph_manager, graph_key="output"
            )
            equality = aug_zx_in.check_equality(aug_zx_out)
            print(f"Equality verification: {equality}")
            print("------------------------------------\n")
