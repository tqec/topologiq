"""Example of how to use Topologiq to perform LS with predefined PyZX graphs."""

from topologiq.assets import pyzx_graphs
from topologiq.core.graph_manager.graph_manager import BlockGraphManager
from topologiq.input.zx_manager import ZXGraphManager

##########
# KWARGS #
##########
# Note 1. Topologiq auto-completes KWARGs, so only give explicitly any KWARGs that deviate from defaults.
# All default parameters are available at the repository root: `src/topologiq/kwargs.py`.

# Note 2. Script uses a all-terrain graph_traverse_mode that is not necessarily optimal across all cases.
# Note 3. To get minimal volumes, change "twins" to `False`.
# Contributions geared to automatically choosing optimal graph traverse mode given graph type are welcome.

# Note 4. Not all "first_id_strategy" and "graph_traverse_mode" are compatible with all graphs.
# Contributinos geared to create fallbacks for when specialised strategies fail are welcome.
kwargs = {
    "debug": 0,  # Verbosity. Change to `3` for step by step visuals.
    "first_id_strategy": "first-spider",  # Strategy for choosing the first spider/cube ID.
    "graph_traverse_mode": "bfs-cross",  # Graph traversing strategy
    "gravity": 7,  # Integer weight that pulls paths towards graph centre
    "twins": True,  # Boolean flag to enable "twins", which safeguards completion at the expense of volume
}
# Available `first_id_strategy` values:
# - [first-spider, "random", "centrality-random", "centrality-majority", "central-qubit", "central-in-first-cycle"]
# Available `graph_traverse_mode` values:
# - ["bfs", "bfs-cross", "bfs-cross-boundaries-last", "bfs-cycles", "bfs-rows", "bfs-cnots", "tfs-cnots", "tfs"]

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
        "ht",
    ]

    # Loop over available encoding functions
    for graph_name in include:
        # Update user
        print(f"\n#####################\nGRAPH NAME: {graph_name}. \n#####################\n")

        # Get PyZX graph
        encoding_fx = getattr(pyzx_graphs, graph_name)
        pyzx_graph = encoding_fx(draw_graph=False)

        # Adjust tree search mode if needed
        if graph_name in ["ht"]:
            kwargs["graph_traverse_mode"] = "tfs"
            kwargs["z_stretch"] = 3
            kwargs["twins"] = False

        # PyZX -> AugmentedZXGraph
        zx_graph_manager = ZXGraphManager(debug=kwargs["debug"])
        aug_zx = zx_graph_manager.add_graph_from_pyzx(pyzx_graph, graph_key="input")

        # AugmentedZXGraph -> BlockGraph
        bgraph_manager = BlockGraphManager(aug_zx, **kwargs)
        bgraph_manager.build()

        # Verification for standard graphs
        if graph_name not in ["yi", "s", "msc", "t", "ht"]:
            # Verify input/output logical equality
            zx_out = bgraph_manager.to_zx_graph()
            equality = aug_zx.check_equality(zx_out)

        # Visualise results
        bgraph_manager.draw_blockgraph()

        # Write results to file
        bgraph_manager.write_bgraph(circuit_name=graph_name)

        # Animate and clean up
        if kwargs.get("animate"):
            bgraph_manager.animate(filename_prefix=graph_name)

        # Say good bye
        print("\n---\nThank you for flying Topologiq.\n")
