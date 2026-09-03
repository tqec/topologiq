"""Example of how to undertake post-processing transformations on a simple BlockGraph."""

from topologiq.assets.pyzx_graphs import ht
from topologiq.core.graph_manager.graph_manager import BlockGraphManager
from topologiq.input.zx_manager import ZXGraphManager

##########
# KWARGS #
##########
kwargs = {
    "debug": 1,  # Verbosity & visualisation control [int]
    "first_id_strategy": "first-spider",  # Strategy for choosing the first spider/cube ID [str]
    "graph_traverse_mode": "tfs",  # Graph traversing strategy [str]
    "gravity": 7,  # Pull paths towards graph centre [int]
    "post_process": True,
    "z_stretch": 3,
    "twins": False,
}

#######
# RUN #
#######
if __name__ == "__main__":
    # Give pretty name to graph
    graph_name = "ht"
    num_t = 5
    print(f"\n#####################\nGRAPH NAME: {graph_name} * {num_t}. \n#####################\n")

    # Get graph
    pyzx_graph = ht(draw_graph=False, num_t=num_t)

    # PyZX -> AugmentedZXGraph
    zx_graph_manager = ZXGraphManager(debug=kwargs["debug"])
    aug_zx = zx_graph_manager.add_graph_from_pyzx(pyzx_graph, graph_key="input")

    # AugmentedZXGraph -> BlockGraph
    bgraph_manager = BlockGraphManager(aug_zx, **kwargs)
    bgraph_manager.build()

    # Visualise results
    bgraph_manager.draw_blockgraph()

    # Write results to file
    bgraph_manager.write_bgraph(circuit_name=graph_name)

    # Extend MSCs to maximum available space
    if kwargs["post_process"]:
        bgraph_manager.stretch_msc_cubes()
        bgraph_manager.draw_blockgraph()

        # Add micro-factories where possible
        bgraph_manager.distributed_msc_factory()
        bgraph_manager.draw_blockgraph()

        # Need to wait for very first MSC that didn't have backups
        bgraph_manager.slice_stretch(slice_at_z=4, shift_z=5)
        bgraph_manager.draw_blockgraph()

        # MSC succeeds; conditional to Y
        bgraph_manager.switch_conditional(17, True)  # Switch conditional -> Y
        bgraph_manager.draw_blockgraph()

        # MSC 44 (factory) fails, 18 (scheduled) succeeds; conditional -> Y
        bgraph_manager.msc_discard(discard_id=44)  # Discard backup factory MSC
        bgraph_manager.switch_conditional(19, True)  # Switch conditional to Y
        bgraph_manager.draw_blockgraph()

        # MSC 45 (factory) succeeds, 20 (scheduled) undetermined, conditional -> X
        bgraph_manager.msc_exchange(include_id=45, discard_id=20)  # Exchange MSCs
        bgraph_manager.switch_conditional(21, False)  # Switch conditional to X
        bgraph_manager.draw_blockgraph()

        # MSCs 46 (factory) fails, 23 (scheduled) succeeds, conditional -> X
        bgraph_manager.msc_discard(discard_id=46)
        bgraph_manager.switch_conditional(23, False)  # Switch conditional to X
        bgraph_manager.draw_blockgraph()

        # MSCs 47 (factory) fails, 25 (scheduled) succeeds, not time to determine conditional switch
        bgraph_manager.msc_discard(discard_id=47)  # Discard failed MSC
        bgraph_manager.slice_stretch(slice_at_z=26, shift_z=3)  # Wait a few beats
        bgraph_manager.draw_blockgraph()

        # Conditional -> Y
        bgraph_manager.switch_conditional(25, True)  # Switch conditional to Y
        bgraph_manager.draw_blockgraph()

    # Animate and clean up
    if kwargs.get("animate"):
        bgraph_manager.animate(filename_prefix=graph_name)

    # Say good bye
    print("\n---\nThank you for flying Topologiq.\n")
