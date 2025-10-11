"""Basic smoke tests to verify test infrastructure and core imports."""


def test_import_topologiq():
    """Test that topologiq package can be imported."""
    import topologiq
    
    assert topologiq is not None


def test_simple_graph_fixture(simple_graph):
    """Test that pytest fixtures work correctly."""
    assert "nodes" in simple_graph
    assert "edges" in simple_graph
    assert len(simple_graph["nodes"]) == 2
    assert len(simple_graph["edges"]) == 1


def test_core_module_imports():
    """Test that key modules can be imported without errors."""
    from topologiq.scripts import runner
    from topologiq.utils import utils_greedy_bfs
    from topologiq.utils import utils_pathfinder
    from topologiq.assets.graphs import simple_graphs
    
    # Verify they're actual module objects
    assert runner is not None
    assert utils_greedy_bfs is not None
    assert utils_pathfinder is not None
    assert simple_graphs is not None


def test_can_instantiate_simple_graph():
    """Test that we can work with the simple graph format."""
    from topologiq.assets.graphs import simple_graphs
    
    # Get a known simple graph
    hadamard_line = simple_graphs.hadamard_line
    
    assert "nodes" in hadamard_line
    assert "edges" in hadamard_line
    assert len(hadamard_line["nodes"]) > 0
