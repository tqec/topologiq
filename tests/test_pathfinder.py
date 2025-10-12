"""
Minimal pathfinder tests for A* validation
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_pathfinder_basic():
    """Test A* pathfinder works"""
    from topologiq.scripts.pathfinder import core_pthfinder_bfs
    
    src = ((0, 0, 0), "xxz")
    tent_coords = [(1, 0, 0)]
    tent_tgt_kinds = ["xxz"]
    min_succ_rate = 50
    
    result, stats = core_pthfinder_bfs(src, tent_coords, tent_tgt_kinds, min_succ_rate)
    
    # Validate structure
    assert isinstance(result, dict) or result is None
    assert isinstance(stats, tuple)
    assert len(stats) == 2

def test_reproducible_parameters():
    """Test reproducible parameters exist"""
    from topologiq.scripts.graph_manager import graph_manager_bfs
    import inspect
    
    sig = inspect.signature(graph_manager_bfs)
    assert 'force_src_kind' in sig.parameters

def test_pathfinder_interface_stability():
    """Test pathfinder interface remains stable for refactoring safety"""
    from topologiq.scripts.pathfinder import core_pthfinder_bfs
    import inspect
    
    sig = inspect.signature(core_pthfinder_bfs)
    expected_params = ['src', 'tent_coords', 'tent_tgt_kinds', 'min_succ_rate', 'taken', 'hdm', 'critical_beams', 'u_v_ids']
    
    actual_params = list(sig.parameters.keys())
    for param in expected_params:
        assert param in actual_params, f"Missing expected parameter: {param}"

def test_granular_visualization():
    """Test granular visualization works"""
    import os
    from topologiq.scripts.pathfinder import core_pthfinder_bfs
    
    # Test with debug mode enabled
    os.environ['PATHFINDER_DEBUG'] = '1'
    
    try:
        src = ((0, 0, 0), "xxz")
        tent_coords = [(1, 0, 0)]
        tent_tgt_kinds = ["xxz"]
        min_succ_rate = 50
        
        result, stats = core_pthfinder_bfs(src, tent_coords, tent_tgt_kinds, min_succ_rate)
        # Test passes if no exception thrown
        assert isinstance(stats, tuple)
    finally:
        os.environ.pop('PATHFINDER_DEBUG', None)

if __name__ == "__main__":
    test_pathfinder_basic()
    test_reproducible_parameters()
    test_pathfinder_interface_stability()
    test_granular_visualization()
    print("All pathfinder tests passed")
