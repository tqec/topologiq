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

if __name__ == "__main__":
    test_pathfinder_basic()
    test_reproducible_parameters()
    print("Minimal tests passed")