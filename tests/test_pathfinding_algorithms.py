"""Minimal pathfinding algorithm tests with performance benchmarking."""
import os, sys, time, statistics
from pathlib import Path
from typing import List, Tuple
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from topologiq.scripts.pathfinder import core_pthfinder_bfs
from topologiq.scripts.graph_manager import graph_manager_bfs
from topologiq.utils.classes import StandardCoord, StandardBlock

class PathfindingBenchmark:
    @staticmethod
    def time_pathfinding(src: StandardBlock, targets: List[StandardCoord], 
                        kinds: List[str], iterations: int = 10) -> Tuple[float, float, float]:
        times, successes = [], 0
        for _ in range(iterations):
            start = time.perf_counter()
            result, _ = core_pthfinder_bfs(src, targets, kinds, 50)
            times.append(time.perf_counter() - start)
            if result: successes += 1
        return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0, successes / iterations

@pytest.mark.benchmark
def test_simple_performance():
    cases = [
        (((0, 0, 0), "xxz"), [(1, 0, 0)], ["xxz"], 20),
        (((0, 0, 0), "xxz"), [(3, 3, 3)], ["xxz"], 15),
        (((0, 0, 0), "xxz"), [(5, 5, 5), (5, 5, -5)], ["xxz", "zxx"], 10)
    ]
    for src, coords, kinds, iters in cases:
        mean_time, _, success_rate = PathfindingBenchmark.time_pathfinding(src, coords, kinds, iters)
        assert success_rate >= 0.8, f"Low success rate: {success_rate}"
        assert mean_time < 2.0, f"Slow execution: {mean_time:.4f}s"
        print(f"Case {src[0]}→{coords}: {mean_time:.4f}s (success: {success_rate:.1%})")

@pytest.mark.parametrize("src,coords,kinds,expected", [
    (((0, 0, 0), "xxz"), [(1, 0, 0)], ["xxz"], True),
    (((0, 0, 0), "xxz"), [(2, 2, 2)], ["zxx"], True),
    (((0, 0, 0), "ooo"), [(3, 3, 3)], ["ooo"], True),
])
def test_pathfinding_correctness(src, coords, kinds, expected):
    result, stats = core_pthfinder_bfs(src, coords, kinds, 50)
    if expected:
        assert result is not None and len(result) > 0
        for target, path in result.items():
            assert len(path) >= 2 and path[0] == src
    assert isinstance(stats, tuple) and len(stats) == 2

@pytest.mark.visualization
def test_exploration_tracking():
    os.environ['TOPOLOGIQ_VERBOSE'] = '1'
    try:
        from topologiq.scripts.exploration_vis import get_tracker, reset_tracker
        reset_tracker()
        tracker = get_tracker()
        result, _ = core_pthfinder_bfs(((0, 0, 0), "xxz"), [(2, 2, 2)], ["zxx"], 50)
        assert len(tracker.steps) > 0
        assert len(tracker.visited) > 0
        report = tracker.report()
        assert "Steps:" in report
    finally:
        os.environ.pop('TOPOLOGIQ_VERBOSE', None)

def test_graph_manager_integration():
    import inspect
    sig = inspect.signature(graph_manager_bfs)
    assert 'force_src_kind' in sig.parameters

def test_performance_validation():
    cases = [{"src": ((0, 0, 0), "xxz"), "coords": [(4, 4, 4)], "kinds": ["zxx"], "iterations": 15}]
    for case in cases:
        mean_time, _, _ = PathfindingBenchmark.time_pathfinding(case["src"], case["coords"], case["kinds"], case["iterations"])
        assert 0 < mean_time < 5.0
        print(f"Performance: {mean_time:.4f}s")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])