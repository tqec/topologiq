"""A* pathfinder exploration tracking and visualization."""
import os
import json
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
from topologiq.utils.classes import StandardCoord, StandardBlock

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    plt = None

class ExplorationTracker:
    """Minimal A* exploration tracker."""
    def __init__(self, enabled: bool = None):
        self.enabled = enabled if enabled is not None else os.environ.get('TOPOLOGIQ_VERBOSE') == '1'
        self.steps: List[Dict] = []
        self.visited: Set[StandardCoord] = set()
        self.paths: Dict[StandardBlock, List[StandardCoord]] = {}
        
    def track_step(self, step: int, block: StandardBlock, queue_size: int, 
                   targets_found: int, f_score: float, g_score: float, h_score: float):
        if not self.enabled: return
        coords, kind = block
        self.visited.add(coords)
        self.steps.append({'step': step, 'coords': coords, 'kind': kind, 'queue_size': queue_size,
                          'targets_found': targets_found, 'f_score': f_score, 'g_score': g_score, 'h_score': h_score})
        if os.environ.get('PATHFINDER_DEBUG') == '1':
            print(f"Step {step:3d}: {coords} [{kind}] F={f_score:.1f} G={g_score} H={h_score:.1f} Queue:{queue_size} Found:{targets_found}")
        
    def track_path(self, block: StandardBlock, path: List[StandardBlock]):
        if self.enabled:
            self.paths[block] = [b[0] for b in path]
        
    def export_data(self, filename: str = None) -> str:
        if not self.enabled or not self.steps:
            return ""
        if filename is None:
            filename = f"exploration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        data = {'steps': self.steps, 'visited': list(self.visited), 'paths': {str(k): v for k, v in self.paths.items()}}
        with open(filename, 'w') as f:
            json.dump(data, f)
        return filename
        
    def visualize(self, save_png: bool = False) -> str:
        if not self.enabled or not self.steps or plt is None:
            return ""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        coords = [s['coords'] for s in self.steps]
        f_scores = [s['f_score'] for s in self.steps]
        if coords:
            x, y, z = zip(*coords)
            ax.scatter(x[0], y[0], z[0], c='red', s=100, marker='s', label='Source')
            if len(coords) > 1:
                scatter = ax.scatter(x[1:], y[1:], z[1:], c=f_scores[1:], s=30, alpha=0.7, cmap='viridis')
                plt.colorbar(scatter, ax=ax, label='F-Score')
        for path in self.paths.values():
            if len(path) > 1:
                px, py, pz = zip(*path)
                ax.plot(px, py, pz, 'g-', linewidth=2, alpha=0.8)
                ax.scatter(px[-1], py[-1], pz[-1], c='gold', s=80, marker='*')
        ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
        ax.set_title('A* Pathfinder Exploration')
        if save_png:
            filename = f"exploration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=150)
            plt.close()
            return filename
        plt.show() if plt else None
        return ""
        
    def report(self) -> str:
        if not self.enabled or not self.steps:
            return "No data"
        return f"Steps: {len(self.steps)}, Visited: {len(self.visited)}, Paths: {len(self.paths)}"

_tracker = ExplorationTracker()
get_tracker = lambda: _tracker
reset_tracker = lambda: globals().update(_tracker=ExplorationTracker())
