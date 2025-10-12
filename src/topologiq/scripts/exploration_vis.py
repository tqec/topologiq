"""
Granular pathfinder visualizations for debugging and analysis.

This module provides "Christmas tree" style visualization of A* pathfinder exploration,
enabling detailed debugging of failed edge cases and performance analysis.
"""

import os
import json
from typing import List, Dict, Tuple, Optional, Set
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime

from topologiq.utils.classes import StandardCoord, StandardBlock


class ExplorationTracker:
    """Tracks A* pathfinding exploration for visualization and debugging."""
    
    def __init__(self, enabled: bool = None):
        """Initialize exploration tracker.
        
        Args:
            enabled: Whether tracking is enabled. If None, checks TOPOLOGIQ_VERBOSE env var.
        """
        if enabled is None:
            enabled = os.environ.get('TOPOLOGIQ_VERBOSE') == '1'
        
        self.enabled = enabled
        self.exploration_steps: List[Dict] = []
        self.visited_coords: Set[StandardCoord] = set()
        self.path_coords: Dict[StandardBlock, List[StandardCoord]] = {}
        
    def track_step(self, step: int, current_block: StandardBlock, queue_size: int, 
                   targets_found: int, f_score: float, g_score: float, h_score: float):
        """Track a single A* exploration step.
        
        Args:
            step: Step number in exploration
            current_block: Current block being explored
            queue_size: Size of priority queue
            targets_found: Number of targets found so far
            f_score: A* f-score (g + h)
            g_score: Cost from start
            h_score: Heuristic to goal
        """
        if not self.enabled:
            return
            
        coords, kind = current_block
        self.visited_coords.add(coords)
        
        step_data = {
            'step': step,
            'coords': coords,
            'kind': kind,
            'queue_size': queue_size,
            'targets_found': targets_found,
            'f_score': f_score,
            'g_score': g_score,
            'h_score': h_score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.exploration_steps.append(step_data)
        
        # Print granular progress
        print(f"🎄 Step {step:3d}: {coords} [{kind}] | F={f_score:.1f} G={g_score} H={h_score:.1f} | Queue:{queue_size} | Found:{targets_found}")
        
    def track_path(self, block: StandardBlock, path: List[StandardBlock]):
        """Track a discovered path for visualization.
        
        Args:
            block: Target block
            path: Complete path from source to target
        """
        if not self.enabled:
            return
            
        self.path_coords[block] = [b[0] for b in path]
        
    def export_exploration_data(self, filename: str = None) -> str:
        """Export exploration data to JSON for analysis.
        
        Args:
            filename: Output filename. If None, generates timestamp-based name.
            
        Returns:
            Path to exported file
        """
        if not self.enabled or not self.exploration_steps:
            return ""
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exploration_data_{timestamp}.json"
            
        data = {
            'exploration_steps': self.exploration_steps,
            'visited_coords': list(self.visited_coords),
            'path_coords': {str(k): v for k, v in self.path_coords.items()},
            'total_steps': len(self.exploration_steps),
            'total_visited': len(self.visited_coords),
            'total_paths': len(self.path_coords)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"📊 Exported exploration data to {filename}")
        return filename
        
    def visualize_christmas_tree(self, save_png: bool = True, show_interactive: bool = False) -> str:
        """Create 'Christmas tree' 3D visualization of exploration pattern.
        
        This visualization shows the exploration pattern as a tree-like structure,
        with the source as the trunk and explorations branching out like tree branches.
        
        Args:
            save_png: Whether to save as PNG file
            show_interactive: Whether to show interactive matplotlib window
            
        Returns:
            Path to saved PNG file, or empty string if not saved
        """
        if not self.enabled or not self.exploration_steps:
            return ""
            
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates and scores for visualization
        coords = [step['coords'] for step in self.exploration_steps]
        f_scores = [step['f_score'] for step in self.exploration_steps]
        g_scores = [step['g_score'] for step in self.exploration_steps]
        
        if not coords:
            return ""
            
        # Separate coordinates
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        z_coords = [c[2] for c in coords]
        
        # Create Christmas tree visualization
        # Trunk (source) in brown
        if coords:
            ax.scatter(x_coords[0], y_coords[0], z_coords[0], 
                      c='brown', s=200, alpha=0.9, marker='s', label='Source')
        
        # Explored nodes colored by f-score (like Christmas lights)
        if len(coords) > 1:
            scatter = ax.scatter(x_coords[1:], y_coords[1:], z_coords[1:], 
                               c=f_scores[1:], s=60, alpha=0.7, cmap='RdYlGn_r',
                               marker='o', label='Explored nodes')
            plt.colorbar(scatter, ax=ax, label='F-Score')
        
        # Draw paths as tree branches
        for block, path_coords_list in self.path_coords.items():
            if len(path_coords_list) > 1:
                px = [c[0] for c in path_coords_list]
                py = [c[1] for c in path_coords_list]
                pz = [c[2] for c in path_coords_list]
                ax.plot(px, py, pz, 'g-', linewidth=3, alpha=0.8)
                # Star on target (top of tree)
                ax.scatter(px[-1], py[-1], pz[-1], 
                          c='gold', s=300, marker='*', 
                          edgecolors='red', linewidths=2, label='Target')
        
        # Styling for Christmas tree effect
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate') 
        ax.set_zlabel('Z Coordinate')
        ax.set_title('🎄 A* Pathfinder Exploration Tree\n(Christmas Tree Visualization)', 
                    fontsize=14, fontweight='bold')
        
        # Add statistics
        stats_text = f"Steps: {len(self.exploration_steps)}\n"
        stats_text += f"Visited: {len(self.visited_coords)}\n"
        stats_text += f"Paths: {len(self.path_coords)}"
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        ax.legend()
        plt.tight_layout()
        
        # Save PNG if requested
        png_path = ""
        if save_png:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            png_path = f"christmas_tree_exploration_{timestamp}.png"
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"🎄 Saved Christmas tree visualization to {png_path}")
        
        # Show interactive if requested
        if show_interactive:
            plt.show()
        else:
            plt.close()
            
        return png_path
        
    def generate_debug_report(self) -> str:
        """Generate comprehensive debug report for failed edge cases.
        
        Returns:
            Formatted debug report string
        """
        if not self.enabled or not self.exploration_steps:
            return "No exploration data available"
            
        report = []
        report.append("=" * 60)
        report.append("🎄 A* PATHFINDER EXPLORATION REPORT")
        report.append("=" * 60)
        
        # Summary statistics
        report.append(f"Total exploration steps: {len(self.exploration_steps)}")
        report.append(f"Unique coordinates visited: {len(self.visited_coords)}")
        report.append(f"Successful paths found: {len(self.path_coords)}")
        
        if self.exploration_steps:
            f_scores = [s['f_score'] for s in self.exploration_steps]
            report.append(f"F-score range: {min(f_scores):.1f} - {max(f_scores):.1f}")
            
        # Path analysis
        if self.path_coords:
            report.append("\n📍 DISCOVERED PATHS:")
            for i, (block, path) in enumerate(self.path_coords.items(), 1):
                coords, kind = block
                report.append(f"  Path {i}: {len(path)} steps to {coords} [{kind}]")
        
        # Recent exploration steps
        report.append("\n🔍 RECENT EXPLORATION STEPS:")
        recent_steps = self.exploration_steps[-min(10, len(self.exploration_steps)):]
        for step in recent_steps:
            report.append(f"  Step {step['step']:3d}: {step['coords']} [{step['kind']}] F={step['f_score']:.1f}")
            
        return "\n".join(report)


# Global exploration tracker instance
_exploration_tracker = ExplorationTracker()


def get_tracker() -> ExplorationTracker:
    """Get the global exploration tracker instance.
    
    Returns:
        Global ExplorationTracker instance
    """
    return _exploration_tracker


def reset_tracker():
    """Reset the global exploration tracker for new pathfinding session."""
    global _exploration_tracker
    _exploration_tracker = ExplorationTracker()
