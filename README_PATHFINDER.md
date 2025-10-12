# A* Pathfinder Usage

## Performance Improvement
The pathfinder now uses A* algorithm for improved performance on quantum circuits.

## Basic Usage
```python
from topologiq.scripts.pathfinder import core_pthfinder_bfs

# Standard pathfinding
result, stats = core_pthfinder_bfs(src, tent_coords, tent_tgt_kinds, min_succ_rate)
```

## Debugging
Enable granular visualization:
```bash
export PATHFINDER_DEBUG=1
python your_script.py
```

## Reproducible Testing
```python
from topologiq.scripts.graph_manager import graph_manager_bfs

# Force specific source kind for reproducible runs  
result = graph_manager_bfs(graph, force_src_kind="xxz")
```

## Testing
```bash
pytest tests/ -v
```