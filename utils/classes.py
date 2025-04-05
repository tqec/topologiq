from dataclasses import dataclass
from typing import Tuple, List
from two_stage_greedy_bfs_config import VALUE_HYPERPARAMS

# QUICK NAMED TYPE TO SAVE SPACE
Position3D = Tuple[int, int, int]

# NO LONGER IN USE, I THINK...
@dataclass(order=True)
class Node:
    position: Position3D
    kind: str
    n_unobstructed_exits: int
    beams: List[Position3D]

# MAIN DATA CLASS TO STORE PATHS AND ENABLE COMPARISONS
@dataclass(order=True)
class Path:
    target_pos: Position3D
    target_kind: str
    target_beams: List[Position3D]
    coords_in_path: List[Position3D]
    all_nodes_in_path: List[Tuple[Position3D, str]]
    beams_broken_by_path: int
    len_of_path: int
    target_unobstructed_exits_n: int

    def weighed_value(self, stage) -> int:
        path_len_hp, beams_broken_hp, target_exits_hp = VALUE_HYPERPARAMS
        return (
            self.len_of_path * path_len_hp
            + self.beams_broken_by_path * beams_broken_hp
            + self.target_unobstructed_exits_n * target_exits_hp
        )
