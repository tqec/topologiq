from itertools import product
import numpy as np
from topologiq.utils.classes import SingleBeam, BeamAxisComponent
from topologiq.utils.classes import StandardCoord

def make_beam(source: np.ndarray, direction: np.ndarray):
    sx, sy, sz = source
    dx, dy, dz = direction
    return SingleBeam(
        BeamAxisComponent(sx, np.inf, dx),
        BeamAxisComponent(sy, np.inf, dy),
        BeamAxisComponent(sz, np.inf, dz)
    )

all_steps = [
    np.array([+1,  0,  0], dtype=np.int32),
    np.array([-1,  0,  0], dtype=np.int32),
    np.array([ 0, +1,  0], dtype=np.int32),
    np.array([ 0, -1,  0], dtype=np.int32),
    np.array([ 0,  0, +1], dtype=np.int32),
    np.array([ 0,  0, -1], dtype=np.int32)
]

ORIGIN = np.zeros(3)
all_positions = list(filter(
    lambda pos : not np.array_equal(ORIGIN, pos),
    [ step1 + step2 + step3 + step4 + step5 for step1, step2, step3, step4, step5
      in product(all_steps, all_steps, all_steps, all_steps, all_steps)
    ]
))

if __name__ == "__main__":
    print(f"Steps: {len(all_steps)}")
    print(f"Positions: {len(all_positions)}")

    origin_beams = [ make_beam(ORIGIN, step) for step in all_steps ]
    all_beams = [ make_beam(position, orientation) for position, orientation in product(all_positions, all_steps) ]

    print(f"Beams: {len(all_beams)}")

    SingleBeam.INTERSECTION_CONSISTENCY_CHECKS = False
    SingleBeam.INTERSECTION_BY_RAYS = True
    for beam1 in origin_beams:
        for beam2 in all_beams:
            beam1.intersects(beam2, len_of_materialised_beam = 50)