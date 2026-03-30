import numpy as np

from topologiq.utils.classes import StandardCoord
from topologiq.core.pathfinder.utils import get_manhattan

class Spacetime:
    ORIGIN = (0, 0, 0)

    XP = (+1, 0, 0)
    XM = (-1, 0, 0)
    YP = (0, +1, 0)
    YM = (0, -1, 0)
    ZP = (0, 0, +1)
    ZM = (0, 0, -1)

    XYZ = (0, 0, 0)
    XY  = (0, 0, 1)
    XZ  = (0, 1, 0)
    YZ  = (1, 0, 0)

    STEPS = [ XP ,YP, ZP, XM, YM, ZM ]
    PLANES = [ XY, XZ, YZ ]

    @staticmethod
    def contains(reach: StandardCoord, step: StandardCoord) -> bool:
        return np.dot(reach, step) == 0

    @staticmethod
    def get_direction(source: StandardCoord, target: StandardCoord) -> StandardCoord:
        differences = [ 1 if cs != ct else 0 for cs, ct in zip(source, target) ]

        if sum(differences) != 1:
            raise Exception(f"Coordinates are not co-linear and thus do not have a line-of-sight [{source}/{target}.")

        deltas = [ +1 if cs - ct < 0 else -1 for cs, ct in zip(source, target) ]

        line_of_sight = StandardCoord.from_list( [ difference * delta for difference, delta in zip(differences, deltas) ] )

        if get_manhattan(Spacetime.ORIGIN, line_of_sight) != 1:
            raise Exception(f"Erroneous computation of line of sight [{source}/{target} = {line_of_sight}].")

        return line_of_sight

    @staticmethod
    def get_step_constellation(reach: StandardCoord) -> list[StandardCoord]:
        return [step for step in Spacetime.STEPS if reach.dot(step) == 0]

    @staticmethod
    def get_orthogonal_plane(plane: StandardCoord, line_of_intersection: StandardCoord) -> StandardCoord:
        if np.dot(plane, line_of_intersection) != 0:
            raise ValueError(f"Line of intersection {line_of_intersection} does not lie in plane {plane}.")

        reach = plane

        if abs(reach.x) == abs(line_of_intersection.x):
            return Spacetime.YZ
        elif abs(reach.y) == abs(line_of_intersection.y):
            return Spacetime.XZ
        else: # abs(reach.z) != abs(line_of_intersection.z)
            return Spacetime.XY

    @staticmethod
    def get_constellation(position: StandardCoord, restriction: StandardCoord = None) -> list[StandardCoord]:
        constellation = []
        for step in Spacetime.STEPS:
            if restriction is None or np.dot(restriction, step) == 0:
                constellation.append(position + step)
        return constellation