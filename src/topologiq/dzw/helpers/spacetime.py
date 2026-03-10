from topologiq.utils.classes import StandardCoord

class Spacetime:
    ORIGIN = StandardCoord(0, 0, 0)

    XP = StandardCoord(+1, 0, 0)
    XM = StandardCoord(-1, 0, 0)
    YP = StandardCoord(0, +1, 0)
    YM = StandardCoord(0, -1, 0)
    ZP = StandardCoord(0, 0, +1)
    ZM = StandardCoord(0, 0, -1)

    XYZ = StandardCoord(0, 0, 0)
    XY  = StandardCoord(0, 0, 1)
    XZ  = StandardCoord(0, 1, 0)
    YZ  = StandardCoord(1, 0, 0)

    STEPS = [ XP ,YP, ZP, XM, YM, ZM ]
    PLANES = [ XY, XZ, YZ ]

    @staticmethod
    def contains(reach: StandardCoord, step: StandardCoord) -> bool:
        return reach.dot(step) == 0

    @staticmethod
    def get_direction(source: StandardCoord, target: StandardCoord) -> StandardCoord:
        differences = [ 1 if cs != ct else 0 for cs, ct in zip(source, target) ]

        if sum(differences) != 1:
            raise Exception(f"Coordinates are not co-linear and thus do not have a line-of-sight [{source}/{target}.")

        deltas = [ +1 if cs - ct < 0 else -1 for cs, ct in zip(source, target) ]

        line_of_sight = StandardCoord.from_list( [ difference * delta for difference, delta in zip(differences, deltas) ] )

        if Spacetime.ORIGIN.get_manhattan_distance(line_of_sight) != 1:
            raise Exception(f"Erroneous computation of line of sight [{source}/{target} = {line_of_sight}].")

        return line_of_sight

    @staticmethod
    def get_step_constellation(reach: StandardCoord) -> list[StandardCoord]:
        return [step for step in Spacetime.STEPS if reach.dot(step) == 0]

    @staticmethod
    def get_orthogonal_plane(plane: StandardCoord, line_of_intersection: StandardCoord) -> StandardCoord:
        if plane.dot(line_of_intersection) != 0:
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
            if restriction is None or restriction.dot(step) == 0:
                constellation.append(position + step)
        return constellation