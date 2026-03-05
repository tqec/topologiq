from topologiq.dzw.common.coordinates import Coordinates

class Spacetime:
    ORIGIN = Coordinates(0, 0, 0)

    XP = Coordinates(+1, 0, 0)
    XM = Coordinates(-1, 0, 0)
    YP = Coordinates(0, +1, 0)
    YM = Coordinates(0, -1, 0)
    ZP = Coordinates(0, 0, +1)
    ZM = Coordinates(0, 0, -1)

    XYZ = Coordinates(0, 0, 0)
    XY  = Coordinates(0, 0, 1)
    XZ  = Coordinates(0, 1, 0)
    YZ  = Coordinates(1, 0, 0)

    STEPS = [ XP ,YP, ZP, XM, YM, ZM ]
    PLANES = [ XY, XZ, YZ ]

    @staticmethod
    def contains(reach: Coordinates, step: Coordinates) -> bool:
        return reach.dot(step) == 0

    @staticmethod
    def get_direction(source: Coordinates, target: Coordinates) -> Coordinates:
        differences = [ 1 if cs != ct else 0 for cs, ct in zip(source, target) ]

        if sum(differences) != 1:
            raise Exception(f"Coordinates are not co-linear and thus do not have a line-of-sight [{source}/{target}.")

        deltas = [ +1 if cs - ct < 0 else -1 for cs, ct in zip(source, target) ]

        line_of_sight = Coordinates.from_list( [ difference * delta for difference, delta in zip(differences, deltas) ] )

        if Spacetime.ORIGIN.get_manhattan_distance(line_of_sight) != 1:
            raise Exception(f"Erroneous computation of line of sight [{source}/{target} = {line_of_sight}].")

        return line_of_sight

    @staticmethod
    def get_step_constellation(reach: Coordinates) -> list[Coordinates]:
        return [step for step in Spacetime.STEPS if reach.dot(step) == 0]

    @staticmethod
    def get_orthogonal_plane(plane: Coordinates, line_of_intersection: Coordinates) -> Coordinates:
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
    def get_constellation(position: Coordinates, restriction: Coordinates = None) -> list[Coordinates]:
        constellation = []
        for step in Spacetime.STEPS:
            if restriction is None or restriction.dot(step) == 0:
                constellation.append(position + step)
        return constellation