from topologiq.dzw.helpers.spacetime import Spacetime
from topologiq.dzw.common.coordinates import Coordinates

from topologiq.dzw.common.components_bg import CubeKind

# from topologiq.utils.classes import NodeBeams

from logging import getLogger
console = getLogger(__name__)

class CubeBeams:
    def __init__(self,
        cube_kind: CubeKind, cube_position: Coordinates,
        extras: list[tuple[CubeKind, Coordinates]] = None,
        occupied: set[Coordinates] = None
    ):
        cube_reach = cube_kind.get_reach()

        self.__cube_kind: CubeKind = cube_kind
        self.__cube_position: Coordinates = cube_position
        self.__available_beams: list[Coordinates] = Spacetime.get_step_constellation(cube_reach)

        console.debug(f"Cube kind : {self.__cube_kind}@{self.__cube_position} [{cube_reach}].")
        console.debug(f"> Occupied : {occupied}")

        if occupied is not None:
            for position in occupied:
                console.debug(f"> {cube_position} colinear with {position} ? {cube_position.colinear(position)}")
                if cube_position.colinear(position):
                    los = Spacetime.get_direction(cube_position, position)
                    console.debug(f">> LOS[Occ] : {los} in {self.__available_beams} : {los in self.__available_beams}")
                    if los in self.__available_beams:
                        self.__available_beams.remove( los )

        if extras is not None:
            for _, position in extras:
                console.debug(f"> {cube_position} colinear with {position} ? {cube_position.colinear(position)}")
                if cube_position.colinear(position):
                    los = Spacetime.get_direction(cube_position, position)
                    console.debug(f">> LOS[Ext] : {los} in {self.__available_beams} : {los in self.__available_beams}")
                    if los in self.__available_beams:
                        self.__available_beams.remove( los )

        console.debug(f"> Available beams : {self.__available_beams}")

    def number_available(self):
        return len(self.__available_beams)

    def count_interrupted(self, lines_of_sight: set[Coordinates]) -> int:
        if any([Spacetime.ORIGIN.get_manhattan_distance(los) != 1 for los in lines_of_sight]):
            raise Exception(f"Computing remaining beam count requires lines-of-sight of unit length.")

        return sum(1 for los in lines_of_sight if los in self.__available_beams)

    def count_intersected(self, other):
        beams_intersected = 0

        return beams_intersected

    def close_beam(self, beam: Coordinates):
        if beam not in self.__available_beams:
            console.warning(f"Closing non-existent beam for cube {self.__cube_kind}@{self.__cube_position} [{beam}].")
        else:
            self.__available_beams.remove(beam)

    def __eq__(self, other):
        return self.__cube_kind == other.__cube_kind and self.__cube_position == other.__cube_position \
            and self.__available_beams == other.__available_beams

    def __repr__(self):
        formatted = f"{self.__cube_kind}@{self.__cube_position} :"
        for beam in self.__available_beams:
            formatted += f" {beam}"
        return formatted

    def __iter__(self):
        return iter(self.__available_beams)

    # # TODO: deal with broken beams (cfr. pathfinder lines 299-329)
    # critical_break = False
    # if critical:
    #     for node, data in critical.items():
    #         broken_beams = 0
    #         min_exit_num, beams = data
    #         for beam in beams:
    #             if any([position in beam for _, position in current_path]):
    #                 broken_beams += 1
    #                 # Additionally, add any number of beam-to-beam clashes for the node
    #                 # currently under investigation because, if they exist, they likely are already
    #                 # using the cushion that allows breaking some beams
    #                 if node in (source, target):
    #                     continue
    #
    #                 for n_id in critical.keys():
    #                     all_beams = critical[n_id][1]
    #                     for single_beam in all_beams:
    #                         if any([position in beam for position in single_beam]):
    #                             broken_beams += 1
    #
    #         adjust_for_source_node = 1 if node in (source, target) else 0
    #         if len(beams) + adjust_for_source_node - broken_beams < min_exit_num:
    #             critical_break = True
    #             break
    #
    #     if critical_break:
    #         continue