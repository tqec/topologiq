from topologiq.dzw.utils.coordinates import Coordinates

from topologiq.dzw.utils.components_zx import EdgeType
from topologiq.dzw.utils.components_bg import CubeId, CubeKind

from topologiq.dzw.utils.cube_beams import CubeBeams

from logging import getLogger
console = getLogger(__name__)

class Path:
    PATH_LEN_HP = -1
    BEAMS_BROKEN_HP = -1

    def __init__(self, source_cube: CubeId, target_cube: CubeId, edge_type: EdgeType, proposed_beams: CubeBeams,
            proposed_cubes: list[tuple[CubeKind, Coordinates]], proposed_pipes: list[EdgeType]
    ):
        self.__source_cube = source_cube
        self.__target_cube = target_cube
        self.__edge_type = edge_type
        proposed_kind, proposed_position = proposed_cubes[-1]
        self.__target_kind = proposed_kind
        self.__target_position = proposed_position
        self.__proposed_cube_beams = proposed_beams
        self.__cubes = proposed_cubes
        self.__pipes = proposed_pipes
        self.__cube_ids = None
        self.__cube_beams = None
        self.__total_beams_interrupted = None

    def get_source_cube(self):
        return self.__source_cube

    def get_target_cube(self):
        return self.__target_cube

    def set_target_cube(self, target_cube):
        self.__target_cube = target_cube

    def get_target_kind(self):
        return self.__target_kind

    def get_target_position(self):
        return self.__target_position

    def set_cube_ids(self, cube_ids: list[CubeId]):
        self.__cube_ids = cube_ids

    def get_cube_ids(self):
        return self.__cube_ids

    def get_cubes(self):
        return self.__cubes

    def get_extra_cubes(self):
        return self.__cubes[1:-1]

    def get_pipes(self):
        return self.__pipes

    def get_cube_beams(self):
        return self.__cube_beams

    def set_cube_beams(self, beams: CubeBeams):
        self.__cube_beams = beams

    def get_total_beams_interrupted(self):
        return self.__total_beams_interrupted

    def set_total_beams_interrupted(self, total_interrupted_beams: int):
        self.__total_beams_interrupted = total_interrupted_beams

    def weight(self):
        return len(self.__pipes) * Path.PATH_LEN_HP + self.__total_beams_interrupted * Path.BEAMS_BROKEN_HP