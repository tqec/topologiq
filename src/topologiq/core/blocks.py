"""Topologiq's ZXBlock class, a dual-purpose ZX spider -meets- BlockGraph block object.

Design notes:
    ZX graphs and blockgraphs share the characteristic that while they
        have many many nodes (spiders | cubes), these are instances of
        a few specific ZXType or CubeKind.

    Topologiq's ZXBlock facilitates storing cubes and pipes in a way that
    can be easily utilised as either/both ZX spiders/legs and/or
        blockgraph cubes/pipes.

"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, lru_cache
from typing import ClassVar

import numpy as np

from topologiq.input.utils import ZXColors
from topologiq.utils.classes import CubeBeams, StandardCoord
from topologiq.utils.misc import kind_to_zx_type


#########################################
# HIGH-LEVEL REGISTRY                   #
# Applicable instance does NOT exist:   #
#   - create & add to registry.         #
# Applicable instance exists:           #
#   - retrieve.                         #
#########################################
class ZXBlockRegistry:
    """Registry to create and hold ZXBlock instances."""

    cache: ClassVar[dict[str, ZXBlock]] = {}

    @classmethod
    def get_create(cls, zx_type: str | None = None, kind: str | None = None):
        """Retrieve an existing ZXBlock or create a new one if applicable ZXBlock does not exist."""

        # Reject if not zx_type or kind is found
        if not zx_type and not kind:
            raise ValueError("Error creating/retrieving ZXBlock. A zx_type *or* a kind are needed.")

        # Create instance if it does not already exist
        key = sys.intern(kind) if kind else sys.intern(zx_type)
        if key not in cls.cache:
            # Enforce consistency in casing
            zx_type = zx_type.upper() if zx_type else zx_type
            kind = kind.upper() if kind else kind

            # Derive zx_type from kind if zx_type not given explicitly
            if not zx_type and kind:
                zx_type = kind_to_zx_type(kind)
            cls.cache[key] = ZXBlock(zx_type, kind)

        return cls.cache[key]


##########################
# ZX BLOCK CLASS         #
# Primary ZX Block class #
##########################
@dataclass(frozen=True)
class ZXBlock:
    """Topologiq's dual-use ZXBlock."""

    zx_type: str
    kind: str | None

    def __post_init__(self) -> None:
        """Post-initialisation actions."""

        # Health checks
        if self.kind:
            # Impossible open faces/axes counts
            num_open_axes = self.kind.count("O")
            if num_open_axes == 2:
                raise ValueError("ERROR. Cannot create ZXBlock: block cannot have two open bases.")

            # Malformed Y
            if "Y" in self.kind:
                if self.kind != "YYO":
                    raise ValueError("ERROR. Cannot create ZXBlock: malformed Y-kind.")

            # Malformed X, Z
            if num_open_axes == 1 and (self.kind[0] == self.kind[1] == self.kind[2]):
                raise ValueError(
                    "ERROR. Cannot create ZXBlock: X/Z cannot have equal basis in all axes."
                )

    @cached_property
    def get_kind_family(self) -> list[str]:
        """Get the family of kinds compatible with block's zx_type."""

        fams: dict[str, list[str]] = {
            "X": ["ZXZ", "ZZX", "XZZ"],
            "Y": ["YYO"],
            "Z": ["XZX", "XXZ", "ZXX"],
            "XZ": ["ZX*", "XZ*"],
            "T": ["TTO"],
            "O": ["OOO"],
            "BOUNDARY": ["OOO"],
            "SIMPLE": ["ZXO", "XZO", "OXZ", "OZX", "XOZ", "ZOX"],
            "HADAMARD": ["ZXOH", "XZOH", "OXZH", "OZXH", "XOZH", "ZOXH"],
        }

        if self.zx_type not in fams:
            print(f"Warning: type '{self.zx_type}' not found.")

        return fams[self.zx_type]

    @cached_property
    def get_basis(self) -> Basis | tuple[Basis, Basis, Basis]:
        """Get the basis of the ZXBlock."""

        # Derive basis from kind or zx_type
        if self.kind:
            return (Basis(self.kind[0]), Basis(self.kind[1]), Basis(self.kind[2]))
        else:
            return Basis(self.zx_type)

    @cached_property
    def get_zx_color(self) -> str:
        """Retrieve the ZXBlock's colours."""
        return ZXColors.lookup(self.zx_type)

    @cached_property
    def get_face_colors(self) -> tuple[str, tuple[str, str, str]]:
        """Retrieve the ZXBlock's colours."""
        return tuple([ZXColors.lookup(c) for c in self.kind[:3]]) * 2

    @cached_property
    def get_open_axes(self) -> tuple[bool, bool, bool]:
        """Return the open axes of a ZXCube.

        The open axes of a ZXCube change according to its kind. For X and Y cubes,
        these are the axes that do NOT correspond to the normal_basis. Y cubes can
        only have open faces along the Z-axis, while ports have open faces in all
        directions.

        """
        if self.get_basis in ((Basis.Y, Basis.Y, Basis.P), (Basis.T, Basis.T, Basis.P)) or "*" in self.kind:
            return (False, False, True)
        if self.get_basis == (Basis.P, Basis.P, Basis.P):
            return (True, True, True)


        basis_x, basis_y, basis_z = self.get_basis
        zx_basis = Basis(self.zx_type)
        open_axes = (basis_x != zx_basis, basis_y != zx_basis, basis_z != zx_basis)
        return open_axes

    @cached_property
    def get_move_vectors(self) -> list[StandardCoord]:
        """Return the normal vectors emanating from all open faces of a cube."""
        all_move_vectors: list[StandardCoord] = [
            (0, 0, 1),
            (1, 0, 0),
            (0, 1, 0),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1),
        ]
        valid_move_vectors = [vec for vec in all_move_vectors if self.check_move_exit(vec)]
        return valid_move_vectors

    @lru_cache
    def nxt_kinds(
        self, move: StandardCoord, is_hadamard: bool = False, tgt_zx_type: str = ""
    ) -> list[str]:
        """Reduce the number of possible kinds for next block.

        Args:
            move: The (x, y, z) displacement between current and target position.
            tgt_zx_type: The ZX type of the final target for the current edge.
            is_hadamard: True if edge being after this ZX cube is a Hadamard.

        Returns:
            ok_kinds: A list kinds that would constitute a topologically-correct placement.

        """

        # All kinds
        all_zx_blocks = [
            ZXBlockRegistry.get_create(kind=kind)
            for kind in ["XXZ", "ZZX", "XZZ", "ZXX", "ZXZ", "XZX"]
        ]

        # Add Y to sequence if target is Y-cube
        if tgt_zx_type in ["Y", "T"]:
            kind = tgt_zx_type * 2 + "O"
            all_zx_blocks.append(ZXBlockRegistry.get_create(kind=kind))

        # Add XZ to sequence if target is ZX-cube
        if tgt_zx_type == "XZ" and move == (0, 0, 1):
            all_zx_blocks.append(ZXBlockRegistry.get_create(kind="XZ*"))
            all_zx_blocks.append(ZXBlockRegistry.get_create(kind="ZX*"))

        if is_hadamard:
            rotated_kind = rotate_block_kind(self.kind, move)
            alt_self = ZXBlockRegistry.get_create(kind=rotated_kind)
        else:
            alt_self = None

        # Narrow down to kinds that would connect via open faces
        ok_tgts = [
            zx_block
            for zx_block in all_zx_blocks
            if self.cube_open_faces_match(move, tgt_zx_block=zx_block, alt_self=alt_self)
        ]

        if self.zx_type in ["O", "Y", "T"]:
            return ok_tgts
        else:
            return [
                ok_tgt_zx_block
                for ok_tgt_zx_block in ok_tgts
                if self.face_match(move, ok_tgt_zx_block, alt_self=alt_self)
            ]

    # Formerly known as `cube_match`.
    def cube_open_faces_match(
        self,
        move: StandardCoord,
        tgt_zx_block: ZXBlock | None = None,
        alt_self: ZXBlock | None = None,
    ) -> bool:
        """Check if two cubes are touching via an axis with open faces/exits.

        Args:
            move: The (x, y, z) displacement between self and other.
            tgt_zx_block: A ZX block to check as potential target connection.
            alt_self: A different ZX block to check instead of self.

        Returns:
            (bool): True if cubes match else False.

        """
        # Check source -> target
        if not self.check_move_exit(move, alt_self=alt_self):
            return False
        # Check target -> source
        if tgt_zx_block:
            if not self.check_move_exit(tuple([-i for i in (move)]), alt_self=tgt_zx_block):
                return False
        return True

    # Formerly known as `check_is_exit`
    @lru_cache  # Independent cache needed: called extensively from outside class
    def check_move_exit(self, move: StandardCoord, alt_self: ZXBlock | None = None) -> bool:
        """Check if a move is exiting block via a valid exit.

        Args:
            move: The (x, y, z) displacement between current and target position.
            alt_self: A different ZX block to check instead of self.

        Returns:
            (bool): True if face is an exit else False.

        """
        exit_idxs = self.get_open_axes if not alt_self else alt_self.get_open_axes
        diff_idx = int(np.nonzero(move)[0])
        return exit_idxs[diff_idx]

    def face_match(
        self, move: StandardCoord, tgt_zx_block: ZXBlock, alt_self: ZXBlock | None = None
    ) -> bool:
        """Check if the faces of two adjacent blocks are a topologically-correct match.

        Args:
            move: The (x, y, z) displacement between self and target block.
            tgt_zx_block: The target block being checked.
            alt_self: A different ZX block to check instead of self.

        Returns:
            (boolean): True if an available exit points towards target coordinate else False.

        """

        # Extract axis of displacement from kinds
        move_idx = int(np.nonzero(move)[0])
        if not alt_self:
            src_kind_new = self.kind[:move_idx] + self.kind[move_idx + 1 :]
        else:
            src_kind_new = alt_self.kind[:move_idx] + alt_self.kind[move_idx + 1 :]
        tgt_kind_new = tgt_zx_block.kind[:move_idx] + tgt_zx_block.kind[move_idx + 1 :]

        if move_idx == 2 and tgt_zx_block.zx_type == "Y":
            src_kind_new = src_kind_new.replace("X", "Y").replace("Z", "Y")

        # Return match
        return src_kind_new[:3] == tgt_kind_new


###############
# AUX OBJECTS #
###############
class Basis(Enum):
    """Defines valid values for a computational basis.

    NB! Class deviates slightly from standard "basis" values
    in quantum computing due to the spatial nature of TQEC. It
    aids comprehension to pack all possible values here than create
    other classes.

    """

    X = "X"
    Y = "Y"
    Z = "Z"
    P = "O"
    T = "T"

    @classmethod
    def _missing_(cls, val):
        if val in ["o", "*", "O", "P", "p"]:
            return cls.P

    def flip_basis(self) -> Basis:
        """Return the opposite basis if applicable."""
        return Basis.X if self == Basis.Z else Basis.Z if self == Basis.X else self

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.value}"


PositionedZXBlock = tuple[StandardCoord, ZXBlock]


@dataclass(order=True)
class CandidatePath:
    """A 3D path between the cubes corresponding to two nodes/spiders in the input ZX graph."""

    full_path: list[tuple[StandardCoord, ZXBlock]]
    tgt_beams: CubeBeams
    tgt_beams_short: CubeBeams
    beams_broken_by_path: int
    tgt_unobstr_exit_n: int

    def weighed_value(self, **kwargs) -> int:
        """Return the weighed value of a given path.

        This function returns the weighed value of a given PathBetweenNodes,
        which can be used for comparing many paths.

        Args:
            **kwargs: Only relevant kwargs listed below.
                weights: A tuple (int, int) of weights used to pick the best of several paths when there are several valid alternatives.

        Returns:
            (int): The weighed value of a path.

        """

        path_len_hp, beams_broken_hp = kwargs["weights"]

        return len(self.full_path) * path_len_hp + self.beams_broken_by_path * beams_broken_hp


##################
# AUX OPERATIONS #
##################
def rotate_block_kind(kind: str, move: StandardCoord) -> str:
    """Rotate a block around a move axis.

    This function enables cube rotation around an arbitrary axis by using the
    move vector as exit marker to create a rotation matrix.

    Args:
        kind: the kind of the block that needs rotation.
        move: The (x, y, z) displacement between current and target position.

    Returns:
        rot_k: a kind with the rotation incorporated into the new name.

    """

    h_flag = False
    if "H" in kind:
        h_flag = True
        kind.replace("H", "")

    # Build rotation matrix based on the direction of the move vector
    idxs = [0, 1, 2]
    idxs.remove(int(np.nonzero(move)[0]))

    new_matrix = {
        int(np.nonzero(move)[0]): np.eye(3, dtype=int)[int(np.nonzero(move)[0])],
        idxs[0]: np.eye(3, dtype=int)[idxs[1]],
        idxs[1]: np.eye(3, dtype=int)[idxs[0]],
    }

    rot_matrix = np.array([new_matrix[0], new_matrix[1], new_matrix[2]])

    # Rotate kind
    rot_k = ""
    for r in rot_matrix:
        entry = ""
        for j, ele in enumerate(r):
            entry += abs(int(ele)) * kind[j]
        rot_k += entry

    if h_flag:
        rot_k += "H"

    return rot_k
