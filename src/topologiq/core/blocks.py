"""Topologiq's ZXBlock class, a dual-purpose ZX-spider-meets-block object.

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

from topologiq.utils.classes import StandardCoord
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
            zx_type = zx_type.lower() if zx_type else zx_type
            kind = kind.lower() if kind else kind

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
            num_open_axes = self.kind.count("o")
            if num_open_axes == 2:
                raise ValueError("ERROR. Cannot create ZXBlock: block cannot have two open bases.")

            # Malformed Y
            if "y" in self.kind.count:
                if self.kind != "yoy":
                    raise ValueError("ERROR. Cannot create ZXBlock: malformed Y-kind.")

            # Malformed X, Z
            if num_open_axes == 1 and (self.kind[0] == self.kind[1] == self.kind[2]):
                raise ValueError(
                    "ERROR. Cannot create ZXBlock: X/Z cannot have equal basis in all axes."
                )

    @cached_property
    def get_basis(self) -> Basis | tuple[Basis, Basis, Basis]:
        """Get the basis of the ZXBlock."""

        # Derive basis from kind or zx_type
        if self.kind and "o":
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
        return tuple([ZXColors.lookup(c) for c in self.kind[:3]])

    @cached_property
    def open_axes(self) -> tuple[bool, bool, bool, bool, bool, bool]:
        """Return the open axes of a ZXCube.

        The open axes of a ZXCube change according to its kind. For X and Y cubes,
        these are the axes that do NOT correspond to the normal_basis. Y cubes can
        only have open faces along the Z-axis, while ports have open faces in all
        directions.

        """

        if self.basis() == (Basis.P, Basis.P, Basis.P):
            return (True, True, True) * 2

        x, y, z = self.basis()
        open_axes = (x != self.zx_type, y != self.zx_type, z != self.zx_type)
        return open_axes

    @lru_cache
    def nxt_kinds(self, move: StandardCoord) -> list[str]:
        """Reduce the number of possible kinds for next block.

        Args:
            move: The (x, y, z) displacement between current and target position.

        Returns:
            ok_kinds: A list kinds that would constitute a topologically-correct placement.

        """
        # All kinds
        cube_kinds = ["xxz", "zzx", "xzz", "zxx", "zxz", "xzx"]
        pipe_kinds = ["zxo", "xzo", "oxz", "ozx", "xoz", "zox"]

        # Narrow down to kinds that would connect via open faces
        ok_kinds = [
            ok_kind
            for ok_kind in (cube_kinds if "o" in self.kind else pipe_kinds)
            if self.cube_open_faces_match(self.kind[:3], move, ok_kind)
        ]

        # Reduce to topologically-correct kinds given direction of move
        return [ok_kind for ok_kind in ok_kinds if self.face_match(self.kind[:3], move, ok_kind)]

    # Formerly known as `cube_match`.
    def cube_open_faces_match(self, move: StandardCoord, tgt_kind: str) -> bool:
        """Check if two cubes are touching via an axis with open faces/exits.

        Args:
            move: The (x, y, z) displacement between self and other.
            tgt_kind: The kind of the target block being checked.

        Returns:
            (bool): True if cubes match else False.

        """
        # Check source -> target
        if not self.check_move_exit(self.kind, move):
            return False
        # Check target -> source
        if not self.check_move_exit(tgt_kind, tuple([-i for i in (move)])):
            return False

        return True

    # Formerly known as `check_is_exit`
    @lru_cache  # Independent cache needed: called extensively from outside class
    def check_move_exit(self, move: StandardCoord) -> bool:
        """Check if a move is exiting block via a valid exit.

        Args:
            move: The (x, y, z) displacement between current and target position.

        Returns:
            (bool): True if face is an exit else False.

        """
        # Exit indexes
        exit_idxs = self.open_axes.index(True)
        # Move axis
        diff_idx = int(np.nonzero(move)[0])

        return diff_idx in exit_idxs

    def face_match(self, move: StandardCoord, tgt_kind: str) -> bool:
        """Check if the faces of two adjacent blocks are a topologically-correct match.

        Args:
            move: The (x, y, z) displacement between current and target position.
            tgt_kind: The kind of the target block being checked.

        Returns:
            (boolean): True if an available exit points towards target coordinate else False.

        """
        # Extract axis of displacement from kinds
        move_idx = int(np.nonzero(move)[0])
        src_kind_new = self.kind[:move_idx] + self.kind[move_idx + 1 :]
        tgt_kind_new = tgt_kind[:move_idx] + tgt_kind[move_idx + 1 :]

        # Return match
        return src_kind_new[:3] == tgt_kind_new


#######
# AUX #
#######
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

    @classmethod
    def _missing_(cls, val):
        if val in ["o", "P", "p"]:
            return cls.P
        else:
            return cls(val.upper())

    def flip_basis(self) -> Basis:
        """Return the opposite basis if applicable."""
        return Basis.X if self == Basis.Z else Basis.Z if self == Basis.X else self

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.value}"


class ZXColors(str, Enum):
    """Colour palette to standardise visualisations."""

    X = "#d7a4a1"
    Y = "#7fff00"
    Z = "#b9cdff"
    P = "#333333"
    HADAMARD = "#ffff00"
    SIMPLE = "#000000"

    @classmethod
    def lookup(cls, char: str) -> str:
        """Get standardised HEX colours for an arbitrary ZXBlock.

        Args:
            char: A character, typically signifying a zx_type or basis.

        Returns:
            zx_color: A colour HEX corresponding to the character.

        """

        try:
            return cls[char.upper()]
        except (KeyError, AttributeError):
            return cls.SIMPLE
