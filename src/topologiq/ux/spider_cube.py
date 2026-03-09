"""Topologiq's SpiderCube, Spider, and Cube class and related methods.

Usage:
    Call a SpiderCube to instantiate the corresponding object.
    Call a SpiderCube method to use an existing SpiderCube.

Design notes:
    ZX graphs and blockgraphs share the characteristic that while they
        have many many nodes (spiders | cubes), these are instances of
        a few specific ZXType or CubeKind.

    Topologiq's SpiderCube is a dual-use object that facilitates storing
        graphs that can be visualised as either/both ZX graphs and/or
        blockgraphs.

"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import ClassVar

from topologiq.utils.misc import kind_to_zx_type


#########################################
# HIGH-LEVEL REGISTRY                   #
# Applicable instance does NOT exist:   #
#   - create & add to registry.         #
# Applicable instance exists:           #
#   - retrieve.                         #
#########################################
class SpiderCubeRegistry:
    """SpiderCube registry to create and hold SpiderCube instances."""

    cache: ClassVar[dict[str, SpiderCube]] = {}

    @classmethod
    def get_create(cls, zx_type: str | None = None, kind: str | None = None):
        """Retrieve an existing SpiderCube or create a new one if applicable SpiderCube does not exist."""

        # Reject if call does not have either a zx_type or a kind
        if not zx_type and not kind:
            raise ValueError(
                "Error creating/retrieving SpiderCube. A zx_type *or* a kind are needed."
            )

        # Set key to Spider or Cube depending on whether call contains a kind
        key = sys.intern(kind) if kind else sys.intern(zx_type)

        # Create instance if it does not already exist
        if key not in cls.cache:

            # Capitalise parameters for consistency
            zx_type = zx_type.lower() if zx_type else zx_type
            kind = kind.lower() if kind else kind

            # Derive zx_type from kind if zx_type not given explicitly
            if not zx_type and kind:
                zx_type = kind_to_zx_type(kind)

            # Derive basis from kind or zx_type
            if kind:
                basis = (Basis(kind[0]), Basis(kind[1]), Basis(kind[2]))
            else:
                basis = (Basis(zx_type), Basis(zx_type), Basis(zx_type))

            cls.cache[key] = SpiderCube(zx_type, kind, basis)

        # Return instance
        return cls.cache[key]


#############################
# SPIDERCUBE                #
# Primary SpiderCube class  #
#############################
@dataclass(frozen=True)
class SpiderCube:
    """Topologiq's dual-use SpiderCube.

    A SpiderCube can double as either a (2D) ZX spider or a
    (3D) BlockGraph cube.

    """

    zx_type: str
    kind: str | None
    basis: tuple[Basis, Basis, Basis] | None = None

    def __post_init__(self) -> None:
        """Post-initialisation actions."""

        # Reject cubes with impossible open faces/axes counts
        if self.kind:
            num_open_axes = self.basis.count(Basis.P)
            if num_open_axes == 2:
                raise ValueError("Error creating SpiderCube. Cubes cannot have two open bases.")

            # Reject malformed Y-cubes
            y_count = self.basis.count(Basis.Y)
            if y_count not in {0, 3}:
                raise ValueError("Error creating SpiderCube. Y-cubes can only have Y-bases.")

            # Reject malformed X or Z cubes
            if num_open_axes == 1 and (self.basis[0] == self.basis[1] == self.basis[2]):
                raise ValueError(
                    "Error creating SpiderCube. X/Z-cubes cannot have equal basis in all faces."
                )

        # Basis
        if self.kind:
            x, y, z = list(self.kind)
            self.basis = (Basis(x), Basis(y), Basis(z))
        else:
            self.basis = (Basis(self.zx_type), Basis(self.zx_type), Basis(self.zx_type))

    @cached_property
    def get_colours(self) -> tuple[str, tuple[str, str, str]]:
        """Retrieve the SpiderCube's colours."""
        return ZXColors.lookup(self)

    @cached_property
    def cube_exits(self) -> tuple[bool, bool, bool, bool, bool, bool]:
        """Return the open axes of a ZXCube.

        The open axes of a ZXCube change according to its kind. For X and Y cubes,
        these are the axes that do NOT correspond to the normal_basis. Y cubes can
        only have open faces along the Z-axis, while ports have open faces in all
        directions.

        """

        if self.basis == (Basis.Y, Basis.Y, Basis.Y):
            return (False, False, True) * 2
        if self.basis == (Basis.P, Basis.P, Basis.P):
            return (True, True, True) * 2

        x, y, z = self.basis
        cube_exits = (x != self.zx_type, y != self.zx_type, z != self.zx_type) * 2
        return cube_exits


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
    Z = "#d7a4a1"
    P = "#333333"
    H = "#ffff00"
    SIMPLE = "#000000"

    @classmethod
    def lookup(cls, spider_cube: SpiderCube) -> tuple[str, tuple[str, str, str]]:
        """Get standardised HEX colours for an arbitrary SpiderCube.

        Args:
            spider_cube: The SpiderCube for which colours are being requested.

        Returns:
            zx_color: The HEX corresponding to the ZX colour of the SpiderCube.
            ax_colors: A tuple of HEX colours corresponding to the colours of (x, y, z) faces of the SpiderCube


        """

        try:
            zx_color = cls[spider_cube.zx_type.upper()].value
            ax_colors = tuple([cls[str(b)] for b in spider_cube.basis])
            return zx_color, ax_colors
        except (KeyError, AttributeError):
            return cls.SIMPLE, (cls.SIMPLE, cls.SIMPLE, cls.SIMPLE)
