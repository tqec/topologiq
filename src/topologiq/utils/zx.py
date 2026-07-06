"""Utilities to assist ZX-calculus operations of various sorts.

Usage:
    Call any function/class from a separate script.

"""


from enum import Enum
from fractions import Fraction

import pyzx as zx
from pyzx.graph.base import BaseGraph
from pyzx.graph.graph_s import GraphS
from pyzx.rewrite_rules.bialgebra_rule import match_bialgebra_op
from pyzx.utils import EdgeType, VertexType


####################
# KIND CONVERSIONS #
####################
def kind_to_zx_type(kind: str) -> str:
    """Get the ZX type corresponding to a given block or pipe kind.

    Args:
        kind: the /kind of a given block.

    Returns:
        zx_type: the ZX type corresponding to the kind.

    """

    if kind == "OOO":
        zx_type = "O"
    elif kind[0] in ["Y", "T"]:
        zx_type = kind[0]
    elif "*" in kind:
        zx_type = "ZX"
    elif "O" in kind:
        zx_type = "HADAMARD" if "h" in kind else "SIMPLE"
    else:
        zx_type = min(set(kind), key=lambda c: kind.count(c)).capitalize()
    return zx_type


######################
# GRAPH MANIPULATION #
######################
def rm_unnecessary_phases(pyzx_graph: zx.Graph):
    """Remove 1/1 phases for prettier visualisation."""
    [
        pyzx_graph.set_phase(i, 0)
        for i in pyzx_graph.vertices()
        if pyzx_graph.phase(i) == Fraction(1, 1)
    ]


def apply_bialgebra(pyzx_graph: BaseGraph | GraphS):
    """Get bi-algebra patterns.

    Args:
        pyzx_graph: An arbitrary PyZX graph.

    """

    matches = _get_bialgebra_candidates(pyzx_graph)
    altered_spiders = set()
    for match in matches:
        if any([m in altered_spiders for m in match]):
            continue
        if match_bialgebra_op(pyzx_graph, match):
            zx.simplify.bialg_op_simp.apply(pyzx_graph, match)
            altered_spiders.update(match)
        else:
            _clean_bialgebra_pattern(pyzx_graph, match)
            if match_bialgebra_op(pyzx_graph, match):
                zx.simplify.bialg_op_simp.apply(pyzx_graph, match)
                altered_spiders.update(match)
    zx.simplify.id_simp(pyzx_graph)


def _get_bialgebra_candidates(pyzx_graph: BaseGraph | GraphS) -> list[list[int]]:
    """Get bi-algebra patterns.

    Args:
        pyzx_graph: An arbitrary PyZX graph.

    Returns:
        matches: A list of lists containing matches for the bi-algebra pattern.

    """

    eligible_z = [
        s_id
        for s_id in pyzx_graph.vertices()
        if pyzx_graph.type(s_id) == VertexType.Z
        and pyzx_graph.vertex_degree(s_id) >= 2  # Must connect to at least 2 X-spiders
        and sum(
            1
            for n_id in pyzx_graph.neighbors(s_id)
            if pyzx_graph.type(n_id) == VertexType.X
            and pyzx_graph.edge_type(pyzx_graph.edge(s_id, n_id)) == EdgeType.SIMPLE
        )
        >= 2
    ]

    eligible_x = [
        s_id
        for s_id in pyzx_graph.vertices()
        if pyzx_graph.type(s_id) == VertexType.X
        and pyzx_graph.vertex_degree(s_id) >= 2  # Must connect to at least 2 Z-spiders
        and sum(
            1
            for n_id in pyzx_graph.neighbors(s_id)
            if pyzx_graph.type(n_id) == VertexType.Z
            and pyzx_graph.edge_type(pyzx_graph.edge(s_id, n_id)) == EdgeType.SIMPLE
        )
        >= 2
    ]

    # Convert Z-candidates to a set for O(1) membership lookups
    matches = []

    num_x = len(eligible_x)
    # 2. Iterate through pairs of filtered X-spiders
    for i in range(num_x):
        x1 = eligible_x[i]

        # Get only the valid Z-neighbors of x1 that passed our strict filters
        z_neighbors_x1 = set(n_id for n_id in pyzx_graph.neighbors(x1) if n_id in eligible_z)

        for k in range(i + 1, num_x):
            x2 = eligible_x[k]
            # Get only the valid Z-neighbors of x2 that passed our strict filters
            z_neighbors_x2 = set(n_id for n_id in pyzx_graph.neighbors(x2) if n_id in eligible_z)

            # The Matrix/Set Intersection step: find shared paths of length 2
            shared_z = z_neighbors_x1.intersection(z_neighbors_x2)

            # A valid K_2,2 reverse bi-algebra match requires exactly or at least 2 shared Zs
            if len(shared_z) >= 2:
                shared_z_list = list(shared_z)

                # If there are exactly 2, extract them. If more, extract combinations.
                for j in range(len(shared_z_list)):
                    for m in range(j + 1, len(shared_z_list)):
                        z1 = shared_z_list[j]
                        z2 = shared_z_list[m]

                        # Append the verified 4-spider structural match
                        matches.append((x1, x2, z1, z2))

    return matches


def _clean_bialgebra_pattern(pyzx_graph: BaseGraph | GraphS, match_candidates: list[int]):
    """Get bi-algebra patterns.

    Args:
        pyzx_graph: An arbitrary PyZX graph.
        match_candidates: A list containing a spiders ID to use as candidates for a bi-algebra transformation.

    """

    for s_id in match_candidates:
        original_neighbours = list(pyzx_graph.neighbors(s_id))
        twin_id = max(pyzx_graph.vertices()) + 1
        qubit = pyzx_graph.qubit(s_id) + 0.3
        row = pyzx_graph.row(s_id) + (-2 if pyzx_graph.row(s_id) == 4 else 2)
        pyzx_graph.add_vertex(ty=pyzx_graph.type(s_id), index=twin_id, qubit=qubit, row=row)
        pyzx_graph.add_edge((s_id, twin_id))

        for neigh_id in original_neighbours:
            if neigh_id not in match_candidates:
                pyzx_graph.remove_edge((s_id, neigh_id))
                pyzx_graph.add_edge((twin_id, neigh_id))
        zx.draw(pyzx_graph, labels=True)


#####################
# ZX TYPES & COLORS #
#####################
class ZXTypes(int, Enum):
    """ZX vertex type conversions to standardise visualisations."""

    BOUNDARY = 0
    X = 2
    Z = 1

    @classmethod
    def from_str(cls, name: str) -> int:
        """Convert a string-based ZX vertex type into its PyZX integer type."""
        try:
            # Normalize to uppercase to match Enum keys
            return cls[name.upper()].value
        except KeyError:
            return cls.BOUNDARY.value  # Safe default


class ZXEdgeTypes(int, Enum):
    """ZX edge type conversions to standardise visualisations."""

    SIMPLE = 1
    HADAMARD = 2

    @classmethod
    def from_str(cls, name: str) -> int:
        """Convert a string-based ZX edge type into its PyZX integer type."""
        try:
            # Normalize to uppercase to match Enum keys
            return cls[name.upper()].value
        except KeyError:
            return cls.SIMPLE.value  # Safe default


class ZXColors(str, Enum):
    """Colour palette to standardise visualisations."""

    X = "#d7a4a1"
    Y = "#7fff00"
    Z = "#b9cdff"
    P = "#777777"
    XZ = "#f2f3fb"
    T = "#f531ff"
    HADAMARD = "#ffff00"
    BOUNDARY = "#777777"
    SIMPLE = "#000000"

    @classmethod
    def lookup(cls, char: str) -> str:
        """Get standardised HEX colours for an arbitrary SpiderBlock.

        Args:
            char: A character, typically signifying a zx_type or basis.

        Returns:
            zx_color: A colour HEX corresponding to the character.

        """

        try:
            char = "XZ" if char == "*" else char
            return cls[char.upper()]
        except (KeyError, AttributeError):
            return cls.BOUNDARY if char.upper() == "O" else cls.SIMPLE
