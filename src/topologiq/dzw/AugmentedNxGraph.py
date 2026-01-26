import random
import pyzx as zx
import networkx as nx

from topologiq.dzw.GraphComponents import NodeType, EdgeType, CubeKind
from topologiq.dzw.BlockGraphSpace import Coordinates, BlockGraphSpace

# TODO: figure out what the other VertexType and EdgeType represent
# TODO: how do we deal with the last four VertexType (i.e. H_BOX, W_INPUT, W_OUTPUT, Z_BOX) ?
# TODO: do we need the last EdgeType (i.e. W_IO) ?
# TODO: how do we deal with the phase of a spider ?
# TODO: benchmarking and timing various parts
# TODO: construction of animation
class AugmentedNxGraph(nx.Graph):
    def __init__(self, pyzx_graph: zx.graph.base.BaseGraph):
        super().__init__()

        # Keeps track of the coordinates in 3D that are occupied by some cube
        # TODO: crude but works. Inefficient for crowded space (Binary Space Partitioning ?)
        self.occupied: set[Coordinates] = set()
        # Keeps track of the paths (i.e. one or more pipes) in the BlockGraph that realize the edges of the ZX-graph
        self.edge_realizations: dict = {}

        for vertex in pyzx_graph.vertices():
            self.add_node(
                vertex,
                # The vertex identifier from the ZX-graph
                # None means this is an extra cube added to realize some edge.
                zx = vertex,
                # NodeType : X, Y, Z or O
                type = NodeType.convert(pyzx_graph.type(vertex)),
                # Attributes of the cube that realizes this vertex
                kind = None,        # CubeKind
                position = None, # 3D coordinates
                beams = None        # Orientation of the I/O ports
            )

        for edge in pyzx_graph.edges():
            self.add_edge(
                edge[0], # Source
                edge[1], # Target
                # EdgeType : IDENTITY or HADAMARD
                type = EdgeType.convert(pyzx_graph.edge_type(edge)),
                # The edge from the ZX-graph.
                # None means this is an extra pipe of a path that realizes some edge from the ZX-graph.
                zx = edge
            )

        # TODO: split any spider with more than 4 edges (cfr. graph_manager.py; prep_3d_g)
        # TODO: does the choice of how to split such spiders affect the minimal achievable volume ?
        # TODO: do we need to check the validity of the constructed nx.Graph ?

    def is_boundary(self, v: int) -> bool:
        return self.nodes[v]['type'] == NodeType.O

    def is_spider(self, v: int) -> bool:
        return self.nodes[v]['type'] != NodeType.O

    # find_first_id(..)
    def pick_root(self, central_spider: bool = True, deterministic: bool = False) -> int:
        """Pick the spider that will serve as the root of the construction.

        Args:
            deterministic (bool, optional):
                True  => return the candidate spider with the lowest ID
                False => return a random candidate spider
            central_spider (bool, optional):
                True  => candidates are all spiders with maximal degree
                False => candidates are all spiders

        Returns:
            ID of the spider that has been selected as the root
        """

        if self.number_of_nodes() == 0:
            raise nx.exception.NodeNotFound("Graph is empty.")

        # n.b. entries of self.degree are of tuples of the form (id, degree)
        if central_spider:
            (_, max_degree) = max(self.degree, key=lambda entry: entry[1])
            candidates = list(filter(lambda v: self.is_spider(v) and self.degree[v] == max_degree, self.nodes))
        else:
            candidates = [v for v in self.nodes if self.is_spider(v)]

        return min(candidates) if deterministic else random.choice(candidates)

    def get_candidate_adjacent(self, source: int) -> list[tuple[Coordinates, CubeKind]]:
        if not self.is_vertex_placed(source):
            raise ValueError(f"{source} is not placed and thus has no kind. Cannot determine its adjacent candidates.")
        source_position = self.nodes[source]['position']
        source_type = self.nodes[source]['type']
        source_kind = self.nodes[source]['kind']
        source_plane = source_kind.get_plane()
        candidates_adjacent = []
        for step in BlockGraphSpace.STEPS:
            # A cube can only have an adjacent cube that lies in the same plane
            if source_plane.contains(step):
                candidate_coordinates = source_position + step.value
                # A cube can only have an adjacent cube at a position that is not occupied by another cube
                if candidate_coordinates not in self.occupied:
                    # A cube can always have an adjacent cube of the same kind
                    candidates_adjacent.append( ( candidate_coordinates, source_kind) )
                    # A cube can always have an adjacent cube of the other color lying in a plane
                    # that is orthogonal to its plane along the step
                    candidate_plane = BlockGraphSpace.get_orthogonal_plane(source_plane, step)
                    candidate_kind = CubeKind.convert(NodeType.flip(source_type), candidate_plane)
                    candidates_adjacent.append( (candidate_coordinates, candidate_kind) )
        return candidates_adjacent

    # TODO: provide number of unobstructed ports, number of legs, information to check the beams
    def get_unobstructed_ports(self, v: int) -> int:
        return 0

    def is_vertex_placed(self, v: int) -> bool:
        return self.nodes[v]['coordinates'] is not None and self.nodes[v]['kind'] is not None

    def place_vertex(self, v: int, kind: CubeKind, position: Coordinates):
        """Realize the vertex as a cube of the given kind placed at the given coordinates."""
        if position in self.occupied:
            raise Exception(f"Requested {position} is already occupied by another cube.")

        if kind not in CubeKind.suitable_kinds(self.nodes[v]['type']):
            raise Exception(f"Requested {kind} is not compatible with {self.nodes[v]['type']}")

        self.nodes[v]['kind'] = kind
        self.nodes[v]['position'] = position

        # TODO: compute the beams of the new cube
        # TODO: prune the beams of other cubes

        self.occupied.add(position)

    def is_edge_realized(self, source: int, target: int) -> bool:
        return (source,target) in self.edge_realizations

    # TODO: this really belongs in the BFS.
    # Used to find the functionalities which belong in AugmentedNxGraph.
    def realize_edge(self, source: int, target: int, maximal_distance: int = 1) -> bool:
        if not self.is_vertex_placed(source):
            raise ValueError(f"{source} is not placed. Cannot realize any of its edges.")

        edge = (source, target) if source < target else (target, source)

        source_kind = self.nodes[source]['kind']
        source_coordinates = self.nodes[source]['coordinates']

        if not self.is_vertex_placed(target):
            # First-pass edge.
            self.number_of_1st_pass_edges += 1
            # TODO: cfr. attempt place_nxt_block(..) with step in [3,6,9]
            raise NotImplemented("First-pass edge processing.")
        elif edge not in self.edge_realizations:
            # Second-pass edge.
            self.number_of_2nd_pass_edges += 1
            target_kind = self.nodes[target]['kind']
            target_coordinates = self.nodes[target]['coordinates']

            # TODO: deal with the critical beams (cfr. graph_manager.py Lines 301-313)

            # Check if edge is Hadamard
            # Call pathfinder for second-pass
            # clean_paths, vis_data = run_pathfinder(
            #       source_coords, source_kind,
            #       target_type, target_coords, target_kind,
            #       edge_type == HADAMARD ?,
            #       init_step=3)
            # update edge_realizations with clean_paths[0]
            # TODO: a path can be a list of (coordinates, kind)

            raise NotImplemented("Second-pass edge processing.")

        self.edge_realizations[edge] = []

        return True