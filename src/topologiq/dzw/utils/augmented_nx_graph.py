from collections import deque

import pyzx as zx
import networkx as nx

from topologiq.dzw.utils.coordinates import Coordinates
from topologiq.dzw.helpers.spacetime import Spacetime
from topologiq.dzw.helpers.blockgraph import BlockGraphHelper

from topologiq.dzw.utils.components_zx import NodeId, NodeType, EdgeType
from topologiq.dzw.utils.components_bg import CubeId, CubeKind
from topologiq.dzw.utils.path import Path

from logging import getLogger
console = getLogger(__name__)

# TODO: figure out what the other VertexType and EdgeType represent
# TODO: how do we deal with the last four VertexType (i.e. H_BOX, W_INPUT, W_OUTPUT, Z_BOX) ?
# TODO: do we need the last EdgeType (i.e. W_IO) ?
# TODO: how do we deal with the phase of a spider ?
# TODO: benchmarking and timing various parts
# TODO: construction of animation
class AugmentedNxGraph:
    KEY_ZX_NODE_TYPE = 'zx_node_type'
    KEY_ZX_EDGE_TYPE = 'zx_edge_type'
    KEY_ZX_EDGES_REALISED = 'zx_edges_realised'
    KEY_ZX_BG_CUBE = 'zx_bg_cube'
    KEY_ZX_BG_PATH = 'zx_bg_path'

    KEY_BG_ZX_NODE   = 'bg_zx_node'
    KEY_BG_CUBE_KIND = 'bg_cube_kind'
    KEY_BG_CUBE_POSITION = 'bg_cube_position'
    KEY_BG_PIPE_TYPE = 'bg_pipe_type'
    KEY_BG_CUBE_BEAMS = 'bg_cube_beams'

    def __init__(self, zx_graph: zx.graph.base.BaseGraph):
        super().__init__()

        # Separate ZX-graph and BG-graph
        self.__zx_graph = nx.Graph()
        self.__bg_graph = nx.Graph()

        # Tracks the order in which nodes and edges from the ZX graph were realised into the Blockgraph
        self.__zx_node_realisation_order = []
        self.__zx_edge_realisation_order = []

        # Keeps track of the coordinates in 3D that are occupied by some cube
        # TODO: Replace with efficient data-structure for crowded space (Binary Space Partitioning ?)
        self.occupied: set[Coordinates] = set()

        for node in zx_graph.vertices():
            self.__zx_graph.add_node(node)
            self.__zx_graph.nodes[node][AugmentedNxGraph.KEY_ZX_NODE_TYPE] = NodeType.convert(zx_graph.type(node))
            self.__zx_graph.nodes[node][AugmentedNxGraph.KEY_ZX_EDGES_REALISED] = 0
            self.__zx_graph.nodes[node][AugmentedNxGraph.KEY_ZX_BG_CUBE] = None

        for edge in zx_graph.edges():
            source = min(edge)
            target = max(edge)
            self.__zx_graph.add_edge(source, target)
            self.__zx_graph.get_edge_data(source, target)[AugmentedNxGraph.KEY_ZX_EDGE_TYPE] = EdgeType.convert(zx_graph.edge_type(edge))
            self.__zx_graph.get_edge_data(source, target)[AugmentedNxGraph.KEY_ZX_BG_PATH] = None

        self.__next_cube_id = self.__zx_graph.number_of_nodes()

        # TODO: split any spider with more than 4 edges (cfr. graph_manager.py; prep_3d_g)
        # TODO: does the choice of how to split such spiders affect the minimal achievable volume ?
        _, max_degree = max(self.__zx_graph.degree, key=lambda entry: entry[1])
        if max_degree > 4:
            raise NotImplemented("Enforcement of no-more-than-four-legs condition not implemented.")

    def get_node_realisation_order(self):
        return self.__zx_node_realisation_order

    def get_edge_realisation_order(self):
        return self.__zx_edge_realisation_order

    def get_nodes(self):
        return self.__zx_graph.nodes()

    def number_of_nodes(self) -> int:
        return self.__zx_graph.number_of_nodes()

    def get_edges(self):
        return self.__zx_graph.edges()

    def number_of_edges(self) -> int:
        return self.__zx_graph.number_of_edges()

    def get_cubes(self):
        return self.__bg_graph.nodes()

    def number_of_cubes(self) -> int:
        return self.__bg_graph.number_of_nodes()

    def get_pipes(self):
        return self.__bg_graph.edges()

    def number_of_pipes(self) -> int:
        return self.__bg_graph.number_of_edges()

    def get_edges_realised(self, node: NodeId):
        return self.__zx_graph.nodes[node].get(AugmentedNxGraph.KEY_ZX_EDGES_REALISED)

    def get_edges_unrealised(self, node: NodeId):
        return self.get_degree(node) - self.get_edges_realised(node)

    def get_node_neighbours(self, node: NodeId):
        return self.__zx_graph.neighbors(node)

    def get_cube_neighbours(self, cube: CubeId):
        return self.__bg_graph.neighbors(cube)

    def get_degree(self, node: NodeId):
        return self.__zx_graph.degree[node]

    def is_boundary(self, node: NodeId) -> bool:
        return self.get_node_type(node) == NodeType.O

    def is_spider(self, node: NodeId) -> bool:
        return self.get_node_type(node) != NodeType.O

    def is_cube_placed(self, cube: CubeId) -> bool:
        return cube in self.__bg_graph

    def get_cube(self, node: NodeId):
        return self.__zx_graph.nodes[node][AugmentedNxGraph.KEY_ZX_BG_CUBE]

    def get_node(self, cube: CubeId):
        return self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_ZX_NODE]

    def get_node_type(self, node: NodeId) -> NodeType:
        return self.__zx_graph.nodes[node][AugmentedNxGraph.KEY_ZX_NODE_TYPE]

    def get_cube_position(self, cube: CubeId) -> Coordinates:
        return self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_CUBE_POSITION]

    def get_cube_kind(self, cube: CubeId) -> CubeKind:
        return self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_CUBE_KIND]

    def get_pipe_type(self, source_cube: CubeId, target_cube: CubeId) -> EdgeType :
        return self.__bg_graph.get_edge_data(source_cube, target_cube).get(AugmentedNxGraph.KEY_BG_PIPE_TYPE)

    def get_edge_type(self, source: NodeId, target: NodeId) -> EdgeType:
        return self.__zx_graph.get_edge_data(source, target).get(AugmentedNxGraph.KEY_ZX_EDGE_TYPE)

    def get_edge_realisation(self, source: NodeId, target: NodeId):
        return self.__zx_graph.get_edge_data(source, target).get(AugmentedNxGraph.KEY_ZX_BG_PATH)

    def is_node_realised(self, node: NodeId) -> bool:
        return self.__zx_graph.nodes[node][AugmentedNxGraph.KEY_ZX_BG_CUBE] is not None

    def realise_node(self, node: NodeId, kind: CubeKind, position: Coordinates) -> CubeId:
        """Realise the node as a cube of the given kind placed at the given coordinates."""
        if kind not in CubeKind.suitable_kinds(self.get_node_type(node)):
            raise Exception(f"Requested {kind} is not compatible with {self.get_node_type(node)}")

        if not self.__zx_graph.has_node(node):
            raise Exception(f"Node #{node} not found in the ZX-graph.")

        cube = self.place_cube(kind, position)
        self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_ZX_NODE] = node
        self.__zx_graph.nodes[node][AugmentedNxGraph.KEY_ZX_BG_CUBE] = cube

        console.info(f"Realising node #{node} [{self.get_node_type(node)}] as cube #{cube} [{kind}@{position}]")
        self.__zx_node_realisation_order.append(node)

        return cube

    def find_realising_cubes(self, node: NodeId) -> set[CubeId]:
        if not self.is_node_realised(node):
            raise Exception(f"Node #{node} is not realised by any cube.")

        node_type = self.get_node_type(node)
        queue: deque[CubeId] = deque([ self.get_cube(node) ])
        realising: set[CubeId] = set()

        # TODO: explore within the BlockGraph
        while queue:
            current = queue.popleft()

            for successor in self.get_cube_neighbours(current):
                successor_type = self.get_node_type(successor)
                pipe_type = self.get_pipe_type(current, successor)
                if successor_type == node_type and pipe_type == EdgeType.IDENTITY and successor not in realising:
                    queue.append(successor)
                    realising.add(successor)

        return realising

    def is_edge_realised(self, source: NodeId, target: NodeId) -> bool:
        return self.__zx_graph.get_edge_data(source, target)[AugmentedNxGraph.KEY_ZX_BG_PATH] is not None

    def realise_edge(self, source: NodeId, target: NodeId, proposed_path: Path):
        if not self.is_node_realised(source):
            raise Exception(f"{source} is not placed; cannot connect with a path.")

        if not self.is_node_realised(target):
            raise Exception(f"{target} is not placed; cannot connect with a path.")

        if not self.__zx_graph.has_edge(source, target):
            raise Exception(f"No edge {source}-{target} found in the ZX-graph.")

        if self.is_edge_realised(source, target):
            raise Exception(f"{source}-{target} is already realized by a path.")

        source_cube = self.get_cube(source)
        target_cube = self.get_cube(target)

        edge_type = self.get_edge_type(source, target)

        # # Reject path if it is invalid.
        if not self.is_path_valid(proposed_path, edge_type):
            raise Exception(f"Proposed path to realise edge {source}-{target} is invalid.")

        if not proposed_path:
            sequence = "[]"
        else:
            sequence = ""
            for position, kind in proposed_path.get_extra_cubes():
                sequence += f"{kind}@{position}"
        console.info(f"Realising edge {source}-{target} [type={self.get_edge_type(source,target)}] with extra cubes : {sequence}")

        # Representation of the path that will go into edge_realisations
        extras = []

        # Add all the extra cubes and pipes of the path to the BlockGraph
        previous_cube: int = source_cube

        proposed_cubes = proposed_path.get_cubes()
        proposed_pipes = proposed_path.get_pipes()

        n = len(proposed_cubes)
        for index in range(1, n-1):
            current_kind, current_position = proposed_cubes[index]
            current_pipe_type = proposed_pipes[index-1]

            # Place the current cube and connect it to the previous cube.
            current_cube = self.place_cube(current_kind, current_position)
            self.__bg_graph.nodes[current_cube][AugmentedNxGraph.KEY_BG_ZX_NODE] = None
            self.connect_pipe(previous_cube, current_cube, current_pipe_type)

            # Extend the sequence of extra node ids
            extras.append(current_cube)

            # Prepare for the next iteration
            previous_cube = current_cube

        # Make the final connection
        target_cube = self.get_cube(target)
        final_pipe_type = proposed_pipes[-1]
        self.connect_pipe(previous_cube, target_cube, final_pipe_type)

        # Associate the path as a realisation of the edge
        self.__zx_graph.get_edge_data(source, target)[AugmentedNxGraph.KEY_ZX_BG_PATH] = extras

        # One more edge has been realised
        self.__zx_graph.nodes[source][AugmentedNxGraph.KEY_ZX_EDGES_REALISED] += 1
        self.__zx_graph.nodes[target][AugmentedNxGraph.KEY_ZX_EDGES_REALISED] += 1
        self.__zx_edge_realisation_order.append( (source, target) )

    def place_cube(self, kind: CubeKind, position: Coordinates):
        if position in self.occupied:
            raise Exception(f"Proposed position for {kind}@{position} is already occupied by another cube.")

        cube = self.__next_cube_id
        self.__next_cube_id += 1

        self.__bg_graph.add_node(cube)

        self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_ZX_NODE] = None
        self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_CUBE_KIND] = kind
        self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_CUBE_POSITION] = position

        self.occupied.add(position)

        return cube

    def connect_pipe(self, source_cube: CubeId, target_cube: CubeId, pipe_type : EdgeType):
        if not self.__bg_graph.has_node(source_cube):
            raise Exception(f"Cube #{source_cube} not found in the BG-graph.")

        if not self.__bg_graph.has_node(target_cube):
            raise Exception(f"Cube #{target_cube} not found in the BG-graph.")

        if self.__bg_graph.has_edge(source_cube, target_cube):
            raise Exception(f"Cubes #{source_cube} and #{target_cube} are already connected by a pipe.")

        source_kind = self.get_cube_kind(source_cube)
        target_kind = self.get_cube_kind(target_cube)
        if not pipe_type in BlockGraphHelper.infer_pipe_type(source_kind, target_kind):
            raise Exception(f"Pipe type {pipe_type} is incompatible with source and target kinds [{source_kind}-{target_kind}].")

        # TODO: validate with respect to inferred pipe type between source and target cubes

        source_position = self.get_cube_position(source_cube)
        target_position = self.get_cube_position(target_cube)
        # TODO: replace 3 with 1 once the pathfinder has been rewritten
        if source_position.get_manhattan_distance(target_position) != 1:
            raise Exception(f"Cubes #{source_cube}@{source_position} and #{target_cube}@{target_position} are not at adjacent positions.")

        self.__bg_graph.add_edge(source_cube, target_cube)
        self.__bg_graph.get_edge_data(source_cube, target_cube)[AugmentedNxGraph.KEY_BG_PIPE_TYPE] = pipe_type

    def is_path_valid(self, path: Path, edge_type: EdgeType) -> bool:
        is_hadamard_path = False

        source_cube = path.get_source_cube()
        target_cube = path.get_target_cube()

        source_kind: CubeKind = self.get_cube_kind(source_cube)
        source_position: Coordinates = self.get_cube_position(source_cube)

        cubes = path.get_cubes()
        pipes = path.get_pipes()
        proposed_target_kind, proposed_target_position = cubes[-1]

        extra_positions = set()

        console.info(f"Checking path validity:")
        console.info(f"> Source cube #{source_cube} [{source_kind}@{source_position}]")
        console.info(f"> Proposed target cube : {proposed_target_kind}@{proposed_target_position}")
        console.info(f"> Path cubes : {cubes}")
        console.info(f"> Path pipes : {pipes}")

        previous_kind = source_kind
        previous_position = source_position
        previous_reach: Coordinates = source_kind.get_reach()

        n = len(cubes)
        for index in range(1, n):
            current_kind, current_position = cubes[index]
            current_reach = current_kind.get_reach()

            if index != n-1:
                # Check that the cube type is either X or Z (Y and boundaries must be leaves)
                if current_kind in [ CubeKind.OOO, CubeKind.YYY ]:
                    console.debug(f"> CubeKind.OOO and CubeKind.YYY can only appear at the ends of a path : {current_kind}.")
                    return False

                # Check that the current_position is not already occupied
                if current_position in self.occupied:
                    console.debug(f"> Current position is already occupied : {current_kind}@{current_position}")
                    return False

            # Check that the step taken lies in both reaches of successive cubes
            step_taken = current_position - previous_position
            if not Spacetime.contains(previous_reach, step_taken) or not Spacetime.contains(current_reach, step_taken):
                console.debug(f"> Previous reach contains step : {Spacetime.contains(previous_reach, step_taken)}")
                console.debug(f"> Current reach contains step : {Spacetime.contains(current_reach, step_taken)}")
                return False

            # Check that the current_position is not already occupied by an extra cube
            if current_position in extra_positions:
                console.debug(f"> Current position is already in path : {current_kind}@{current_position}")
                return False
            extra_positions.add(current_position)

            # Check that the current pipe has a type consistent with what is allowed
            current_pipe_type = pipes[index-1]
            inferred = BlockGraphHelper.infer_pipe_type(previous_kind, current_kind)
            if not current_pipe_type in inferred:
                console.debug(f"> Current pipe type is not allowed between {previous_kind} and {current_kind} [{current_pipe_type} not in {inferred}].")
                return False

            if current_pipe_type == EdgeType.HADAMARD:
                is_hadamard_path = not is_hadamard_path

            previous_position = current_position
            previous_kind = current_kind
            previous_reach = current_reach

        console.debug(f"> Is cube {target_cube} placed ? {self.is_cube_placed(target_cube)}")
        console.debug(f"> cubes {self.get_cubes()}")

        if self.is_cube_placed(target_cube):
            if proposed_target_kind != self.get_cube_kind(target_cube):
                console.debug(f"> Proposed target kind does not match its existing realisation.")
                return False
            if proposed_target_position != self.get_cube_position(target_cube):
                console.debug(f"> Proposed target position does not match its existing realisation.")
                return False
        elif proposed_target_position in self.occupied:
            occupant = self.__identify_cube_at_position(proposed_target_position)
            occupant_kind = self.get_cube_kind(occupant)
            occupant_position = self.get_cube_position(occupant)
            console.debug(f"> Proposed target position is already occupied [{occupant_kind}@{occupant_position}].")
            return False

        hadamard_consistent = is_hadamard_path == (edge_type == EdgeType.HADAMARD)

        if not hadamard_consistent:
            console.debug(f"> Proposed path is Hadamard-inconsistent with its purported edge [{edge_type}].")

        return hadamard_consistent

    def __identify_cube_at_position(self, position: Coordinates) -> int:
        for cube in self.get_cubes():
            if self.get_cube_position(cube) == position:
                return cube

        return -1