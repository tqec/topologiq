from collections import deque, defaultdict
from enum import Enum
from typing import Iterable
from ast import literal_eval as make_tuple

import numpy as np
import pyzx as zx
import networkx as nx

from topologiq.dzw.common.coordinates import Coordinates
from topologiq.utils.classes import StandardCoord

from topologiq.dzw.helpers.spacetime import Spacetime
from topologiq.dzw.helpers.blockgraph import BlockGraphHelper

from topologiq.dzw.common.components_zx import NodeId, NodeType, EdgeId, EdgeType
from topologiq.dzw.common.components_bg import CubeId, CubeKind, PipeId
from topologiq.dzw.common.path import PathSpecification

from logging import getLogger

from topologiq.utils.classes import SimpleDictGraph
from topologiq.core.pathfinder.utils import get_manhattan

console = getLogger(__name__)

QubitId = int
LayerId = int

class LayerTransitionType(Enum):
    EVERY = 0
    LOWER = 1
    INTRA = 2
    UPPER = 3
    OUTER = 4

# TODO: figure out what the other VertexType and EdgeType represent
# TODO: how do we deal with the last four VertexType (i.e. H_BOX, W_INPUT, W_OUTPUT, Z_BOX) ?
# TODO: do we need the last EdgeType (i.e. W_IO) ?
# TODO: how do we deal with the phase of a spider ?
# TODO: benchmarking and timing various parts
# TODO: construction of animation
class AugmentedNxGraph(nx.Graph):
    KEY_ZX_NODE_TYPE = 'zx_node_type'
    KEY_ZX_NODE_QUBIT = 'zx_node_qubit'
    KEY_ZX_NODE_LAYER = 'zx_node_layer'
    KEY_ZX_EDGE_TYPE = 'zx_edge_type'
    KEY_ZX_EDGES_REALISED = 'zx_edges_realised'
    KEY_ZX_BG_CUBE = 'zx_bg_cube'
    KEY_ZX_BG_PATH = 'zx_bg_path'

    KEY_BG_ZX_NODE   = 'bg_zx_node'
    KEY_BG_CUBE_KIND = 'bg_cube_kind'
    KEY_BG_CUBE_POSITION = 'bg_cube_position'
    KEY_BG_PIPE_TYPE = 'bg_pipe_type'
    KEY_BG_CUBE_BEAMS = 'bg_cube_beams'

    def __init__(self, nodes: Iterable[tuple[NodeId, NodeType]] | None = None, edges: Iterable[tuple[tuple[NodeId, NodeId], EdgeType]] | None = None):
        # Separate ZX-graph and BG-graph
        super(AugmentedNxGraph, self).__init__()
        self.__bg_graph: nx.Graph = nx.Graph()

        # Keeps track of which nodes appear on which qubit-line or layer of the ZX-graph
        self.__zx_qubits: dict[QubitId, list[NodeId]] = defaultdict(list)
        self.__zx_layers: dict[LayerId, list[NodeId]] = defaultdict(list)

        # Tracks the order in which nodes and edges from the ZX graph were realised into the Blockgraph
        self.__zx_node_realisation_order: list[NodeId] = []
        self.__zx_edge_realisation_order: list[EdgeId] = []

        # Keeps track of the coordinates in 3D that are occupied by some cube
        self.occupied: set[StandardCoord] = set()

        if nodes is not None:
            for node, node_type in nodes:
                self.add_node(node)
                self.nodes[node][AugmentedNxGraph.KEY_ZX_NODE_TYPE] = node_type
                self.nodes[node][AugmentedNxGraph.KEY_ZX_EDGES_REALISED] = 0
                self.nodes[node][AugmentedNxGraph.KEY_ZX_BG_CUBE] = None

        if edges is not None:
            for edge, edge_type in edges:
                source = min(edge)
                target = max(edge)
                self.add_edge(source, target)
                self.get_edge_data(source, target)[AugmentedNxGraph.KEY_ZX_EDGE_TYPE] = edge_type
                self.get_edge_data(source, target)[AugmentedNxGraph.KEY_ZX_BG_PATH] = None

        self.__next_cube_id = self.number_of_nodes()

        # TODO: split any spider with more than 4 edges (cfr. graph_manager.py; prep_3d_g)
        # TODO: does the choice of how to split such spiders affect the minimal achievable volume ?
        self.enforce_nmtfl()

    def enforce_nmtfl(self):
        """Ensures that all spiders of a AugmentedNxGraph have at most four legs (i.e. No-More-Than-Four-Legs).
            N.B. the provided graph is altered by this method to perform the enforcing.

        Args:
            self: An AugmentedNxGraph

        Returns:
            graph: An equivalent (under spider fusion) AugmentedNxGraph with every node having at most four edges.

        """

        # Process all zx-nodes with more than four zx-edges
        for node in filter(lambda nd: self.degree[nd] > 4, self.nodes):
            neighbors = self.neighbors(node)
            degree = self.degree[node]
            node_type = self.get_node_type(node)
            node_qubit = self.get_node_qubit(node)
            node_layer = self.get_node_layer(node)

            remaining_neighbors = degree - 3
            next(neighbors); next(neighbors); next(neighbors)
            previous = node
            while remaining_neighbors > 0:
                extra_node_id = self.number_of_nodes()

                self.add_node(extra_node_id)
                extra_node = self.nodes[extra_node_id]
                extra_node[AugmentedNxGraph.KEY_ZX_NODE_TYPE] = node_type
                extra_node[AugmentedNxGraph.KEY_ZX_NODE_QUBIT] = node_qubit
                extra_node[AugmentedNxGraph.KEY_ZX_NODE_LAYER] = node_layer
                extra_node[AugmentedNxGraph.KEY_ZX_BG_CUBE] = None

                # Make zx-edge between previous node of chain and new extra_node
                self.add_edge(previous, extra_node_id)

                neighbors_to_graft = remaining_neighbors if remaining_neighbors <= 3 else 2
                for _ in range(neighbors_to_graft):
                    neighbor = next(neighbors)
                    edge_type = self.get_edge_type(extra_node_id, neighbor)

                    self.remove_edge(node, neighbor)
                    self.add_edge(extra_node_id, neighbor)
                    extra_edge = self.edges[extra_node_id, neighbor]
                    extra_edge[AugmentedNxGraph.KEY_ZX_EDGE_TYPE] = edge_type
                    extra_edge[AugmentedNxGraph.KEY_ZX_BG_PATH] = []

                remaining_neighbors -= neighbors_to_graft
                previous = extra_node_id

    @staticmethod
    def from_pyzx_graph(zx_graph: zx.graph.base.BaseGraph):
        converted_node_ids: dict[NodeId, NodeId] = dict()
        nodes: list[tuple[NodeId, NodeType]] = []
        for node in zx_graph.vertices():
            node_id = len(converted_node_ids)
            converted_node_ids[node] = node_id
            nodes.append( (node_id, NodeType.convert(zx_graph.type(node))) )

        edges: list[tuple[EdgeId, EdgeType]] = []
        for edge in zx_graph.edges():
            source = converted_node_ids[min(edge)]
            target = converted_node_ids[max(edge)]
            edges.append( ( (source,target) , EdgeType.convert(zx_graph.edge_type(edge))) )

        ang = AugmentedNxGraph(nodes, edges)

        # Add qubit and layer information
        for node, node_id in converted_node_ids.items():
            node_qubit = int(zx_graph.qubit(node))
            ang.nodes[node_id][AugmentedNxGraph.KEY_ZX_NODE_QUBIT] = node_qubit
            ang.__zx_qubits[node_qubit].append(node_id)

            node_layer = int(zx_graph.row(node))
            ang.nodes[node_id][AugmentedNxGraph.KEY_ZX_NODE_LAYER] = node_layer
            ang.__zx_layers[node_layer].append(node_id)

        return ang

    @staticmethod
    def from_simple_graph(simple_graph: SimpleDictGraph):
        console.warning(f"SimpleDictGraph does not provide qubit and layer information.")

        nodes = map( lambda nt : (nt[0], NodeType.convert_simple(nt[1])) , simple_graph["nodes"] )
        edges = map( lambda et : (et[0], EdgeType.convert_simple(et[1])) , simple_graph["edges"] )

        ang = AugmentedNxGraph(nodes, edges)

        return ang

    def get_node_realisation_order(self) -> list[NodeId]:
        return self.__zx_node_realisation_order

    def get_edge_realisation_order(self) -> list[tuple[NodeId, NodeId]]:
        return self.__zx_edge_realisation_order

    def get_qubits(self):
        return self.__zx_qubits.keys()

    def get_node_qubit(self, node) -> QubitId:
        return self.nodes[node][AugmentedNxGraph.KEY_ZX_NODE_QUBIT]

    def get_nodes(self):
        return self.nodes()

    def get_depth(self):
        return len(self.__zx_layers)

    def get_layers(self):
        return self.__zx_layers.keys()

    def get_layer(self, layer) -> list[NodeId]:
        return self.__zx_layers[layer]

    def get_layer_density(self, layer: int) -> tuple[int,int]:
        layer_nodes = self.get_layer(layer)
        number_of_nodes = len(layer_nodes)
        number_of_edges = 0
        for node in layer_nodes:
            number_of_edges += sum(1 for _ in self.get_node_neighbours(node, transition = LayerTransitionType.INTRA))
        number_of_edges = number_of_edges // 2
        return number_of_nodes, number_of_edges

    def get_edges(self):
        return self.edges()

    def get_layered_edges(self, layer: int, transition: LayerTransitionType = LayerTransitionType.EVERY):
        if transition == LayerTransitionType.LOWER:
            filtering = lambda edge : self.get_node_layer(edge[0]) <  layer == self.get_node_layer(edge[1])
        elif transition == LayerTransitionType.INTRA:
            filtering = lambda edge : self.get_node_layer(edge[0]) == layer == self.get_node_layer(edge[1])
        elif transition == LayerTransitionType.UPPER:
            filtering = lambda edge : self.get_node_layer(edge[0]) == layer <  self.get_node_layer(edge[1])
        elif transition == LayerTransitionType.OUTER:
            filtering = lambda edge : self.get_node_layer(edge[0]) <  layer <  self.get_node_layer(edge[1])
        else:
            filtering = lambda edge : True

        return filter(filtering, self.edges())

    def get_cubes(self):
        return self.__bg_graph.nodes()

    def number_of_cubes(self) -> int:
        return self.__bg_graph.number_of_nodes()

    def get_pipes(self):
        return self.__bg_graph.edges()

    def number_of_pipes(self) -> int:
        return self.__bg_graph.number_of_edges()

    def get_edges_realised(self, node: NodeId):
        return self.nodes[node].get(AugmentedNxGraph.KEY_ZX_EDGES_REALISED)

    def get_edges_unrealised(self, node: NodeId):
        return self.get_degree(node) - self.get_edges_realised(node)

    def get_node_neighbours(self,
        node: NodeId, transition: LayerTransitionType = LayerTransitionType.EVERY
    ) -> Iterable[NodeId]:
        if transition == LayerTransitionType.EVERY:
            filtering = lambda other : True
        elif transition == LayerTransitionType.LOWER:
            filtering = lambda other : self.get_node_layer(other) < self.get_node_layer(node)
        elif transition == LayerTransitionType.INTRA:
            filtering = lambda other : self.get_node_layer(other) == self.get_node_layer(node)
        elif transition == LayerTransitionType.UPPER:
            filtering = lambda other : self.get_node_layer(node) < self.get_node_layer(other)
        else: #transition == LayerTransitionType.OUTER
            raise Exception(f"Requesting OUTER transition type for node neighbours. Will always be empty.")

        return filter(filtering, self.neighbors(node))

    def get_cube_neighbours(self, cube: CubeId):
        return self.__bg_graph.neighbors(cube)

    def get_degree(self, node: NodeId) -> float:
        return self.degree[node]

    def is_boundary(self, node: NodeId) -> bool:
        return self.get_node_type(node) == NodeType.O

    def is_spider(self, node: NodeId) -> bool:
        return self.get_node_type(node) != NodeType.O

    def is_cube_placed(self, cube: CubeId) -> bool:
        return cube in self.__bg_graph

    def get_cube(self, node: NodeId) -> CubeId:
        return self.nodes[node][AugmentedNxGraph.KEY_ZX_BG_CUBE]

    def get_node(self, cube: CubeId) -> NodeId:
        return self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_ZX_NODE]

    def get_node_type(self, node: NodeId) -> NodeType:
        return self.nodes[node][AugmentedNxGraph.KEY_ZX_NODE_TYPE]

    def get_node_layer(self, node: int) -> int:
        return self.nodes[node][AugmentedNxGraph.KEY_ZX_NODE_LAYER]

    def get_cube_position(self, cube: CubeId) -> StandardCoord:
        return self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_CUBE_POSITION]

    def get_cube_kind(self, cube: CubeId) -> CubeKind:
        return self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_CUBE_KIND]

    def get_pipe_type(self, source_cube: CubeId, target_cube: CubeId) -> EdgeType :
        return self.__bg_graph.get_edge_data(source_cube, target_cube).get(AugmentedNxGraph.KEY_BG_PIPE_TYPE)

    def get_edge_type(self, source: NodeId, target: NodeId) -> EdgeType:
        return self.get_edge_data(source, target).get(AugmentedNxGraph.KEY_ZX_EDGE_TYPE)

    def get_edge_realisation(self, source: NodeId, target: NodeId) -> list[tuple[CubeId, CubeId]]:
        return self.get_edge_data(source, target).get(AugmentedNxGraph.KEY_ZX_BG_PATH)

    def is_node_realised(self, node: NodeId) -> bool:
        return self.nodes[node][AugmentedNxGraph.KEY_ZX_BG_CUBE] is not None

    def realise_node(self, node: NodeId, kind: CubeKind, position: StandardCoord) -> CubeId:
        """Realise the node as a cube of the given kind placed at the given coordinates."""
        if kind not in CubeKind.suitable_kinds(self.get_node_type(node)):
            raise Exception(f"Requested {kind} is not compatible with {self.get_node_type(node)}")

        if not self.has_node(node):
            raise Exception(f"Node #{node} not found in the ZX-graph.")

        cube = self.place_cube(kind, position)
        self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_ZX_NODE] = node
        self.nodes[node][AugmentedNxGraph.KEY_ZX_BG_CUBE] = cube

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
        return self.get_edge_data(source, target)[AugmentedNxGraph.KEY_ZX_BG_PATH] is not None

    def realise_edge(self, source: NodeId, target: NodeId, proposal: PathSpecification):
        if not self.is_node_realised(source):
            raise Exception(f"{source} is not placed; cannot connect with a path.")

        if not self.is_node_realised(target):
            raise Exception(f"{target} is not placed; cannot connect with a path.")

        if not self.has_edge(source, target):
            raise Exception(f"No edge {source}-{target} found in the ZX-graph.")

        if self.is_edge_realised(source, target):
            raise Exception(f"{source}-{target} is already realized by a path.")

        source_cube = self.get_cube(source)
        target_cube = self.get_cube(target)

        edge_type = self.get_edge_type(source, target)

        # # Reject path if it is invalid.
        if not self.is_path_valid(proposal, edge_type):
            raise Exception(f"Proposed path to realise edge {source}-{target} is invalid.")

        if not proposal:
            sequence = "[]"
        else:
            sequence = ""
            for position, kind in proposal.get_extra_cubes():
                sequence += f"{kind}@{position}"
        console.info(f"Realising edge {source}-{target} [type={self.get_edge_type(source,target)}] with extra cubes : {sequence}")

        # Representation of the path that will go into edge_realisations
        pipe_ids = []

        # Add all the extra cubes and pipes of the path to the BlockGraph
        previous_cube: int = source_cube

        proposed_cubes = proposal.get_cubes()
        proposed_pipes = proposal.get_pipes()

        n = len(proposed_cubes)
        for index in range(1, n-1):
            current_kind, current_position = proposed_cubes[index]
            current_pipe_type = proposed_pipes[index-1]

            # Place the current cube and connect it to the previous cube.
            current_cube = self.place_cube(current_kind, current_position)
            self.__bg_graph.nodes[current_cube][AugmentedNxGraph.KEY_BG_ZX_NODE] = None
            self.connect_pipe(previous_cube, current_cube, current_pipe_type)

            # Extend the sequence of extra node ids
            pipe = tuple(sorted((previous_cube, current_cube)))
            pipe_ids.append( pipe )

            # Prepare for the next iteration
            previous_cube = current_cube

        # Make the final connection
        target_cube = self.get_cube(target)
        final_pipe_type = proposed_pipes[-1]
        self.connect_pipe(previous_cube, target_cube, final_pipe_type)

        pipe = tuple(sorted((previous_cube, target_cube)))
        pipe_ids.append( pipe )

        # Associate the path as a realisation of the edge
        self.get_edge_data(source, target)[AugmentedNxGraph.KEY_ZX_BG_PATH] = pipe_ids

        # One more edge has been realised
        self.nodes[source][AugmentedNxGraph.KEY_ZX_EDGES_REALISED] += 1
        self.nodes[target][AugmentedNxGraph.KEY_ZX_EDGES_REALISED] += 1
        self.__zx_edge_realisation_order.append( (source, target) )

    def place_cube(self, kind: CubeKind, position: StandardCoord) -> CubeId:
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
        if get_manhattan(source_position, target_position) != 3:
            raise Exception(f"Cubes #{source_cube}@{source_position} and #{target_cube}@{target_position} are not at adjacent positions.")

        self.__bg_graph.add_edge(source_cube, target_cube)
        self.__bg_graph.get_edge_data(source_cube, target_cube)[AugmentedNxGraph.KEY_BG_PIPE_TYPE] = pipe_type

    def is_path_valid(self, path: PathSpecification, edge_type: EdgeType) -> bool:
        is_hadamard_path = False

        source_cube = path.get_source_cube()
        target_cube = path.get_target_cube()

        source_kind: CubeKind = self.get_cube_kind(source_cube)
        source_position: StandardCoord = self.get_cube_position(source_cube)

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
        previous_reach: StandardCoord = source_kind.get_reach()

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
            difference = np.subtract(current_position, previous_position)
            step_taken = (difference[0], difference[1], difference[2])
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

    def log_summary(self):
        for node_type in [NodeType.O, NodeType.X, NodeType.Y, NodeType.Z]:
            content = ""
            for node in self.nodes:
                if node_type == self.get_node_type(node):
                    content += f"{node} "
            console.info(f"Nodes {node_type.name} : {content}")

        content = ""
        for edge in self.edges:
            content += f"{edge} "
        console.info(f"Edges   : {content}")

        for layer in self.get_layers():
            console.info(f"Layer {layer}  : {self.get_layer(layer)}")

        for qubit in self.get_qubits():
            nodes = filter(lambda nd: qubit == self.get_node_qubit(nd), self.nodes)
            console.info(f"Qubit {qubit}  : {list(nodes)}")

    def print_summary(self):
        for node_type in [NodeType.O, NodeType.X, NodeType.Y, NodeType.Z]:
            content = ""
            for node in self.nodes:
                if node_type == self.get_node_type(node):
                    content += f"{node} "
            print(f"Nodes {node_type.name}: {content}")

        content = ""
        for edge in self.edges:
            content += f"{edge} "
        print(f"Edges  : {content}")

        for layer in self.get_layers():
            print(f"Layer {layer}  : {self.get_layer(layer)}")

        for qubit in self.get_qubits():
            nodes = filter(lambda nd: qubit == self.get_node_qubit(nd), self.nodes)
            print(f"Qubit {qubit}  : {list(nodes)}")

        for cube in self.get_cubes():
            print(f"Cube {cube}  : {self.get_cube_kind(cube)}@{self.get_cube_position(cube)}")

        for pipe in self.get_pipes():
            print(f"Pipe {pipe}  : {self.get_pipe_type(*pipe)}")

    @staticmethod
    def __make_tuple(tpl: str):
        source, target = tpl.split('-')
        return int(source), int(target)

    @staticmethod
    def from_file(filepath: str):
        cubes: dict[CubeId, tuple[CubeKind, StandardCoord]] = dict()
        pipes: dict[PipeId, EdgeType] = dict()
        with open(filepath, 'r') as file:
            # Read the blockgraph header
            header = file.readline()
            if header != "BLOCKGRAPH 0.1.0;\n":
                raise Exception(f"Invalid file format. Header for BLOCKGRAPH 0.1.0 not found [got={header}].")

            # Read the empty line between the blockgraph header and the cubes header
            file.readline()

            # Read the cubes header
            header = file.readline()
            if header != "CUBES: index;x;y;z;kind;label;\n":
                raise Exception(f"Invalid file format. Header for CUBES not found [got={header}].")

            # Read all the lines describing cubes
            current_cube = file.readline()
            while current_cube and current_cube != "\n":
                print(f"Cube : {current_cube}")
                if current_cube != "":
                    cube_id, x, y, z, kind, _, _ = current_cube.split(';')
                    cubes[ int(cube_id) ] = (CubeKind[kind.upper()], (int(x), int(y), int(z)))
                current_cube = file.readline()

            # Read the pipes header
            header = file.readline()
            if header != "PIPES: src;tgt;kind;\n":
                raise Exception(f"Invalid file format. Header for PIPES not found [got={header}].")

            # Read all the lines describing pipes
            current_pipe = file.readline()
            while current_pipe and current_pipe != "\n":
                if current_pipe != "":
                    print(f"Pipe : {current_pipe}")
                    src, tgt, kind, _ = current_pipe.split(';')
                    pipes[ (int(src), int(tgt)) ] = EdgeType.IDENTITY if 'h' not in kind else EdgeType.HADAMARD
                current_pipe = file.readline()

            # Instantiate the new ANG
            vzx = AugmentedNxGraph()

            # Populate with cubes, storing the original ID to correctly connect the pipes
            cube_new_ids: dict[CubeId, CubeId] = dict()
            for cube, cube_attributes in cubes.items():
                cube_kind, cube_position = cube_attributes
                cube_new_ids[cube] = vzx.place_cube(cube_kind, cube_position)

            # Populate with pipes, using the correspondence between original IDs and new IDs
            for pipe_id, pipe_type in pipes.items():
                source, target = pipe_id
                vzx.connect_pipe(cube_new_ids[source], cube_new_ids[target], pipe_type)

            return vzx

    @staticmethod
    def __scaled_position(position: tuple[int,int,int]) -> str:
        return '(' + ','.join(str(int(p / 3.0)) for p in position) + ')'

    def __infer_pipe_kind(self, src: CubeId, tgt: CubeId) -> str:
        # cellcolors are for faces (+X, -X, +Y, -Y, +Z, -Z)
        colors = ""
        src_kind = self.get_cube_kind(src)
        tgt_kind = self.get_cube_kind(tgt)
        src_position = Coordinates.from_tuple(self.get_cube_position(src))
        tgt_position = Coordinates.from_tuple(self.get_cube_position(tgt))
        step_taken = tgt_position - src_position
        distances = step_taken.as_tuple()
        for c in range(3):
            if distances[c] == 0 and (
                    src_kind not in [CubeKind.OOO, CubeKind.YYY] or tgt_kind not in [CubeKind.OOO, CubeKind.YYY]):
                color = src_kind.name[c] if src_kind not in [CubeKind.OOO, CubeKind.YYY] else tgt_kind.name[c]
            else:
                color = 'o'
            colors += color

        if self.get_pipe_type(src, tgt) == EdgeType.HADAMARD:
            colors += 'h'

        return colors.lower()

    def __format_label(self, cube: CubeId):
        label = ""
        if self.get_cube_kind(cube) == CubeKind.OOO:
            node = self.get_node(cube)
            qubit = self.get_node_qubit(node)
            layer = self.get_node_layer(node)
            if layer == 0:
                label = f"in_{qubit}"
            elif layer == self.get_depth() - 1:
                label = f"out_{qubit}"
        return label

    def into_file(self, filepath: str, include_zx_graph: bool = False):
        with (open(filepath, 'w') as file):
            file.write(f"BLOCKGRAPH 0.1.0;\n")

            # Store cube information
            file.write("\nCUBES: index;x;y;z;kind;label;\n")
            file.writelines(
                [
                    f"{cube};{';'.join(map(str, iter(self.get_cube_position(cube))))};{self.get_cube_kind(cube).name.lower()};{self.__format_label(cube)};\n"
                    for cube in self.get_cubes()
                ]
            )

            # Store pipe information
            file.write("\nPIPES: src;tgt;kind;\n")
            file.writelines(
                [
                    f"{src};{tgt};{self.__infer_pipe_kind(src, tgt)};\n"
                    for src, tgt in self.get_pipes()
                ]
            )

            if include_zx_graph:
                # Store node information
                file.write("\nNODES: index;type;qubit;layer;\n")
                file.writelines(
                    [
                        f"{node};{self.get_node_type(node)};{self.get_node_qubit(node)};{self.get_node_layer(node)};\n"
                        for node in self.nodes
                    ]
                )

                # Store edge information
                file.write("\nEDGES: src;tgt;type;\n")
                file.writelines(
                    [
                        f"{src};{tgt};{self.get_edge_type(src, tgt)};\n"
                        for src, tgt in self.edges
                    ]
                )

                # Store node-cube correspondence
                file.write("\nNODES-CUBES: node_id;cube_id;\n")
                file.writelines(
                    [
                        f"{node};{self.get_cube(node)}\n"
                        for node in self.nodes
                    ]
                )

                # Store edge-pipes correspondence
                file.write("\nEDGES-PIPES: edge;pipes;\n")
                file.writelines(
                    [
                        f"{edge};{self.get_edge_realisation(*edge)}\n"
                        for edge in self.edges
                    ]
                )

    def __identify_cube_at_position(self, position: StandardCoord) -> int:
        for cube in self.get_cubes():
            if self.get_cube_position(cube) == position:
                return cube

        return -1