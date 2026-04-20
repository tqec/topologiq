from collections import deque, defaultdict
from enum import Enum
from typing import Iterable

import numpy as np
import pyzx as zx
import networkx as nx

from topologiq.dzw.common.components import ZxNode, ZxEdge, BgCube, BgPipe
from topologiq.dzw.common.coordinates import Coordinates
from topologiq.utils.classes import StandardCoord

from topologiq.dzw.helpers.spacetime import Spacetime
from topologiq.dzw.helpers.blockgraph import BlockGraphHelper

from topologiq.dzw.common.attributes_zx import NodeId, NodeType, EdgeId, EdgeType, QubitId, LayerId
from topologiq.dzw.common.attributes_bg import CubeId, CubeKind, PipeId
from topologiq.dzw.common.path import PathSpecification

from topologiq.utils.classes import SimpleDictGraph
from topologiq.core.pathfinder.utils import get_manhattan

from logging import getLogger
console = getLogger(__name__)

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
    KEY_ZX_NODE = 'zx_node'
    KEY_ZX_EDGE = 'zx_edge'

    KEY_BG_CUBE = 'bg_cube'
    KEY_BG_PIPE = 'bg_pipe'

    def __init__(self,
            nodes: Iterable[tuple[NodeId, NodeType]] | None = None,
            edges: Iterable[tuple[EdgeId, EdgeType]] | None = None
    ):
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
                self.nodes[node][AugmentedNxGraph.KEY_ZX_NODE] = ZxNode(id = node, type = node_type)

        if edges is not None:
            for edge, edge_type in edges:
                self.add_edge(*edge)
                self.edges[edge][AugmentedNxGraph.KEY_ZX_EDGE] = ZxEdge(source = edge[0], target = edge[1], type = edge_type)

        self.__next_id = max(self.nodes.keys()) + 1

        # TODO: split any spider with more than 4 edges (cfr. graph_manager.py; prep_3d_g)
        # TODO: does the choice of how to split such spiders affect the minimal achievable volume ?
        self.__enforce_nmtfl()

    def __enforce_nmtfl(self):
        """Ensures that all spiders of a AugmentedNxGraph have at most four legs (i.e. No-More-Than-Four-Legs).
            N.B. the provided graph is altered by this method to perform the enforcing.

        Args:
            self: An AugmentedNxGraph

        Returns:
            The original graph, transformed into an equivalent (under spider fusion) one with every node having at most four edges.

        """

        # Process all zx-nodes with more than four zx-edges
        for node in filter(lambda nd: self.degree[nd] > 4, self.nodes):
            neighbors = self.neighbors(node)
            degree = self.degree[node]
            zx_node = self.get_zx_node(node)

            remaining_neighbors = degree - 3
            next(neighbors); next(neighbors); next(neighbors)
            previous = node
            while remaining_neighbors > 0:
                extra_node_id = self.__next_id
                self.__next_id += 1

                self.add_node(extra_node_id)
                extra_zx_node = ZxNode(id = extra_node_id, type = zx_node.type, qubit = zx_node.qubit, layer = zx_node.layer)
                self.nodes[extra_node_id][AugmentedNxGraph.KEY_ZX_NODE] = extra_zx_node

                # Make zx-edge between previous node of chain and new extra_node
                self.add_edge(previous, extra_node_id)

                neighbors_to_graft = remaining_neighbors if remaining_neighbors <= 3 else 2
                for _ in range(neighbors_to_graft):
                    neighbor = next(neighbors)
                    edge_type = self.get_zx_edge(extra_node_id, neighbor).type

                    self.remove_edge(node, neighbor)
                    self.add_edge(extra_node_id, neighbor)
                    extra_zx_edge = ZxEdge(source = extra_node_id, target = neighbor, type = edge_type)
                    self.get_edge_data(extra_node_id, extra_zx_edge)[AugmentedNxGraph.KEY_ZX_EDGE] = extra_zx_edge

                remaining_neighbors -= neighbors_to_graft
                previous = extra_node_id

    @staticmethod
    def from_pyzx_graph(zx_graph: zx.graph.base.BaseGraph):
        nodes: list[tuple[NodeId, NodeType]] = []
        for node in zx_graph.vertices():
            nodes.append( (node, NodeType.convert(zx_graph.type(node))) )

        edges: list[tuple[EdgeId, EdgeType]] = []
        for edge in zx_graph.edges():
            edges.append( ( edge , EdgeType.convert(zx_graph.edge_type(edge))) )

        ang = AugmentedNxGraph(nodes, edges)

        # Add qubit and layer information
        for node, node_type in nodes:
            node_qubit = int(zx_graph.qubit(node))
            ang.nodes[node][AugmentedNxGraph.KEY_ZX_NODE].qubit = node_qubit
            ang.__zx_qubits[node_qubit].append(node)

            node_layer = int(zx_graph.row(node))
            ang.nodes[node][AugmentedNxGraph.KEY_ZX_NODE].layer = node_layer
            ang.__zx_layers[node_layer].append(node)

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

    def get_zx_node(self, node: NodeId) -> ZxNode:
        return self.nodes[node][AugmentedNxGraph.KEY_ZX_NODE]

    def get_zx_edge(self, source: NodeId, target: NodeId) -> ZxEdge:
        return self.get_edge_data(source, target)[AugmentedNxGraph.KEY_ZX_EDGE]

    def get_bg_cube(self, cube: CubeId) -> BgCube:
        return self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_CUBE]

    def get_bg_pipe(self, source: CubeId, target: CubeId) -> BgPipe:
        return self.__bg_graph.get_edge_data(source, target)[AugmentedNxGraph.KEY_BG_PIPE]

    def get_nodes(self, node_type: NodeType | None = None, qubit: QubitId | None = None, layer: LayerId | None = None):
        if node_type or qubit or layer:
            return filter(
                lambda node : (node_type is None or self.get_zx_node(node).type == node_type) and
                              (qubit is None or self.get_zx_node(node).qubit == qubit) and
                              (layer is None or self.get_zx_node(node).layer == layer),
                self.nodes
            )
        else:
            return self.nodes

    def get_depth(self):
        return len(self.__zx_layers)

    def get_layers(self):
        return self.__zx_layers.keys()

    def get_layer(self, layer) -> list[NodeId]:
        return self.__zx_layers[layer]

    def get_layer_density(self, layer: int) -> tuple[int,int]:
        layer_nodes = list(self.get_nodes(layer = layer))
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
            filtering = lambda edge : self.get_zx_node(edge[0]).layer <  layer == self.get_zx_node(edge[1]).layer
        elif transition == LayerTransitionType.INTRA:
            filtering = lambda edge : self.get_zx_node(edge[0]).layer == layer == self.get_zx_node(edge[1]).layer
        elif transition == LayerTransitionType.UPPER:
            filtering = lambda edge : self.get_zx_node(edge[0]).layer == layer <  self.get_zx_node(edge[1]).layer
        elif transition == LayerTransitionType.OUTER:
            filtering = lambda edge : self.get_zx_node(edge[0]).layer <  layer <  self.get_zx_node(edge[1]).layer
        else:
            filtering = lambda edge : True

        return filter(filtering, self.edges())

    def get_cubes(self, cube_kind: CubeKind | None = None):
        return filter(
            lambda cb: (cube_kind is None or self.get_bg_cube(cb).kind == cube_kind), self.__bg_graph.nodes()
        )

    def number_of_cubes(self) -> int:
        return self.__bg_graph.number_of_nodes()

    def get_pipes(self):
        return self.__bg_graph.edges()

    def number_of_pipes(self) -> int:
        return self.__bg_graph.number_of_edges()

    def get_node_neighbours(self,
        node: NodeId, transition: LayerTransitionType = LayerTransitionType.EVERY
    ) -> Iterable[NodeId]:
        if transition == LayerTransitionType.EVERY:
            filtering = lambda other : True
        elif transition == LayerTransitionType.LOWER:
            filtering = lambda other : self.get_zx_node(other).layer < self.get_zx_node(node).layer
        elif transition == LayerTransitionType.INTRA:
            filtering = lambda other : self.get_zx_node(other).layer == self.get_zx_node(node).layer
        elif transition == LayerTransitionType.UPPER:
            filtering = lambda other : self.get_zx_node(node).layer < self.get_zx_node(other).layer
        else: #transition == LayerTransitionType.OUTER
            raise Exception(f"Requesting OUTER transition type for node neighbours. Will always be empty.")

        return filter(filtering, self.neighbors(node))

    def get_cube_neighbours(self, cube: CubeId):
        return self.__bg_graph.neighbors(cube)

    def get_degree(self, node: NodeId) -> float:
        return self.degree[node]

    def is_boundary(self, node: NodeId) -> bool:
        return self.get_zx_node(node).type == NodeType.O

    def is_spider(self, node: NodeId) -> bool:
        return self.get_zx_node(node).type != NodeType.O

    def is_cube_placed(self, cube: CubeId) -> bool:
        return cube in self.__bg_graph

    # def get_pipe_type(self, source_cube: CubeId, target_cube: CubeId) -> EdgeType :
    #     return self.__bg_graph.get_edge_data(source_cube, target_cube)[AugmentedNxGraph.KEY_BG_PIPE_TYPE]

    def is_node_realised(self, node: NodeId) -> bool:
        return self.get_zx_node(node).realising_cube != -1

    def realise_node(self, node: NodeId, kind: CubeKind, position: StandardCoord) -> CubeId:
        """Realise the node as a cube of the given kind placed at the given coordinates."""
        if kind not in CubeKind.suitable_kinds(self.get_zx_node(node).type):
            raise Exception(f"Requested {kind} is not compatible with {self.get_zx_node(node).type}")

        if not self.has_node(node):
            raise Exception(f"Node #{node} not found in the ZX-graph.")

        cube = self.place_cube(kind, position)
        self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_CUBE].realised_node = node
        self.nodes[node][AugmentedNxGraph.KEY_ZX_NODE].realising_cube = cube

        console.info(f"Realising node #{node} [{self.get_zx_node(node).type}] as cube #{cube} [{kind}@{position}]")
        self.__zx_node_realisation_order.append(node)

        return cube

    def find_realising_cubes(self, node: NodeId) -> set[CubeId]:
        if not self.is_node_realised(node):
            raise Exception(f"Node #{node} is not realised by any cube.")

        node_type = self.get_zx_node(node).type
        queue: deque[CubeId] = deque([ self.get_zx_node(node).realising_cube ])
        realising: set[CubeId] = set()

        # TODO: explore within the BlockGraph
        while queue:
            current = queue.popleft()

            for successor in self.get_cube_neighbours(current):
                successor_type = self.get_zx_node(successor).type
                pipe_type = self.get_bg_pipe(current, successor).type
                if successor_type == node_type and pipe_type == EdgeType.IDENTITY and successor not in realising:
                    queue.append(successor)
                    realising.add(successor)

        return realising

    def is_edge_realised(self, source: NodeId, target: NodeId) -> bool:
        return len(self.get_edge_data(source, target)[AugmentedNxGraph.KEY_ZX_EDGE].realisation) > 0

    def realise_edge(self, source: NodeId, target: NodeId, proposal: PathSpecification):
        if not self.is_node_realised(source):
            raise Exception(f"{source} is not placed; cannot connect with a path.")

        if not self.is_node_realised(target):
            raise Exception(f"{target} is not placed; cannot connect with a path.")

        if not self.has_edge(source, target):
            raise Exception(f"No edge {source}-{target} found in the ZX-graph.")

        if self.is_edge_realised(source, target):
            raise Exception(f"{source}-{target} is already realized by a path.")

        source_cube = self.get_zx_node(source).realising_cube
        target_cube = self.get_zx_node(target).realising_cube

        edge_type = self.get_zx_edge(source, target).type

        # # Reject path if it is invalid.
        if not self.is_path_valid(proposal, edge_type):
            raise Exception(f"Proposed path to realise edge {source}-{target} is invalid.")

        if not proposal:
            sequence = "[]"
        else:
            sequence = ""
            for position, kind in proposal.get_extra_cubes():
                sequence += f"{kind}@{position}"
        console.info(f"Realising edge {source}-{target} [type={self.get_zx_edge(source, target).type}] with extra cubes : {sequence}")

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
            # self.__bg_graph.nodes[current_cube][AugmentedNxGraph.KEY_BG_ZX_NODE] = None
            self.connect_pipe(previous_cube, current_cube, current_pipe_type)

            # Extend the sequence of extra node ids
            pipe = (previous_cube, current_cube)
            pipe_ids.append( pipe )

            # Prepare for the next iteration
            previous_cube = current_cube

        # Make the final connection
        target_cube = self.get_zx_node(target).realising_cube
        final_pipe_type = proposed_pipes[-1]
        self.connect_pipe(previous_cube, target_cube, final_pipe_type)

        pipe = (previous_cube, target_cube)
        pipe_ids.append( pipe )

        # Associate the path as a realisation of the edge
        self.get_edge_data(source, target)[AugmentedNxGraph.KEY_ZX_EDGE].realisation = pipe_ids

        # One more edge has been realised
        self.__zx_edge_realisation_order.append( (source, target) )

    def place_cube(self, kind: CubeKind, position: StandardCoord) -> CubeId:
        if position in self.occupied:
            raise Exception(f"Proposed position for {kind}@{position} is already occupied by another cube.")

        cube = self.__next_id
        self.__next_id += 1

        self.__bg_graph.add_node(cube)

        self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_CUBE] = BgCube(id = cube, kind = kind, position = position)

        self.occupied.add(position)

        return cube

    def connect_pipe(self, source_cube: CubeId, target_cube: CubeId, pipe_type : EdgeType):
        if not self.__bg_graph.has_node(source_cube):
            raise Exception(f"Cube #{source_cube} not found in the BG-graph.")

        if not self.__bg_graph.has_node(target_cube):
            raise Exception(f"Cube #{target_cube} not found in the BG-graph.")

        if self.__bg_graph.has_edge(source_cube, target_cube):
            raise Exception(f"Cubes #{source_cube} and #{target_cube} are already connected by a pipe.")

        source_kind = self.get_bg_cube(source_cube).kind
        target_kind = self.get_bg_cube(target_cube).kind
        if not pipe_type in BlockGraphHelper.infer_pipe_type(source_kind, target_kind):
            raise Exception(f"Pipe type {pipe_type} is incompatible with source and target kinds [{source_kind}-{target_kind}].")

        # TODO: validate with respect to inferred pipe type between source and target cubes

        source_position = self.get_bg_cube(source_cube).position
        target_position = self.get_bg_cube(target_cube).position
        # TODO: replace 3 with 1 once the pathfinder has been rewritten
        if get_manhattan(source_position, target_position) != 3:
            raise Exception(f"Cubes #{source_cube}@{source_position} and #{target_cube}@{target_position} are not at adjacent positions.")

        self.__bg_graph.add_edge(source_cube, target_cube)
        bg_pipe = BgPipe(source = source_cube, target = target_cube, type = pipe_type)
        self.__bg_graph.get_edge_data(source_cube, target_cube)[AugmentedNxGraph.KEY_BG_PIPE] = bg_pipe

    def is_path_valid(self, path: PathSpecification, edge_type: EdgeType) -> bool:
        is_hadamard_path = False

        source_cube = path.get_source_cube()
        target_cube = path.get_target_cube()

        source_kind: CubeKind = self.get_bg_cube(source_cube).kind
        source_position: StandardCoord = self.get_bg_cube(source_cube).position

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
            if proposed_target_kind != self.get_bg_cube(target_cube).kind:
                console.debug(f"> Proposed target kind does not match its existing realisation.")
                return False
            if proposed_target_position != self.get_bg_cube(target_cube).position:
                console.debug(f"> Proposed target position does not match its existing realisation.")
                return False
        elif proposed_target_position in self.occupied:
            occupant = self.__identify_cube_at_position(proposed_target_position)
            occupant_kind = self.get_bg_cube(occupant).kind
            occupant_position = self.get_bg_cube(occupant).position
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
                if node_type == self.get_zx_node(node).type:
                    content += f"{node} "
            console.info(f"Nodes {node_type.name} : {content}")

        content = ""
        for edge in self.edges:
            content += f"{edge} "
        console.info(f"Edges   : {content}")

        for layer in self.get_layers():
            console.info(f"Layer {layer}  : {self.get_layer(layer)}")

        for qubit in self.get_qubits():
            nodes = filter(lambda nd: qubit == self.get_zx_node(nd).qubit, self.nodes)
            console.info(f"Qubit {qubit}  : {list(nodes)}")

    def print_summary(self):
        for node_type in [NodeType.O, NodeType.X, NodeType.Y, NodeType.Z]:
            content = ""
            for node in self.nodes:
                if node_type == self.get_zx_node(node).type:
                    content += f"{node} "
            print(f"Nodes {node_type.name}: {content}")

        content = ""
        for edge in self.edges:
            content += f"{edge} "
        print(f"Edges  : {content}")

        for layer in self.get_layers():
            print(f"Layer {layer}  : {self.get_layer(layer)}")

        for qubit in self.get_qubits():
            nodes = filter(lambda nd: qubit == self.get_zx_node(nd).qubit, self.nodes)
            print(f"Qubit {qubit}  : {list(nodes)}")

        for cube in self.get_cubes():
            print(f"Cube {cube}  : {self.get_bg_cube(cube).kind}@{self.get_bg_cube(cube).position}")

        for pipe in self.get_pipes():
            print(f"Pipe {pipe}  : {self.get_bg_pipe(*pipe).type}")

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
        src_kind = self.get_bg_cube(src).kind
        tgt_kind = self.get_bg_cube(tgt).kind
        src_position = Coordinates.from_tuple(self.get_bg_cube(src).position)
        tgt_position = Coordinates.from_tuple(self.get_bg_cube(tgt).position)
        step_taken = tgt_position - src_position
        distances = step_taken.as_tuple()
        for c in range(3):
            if distances[c] == 0 and (
                    src_kind not in [CubeKind.OOO, CubeKind.YYY] or tgt_kind not in [CubeKind.OOO, CubeKind.YYY]):
                color = src_kind.name[c] if src_kind not in [CubeKind.OOO, CubeKind.YYY] else tgt_kind.name[c]
            else:
                color = 'o'
            colors += color

        if self.get_bg_pipe(src, tgt).type == EdgeType.HADAMARD:
            colors += 'h'

        return colors.lower()

    def __format_label(self, cube: CubeId):
        label = ""
        if self.get_bg_cube(cube).kind == CubeKind.OOO:
            zx_node = self.get_zx_node(self.get_bg_cube(cube).realised_node)
            if zx_node.layer == 0:
                label = f"in_{zx_node.qubit}"
            elif zx_node.layer == self.get_depth() - 1:
                label = f"out_{zx_node.qubit}"
        return label

    def into_file(self, filepath: str, include_zx_graph: bool = False):
        with (open(filepath, 'w') as file):
            file.write(f"BLOCKGRAPH 0.1.0;\n")

            # Store cube information
            file.write("\nCUBES: index;x;y;z;kind;label;\n")
            file.writelines(
                [
                    f"{cube};{';'.join(map(str, iter(self.get_bg_cube(cube).position)))};{self.get_bg_cube(cube).kind.name.lower()};{self.__format_label(cube)};\n"
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
                        f"{node};{self.get_zx_node(node).type};{self.get_zx_node(node).qubit};{self.get_zx_node(node).layer};\n"
                        for node in self.nodes
                    ]
                )

                # Store edge information
                file.write("\nEDGES: src;tgt;type;\n")
                file.writelines(
                    [
                        f"{src};{tgt};{self.get_zx_edge(src, tgt).type};\n"
                        for src, tgt in self.edges
                    ]
                )

                # Store node-cube correspondence
                file.write("\nNODES-CUBES: node_id;cube_id;\n")
                file.writelines(
                    [
                        f"{node};{self.get_zx_node(node).realising_cube}\n"
                        for node in self.nodes
                    ]
                )

                # Store edge-pipes correspondence
                file.write("\nEDGES-PIPES: edge;pipes;\n")
                file.writelines(
                    [
                        f"{edge};{self.get_zx_edge(*edge).realisation}\n"
                        for edge in self.edges
                    ]
                )

    def __identify_cube_at_position(self, position: StandardCoord) -> int:
        for cube in self.get_cubes():
            if self.get_bg_cube(cube).position == position:
                return cube

        return -1