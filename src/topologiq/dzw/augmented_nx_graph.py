from collections import deque, defaultdict
from enum import Enum
from itertools import chain
from typing import Iterable

import numpy as np
import pyzx
import networkx as nx
from ast import literal_eval as make_tuple

from topologiq.utils.classes import StandardCoord
from topologiq.utils.classes import SimpleDictGraph
from topologiq.core.pathfinder.utils import get_manhattan

from topologiq.dzw.helpers.spacetime import Spacetime
from topologiq.dzw.helpers.blockgraph import BlockGraphHelper

from topologiq.dzw.common.attributes_zx import NodeId, NodeType, EdgeId, EdgeType, QubitId, LayerId
from topologiq.dzw.common.attributes_bg import CubeId, CubeKind, PipeId
from topologiq.dzw.common.components import ZxNode, ZxEdge, BgCube, BgPipe
from topologiq.dzw.common.coordinates import Coordinates
from topologiq.dzw.common.path import PathSpecification

from logging import getLogger
console = getLogger(__name__)

class LayerTransition(Enum):
    EVERY = 0
    LOWER = 1
    INTRA = 2
    UPPER = 3
    OUTER = 4

    def matches(self, source_layer: LayerId, target_layer: LayerId):
        if self == LayerTransition.EVERY:
            return True
        elif self == LayerTransition.LOWER:
            return target_layer < source_layer
        elif self == LayerTransition.INTRA:
            return source_layer == target_layer
        elif self == LayerTransition.UPPER:
            return source_layer < target_layer
        else: # self == LayerTransition.OUTER
            return False

# TODO: figure out whether anything can be added as a node/edge in networkx
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
                zx_source = self.get_zx_node(min(edge))
                zx_target = self.get_zx_node(max(edge))
                self.add_edge(*edge)
                self.edges[edge][AugmentedNxGraph.KEY_ZX_EDGE] = ZxEdge(zx_source, zx_target, edge_type)

        # Assume that the nodes and cubes are identified by random numbers, not necessarily consecutive ones
        zx_max_id = max(self.nodes) if len(self.nodes) > 0 else 0
        bg_max_id = max(self.__bg_graph.nodes) if len(self.__bg_graph.nodes) > 0 else 0
        self.__next_id = max(zx_max_id, bg_max_id) + 1

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
        for node in filter(lambda nd: self.get_zx_degree(nd) > 4, self.get_zx_nodes()):
            neighbors = iter(self.get_zx_neighbors(node))

            remaining_neighbors = self.get_zx_degree(node) - 3
            next(neighbors); next(neighbors); next(neighbors)
            previous: ZxNode = node
            while remaining_neighbors > 0:
                extra = ZxNode(id = self.__next_id, type = node.type, qubit = node.qubit, layer = node.layer)
                self.__next_id += 1

                self.add_node(extra.id)
                self.nodes[extra.id][AugmentedNxGraph.KEY_ZX_NODE] = extra

                # Make zx-edge between previous node of chain and new extra_node
                self.add_edge(previous.id, extra.id)

                neighbors_to_graft = remaining_neighbors if remaining_neighbors <= 3 else 2
                for _ in range(neighbors_to_graft):
                    neighbor = next(neighbors)
                    edge_type = self.get_zx_edge(extra.id, neighbor.id).type

                    self.remove_edge(node.id, neighbor.id)
                    self.add_edge(extra.id, neighbor.id)
                    extra_zx_edge = ZxEdge(source = extra, target = neighbor, type = edge_type)
                    self.get_edge_data(extra.id, extra_zx_edge)[AugmentedNxGraph.KEY_ZX_EDGE] = extra_zx_edge

                remaining_neighbors -= neighbors_to_graft
                previous = extra

    @staticmethod
    def from_pyzx_graph(zx_graph: pyzx.graph.base.BaseGraph):
        nodes: list[tuple[NodeId, NodeType]] = []
        for node in zx_graph.vertices():
            nodes.append((node, NodeType.convert_from_pyzx(zx_graph.type(node))))

        edges: list[tuple[EdgeId, EdgeType]] = []
        for edge in zx_graph.edges():
            edges.append(( edge , EdgeType.convert_from_pyzx(zx_graph.edge_type(edge))))

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

    def to_pyzx_graph(self, filepath: str = None, planar_scale: int = 8):
        pyzx_graph = pyzx.Graph()
        if len(self.get_zx_qubits()) == 0 or len(self.get_zx_layers()) == 0:
            layout = nx.planar_layout(self.__bg_graph, scale = planar_scale)
        else:
            # TODO: infer row and qubit of added cubes
            layout = dict()
            for cube in self.get_bg_cubes():
                layout[cube.id] = (
                    cube.realised_node.layer if cube.realised_node else -1,
                    cube.realised_node.qubit if cube.realised_node else -1
                )
        for cube in self.get_bg_cubes():
            layer, qubit = layout[cube.id]
            pyzx_graph.add_vertex(
                index = cube.id, ty = NodeType.convert_into_pyzx(cube.kind.get_type()), row = layer, qubit = qubit
            )
        for pipe in self.get_bg_pipes():
            pyzx_graph.add_edge((pipe.source.id, pipe.target.id), EdgeType.convert_into_pyzx(pipe.type))

        if filepath is not None:
            with open(filepath, 'w') as file:
                file.write(pyzx_graph.to_json())

        return pyzx_graph

    @staticmethod
    def from_simple_graph(simple_graph: SimpleDictGraph):
        console.warning(f"SimpleDictGraph does not provide qubit and layer information.")

        nodes = map( lambda nt : (nt[0], NodeType.convert_simple(nt[1])) , simple_graph["nodes"] )
        edges = map( lambda et : (et[0], EdgeType.convert_simple(et[1])) , simple_graph["edges"] )

        ang = AugmentedNxGraph(nodes, edges)

        return ang

    def get_node_realisation_order(self) -> list[NodeId]:
        return self.__zx_node_realisation_order

    def get_edge_realisation_order(self) -> list[EdgeId]:
        return self.__zx_edge_realisation_order

    def get_zx_node(self, node: NodeId) -> ZxNode:
        return self.nodes[node][AugmentedNxGraph.KEY_ZX_NODE]

    def get_zx_edge(self, source: NodeId, target: NodeId) -> ZxEdge:
        return self.get_edge_data(source, target)[AugmentedNxGraph.KEY_ZX_EDGE]

    def get_zx_nodes(
            self, node_type: NodeType | None = None, qubit: QubitId | None = None, layer: LayerId | None = None
    ) -> Iterable[ZxNode]:
        return filter(
            lambda node: (node_type is None or node.type == node_type) and
                         (qubit is None or node.qubit == qubit) and
                         (layer is None or node.layer == layer),
            map(lambda nd: self.get_zx_node(nd), self.nodes)
        )

    def get_zx_edges(
            self, edge_type: EdgeType | None = None, layered: tuple[LayerId, LayerTransition] | None = None
    ) -> Iterable[ZxEdge]:
        if layered is None:
            edges = map(lambda edge: self.get_zx_edge(*edge), self.edges)
        else:
            layer, transition = layered
            if len(self.__zx_layers) == 0:
                console.warning(f"Requesting layered edges but VolumetricZxGraph does not contain layer information.")
            edges = chain.from_iterable(map(
                lambda zxn: map(
                    lambda neighbor : self.get_zx_edge(zxn.id, neighbor.id),
                    filter(
                        lambda nb: transition != LayerTransition.INTRA or zxn.id < nb.id,
                        self.get_zx_neighbors(zxn, transition)
                    )
                ),
                self.get_zx_nodes(layer = layer)
            ))

        return filter(lambda edge: edge_type is None or edge.type == edge_type, edges)

    def get_zx_neighbors(
            self, node: ZxNode, transition: LayerTransition = LayerTransition.EVERY
    ) -> Iterable[ZxNode]:
        return filter(
            lambda neighbor: transition.matches(node.layer, neighbor.layer),
            map(self.get_zx_node, self.neighbors(node.id))
        )

    def get_zx_degree(self, node: ZxNode) -> int:
        return int(self.degree[node.id])

    def get_depth(self):
        return len(self.__zx_layers)

    def get_zx_layers(self):
        return self.__zx_layers.keys()

    def get_zx_qubits(self):
        return self.__zx_qubits.keys()

    def get_bg_cube(self, cube: CubeId) -> BgCube:
        return self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_CUBE]

    def get_bg_pipe(self, source: CubeId, target: CubeId) -> BgPipe:
        return self.__bg_graph.get_edge_data(source, target)[AugmentedNxGraph.KEY_BG_PIPE]

    def get_bg_cubes(self, kind: CubeKind | None = None) -> Iterable[BgCube]:
        return map(lambda cb: self.get_bg_cube(cb),
            filter(
                lambda cb: (kind is None or self.get_bg_cube(cb).kind == kind), self.__bg_graph.nodes()
            )
        )

    def get_bg_pipes(self, pipe_type: EdgeType | None = None) -> Iterable[BgPipe]:
        return map(lambda pp: self.get_bg_pipe(*pp),
            filter(
                lambda pp: (pipe_type is None or self.get_bg_pipe(*pp).type == pipe_type),
                self.__bg_graph.edges()
            )
        )

    def number_of_cubes(self) -> int:
        return self.__bg_graph.number_of_nodes()

    def number_of_pipes(self) -> int:
        return self.__bg_graph.number_of_edges()

    # TODO: remove after replacing the calls to this with calls to get_zx_neighbors(..)
    def get_node_neighbours(self,
        node: NodeId, transition: LayerTransition = LayerTransition.EVERY
    ) -> Iterable[NodeId]:
        return map(lambda zxn: zxn.id, self.get_zx_neighbors(self.get_zx_node(node), transition))

    def get_bg_neighbours(self, cube: BgCube, pipe_type: EdgeType | None = None) -> Iterable[BgCube]:
        return filter(
            lambda nb : (pipe_type is None or self.get_bg_pipe(cube.id, nb.id).type == pipe_type),
            map(self.get_bg_cube, self.__bg_graph.neighbors(cube.id))
        )

    def is_cube_placed(self, cube: CubeId) -> bool:
        return cube in self.__bg_graph

    def is_node_realised(self, node: NodeId) -> bool:
        return self.get_zx_node(node).is_realised()

    def realise_node(self, node: NodeId, kind: CubeKind, position: StandardCoord) -> CubeId:
        """Realise the node as a cube of the given kind placed at the given coordinates."""
        if kind not in CubeKind.suitable_kinds(self.get_zx_node(node).type):
            raise Exception(f"Requested {kind} is not compatible with {self.get_zx_node(node).type}")

        if not self.has_node(node):
            raise Exception(f"Node #{node} not found in the ZX-graph.")

        cube = self.place_cube(kind, position)
        self.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_CUBE].realised_node = self.get_zx_node(node)
        self.nodes[node][AugmentedNxGraph.KEY_ZX_NODE].realising_cube = self.get_bg_cube(cube)

        console.info(f"Realising node #{node} [{self.get_zx_node(node).type}] as cube #{cube} [{kind}@{position}]")
        self.__zx_node_realisation_order.append(node)

        return cube

    def is_edge_realised(self, source: NodeId, target: NodeId) -> bool:
        return len(self.edges[source, target][AugmentedNxGraph.KEY_ZX_EDGE].realisation) > 0

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

        # The sequence of pipes ids will serve as the realisation of the edge
        realisation = []

        # Add all the extra cubes and pipes of the path to the BlockGraph
        previous_cube: int = source_cube.id

        proposed_cubes = proposal.get_cubes()
        proposed_pipes = proposal.get_pipes()

        n = len(proposed_cubes)
        for index in range(1, n-1):
            current_kind, current_position = proposed_cubes[index]
            current_pipe_type = proposed_pipes[index-1]

            # Place the current cube and connect it to the previous cube.
            current_cube = self.place_cube(current_kind, current_position)
            self.connect_pipe(previous_cube, current_cube, current_pipe_type)

            # Extend the sequence of extra node ids
            pipe = (previous_cube, current_cube)
            realisation.append( pipe )

            # Prepare for the next iteration
            previous_cube = current_cube

        # Make the final connection
        final_pipe_type = proposed_pipes[-1]
        self.connect_pipe(previous_cube, target_cube.id, final_pipe_type)

        pipe = (previous_cube, target_cube.id)
        realisation.append( pipe )

        # Store the realisation of the edge
        self.edges[source, target][AugmentedNxGraph.KEY_ZX_EDGE].realisation = realisation

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

    def connect_pipe(self, source_cube_id: CubeId, target_cube_id: CubeId, pipe_type : EdgeType):
        if not self.__bg_graph.has_node(source_cube_id):
            raise Exception(f"Cube #{source_cube_id} not found in the BG-graph.")

        if not self.__bg_graph.has_node(target_cube_id):
            raise Exception(f"Cube #{target_cube_id} not found in the BG-graph.")

        if self.__bg_graph.has_edge(source_cube_id, target_cube_id):
            raise Exception(f"Cubes #{source_cube_id} and #{target_cube_id} are already connected by a pipe.")

        source_cube = self.get_bg_cube(source_cube_id)
        target_cube = self.get_bg_cube(target_cube_id)
        source_kind = source_cube.kind
        target_kind = target_cube.kind
        if not pipe_type in BlockGraphHelper.infer_pipe_type(source_kind, target_kind):
            raise Exception(f"Pipe type {pipe_type} is incompatible with source and target kinds [{source_kind}-{target_kind}].")

        # TODO: validate with respect to inferred pipe type between source and target cubes

        source_position = source_cube.position
        target_position = target_cube.position
        # TODO: replace 3 with 1 once the pathfinder has been rewritten
        if get_manhattan(source_position, target_position) != 3:
            raise Exception(f"Cubes #{source_cube_id}@{source_position} and #{target_cube_id}@{target_position} are not at adjacent positions.")

        self.__bg_graph.add_edge(source_cube.id, target_cube.id)
        bg_pipe = BgPipe(source_cube, target_cube, pipe_type)
        self.__bg_graph.get_edge_data(source_cube.id, target_cube.id)[AugmentedNxGraph.KEY_BG_PIPE] = bg_pipe

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

        for layer in self.get_zx_layers():
            console.info(f"Layer {layer}  : {list(self.get_zx_nodes(layer = layer))}")

        for qubit in self.get_zx_qubits():
            console.info(f"Qubit {qubit}  : {list(self.get_zx_nodes(qubit = qubit))}")

    def print_summary(self):
        for node_type in [NodeType.O, NodeType.X, NodeType.Y, NodeType.Z]:
            content = ""
            for node in self.get_zx_nodes(node_type = node_type):
                content += f"{node} "
            print(f"Nodes {node_type.name}: {content}")

        content = ""
        for edge in self.get_zx_edges():
            content += f"({edge.source.id},{edge.target.id}) "
        print(f"Edges  : {content}")

        for layer in self.get_zx_layers():
            print(f"Layer {layer}  : {list(self.get_zx_nodes(layer = layer))}")

        for qubit in self.get_zx_qubits():
            print(f"Qubit {qubit}  : {list(self.get_zx_nodes(qubit = qubit))}")

        for cube in self.get_bg_cubes():
            print(f"Cube : {cube}")

        for pipe in self.get_bg_pipes():
            print(f"Pipe {pipe}  : {pipe.type}")

    @staticmethod
    def from_file(filepath: str, include_zx_graph: bool = False):
        with open(filepath, 'r') as file:
            # Read the blockgraph header
            header = file.readline()
            if header != "BLOCKGRAPH 0.1.0;\n":
                raise Exception(f"Invalid file format. Header for BLOCKGRAPH 0.1.0 not found [got={header}].")

            # Read the empty line between the blockgraph header and the cubes header
            file.readline()

            # Instantiate the new ANG
            ang = AugmentedNxGraph()

            if include_zx_graph:
                # Read the nodes header
                header = file.readline()
                if header != "NODES: id;type;qubit;layer;realising_cube\n":
                    raise Exception(f"Invalid file format. Header for NODES not found [got={header}].")

                zx_node_bg_cube: dict[ZxNode, CubeId] = dict()

                # Read all the lines describing nodes
                current_line = file.readline()
                while current_line and current_line != "\n":
                    if current_line != "":
                        node_id, node_type, qubit, layer, realising_cube = current_line.split(';')
                        node = int(node_id)
                        ang.add_node(node)
                        zx_node = ZxNode(id=node, type=NodeType[node_type], qubit=int(qubit), layer=int(layer))
                        ang.nodes[node][AugmentedNxGraph.KEY_ZX_NODE] = zx_node
                        zx_node_bg_cube[zx_node] = int(realising_cube)
                        ang.__zx_qubits[int(qubit)].append(node)
                        ang.__zx_layers[int(layer)].append(node)
                    current_line = file.readline()

                # Read the edges header
                header = file.readline()
                if header != "EDGES: source;target;type;realisation\n":
                    raise Exception(f"Invalid file format. Header for EDGES not found [got={header}].")

                zx_edge_bg_pipe: dict[ZxEdge, list[PipeId]] = dict()

                # Read all the lines describing edges
                current_line = file.readline()
                while current_line and current_line != "\n":
                    if current_line != "":
                        source_id, target_id, edge_type, realisation = current_line.split(';')
                        zx_source = ang.get_zx_node(int(source_id))
                        zx_target = ang.get_zx_node(int(target_id))
                        ang.add_edge(zx_source.id, zx_target.id)
                        zx_edge = ZxEdge(source = zx_source, target = zx_target, type = EdgeType[edge_type])
                        ang.edges[zx_source.id, zx_target.id][AugmentedNxGraph.KEY_ZX_EDGE] = zx_edge
                        zx_edge_bg_pipe[zx_edge] = [ make_tuple(pair) for pair in realisation[1:-2].split(':') ]
                    current_line = file.readline()

            # Read the cubes header
            header = file.readline()
            if header != "CUBES: index;x;y;z;kind;label;\n":
                raise Exception(f"Invalid file format. Header for CUBES not found [got={header}].")

            # Read all the lines describing cubes
            current_line = file.readline()
            while current_line and current_line != "\n":
                if current_line != "":
                    cube_id, x, y, z, kind, realised_node, _ = current_line.split(';')
                    cube = int(cube_id)
                    ang.__bg_graph.add_node(cube)
                    bg_cube = BgCube(
                        id = cube, kind = CubeKind[kind.upper()], position = Coordinates(int(x), int(y), int(z)) / 3.0
                    )
                    ang.__bg_graph.nodes[cube][AugmentedNxGraph.KEY_BG_CUBE] = bg_cube
                current_line = file.readline()

            # Read the pipes header
            header = file.readline()
            if header != "PIPES: src;tgt;kind;\n":
                raise Exception(f"Invalid file format. Header for PIPES not found [got={header}].")

            # Read all the lines describing pipes
            current_line = file.readline()
            while current_line and current_line != "\n":
                if current_line != "":
                    source_id, target_id, pipe_kind, _ = current_line.split(';')
                    bg_source = ang.get_bg_cube(int(source_id))
                    bg_target = ang.get_bg_cube(int(target_id))
                    ang.__bg_graph.add_edge(bg_source.id, bg_target.id)
                    ang.__bg_graph.edges[bg_source.id, bg_target.id][AugmentedNxGraph.KEY_BG_PIPE] = BgPipe(
                        source = bg_source, target = bg_target, type = EdgeType.HADAMARD if 'h' in pipe_kind else EdgeType.IDENTITY
                    )
                current_line = file.readline()

            if include_zx_graph:
                for zx_node, bg_cube_id in zx_node_bg_cube.items():
                    bg_cube = ang.get_bg_cube(bg_cube_id)
                    bg_cube.realised_node = zx_node
                    zx_node.realising_cube = bg_cube

                for zx_edge, bg_pipe_ids in zx_edge_bg_pipe.items():
                    zx_edge.realisation = list(map(lambda pp: ang.get_bg_pipe(*pp), bg_pipe_ids))

            return ang

    @staticmethod
    def __scaled_position(position: tuple[int,int,int]) -> str:
        return '(' + ','.join(str(int(p / 3.0)) for p in position) + ')'

    def __infer_pipe_kind(self, source: BgCube, target: BgCube) -> str:
        # cellcolors are for faces (X, Y, Z) +  'h' if Hadamard pipe
        colors = ""
        src_position = Coordinates.from_tuple(source.position)
        tgt_position = Coordinates.from_tuple(target.position)
        step_taken = tgt_position - src_position
        distances = step_taken.as_tuple()
        for c in range(3):
            if distances[c] == 0 and (
                    source.kind not in [CubeKind.OOO, CubeKind.YYY] or target.kind not in [CubeKind.OOO, CubeKind.YYY]):
                color = source.kind.name[c] if source.kind not in [CubeKind.OOO, CubeKind.YYY] else target.kind.name[c]
            else:
                color = 'o'
            colors += color

        if self.get_bg_pipe(source.id, target.id).type == EdgeType.HADAMARD:
            colors += 'h'

        return colors.lower()

    def __format_label(self, cube: BgCube):
        label = ""
        if cube.kind == CubeKind.OOO and cube.realised_node is not None:
            zx_node = cube.realised_node
            if zx_node.layer == 0:
                label = f"in_{zx_node.qubit}"
            elif zx_node.layer == self.get_depth() - 1:
                label = f"out_{zx_node.qubit}"
        return label

    def into_file(self, filepath: str, include_zx_graph: bool = False):
        with (open(filepath, 'w') as file):
            file.write(f"BLOCKGRAPH 0.1.0;\n")

            if include_zx_graph:
                # Store node information
                file.write("\nNODES: id;type;qubit;layer;realising_cube\n")
                file.writelines(
                    [
                        f"{node.id};{node.type};{node.qubit};{node.layer};{node.realising_cube.id}\n"
                        for node in self.get_zx_nodes()
                    ]
                )

                # Store edge information
                file.write("\nEDGES: source;target;type;realisation\n")
                file.writelines(
                    [
                        f"{edge.source.id};{edge.target.id};{edge.type};[{':'.join(map(str, edge.realisation))}]\n"
                        for edge in self.get_zx_edges()
                    ]
                )

            # Store cube information
            file.write("\nCUBES: index;x;y;z;kind;label;\n")
            file.writelines(
                [
                    f"{cube.id};{';'.join(map(str, iter(cube.position)))};{cube.kind.name.lower()};{self.__format_label(cube)};\n"
                    for cube in self.get_bg_cubes()
                ]
            )

            # Store pipe information
            file.write("\nPIPES: src;tgt;kind;\n")
            file.writelines(
                [
                    f"{pipe.source.id};{pipe.target.id};{self.__infer_pipe_kind(pipe.source, pipe.target)};\n"
                    for pipe in self.get_bg_pipes()
                ]
            )

    def __identify_cube_at_position(self, position: StandardCoord) -> int:
        for cube in self.get_bg_cubes():
            if cube.position == position:
                return cube.id

        return -1