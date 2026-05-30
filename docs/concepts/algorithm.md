# Topologiq: High-level algorithm description

## Abstract
We introduce an algorithm that converts ZX-circuits into topologically-correct space-time diagrams using the surface code and lattice surgery. The algorithm is best described as a graph-traversing -meets- three-dimensional (3D) routing algorithm where all topological constraints associated with lattice surgery are enforced by geometrical constraints. The algorithm traverses a ZX graph breadth-first and uses these geometrical constraints to iteratively place 3D representations of each spider in a way that preserves the topology of the underlying quantum computation.

## Dual-stage BFS
Topologiq works by collaboration between a graph-traversing subroutine that we call the ***graph manager*** and a 3D path-creating or path-finding subrouting that we call the ***pathfinder***.

For the time being, both subroutines follow a breadth-first-search (BFS) rationale.

### Graph Manager
The process is orchestrated by an (outer) BFS graph manager algorithm. 
- Pick FIRST spider (S) as root for the construction and place its corresponding CUBE at origin (x: 0, y: 0, z: 0). 
  - ID selection strategies: lowest ID, centrality via majority vote, centrality random.
  - KIND selection strategies: pre-determined first kind, random.
- Pre-generate edge queue: [(S, v0), (u1, v1), (u2, v2), …, (un, vn)].
  - BFS: Visit every neighbour of each spider, then revisit any edges missed. 
  - BFS with priority cross-edges. A reshuffled BFS queue where cross-edges are handled as soon as they appear.
- Exhaust the queue.
  - For every neighbour T of S, considering **GEOMETRIC CONSTRAINTS**:
    - If T has no assigned coordinates (standard edge): call pathfinder to place T as close as possible to S
    - If T has assigned coordinates (cross edge): call pathfinder to find the shortest path from S to T
    - Always: add path to the main BlockGraph.
  - When queue ends, SUCCESS.
- Verify correctness computationally.

### Pathfinder
Each edge is rendered into a topologically-correct 3D path by the (inner) pathfinder algorithm (currently, Djikstra).
- Considering GEOMETRIC CONSTRAINTS
  - Standard edge (a new cube is being added to the 3D space): multi-target BFS search.
    - Returns many topologically correct paths to alternative tentative position where a cube of the respective kind could be placed (final path selection happens in graph manager, via a value function).
  - Cross edge (both source and target cubes have already been placed in the 3D space): shortest-path BFS
    - Finds the shortest path between the (src, tgt) pair.

## Considering GEOMETRIC CONSTRAINTS
There is a number of objects that support Topologiq's ability to deliver topologically-correct 3D paths. Several of these objects are algorithmically relevant and computationally interesting. 

**Beams.** Topologiq has an object called "beams", which are like beams of light that emanate from any open face (i.e. a face that could accept an incoming or outgoing connection) of any pending cube (i.e. a cube that still needs connections). Beams are the most significant contributor to Topologiq’s ability to complete circuits, because they clear the path to/from the “exits” of any block that still needs connections.

> *Without beams, Topologiq would need an alternative way to ensure paths do not crash into themselves.*

**Symbolic matching.** Topologiq enforces topological constraints with a series of symbolic micro-methods. Each method uses the characteristics of an arbitrary cube to determine small things such as, for instance, if a face could acccept a connection, or if the colours (encoded as strings) of two adjacent faces match. When all these checks clear, topology is maintained.

> *Without symbolic matching, Topologiq would need an alternative way to ensure the continuity of the computation is preseved.*

**Value function.** Topologiq delivers the lowest volume it can thanks to a value function that chooses between paths when the pathfinder returns more than one topologically-correct path candidate. The value function primarily considers the lenght of candidata paths against the risk each path represents for the completion of any pending edges.

> *Without the value function, Topologiq would need an alternative way to ensure the BlockGraph does not become unnecessarily large.*