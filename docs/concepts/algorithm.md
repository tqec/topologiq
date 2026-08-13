# Topologiq: High-level algorithm description

## Abstract
We introduce an algorithm that converts ZX-circuits into topologically-correct space-time diagrams using the surface code and lattice surgery. The algorithm is best described as a graph-traversing -meets- three-dimensional (3D) routing algorithm where all topological constraints associated with lattice surgery are enforced by geometrical constraints. The algorithm traverses a ZX graph breadth-first and uses these geometrical constraints to iteratively place 3D representations of each spider in a way that preserves the topology of the underlying quantum computation.

## Dual-stage BFS
Topologiq works by collaboration between a graph-traversing subroutine that we call the ***graph manager*** and a 3D path-creating or path-finding subrouting that we call the ***pathfinder***.

### Graph Manager
The general aspects of the process is orchestrated by an (outer) BFS graph manager algorithm. The graph manager is broadly in charge of administering the edge discovery process, choosing best of several 3D paths returned by the pathfinder, as well as incorporating each new 3D path into the running surgery. 

#### Definitions
- ***ID:*** A numerical unique identifier given to any given spider or cube in the ZX graph.
- ***ZX_SPIDER:*** A spider or node in the input ZX graph.
- ***ZX_EDGE:*** An edge in the input ZX graph, i.e., a connected *(u, v)* pair where *u* and *v* are arbitrary *SPIDERs* connected to one another.
- ***ZX_TYPE:*** The type of a *ZX_SPIDER* or *ZX_EDGE*.
- ***CUBE:*** A 3D block of equal width, length and height, which materialises a *ZX_SPIDER* into a topologically correct lattice surgery primitive.
- ***KIND:*** A string identifier used to denote the type of a *CUBE*.
- ***COORDINATES:*** A (x, y, z) tuple representing the position of a *CUBE*.
- ***FIRST_SPIDER:*** The *ID* of the first *ZX_SPIDER* processed.
- ***FIRST_CUBE:*** The (*ID*, *KIND*) of the first *CUBE* processed.
- ***ORIGIN:*** The coordinates given to the *FIRST_CUBE*.
- ***EDGE_QUEUE:*** A list of *ZX_EDGEs* in the order they will be visited by the algorithm, starting with FIRST_SPIDER and organised so that every *u* in every *(u, v)* pair has already appeared as *v* in a previous edge (i.e. has been visited earlier in the process).
- ***T_PATTERN:*** A list of *ZX_EDGEs* that altogether realise a ZX_SPIDER with pi/4 phase (i.e., a T-gate) into a topologically correct lattice surgery subroutine composed of several CUBEs.
- ***CANDIDATE_PATH***: A series of CUBEs that altogether deliver a topologically correct lattice surgery realisation of a ZX_EDGE.
- ***WINNER_PATH***: The best CANDIDATE_PATH, as chosen by a tunable value function. 
- ***GEOMETRIC CONSTRAINTS:*** Geometric objects representing obstacles.


#### Description
1. **Pick FIRST_CUBE** as root for the build process and place at ORIGIN. 
    - ID selection:
        * ***First-spider:*** Lowest ID from inputs, lowest non-boundary ID if inputs not available, lowest ID if all else fails.
        * ***Random:*** Random non-boundary, non-special (T, S, etc) ID.
        * ***Centrality-random:*** Random ID from a list of central IDs.
        * ***Centrality-majority:*** ID corresponding to the majority vote from several centrality measures (per NetworkX).
        * ***Central-qubit:*** Lowest ID in most central qubit (the qubit with most CNOTs).
        * ***Central-in-first-cycle:*** ID of the central spider in the first cycle of the graph (per NetworkX's `cycle_basis`).
    - KIND selection:
        * ***Pre-determined:*** "ZXZ" if ZX_TYPE of FIRST_SPIDER is "X", "XZX" if "Z", or "OOO" if "BOUNDARY".
        * ***Random:*** Random choice between all KINDS compatible with the ZX_TYPE of FIRST_SPIDER.
2. **Pre-generate EDGE_QUEUE**: *[(FIRST_SPIDER, v0), (u1, v1), (u2, v2), …, (un, vn)]*.
    - ***BFS***: Subroutines following traditional or near-traditional BFS rationales.
        * ***BFS:*** Starting at FIRST_SPIDER, for each ID in queue, remove ID from queue, visit each neighbour, and add neighbour to queue, then query the graph again for any edges not completed in the main pass (aka. "cross" edges).
        * ***BFS priority cross-edges:*** *BFS* as above, reshuffled to handle cross-edges as soon as their respective (u,v) nodes are visited rather than at the end of queue.
        * ***BFS cycles:*** For each cycle in graph, *BFS* over cycle if cycle is connected to previously visited IDs or connect cycle to visited spiders using a shortest topologically-correct path and then *BFS* over cycle, then *BFS* over remainders of the graph not included in the main pass due to not being part of a cycle (requires *central-in-first-cycle* ID selection strategy).
    - ***Experimental (almost-no-longer) BFS:***
        * ***BFS CNOTS:*** Starting at FIRST_SPIDER, visit all spiders in central qubit, then *BFS* the remainder of the graph starting by any CNOTs coming out of central qubit (requires *central-qubit* ID selection strategy).
        * ***BFS layers:*** Find a shortest path to all inputs, then visit all spiders in central qubit, then fulfill all T-gate patterns, then resolve the rest of the graph using BFS.
        * ***TFS CNOTS:*** Same as *BFS CNOTS* but resolving T-gate patterns as a priority (requires *central-qubit* ID selection strategy).
3. **Call the pathfinder for each *ZX_EDGE* in *EDGE_QUEUE***.
    - For every *(u, v)* in *EDGE_QUEUE*, considering **GEOMETRIC CONSTRAINTS**:
        * If *v* has no assigned coordinates (standard edge):
            + Assemble a list of n-potential coordinates where *u* could theoretically be placed
            + Call pathfinder to get a number of topologically *CANDIDATE_PATHs* between *u* and *v*.
            + Discard any *CANDIDATE_PATHs* that, while topologically correct, risk completion of the full circuit.
            + Use value function to choose a *WINNER_PATH* from *CANDIDATE_PATHs* that survive elimination.
        * If *v* has assigned coordinates (cross edge)
            + Call pathfinder to find the shortest topologically-correct path from *u* to *v*.
            + Set shortest topologically-correct path as *WINNER_PATH*.
        Add *WINNER_PATH* path to the main BlockGraph.
    - When queue ends, call SUCCESS.
4. Verify correctness computationally
    - ***Pending:*** only partially implemented as of yet.

### Pathfinder
The pathfinder is in charge of rendering each *ZX_EDGE* sent to it into a topologically-correct 3D path made up of as many CUBES as needed to clear the edge in 3D.

#### Definitions
- ***ZX_BLOCK:*** A dual-purpose ZX_SPIDER -meets- BlockGraph CUBE which encodes **GEOMETRIC CONSTRAINTS** needed to ensure paths composed of several blocks are topologically correct.
- ***POSITIONED_ZX_BLOCK:*** A (coordinate, ZX_BLOCK) tuple containing the 3D position and meta-information for any CUBE

#### Description
> Note. Any call to the pathfinder has been checked to ensure the *u* component of the current *(u,v)* pair in EDGE_QUEUE has already been incorporated into the BLOCKGRAPH as a fixed (indeed unchangeable) *POSITIONED_ZX_BLOCK*

1. Examine the incoming list of list of n-potential coordinates where *v* could theoretically be placed.
    - If there is only one potential coordinate, that's the only possible location for *v*: go into shortest topologically-correct path mode.
        * If *v* has already been placed into the BLOCKGRAPH (cross-edge):
            + Considering **GEOMETRIC CONSTRAINTS** encoded in any ZX_BLOCK used across the pathfinding process
                + Find the shortest topologically-correct path between the pre-existent CUBE with *ID* *u* and the pre-existent *CUBE* with *ID* *v*.
        * If *v* has not already been placed into the BLOCKGRAPH (a standard edge with very limited room for maneuvring):
            + Considering **GEOMETRIC CONSTRAINTS** encoded in any ZX_BLOCK used across the pathfinding process
                + Find the shortest topologically-correct path between the pre-existent CUBE with *ID* *u* and the potential coordinate, so long as the *KIND* of the final *CUBE* in path is compatible with the *ZX_KIND* of the *ZX_SPIDER* with *ID* *v*.
    - If there are many potential coordinates:
        * Considering **GEOMETRIC CONSTRAINTS** encoded in any ZX_BLOCK used across the pathfinding process
            + Find shortest topologically-correct paths from *u* to at least 60% of potential coordinates, so long as the *KIND* of the final *CUBE* in each path is compatible with the *ZX_KIND* of the *ZX_SPIDER* with *ID* *v*.
2. Return any and all shortest paths found. 


## Considering **GEOMETRIC CONSTRAINTS**
Topologiq enforces topology via a series of flexible micro-subroutines spread across the algorithm.

### Definitions
- ***TAKEN:*** All coordinates taken by all CUBES already in the BLOCKGRAPH.
- ***FACES:*** The sides of a CUBE, which have coloured faces arranged in different patterns depending on its KIND.
- ***EXITS:*** A FACE that allows connecting an edge.
- ***BEAMS:*** Variable-length one-dimensional line segments attached to EXITS, which clear the way to and from the given EXIT.

### Usage
- *TAKEN:* TAKEN is used whenever there is a need to establish if a given coordinate is occupied and can therefore not be used to place a CUBE. So, for example, if the algorithm need to cross over a coordinate in TAKEN to reach a potential coordinate, it will used TAKEN to detect the obstacle and go around it rather than through it.
- *EXITS:* EXITs are used to determine if an edge can go out or into a cube from a given direction. For example, if a CUBE has EXITS on the X- and Y-axes, the algorithm will skip visiting neighbouring coordinates along the Z-axis because the CUBE is not connectable along its Z-axis.
- *FACES:* The FACEs of a CUBE are used to determine EXITS, as well as to match two CUBEs against one another. So, for example, if a CUBE has a red FACE on its X-axis, the algorithm can check that any CUBE connected CUBE also has a red FACE on its X-axis, else the cubes are not compatible.
- *EXIST and FACES:* Any match between two CUBES can be guaranteed to be topologically correct by combining EXITS and FACES, as this checks that the connection is taken place via an EXIT on both CUBES and that the FACES of both CUBES are compatible with one another. This is also true of special lattice surgery primitives such as the Y-cube, cultivation/distillation, and conditionals, because all of these CUBES must connect to more standard surface code CUBES and must therefore match at the boundary irrespective of internal specifics.
- *BEAMS:* For practical purposes, BEAMS can be imagined as beams of light emanating from any face of CUBE that could accept an incoming or outgoing connection, and are only attached to CUBES that still need edges. While mathematically distinct, the principle used with BEAMS is similar to what is done with TAKEN. The algorithm looks for potential clashes against any coordinate covered by a BEAM, and goes around them when needed. Having said that, while a coordinate in TAKEN is a final obstacle, a beam clash may or may not be tolerable, as there are tolerance tresholds that permit some BEAMS get broken.