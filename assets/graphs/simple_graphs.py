from utils.classes import SimpleDictGraph

cnot: SimpleDictGraph = {
    "nodes": [
        (0, "X"),
        (1, "Z"),
        (2, "X"),
        (3, "Z"),
        (4, "X"),
        (5, "Z"),
    ],
    "edges": [
        ((0, 2), "SIMPLE"),
        ((1, 3), "SIMPLE"),
        ((2, 3), "SIMPLE"),
        ((2, 4), "SIMPLE"),
        ((3, 5), "SIMPLE"),
    ],
}


steane: SimpleDictGraph = {
    "nodes": [
        (1, "X"),
        (2, "Z"),
        (3, "Z"),
        (4, "Z"),
        (5, "X"),
        (6, "X"),
        (7, "X"),
        (8, "O"),
        (9, "O"),
        (10, "O"),
        (11, "O"),
        (12, "O"),
        (13, "O"),
        (14, "O"),
    ],
    "edges": [
        ((1, 2), "SIMPLE"),
        ((1, 3), "SIMPLE"),
        ((1, 4), "SIMPLE"),
        ((5, 2), "SIMPLE"),
        ((5, 3), "SIMPLE"),
        ((6, 2), "SIMPLE"),
        ((6, 4), "SIMPLE"),
        ((7, 3), "SIMPLE"),
        ((7, 4), "SIMPLE"),
        ((8, 1), "SIMPLE"),
        ((9, 5), "SIMPLE"),
        ((10, 6), "SIMPLE"),
        ((11, 7), "SIMPLE"),
        ((2, 12), "SIMPLE"),
        ((3, 13), "SIMPLE"),
        ((4, 14), "SIMPLE"),
    ],
}

hadamard_line: SimpleDictGraph = {
    "nodes": [
        (0, "Z"),
        (1, "X"),
        (2, "Z"),
        (3, "X"),
        (4, "Z"),
        (5, "X"),
    ],
    "edges": [
        ((0, 1), "HADAMARD"),
        ((1, 2), "HADAMARD"),
        ((2, 3), "HADAMARD"),
        ((3, 4), "HADAMARD"),
        ((4, 5), "HADAMARD"),
    ],
}

hadamard_bend: SimpleDictGraph = {
    "nodes": [
        (0, "Z"),
        (1, "X"),
        (2, "Z"),
        (3, "X"),
        (4, "Z"),
        (5, "X"),
    ],
    "edges": [
        ((0, 1), "HADAMARD"),
        ((0, 2), "HADAMARD"),
        ((0, 5), "HADAMARD"),
        ((3, 4), "HADAMARD"),
        ((4, 5), "HADAMARD"),
    ],
}

mess_of_hadamards: SimpleDictGraph = {
    "nodes": [
        (1, "X"),
        (2, "Z"),
        (3, "Z"),
        (4, "Z"),
        (5, "X"),
        (6, "X"),
        (7, "X"),
        (8, "O"),
        (9, "O"),
        (10, "O"),
        (11, "O"),
        (12, "O"),
        (13, "O"),
        (14, "O"),
    ],
    "edges": [
        ((1, 2), "HADAMARD"),
        ((1, 3), "SIMPLE"),
        ((1, 4), "SIMPLE"),
        ((5, 2), "HADAMARD"),
        ((5, 3), "SIMPLE"),
        ((6, 2), "SIMPLE"),
        ((6, 4), "HADAMARD"),
        ((7, 3), "HADAMARD"),
        ((7, 4), "HADAMARD"),
        ((8, 1), "SIMPLE"),
        ((9, 5), "SIMPLE"),
        ((10, 6), "SIMPLE"),
        ((11, 7), "SIMPLE"),
        ((2, 12), "SIMPLE"),
        ((3, 13), "SIMPLE"),
        ((4, 14), "SIMPLE"),
    ],
}
