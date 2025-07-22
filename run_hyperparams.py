# This file contains hyperparameters that the algorithm uses to make key decisions

# Weights for the main value function to choose best of several valid paths (length of path, beams broken by path)
VALUE_FUNCTION_HYPERPARAMS = (-1, -0.5)

# Length of beams
LENGTH_OF_BEAMS = 3

# Maximum size of the search space (measured from the current starting block)
MAX_PATHFINDER_SEARCH_SPACE = 3