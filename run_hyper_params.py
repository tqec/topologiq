# THIS FILE HOLDS INITIAL HYPER-PARAMETERS

# Weights for the main value functoin
# Used to choose between several valid paths between blocks
# (length of path, beams broken by path, # of unobstructed exits in next cube)
VALUE_FUNCTION_HYPERPARAMS = (-10, -2, 0)

# Length of beams
LENGTH_OF_BEAMS = 3

# Maximum size of the search space (measured from the current starting block)
MAX_PATHFINDER_SEARCH_SPACE = 3