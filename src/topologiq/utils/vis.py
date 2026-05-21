"""Visualisation utils shared by features across the codebase.

Usage:
    Call any constant/function/class from a separate script.

"""

#############
# CONSTANTS #
#############
node_hex_map = {
    "xxz": ["#d7a4a1", "#d7a4a1", "#b9cdff"],
    "xzz": ["#d7a4a1", "#b9cdff", "#b9cdff"],
    "xzx": ["#d7a4a1", "#b9cdff", "#d7a4a1"],
    "zzx": ["#b9cdff", "#b9cdff", "#d7a4a1"],
    "zxx": ["#b9cdff", "#d7a4a1", "#d7a4a1"],
    "zxz": ["#b9cdff", "#d7a4a1", "#b9cdff"],
    "zxo": ["#b9cdff", "#d7a4a1", "gray"],
    "xzo": ["#d7a4a1", "#b9cdff", "gray"],
    "oxz": ["gray", "#d7a4a1", "#b9cdff"],
    "ozx": ["gray", "#b9cdff", "#d7a4a1"],
    "xoz": ["#d7a4a1", "gray", "#b9cdff"],
    "zox": ["#b9cdff", "gray", "#d7a4a1"],
    "zxoh": ["#b9cdff", "#d7a4a1", "gray"],
    "xzoh": ["#d7a4a1", "#b9cdff", "gray"],
    "oxzh": ["gray", "#d7a4a1", "#b9cdff"],
    "ozxh": ["gray", "#b9cdff", "#d7a4a1"],
    "xozh": ["#d7a4a1", "gray", "#b9cdff"],
    "zoxh": ["#b9cdff", "gray", "#d7a4a1"],
    "xxx": ["red", "red", "red"],
    "yyy": ["green", "green", "green"],
    "zzz": ["blue", "blue", "blue"],
    "X": ["red", "red", "red"],
    "Y": ["green", "green", "green"],
    "Z": ["blue", "blue", "blue"],
    "aaa": ["silver", "silver", "silver"],
    "hadamard": ["yellow"],
    "beam": ["yellow"],
}
