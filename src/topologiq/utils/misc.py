"""Misc utils of various sorts.

Usage:
    Call any function/class from a separate script.

"""

def kind_to_zx_type(kind: str) -> str:
    """Get the ZX type corresponding to a given block or pipe kind.

    Args:
        kind: the /kind of a given block.

    Returns:
        zx_type: the ZX type corresponding to the kind.

    """

    if kind == "ooo":
        zx_type = "BOUNDARY"
    elif "o" in kind:
        zx_type = "HADAMARD" if "h" in kind else "SIMPLE"
    else:
        zx_type = min(set(kind), key=lambda c: kind.count(c)).capitalize()
    return zx_type
