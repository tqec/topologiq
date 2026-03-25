"""UX misc. utils.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""

GLUE_CODE = """
import sys
import json
import traceback

try:
    import pyzx as zx
    # The most reliable way to check for ANY PyZX graph
    from pyzx.graph.base import BaseGraph
except ImportError:
    zx = None
    BaseGraph = None

def run_user_code():
{source_design}
    return locals()

try:
    user_vars = run_user_code()
    target = user_vars.get('{var_name}') or globals().get('{var_name}')

    if target is not None:
        # DEBUG: Tell the manager what we found
        sys.stderr.write(f"DEBUG: Found target type: {{type(target)}}\\n")

        # PATH A: Direct PyZX Handling
        # We check for BaseGraph OR the presence of the to_json method
        if (BaseGraph and isinstance(target, BaseGraph)) or hasattr(target, 'to_json'):
            sys.stderr.write("DEBUG: Path A (PyZX) selected\\n")
            sys.stdout.write("---BEGIN_PYZX_JSON---\\n")
            sys.stdout.write(target.to_json())
            sys.stdout.write("\\n---END_PYZX_JSON---\\n")

        # PATH B: Standard qBraid fallback
        else:
            sys.stderr.write("DEBUG: Path B (qBraid) selected\\n")
            import qbraid
            qasm_str = qbraid.transpiler.transpile(target, "qasm2")
            sys.stdout.write("---BEGIN_QASM---\\n")
            sys.stdout.write(qasm_str)
            sys.stdout.write("\\n---END_QASM---\\n")

        sys.stdout.flush()
    else:
        sys.stderr.write(f"DEBUG: Variable '{{var_name}}' was None or not found.\\n")
except Exception:
    sys.stderr.write(traceback.format_exc())
    sys.stderr.flush()
"""
