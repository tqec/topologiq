"""UX misc. utils.

AI disclaimer:
    category: Coding partner (see CONTRIBUTING.md for details).
    model: Gemini, 3.0 Fast.
    details: The AI assisted in architectural patterns, multi-framework type handling,
        and boilerplate generation, while the domain logic and integration requirements
        were directed by the human author.

"""


GLUE_CODE = """import qiskit
import qbraid
import sys

def run_user_code():
{source_design}
    return locals()

try:
    user_vars = run_user_code()
    target = user_vars.get('{var_name}') or globals().get('{var_name}')
    
    if target is not None:
        # Robust conversion using qbraid's high-level transpile
        # This avoids the missing 'circuit_wrapper' name error
        qasm_str = qbraid.transpiler.transpile(target, "qasm2")
        
        sys.stdout.write("---BEGIN_QASM---\\n")
        sys.stdout.write(qasm_str)
        sys.stdout.write("\\n---END_QASM---\\n")
        sys.stdout.flush()
    else:
        print(f"ERROR: Variable '{{var_name}}' not found.", file=sys.stderr)
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.stderr.flush()
"""
