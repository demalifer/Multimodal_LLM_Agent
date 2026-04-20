"""Execution utilities shared across tool modules."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

SAFE_BUILTINS: dict[str, object] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}


def run_python_code(code: str) -> str:
    """Run Python code in a constrained builtins scope and return stdout text."""
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code 必须是非空字符串")

    stdout_buffer = io.StringIO()
    try:
        globals_scope = {"__builtins__": SAFE_BUILTINS}
        locals_scope: dict[str, object] = {}
        with redirect_stdout(stdout_buffer):
            exec(code, globals_scope, locals_scope)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"执行异常: {exc}") from exc

    return stdout_buffer.getvalue().rstrip("\n")
