"""Simple JSON-based tool executor."""

import json

from execution import run_python_code


def execute_tool(json_str: str) -> str:
    """Execute a tool call described by a JSON string.

    Expected input format:
    {"tool": "python", "code": "print(1+1)"}
    """
    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return f"JSON解析失败: {exc}"

    tool = payload.get("tool")
    if tool != "python":
        return f"不支持的工具: {tool}"

    code = payload.get("code")
    if not isinstance(code, str):
        return "参数错误: code 必须是字符串"

    try:
        return run_python_code(code)
    except Exception as exc:  # noqa: BLE001
        return str(exc)
