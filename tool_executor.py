"""Simple JSON-based tool executor."""

import io
import json
from contextlib import redirect_stdout


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

    stdout_buffer = io.StringIO()
    try:
        # 使用空globals/local，降低对外部作用域的影响
        with redirect_stdout(stdout_buffer):
            exec(code, {}, {})
    except Exception as exc:  # noqa: BLE001
        return f"执行异常: {exc}"

    return stdout_buffer.getvalue()
