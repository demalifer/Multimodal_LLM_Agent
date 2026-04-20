"""Tool routing module.

根据输入 JSON 中的 ``tool`` 字段分发到不同工具函数执行。
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from execution import run_python_code


class ToolRouter:
    """根据 JSON 指令路由并执行不同工具。"""

    def __init__(self) -> None:
        """初始化内存 SQLite 数据库连接。"""
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row

    def execute_python(self, code: str) -> str:
        """执行 Python 代码并返回标准输出。"""
        if not isinstance(code, str) or not code.strip():
            raise ValueError("python 工具参数错误: code 必须是非空字符串")

        try:
            return run_python_code(code)
        except RuntimeError as exc:
            raise RuntimeError(f"python 执行异常: {exc}") from exc

    def execute_sql(self, query: str) -> dict[str, Any]:
        """在内存数据库中执行 SQL 并返回结果。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("sql 工具参数错误: query 必须是非空字符串")

        try:
            cursor = self._conn.cursor()
            cursor.execute(query)

            if cursor.description is not None:
                rows = cursor.fetchall()
                return {
                    "type": "select",
                    "columns": [col[0] for col in cursor.description],
                    "rows": [dict(row) for row in rows],
                }

            self._conn.commit()
            return {
                "type": "mutation",
                "rowcount": cursor.rowcount,
                "message": "SQL 执行成功",
            }
        except sqlite3.Error as exc:
            raise RuntimeError(f"sql 执行异常: {exc}") from exc

    def call_api(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """模拟 API 调用，返回固定结构的 mock 数据。"""
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("api 工具参数错误: endpoint 必须是非空字符串")
        if params is not None and not isinstance(params, dict):
            raise ValueError("api 工具参数错误: params 必须是对象")

        return {
            "status": "success",
            "endpoint": endpoint,
            "params": params or {},
            "data": {
                "message": "This is mocked API response",
                "code": 200,
            },
        }

    def route(self, json_str: str) -> dict[str, Any]:
        """解析 JSON 并根据 tool 字段分发执行。"""
        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"JSON 解析失败: {exc}"}

        if not isinstance(payload, dict):
            return {"ok": False, "error": "请求体必须是 JSON 对象"}

        tool = payload.get("tool")
        args = payload.get("args", {})

        if not isinstance(args, dict):
            return {"ok": False, "error": "args 必须是对象"}

        try:
            if tool == "python":
                result = self.execute_python(args.get("code"))
            elif tool == "sql":
                result = self.execute_sql(args.get("query"))
            elif tool == "api":
                result = self.call_api(args.get("endpoint"), args.get("params"))
            else:
                return {"ok": False, "error": f"不支持的工具: {tool}"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

        return {"ok": True, "tool": tool, "result": result}

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()


__all__ = ["ToolRouter"]
