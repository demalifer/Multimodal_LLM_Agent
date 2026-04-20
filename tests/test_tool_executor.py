from multimodal_agent.tools.tool_executor import execute_tool


def test_execute_python_success() -> None:
    output = execute_tool('{"tool": "python", "code": "print(1+1)"}')
    assert output == "2"


def test_execute_tool_invalid_tool() -> None:
    output = execute_tool('{"tool": "sql", "code": "select 1"}')
    assert "不支持的工具" in output


def test_execute_python_restricted_builtins() -> None:
    output = execute_tool('{"tool": "python", "code": "print(open)"}')
    assert "执行异常" in output
