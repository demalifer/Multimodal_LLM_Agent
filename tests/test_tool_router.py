from multimodal_agent.tools.tool_router import ToolRouter


def test_route_python() -> None:
    router = ToolRouter()
    try:
        result = router.route('{"tool":"python","args":{"code":"print(3)"}}')
        assert result["ok"] is True
        assert result["result"] == "3"
    finally:
        router.close()


def test_route_sql_select() -> None:
    router = ToolRouter()
    try:
        router.route('{"tool":"sql","args":{"query":"create table t(id int);"}}')
        router.route('{"tool":"sql","args":{"query":"insert into t values(1);"}}')
        result = router.route('{"tool":"sql","args":{"query":"select * from t;"}}')
        assert result["ok"] is True
        assert result["result"]["type"] == "select"
        assert result["result"]["rows"] == [{"id": 1}]
    finally:
        router.close()


def test_route_unknown_tool() -> None:
    router = ToolRouter()
    try:
        result = router.route('{"tool":"abc","args":{}}')
        assert result["ok"] is False
    finally:
        router.close()
