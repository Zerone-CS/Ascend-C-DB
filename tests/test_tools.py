import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent.tools import get_tool_definitions, handle_tool_call


def test_tool_definitions_schema():
    tools = get_tool_definitions()
    assert len(tools) == 1
    tool = tools[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "search_ascend_docs"
    params = tool["function"]["parameters"]
    assert "query" in params["properties"]
    assert "query" in params["required"]


def test_handle_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool"):
        handle_tool_call("nonexistent_tool", {})
