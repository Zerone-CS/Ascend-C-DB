import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.ascend_agent import AscendCAgent
from agent.prompts import SYSTEM_PROMPT


def test_agent_init():
    agent = AscendCAgent(
        api_key="test",
        base_url="http://localhost:9999/v1",
        model="test-model",
    )
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"
    assert agent.mode == "tool_call"


def test_agent_reset():
    agent = AscendCAgent(
        api_key="test",
        base_url="http://localhost:9999/v1",
        model="test-model",
    )
    agent.messages.append({"role": "user", "content": "hello"})
    assert len(agent.messages) == 2
    agent.reset()
    assert len(agent.messages) == 1


def test_system_prompt_has_key_concepts():
    for keyword in ["Ascend C", "Tiling", "DoubleBuffer", "SPMD", "search_ascend_docs"]:
        assert keyword in SYSTEM_PROMPT, "Missing: {}".format(keyword)
