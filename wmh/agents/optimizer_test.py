"""Tests for the built-in complete-source optimization agent."""

from wmh.agents.default import default_agent
from wmh.agents.optimizer import optimizer_agent


def test_optimizer_agent_is_a_contained_complete_source_editor() -> None:
    agent = optimizer_agent()

    assert agent.runtime_kind() == "pi-node"
    assert agent.tools() == ["bash", "read_file", "submit"]
    assert agent.max_turns() == 60
    assert agent.max_output_tokens() == 16384
    prompt = agent.system_prompt().lower()
    assert "complete harness source tree" in prompt
    assert "do not solve" in prompt
    assert "parent" not in prompt
    assert {surface.path: surface.content for surface in agent.code_files()} == {
        surface.path: surface.content for surface in default_agent().code_files()
    }
