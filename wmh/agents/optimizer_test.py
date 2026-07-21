"""Tests for the built-in complete-source optimization agent."""

from wmh.agents.default import default_agent
from wmh.agents.optimizer import optimizer_agent


def test_optimizer_agent_is_a_contained_complete_source_editor() -> None:
    agent = optimizer_agent()

    assert agent.runtime_kind() == "pi-node"
    assert agent.tools() == ["bash", "read_file", "submit"]
    assert agent.max_turns() == 60
    assert agent.max_output_tokens() == 16384
    prompt = " ".join(agent.system_prompt().lower().split())
    assert "complete harness source tree" in prompt
    assert "do not solve" in prompt
    assert "preinitialized output directory" in prompt
    assert "empty output directory" not in prompt
    assert "general-purpose harness mechanisms" in prompt
    assert "particular evaluation instances" in prompt
    assert "literal instance names or identifiers" in prompt
    assert "instance-specific strings" in prompt
    assert "expected answers" in prompt
    assert "fixture details" in prompt
    assert "special-case branches" in prompt
    assert "many unfamiliar tasks" in prompt
    assert "complete portable source remains freely rewritable" in prompt
    for candidate_surface in (
        "path",
        "filename",
        "source file",
        "prompt",
        "comment",
        "skill",
        "configuration",
        "test",
    ):
        assert candidate_surface in prompt
    assert "parent" not in prompt
    assert {surface.path: surface.content for surface in agent.code_files()} == {
        surface.path: surface.content for surface in default_agent().code_files()
    }
