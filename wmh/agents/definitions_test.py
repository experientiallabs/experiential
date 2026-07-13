"""Tests for the built-in default and meta agent definitions."""

from wmh.agents.default import default_agent
from wmh.agents.meta import meta_agent


def test_default_and_meta_agents_are_independent_pi_documents() -> None:
    """Both agents share the pinned pi source while owning separate prompts."""
    default = default_agent("default")
    meta = meta_agent("meta")

    assert default.runtime_kind() == meta.runtime_kind() == "pi-node"
    assert default.system_prompt() != meta.system_prompt()
    assert "optimizer project" in meta.system_prompt().lower()
    assert {surface.path: surface.content for surface in default.code_files()} == {
        surface.path: surface.content for surface in meta.code_files()
    }


def test_meta_agent_has_a_larger_turn_budget_without_mutating_default() -> None:
    """Project exploration gets its own budget on its own HarnessDoc."""
    default = default_agent("default")
    meta = meta_agent("meta")

    assert default.max_turns() == 20
    assert meta.max_turns() == 60


def test_meta_agent_uses_only_project_scoped_file_tools() -> None:
    """The optimizer agent has no shell escape from its persistent workspace."""
    assert meta_agent().tools() == ["read_file", "write_file", "submit"]
