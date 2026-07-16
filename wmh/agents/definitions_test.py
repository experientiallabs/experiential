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
    assert "within the first 12 read_file calls" in meta.system_prompt().lower()
    assert "durable checkpoints" in meta.system_prompt().lower()
    assert {surface.path: surface.content for surface in default.code_files()} == {
        surface.path: surface.content for surface in meta.code_files()
    }


def test_meta_agent_has_larger_budgets_without_mutating_default() -> None:
    """Project exploration gets its own turn and model-output budgets on its own HarnessDoc."""
    default = default_agent("default")
    meta = meta_agent("meta")

    assert default.max_turns() == 20
    assert meta.max_turns() == 60
    assert default.max_output_tokens() == 4096
    assert meta.max_output_tokens() == 16384
    assert default.surface("param:max-output-tokens") is not None
    assert meta.surface("param:max-output-tokens") is not None


def test_meta_agent_uses_only_project_scoped_tools() -> None:
    """The optimizer agent gets only contained project-file tools plus submit."""
    assert meta_agent().tools() == ["read_file", "write_file", "submit"]
