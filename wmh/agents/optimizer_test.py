"""Tests for the complete-source optimizer persona."""

from wmh.agents.optimizer import optimizer_agent


def test_optimizer_agent_locks_tools_budgets_and_the_contract() -> None:
    doc = optimizer_agent()
    assert doc.tools() == ["bash", "read_file", "submit"]
    assert doc.max_turns() == 60
    assert doc.max_output_tokens() == 16384
    prompt = doc.system_prompt()
    # Two load-bearing phrases: the anti-overfitting contract and the filename grammar.
    assert "Never hard-code" in prompt
    assert "kebab-case" in prompt
