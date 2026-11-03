"""Refusal feedback keeps the original request and never modifies a saved proposal."""

from exp.common.core.artifacts import JsonObject
from exp.simulation.engines.reasks import reask_proposal


def test_reask_retains_schema_and_original_messages() -> None:
    """Only appended feedback distinguishes the next purchased proposal."""
    body: JsonObject = {"messages": [{"role": "user", "content": "act"}], "tools": []}
    raw: JsonObject = {"output": {"text": "invalid"}}
    result = reask_proposal(body, raw, "bad path")
    assert result["tools"] == body["tools"]
    assert isinstance(result["messages"], list) and len(result["messages"]) == 3
    assert len(body["messages"]) == 1 and raw["output"] == {"text": "invalid"}
