"""Bounds of the agent memory spec."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.optimize.evaluation.learning.agent_memory_contracts import AgentMemorySpec


def test_one_trace_and_one_message_must_fit_their_enclosing_bounds() -> None:
    """A trace larger than a request, or a message larger than a trace, is refused."""
    base = {"traces": Path("t.jsonl"), "output": Path("out"), "alias": "learner"}
    AgentMemorySpec.model_validate(base)
    with pytest.raises(ValidationError, match="maximum_trace_bytes must not exceed batch_bytes"):
        AgentMemorySpec.model_validate(
            {**base, "batch_bytes": 10_000, "maximum_trace_bytes": 20_000}
        )
    with pytest.raises(ValidationError, match="maximum_message_bytes must not exceed"):
        AgentMemorySpec.model_validate(
            {**base, "maximum_trace_bytes": 2_000, "maximum_message_bytes": 3_000}
        )
