"""Behavior tests for the two-axis realism judge over a scripted model client."""

from __future__ import annotations

import json

import pytest

from exp.common.judging.realism import (
    REALISM_PROMPT,
    RealismJudge,
    RealismJudgmentError,
)
from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelFinishReason,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    ToolCall,
)

_DIGEST = "b" * 64


def _model() -> ModelSnapshot:
    """Return one deterministic judge model identity."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="scripted",
        model_id="realism-judge",
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )


class _ScriptedClient:
    """Deterministic client returning one preconfigured completion."""

    def __init__(
        self,
        content: str | None,
        *,
        tool_calls: tuple[ToolCall, ...] = (),
        finish_reason: ModelFinishReason = ModelFinishReason.COMPLETED,
    ) -> None:
        """Store the scripted reply and start capturing requests."""
        self._content = content
        self._tool_calls = tool_calls
        self._finish_reason = finish_reason
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Capture the request and return the scripted completion."""
        self.requests.append(request)
        return ModelResponse(
            output=AssistantAction(content=self._content, tool_calls=self._tool_calls),
            model=_model(),
            economics=OperationEconomics(),
            finish_reason=self._finish_reason,
        )


def _verdict(likelihood: float, feasibility: float) -> str:
    """Return a contract-valid verdict payload."""
    return json.dumps(
        {
            "likelihood": likelihood,
            "feasibility": feasibility,
            "rationale": "Rare but entirely possible for this agent.",
        }
    )


def test_keeps_the_two_axes_separate() -> None:
    """A rare-but-real verdict keeps low likelihood beside high feasibility."""
    client = _ScriptedClient(_verdict(0.05, 0.95))
    assessment = RealismJudge(client).assess("A customer asks to merge 400 accounts at once.")
    assert assessment.likelihood == 0.05
    assert assessment.feasibility == 0.95
    assert assessment.rationale


def test_sends_the_anti_conflation_prompt_and_the_scenario_verbatim() -> None:
    """The system prompt forbids conflating rare with impossible; the task rides as is."""
    client = _ScriptedClient(_verdict(0.5, 0.5))
    RealismJudge(client, maximum_output_tokens=123).assess("Scenario text.")
    (request,) = client.requests
    assert request.messages[0].role == "system"
    assert request.messages[0].content == REALISM_PROMPT
    assert "never call it infeasible because it is rare" in str(request.messages[0].content)
    assert request.messages[1].content == "Scenario text."
    assert request.maximum_output_tokens == 123


def test_accepts_a_fenced_json_verdict() -> None:
    """A single json-fenced verdict parses like a bare one."""
    client = _ScriptedClient(f"```json\n{_verdict(0.2, 0.8)}\n```")
    assessment = RealismJudge(client).assess("Scenario text.")
    assert assessment.likelihood == 0.2


def test_rejects_non_json_output() -> None:
    """Prose instead of JSON fails loudly instead of scoring nothing."""
    client = _ScriptedClient("I would rate this fairly unlikely overall.")
    with pytest.raises(RealismJudgmentError, match="non-JSON"):
        RealismJudge(client).assess("Scenario text.")


def test_rejects_a_verdict_outside_the_contract() -> None:
    """Out-of-range axes and missing fields break the contract."""
    out_of_range = _ScriptedClient(
        json.dumps({"likelihood": 1.5, "feasibility": 0.5, "rationale": "x"})
    )
    with pytest.raises(RealismJudgmentError, match="contracted shape"):
        RealismJudge(out_of_range).assess("Scenario text.")
    combined_score = _ScriptedClient(json.dumps({"realism": 0.4, "rationale": "x"}))
    with pytest.raises(RealismJudgmentError, match="contracted shape"):
        RealismJudge(combined_score).assess("Scenario text.")
    blank_rationale = _ScriptedClient(
        json.dumps({"likelihood": 0.4, "feasibility": 0.6, "rationale": "   "})
    )
    with pytest.raises(RealismJudgmentError, match="contracted shape"):
        RealismJudge(blank_rationale).assess("Scenario text.")


def test_rejects_a_tool_call_only_reply() -> None:
    """A reply with tool calls and no text cannot carry a verdict."""
    client = _ScriptedClient(
        None,
        tool_calls=(ToolCall(call_id="call-1", name="noise", arguments={}),),
    )
    with pytest.raises(RealismJudgmentError, match="no text"):
        RealismJudge(client).assess("Scenario text.")


def test_rejects_a_reply_truncated_at_the_token_limit() -> None:
    """A length-limited reply is refused before parsing a half verdict."""
    client = _ScriptedClient(_verdict(0.4, 0.6), finish_reason=ModelFinishReason.LENGTH)
    with pytest.raises(RealismJudgmentError, match="output-token limit"):
        RealismJudge(client).assess("Scenario text.")


def test_rejects_empty_scenario_text_and_bad_token_limits() -> None:
    """Empty scenarios and non-positive token limits fail before any model call."""
    client = _ScriptedClient(_verdict(0.4, 0.6))
    with pytest.raises(ValueError, match="non-empty scenario"):
        RealismJudge(client).assess("")
    with pytest.raises(ValueError, match="maximum_output_tokens"):
        RealismJudge(client, maximum_output_tokens=0)
    assert client.requests == []
