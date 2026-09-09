"""Behavior tests for the one-shot answer-grading judge over a scripted client."""

from __future__ import annotations

import json

import pytest

from exp.common.judging.grading import (
    GRADING_PROMPT,
    AnswerGradeError,
    AnswerGradeJudge,
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

_DIGEST = "c" * 64


def _model() -> ModelSnapshot:
    """Return one deterministic judge model identity."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="scripted",
        model_id="grading-judge",
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


def _verdict(score: float) -> str:
    """Return a contract-valid grade payload."""
    return json.dumps({"score": score, "rationale": "Covers the task with one small gap."})


def test_grades_one_answer_against_one_task() -> None:
    """A contract-valid verdict yields the parsed score and rationale."""
    client = _ScriptedClient(_verdict(0.5))
    grade = AnswerGradeJudge(client).grade(
        task="Reset the billing password.",
        answer="Open Settings, choose Billing, then Reset password.",
    )
    assert grade.score == 0.5
    assert grade.rationale == "Covers the task with one small gap."


def test_sends_the_completion_only_prompt_and_both_texts() -> None:
    """The prompt forbids style judging; task and answer ride labeled and verbatim."""
    client = _ScriptedClient(_verdict(1.0))
    AnswerGradeJudge(client, maximum_output_tokens=321).grade(
        task="Task text.",
        answer="Answer text.",
    )
    (request,) = client.requests
    assert request.messages[0].role == "system"
    assert request.messages[0].content == GRADING_PROMPT
    assert "Judge task completion only" in str(request.messages[0].content)
    assert (
        request.messages[1].content
        == "SCENARIO TASK:\nTask text.\n\nCANDIDATE ANSWER:\nAnswer text."
    )
    assert request.maximum_output_tokens == 321


def test_accepts_a_fenced_json_verdict_and_integer_scores() -> None:
    """A single json-fenced verdict parses, and integer 0 and 1 scores are valid."""
    client = _ScriptedClient(f"```json\n{json.dumps({'score': 1, 'rationale': 'Complete.'})}\n```")
    assert AnswerGradeJudge(client).grade(task="Task.", answer="Answer.").score == 1.0


def test_rejects_non_json_output() -> None:
    """Prose instead of JSON fails loudly instead of scoring nothing."""
    client = _ScriptedClient("I would give this about half marks.")
    with pytest.raises(AnswerGradeError, match="non-JSON"):
        AnswerGradeJudge(client).grade(task="Task.", answer="Answer.")


def test_rejects_a_grade_outside_the_contract() -> None:
    """Out-of-range scores and blank rationales break the contract."""
    out_of_range = _ScriptedClient(json.dumps({"score": 1.5, "rationale": "x"}))
    with pytest.raises(AnswerGradeError, match="contracted shape"):
        AnswerGradeJudge(out_of_range).grade(task="Task.", answer="Answer.")
    blank_rationale = _ScriptedClient(json.dumps({"score": 0.5, "rationale": "   "}))
    with pytest.raises(AnswerGradeError, match="contracted shape"):
        AnswerGradeJudge(blank_rationale).grade(task="Task.", answer="Answer.")


def test_rejects_a_tool_call_only_reply() -> None:
    """A reply with tool calls and no text cannot carry a verdict."""
    client = _ScriptedClient(
        None,
        tool_calls=(ToolCall(call_id="call-1", name="noise", arguments={}),),
    )
    with pytest.raises(AnswerGradeError, match="no text"):
        AnswerGradeJudge(client).grade(task="Task.", answer="Answer.")


def test_rejects_a_reply_truncated_at_the_token_limit() -> None:
    """A length-limited reply is refused before parsing a half verdict."""
    client = _ScriptedClient(_verdict(0.5), finish_reason=ModelFinishReason.LENGTH)
    with pytest.raises(AnswerGradeError, match="output-token limit"):
        AnswerGradeJudge(client).grade(task="Task.", answer="Answer.")


def test_rejects_blank_inputs_and_bad_token_limits() -> None:
    """Blank task or answer text and non-positive limits fail before any model call."""
    client = _ScriptedClient(_verdict(0.5))
    with pytest.raises(ValueError, match="scenario task"):
        AnswerGradeJudge(client).grade(task="   ", answer="Answer.")
    with pytest.raises(ValueError, match="candidate answer"):
        AnswerGradeJudge(client).grade(task="Task.", answer="")
    with pytest.raises(ValueError, match="maximum_output_tokens"):
        AnswerGradeJudge(client, maximum_output_tokens=0)
    assert client.requests == []
