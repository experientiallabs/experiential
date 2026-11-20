"""Behavior tests for frontier probe generation over a scripted model client."""

from __future__ import annotations

import json

import pytest

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
from exp.simulation.synthesis import (
    FRONTIER_PROBE_PROMPT,
    FrontierProbe,
    FrontierProbeError,
    generate_frontier_probes,
)

_DIGEST = "a" * 64


def _model() -> ModelSnapshot:
    """Return one deterministic generator model identity."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="scripted",
        model_id="probe-generator",
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


def _probe_payload(count: int) -> str:
    """Return a contract-valid JSON array with the given probe count."""
    return json.dumps(
        [
            {"task": f"Probe scenario {index}.", "rationale": f"Frontier reason {index}."}
            for index in range(count)
        ]
    )


def test_generates_probes_from_the_observed_sample() -> None:
    """One call sends the frontier prompt plus the sample and parses the reply."""
    client = _ScriptedClient(_probe_payload(2))
    batch = generate_frontier_probes(
        client,
        ("Reset a billing password.", "Merge duplicate accounts."),
        probe_count=2,
    )
    assert [probe.task for probe in batch.probes] == ["Probe scenario 0.", "Probe scenario 1."]
    assert batch.model.model_id == "probe-generator"
    (request,) = client.requests
    assert request.messages[0].role == "system"
    assert request.messages[0].content == FRONTIER_PROBE_PROMPT
    assert request.messages[1].role == "user"
    assert "Reset a billing password." in str(request.messages[1].content)
    assert '"probe_count":2' in str(request.messages[1].content)
    assert request.maximum_output_tokens == 4_000


def test_truncates_an_overlong_reply_to_the_requested_count() -> None:
    """A reply with extra probes is cut to the requested count, never widened."""
    client = _ScriptedClient(_probe_payload(4))
    batch = generate_frontier_probes(client, ("Observed task.",), probe_count=2)
    assert len(batch.probes) == 2


def test_accepts_a_fenced_json_reply_and_a_short_batch() -> None:
    """A single json-fenced reply parses, and fewer probes than asked is valid."""
    client = _ScriptedClient(f"```json\n{_probe_payload(1)}\n```")
    batch = generate_frontier_probes(client, ("Observed task.",), probe_count=3)
    assert len(batch.probes) == 1


def test_task_sha256_is_content_derived() -> None:
    """Identical probe text yields the identical digest, so re-generation converges."""
    first = FrontierProbe(task="Same probe.", rationale="Reason one.")
    second = FrontierProbe(task="Same probe.", rationale="Reason two.")
    assert first.task_sha256 == second.task_sha256
    assert len(first.task_sha256) == 64


def test_rejects_non_json_output() -> None:
    """Prose instead of JSON fails loudly instead of yielding zero probes."""
    client = _ScriptedClient("Here are some ideas you might like.")
    with pytest.raises(FrontierProbeError, match="non-JSON"):
        generate_frontier_probes(client, ("Observed task.",))


def test_rejects_a_json_object_reply() -> None:
    """A JSON object where the array should be breaks the contract."""
    client = _ScriptedClient(json.dumps({"task": "x", "rationale": "y"}))
    with pytest.raises(FrontierProbeError, match="not an array"):
        generate_frontier_probes(client, ("Observed task.",))


def test_rejects_a_probe_outside_the_contract() -> None:
    """A probe entry missing its rationale breaks the contract."""
    client = _ScriptedClient(json.dumps([{"task": "Probe without rationale."}]))
    with pytest.raises(FrontierProbeError, match="contracted shape"):
        generate_frontier_probes(client, ("Observed task.",))


def test_rejects_whitespace_only_probe_text_and_trims_edges() -> None:
    """Blank-looking probe fields break the contract; edge whitespace is trimmed."""
    blank = _ScriptedClient(json.dumps([{"task": "   ", "rationale": "Reason."}]))
    with pytest.raises(FrontierProbeError, match="contracted shape"):
        generate_frontier_probes(blank, ("Observed task.",))
    padded = _ScriptedClient(json.dumps([{"task": "  Probe.  ", "rationale": " Reason. "}]))
    (probe,) = generate_frontier_probes(padded, ("Observed task.",)).probes
    assert probe.task == "Probe."
    assert probe.rationale == "Reason."


def test_rejects_a_tool_call_only_reply() -> None:
    """A reply with tool calls and no text cannot carry probes."""
    client = _ScriptedClient(
        None,
        tool_calls=(ToolCall(call_id="call-1", name="noise", arguments={}),),
    )
    with pytest.raises(FrontierProbeError, match="no text"):
        generate_frontier_probes(client, ("Observed task.",))


def test_rejects_a_reply_truncated_at_the_token_limit() -> None:
    """A length-limited reply is refused before parsing half a JSON array."""
    client = _ScriptedClient(_probe_payload(1), finish_reason=ModelFinishReason.LENGTH)
    with pytest.raises(FrontierProbeError, match="output-token limit"):
        generate_frontier_probes(client, ("Observed task.",))


def test_rejects_an_empty_observed_sample() -> None:
    """Probing the frontier of nothing is a caller bug."""
    client = _ScriptedClient(_probe_payload(1))
    with pytest.raises(ValueError, match="at least one observed task"):
        generate_frontier_probes(client, ())


def test_rejects_empty_tasks_and_non_positive_counts() -> None:
    """Empty sample entries and non-positive knobs fail before any model call."""
    client = _ScriptedClient(_probe_payload(1))
    with pytest.raises(ValueError, match="non-empty"):
        generate_frontier_probes(client, ("",))
    with pytest.raises(ValueError, match="probe_count"):
        generate_frontier_probes(client, ("Observed task.",), probe_count=0)
    with pytest.raises(ValueError, match="maximum_output_tokens"):
        generate_frontier_probes(client, ("Observed task.",), maximum_output_tokens=0)
    assert client.requests == []
