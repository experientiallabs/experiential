"""Judge-visible rollout evidence and the shared judge output-token budget.

Every LM judge renderer builds its rollout section from :func:`visible_rollout_evidence`, so the
judge sees exactly one documented evidence contract: the task framing, each span's externally
visible outputs, tool calls, tool responses, structured failures, and the final output. Internal
provider request payloads (the per-turn message histories repeated inside every model-call span)
and candidate reasoning or chain-of-thought content never reach the judge. The projection is a
pure function of the immutable rollout, so rendered requests stay deterministic and digest-stable.
"""

from __future__ import annotations

from typing import Final, cast

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject
from wmo.common.rollouts import RolloutArtifact, RolloutSpan

DEFAULT_JUDGE_OUTPUT_TOKENS: Final = 16_384
"""Per-call output-token budget reserved for every LM judge dispatch.

Reasoning-effort judge models can spend thousands of tokens on hidden reasoning before any
visible text, so the reservation leaves room for both reasoning and the structured verdict.
"""

_HIDDEN_EVIDENCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "request",
        "gen_ai.input.messages",
        "reasoning",
        "reasoning_content",
        "redacted_thinking",
        "thinking",
        "chain_of_thought",
    }
)
"""JSON keys whose values are internal provider input or hidden model reasoning.

``request`` carries the complete provider request for one model-call span, including the whole
prior message history, so a ten-turn rollout would repeat its transcript quadratically.
``gen_ai.input.messages`` is the equivalent per-turn input history on normalized production
trace spans. The remaining keys name provider reasoning containers that must never be shown to
a judge scoring externally visible behavior.
"""


def visible_rollout_evidence(rollout: RolloutArtifact) -> JsonObject:
    """Project one verified rollout onto its judge-visible evidence contract.

    Args:
        rollout: Verified immutable rollout to render for a judge.

    Returns:
        Deterministic payload with rollout identity, stop reason, the initial task framing
        messages, per-span visible evidence, and the final output. Provider request payloads
        and reasoning content are excluded by :data:`_HIDDEN_EVIDENCE_KEYS`.
    """
    return {
        "rollout_id": rollout.rollout_id,
        "task_id": rollout.task_id,
        "stop_reason": rollout.stop_reason.value,
        "task_context": _task_context(rollout.spans),
        "final_output": (
            rollout.final_output.model_dump(mode="json")
            if rollout.final_output is not None
            else None
        ),
        "spans": [_visible_span_evidence(span) for span in rollout.spans],
    }


def _visible_span_evidence(span: RolloutSpan) -> JsonObject:
    """Return the externally visible evidence for one rollout span.

    Args:
        span: One ordered immutable rollout span.

    Returns:
        Span identity, kind, tool name, structured failure, and the span payload with every
        hidden provider-input and reasoning key removed recursively.
    """
    return {
        "span_id": span.span_id,
        "kind": span.kind.value,
        "tool_name": span.tool_name,
        "payload": _without_hidden_evidence(cast(JsonValue, span.payload)),
        "failure": span.failure.model_dump(mode="json") if span.failure is not None else None,
    }


def _task_context(spans: tuple[RolloutSpan, ...]) -> list[JsonValue]:
    """Return the initial task framing messages shown before any assistant output.

    Model-call spans keep the task text only inside their excluded provider request payloads,
    so the seed system and user messages of the first recorded request are retained once here.
    Production trace spans carry their task text in retained span attributes instead and yield
    an empty framing list.

    Args:
        spans: Ordered immutable rollout spans.

    Returns:
        Messages from the first recorded request up to the first assistant message, with
        hidden keys removed, or an empty list when no span records a request.
    """
    for span in spans:
        request = span.payload.get("request")
        if not isinstance(request, dict):
            continue
        messages = request.get("messages")
        if not isinstance(messages, list):
            continue
        framing: list[JsonValue] = []
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "assistant":
                break
            framing.append(_without_hidden_evidence(message))
        return framing
    return []


def _without_hidden_evidence(value: JsonValue) -> JsonValue:
    """Remove every hidden provider-input and reasoning key from one JSON value.

    Args:
        value: Arbitrary JSON value from an immutable span payload.

    Returns:
        The value with :data:`_HIDDEN_EVIDENCE_KEYS` removed from every nested object,
        preserving key and item order for deterministic rendering.
    """
    if isinstance(value, dict):
        return {
            key: _without_hidden_evidence(item)
            for key, item in value.items()
            if key not in _HIDDEN_EVIDENCE_KEYS
        }
    if isinstance(value, list):
        return [_without_hidden_evidence(item) for item in value]
    return value
