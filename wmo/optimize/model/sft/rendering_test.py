"""Tests for canonical context-target SFT rendering."""

from __future__ import annotations

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.models import AssistantAction, ToolCall
from wmo.optimize.model.sft.contracts import (
    AssistantActionEvent,
    RolloutExampleSource,
    SFTExample,
    SFTMessage,
    ToolEvent,
    TraceExampleSource,
)
from wmo.optimize.model.sft.rendering import context_target_fingerprint

_DIGEST = "a" * 64


def _action() -> AssistantAction:
    return AssistantAction(
        content="I will look up the order, then issue the refund.",
        tool_calls=(
            ToolCall(call_id="lookup-1", name="lookup_order", arguments={"order_id": "o-17"}),
            ToolCall(call_id="refund-1", name="issue_refund", arguments={"order_id": "o-17"}),
        ),
    )


def test_source_only_approval_does_not_change_normalized_fingerprint() -> None:
    """Eligibility metadata cannot stop identical model-visible examples from being deduplicated."""
    action = AssistantAction(content="Investigating the account.")
    approved_history = (AssistantActionEvent(action=action, approved=True),)
    unapproved_history = (AssistantActionEvent(action=action, approved=False),)

    assert context_target_fingerprint(
        task="Check the account.", history=approved_history, target=_action()
    ) == context_target_fingerprint(
        task="Check the account.", history=unapproved_history, target=_action()
    )


def test_raw_tool_argument_formatting_does_not_change_normalized_fingerprint() -> None:
    """Provider whitespace and key order cannot split equivalent rows across partitions."""
    compact = AssistantAction(
        tool_calls=(
            ToolCall(
                call_id="lookup-1",
                name="lookup_order",
                arguments={"limit": 1, "order_id": "o-17"},
                raw_arguments='{"limit":1,"order_id":"o-17"}',
            ),
        )
    )
    provider_order = AssistantAction(
        tool_calls=(
            ToolCall(
                call_id="lookup-1",
                name="lookup_order",
                arguments={"limit": 1, "order_id": "o-17"},
                raw_arguments='{ "order_id": "o-17", "limit": 1 }',
            ),
        )
    )

    assert context_target_fingerprint(task="Look up the order.", history=(), target=compact) == (
        context_target_fingerprint(task="Look up the order.", history=(), target=provider_order)
    )
    assert compact.tool_calls[0].arguments_json() == '{"limit":1,"order_id":"o-17"}'
    assert provider_order.tool_calls[0].arguments_json() == ('{ "order_id": "o-17", "limit": 1 }')


def test_structured_example_round_trips_with_complete_target_and_tool_context() -> None:
    """The frozen SFT example contract preserves production-style structure losslessly."""
    example = SFTExample(
        example_id="sft-example-1",
        leakage_group_id="sft-lineage-1",
        task="Refund order o-17.",
        history=(
            SFTMessage(role="user", content="Please refund my order."),
            ToolEvent(tool_call_id="lookup-1", tool_name="lookup_order", content="eligible"),
        ),
        target=_action(),
        source=TraceExampleSource(
            trace_id="trace-1",
            acceptance_evidence=ArtifactInput(
                artifact_id="production-evidence-1",
                sha256=_DIGEST,
            ),
        ),
        source_step_index=1,
    )

    assert SFTExample.model_validate_json(example.model_dump_json()) == example


def test_teacher_structured_example_round_trips_with_complete_target_and_tool_context() -> None:
    """The frozen SFT example contract preserves teacher rollout structure losslessly."""
    example = SFTExample(
        example_id="sft-example-teacher-1",
        leakage_group_id="sft-lineage-task-1",
        task="Refund order o-17.",
        history=(
            SFTMessage(role="user", content="Please refund my order."),
            AssistantActionEvent(action=AssistantAction(content="I will verify eligibility.")),
            ToolEvent(tool_call_id="lookup-1", tool_name="lookup_order", content="eligible"),
        ),
        target=_action(),
        source=RolloutExampleSource(
            rollout_id="rollout-1",
            acceptance_evidence=ArtifactInput(
                artifact_id="teacher-evidence-1",
                sha256=_DIGEST,
            ),
        ),
        source_step_index=3,
        score=1.0,
    )

    assert SFTExample.model_validate_json(example.model_dump_json()) == example
