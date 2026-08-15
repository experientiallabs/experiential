"""Canonical source-independent rendering and hashing for structured SFT examples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import Field

from wmo.common.core.artifacts import (
    ContractModel,
    Sha256,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
    sha256_json,
)
from wmo.common.models import AssistantAction
from wmo.optimize.model.sft.contracts import (
    AssistantActionEvent,
    PartitionedSFTExample,
    SFTContextEvent,
    SFTMessage,
    ToolEvent,
)


class CanonicalSFTMessage(ContractModel):
    """One canonical non-assistant message in a normalized SFT context."""

    kind: Literal["message"] = "message"
    role: Literal["system", "user", "observation"]
    content: str


class CanonicalSFTAssistantAction(ContractModel):
    """One canonical prior assistant action without source-only approval metadata."""

    kind: Literal["assistant"] = "assistant"
    action: AssistantAction


class CanonicalSFTToolEvent(ContractModel):
    """One canonical tool result in a normalized SFT context."""

    kind: Literal["tool"] = "tool"
    tool_call_id: str
    content: str
    tool_name: str | None = None


CanonicalSFTContextEvent = Annotated[
    CanonicalSFTMessage | CanonicalSFTAssistantAction | CanonicalSFTToolEvent,
    Field(discriminator="kind"),
]


class CanonicalSFTTurn(ContractModel):
    """The source-independent task, ordered context, and complete assistant-action target."""

    task: str
    history: tuple[CanonicalSFTContextEvent, ...]
    target: AssistantAction


def render_context_target(
    *, task: str, history: tuple[SFTContextEvent, ...], target: AssistantAction
) -> str:
    """Render one complete context and target as deterministic JSON.

    Source-only approval metadata is intentionally absent from the rendering. The target retains
    the whole `AssistantAction`, including text and all ordered tool calls.

    Args:
        task: Task text supplied with the source trace or rollout.
        history: Ordered previous messages, assistant actions, and tool results.
        target: The complete next assistant action to learn.

    Returns:
        Canonical compact JSON that round-trips through `parse_rendered_turn`.
    """
    return canonical_json_bytes(_canonical_turn(task=task, history=history, target=target)).decode(
        "utf-8"
    )


def parse_rendered_turn(rendered: str) -> CanonicalSFTTurn:
    """Parse one canonical context-target rendering without losing action or tool-call structure."""
    return CanonicalSFTTurn.model_validate_json(rendered)


def context_target_fingerprint(
    *, task: str, history: tuple[SFTContextEvent, ...], target: AssistantAction
) -> Sha256:
    """Return the stable SHA-256 fingerprint used for lineage union and deduplication."""
    return sha256_json(_canonical_turn(task=task, history=history, target=target))


def canonical_partitioned_rows_jsonl(rows: Sequence[PartitionedSFTExample]) -> bytes:
    """Render ordered partitioned SFT rows as deterministic newline-terminated JSONL.

    Args:
        rows: Already ordered examples to serialize into the frozen dataset payload.

    Returns:
        Canonical JSONL bytes, or empty bytes when no examples are present.
    """
    return canonical_jsonl_bytes(rows)


def partitioned_rows_sha256(rows: Sequence[PartitionedSFTExample]) -> Sha256:
    """Return the SHA-256 digest of persisted canonical partitioned SFT JSONL rows."""
    return sha256_bytes(canonical_partitioned_rows_jsonl(rows))


def _canonical_turn(
    *, task: str, history: tuple[SFTContextEvent, ...], target: AssistantAction
) -> CanonicalSFTTurn:
    """Build one typed canonical turn from source-visible context and its assistant target."""
    return CanonicalSFTTurn(
        task=task,
        history=tuple(_canonical_event(event) for event in history),
        target=target,
    )


def _canonical_event(event: SFTContextEvent) -> CanonicalSFTContextEvent:
    """Remove source-only approval metadata while preserving one context event exactly."""
    if isinstance(event, SFTMessage):
        return CanonicalSFTMessage(role=event.role, content=event.content)
    if isinstance(event, AssistantActionEvent):
        return CanonicalSFTAssistantAction(action=event.action)
    if isinstance(event, ToolEvent):
        return CanonicalSFTToolEvent(
            tool_call_id=event.tool_call_id,
            content=event.content,
            tool_name=event.tool_name,
        )
    raise ValueError("canonical SFT context cannot contain an infrastructure failure")
