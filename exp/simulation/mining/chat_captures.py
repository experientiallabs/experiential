"""Representative-task mining over captured chat-completion prompts.

Chat captures are request-time records of OpenAI-style chat prompts: a request
id, the ordered role/content message array, and the capture time. This module
adapts each capture into one canonical ``Trace`` (the task text is the system
and developer contents plus the first user turn, carried by one fabricated
span) and runs the canonical mining service over the result, so capture-fed
consumers inherit deduplication, leakage-safe partitioning, weighting, and
coverage evidence instead of reimplementing them. Mined ``TaskCase.task_id``
values are content derived, so re-mining an overlapping capture window
converges on the same tasks instead of duplicating them. The Experiential
Labs platform gateway is the first consumer of this seam.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import ContractModel, JsonObject, SourceIdentity
from exp.common.tasks import TaskCase
from exp.common.traces import Trace, TraceSource, TraceSpan
from exp.simulation.mining.descriptors import DescriptorEmbedder
from exp.simulation.mining.service import MiningSpec, TaskMiningResult, mine_tasks

# Semantic-convention marker carried by every trace fabricated from a capture.
_CHAT_CAPTURE_SEMANTIC_CONVENTION = "exp-chat-capture-v1"


class ChatCapture(ContractModel):
    """One captured chat-completion prompt ready for task mining.

    Args:
        request_id: Unique capture identity; becomes the trace and span id.
        messages: Ordered OpenAI-style chat messages as raw JSON objects. Only
            ``system``, ``developer``, and ``user`` messages whose ``content``
            is a non-empty string contribute task text; tool calls and
            structured content parts carry none.
        captured_at: Timezone-aware capture time of the request.
        group_key: Optional caller-owned grouping key (for example a prompt
            digest) resolved per mined task through the task's selected
            representative capture.
    """

    request_id: str = Field(min_length=1, max_length=512)
    messages: tuple[JsonObject, ...] = Field(min_length=1)
    captured_at: AwareDatetime
    group_key: str | None = None


class ChatCaptureMiningSummary(ContractModel):
    """Scalar selection-honesty summary of one chat-capture mining run.

    Every value restates the run's ``CoverageReport`` (plus the capture-level
    exclusion count) as flat scalars, so a consumer can persist or display
    what the mining actually covered without re-deriving numbers that drift
    across runs. The per-partition ``*_workload_covered`` fractions are the
    selected workload mass over the partition's eligible traces. When no
    capture was minable every count is zero and ``split_separation_verified``
    is vacuously true.

    Args:
        input_capture_count: All captures given to the run.
        captures_without_text: Captures excluded because no message carried
            usable task text.
        eligible_trace_count: Adapted traces that entered mining.
        duplicate_trace_count: Eligible traces removed as duplicates.
        selected_task_count: Representative tasks selected across partitions.
        fit_task_budget: Task budget requested of the fit partition.
        held_out_task_budget: Task budget requested of the held-out partition.
        fit_selected_task_count: Tasks selected in the fit partition.
        held_out_selected_task_count: Tasks selected in the held-out partition.
        fit_workload_covered: Fit selected workload mass over fit eligible
            traces.
        held_out_workload_covered: Held-out selected workload mass over
            held-out eligible traces.
        split_separation_verified: Whether fit and held-out lineage groups
            were verified disjoint.
    """

    input_capture_count: int = Field(ge=0)
    captures_without_text: int = Field(ge=0)
    eligible_trace_count: int = Field(ge=0)
    duplicate_trace_count: int = Field(ge=0)
    selected_task_count: int = Field(ge=0)
    fit_task_budget: int = Field(ge=0)
    held_out_task_budget: int = Field(ge=0)
    fit_selected_task_count: int = Field(ge=0)
    held_out_selected_task_count: int = Field(ge=0)
    fit_workload_covered: float = Field(ge=0.0, le=1.0)
    held_out_workload_covered: float = Field(ge=0.0, le=1.0)
    split_separation_verified: bool


@dataclass(frozen=True)
class ChatCaptureMiningResult:
    """One chat-capture mining run: full mining evidence plus capture links.

    Args:
        mining: The complete canonical mining result, or ``None`` when no
            capture carried usable task text (a normal state for a quiet
            traffic window, not an error).
        group_keys_by_task_id: Each selected task's ``group_key``, resolved
            through the task's representative source capture.
        summary: Scalar selection-honesty numbers for the run.
    """

    mining: TaskMiningResult | None
    group_keys_by_task_id: Mapping[str, str | None]
    summary: ChatCaptureMiningSummary

    @property
    def tasks(self) -> tuple[TaskCase, ...]:
        """Selected representative tasks; empty when no capture was minable."""
        return () if self.mining is None else self.mining.tasks


def _text_content(message: JsonObject) -> str | None:
    """Return the message's text content; None for tool calls and structured parts."""
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    return None


def _task_text(messages: tuple[JsonObject, ...]) -> str | None:
    """Return the mined task basis: system and developer contents plus the first user turn."""
    system_parts = [
        text
        for message in messages
        if message.get("role") in ("system", "developer")
        and (text := _text_content(message)) is not None
    ]
    first_user = next(
        (
            text
            for message in messages
            if message.get("role") == "user" and (text := _text_content(message)) is not None
        ),
        None,
    )
    parts = [*system_parts, *([first_user] if first_user is not None else [])]
    if not parts:
        return None
    return "\n\n".join(parts)


def _capture_trace(capture: ChatCapture, task: str) -> Trace:
    """Return one canonical trace with a single fabricated span for the capture."""
    return Trace(
        trace_id=capture.request_id,
        task=task,
        spans=(
            TraceSpan(
                span_id=capture.request_id,
                name="chat.capture",
                started_at=capture.captured_at,
                ended_at=capture.captured_at,
            ),
        ),
        outcome=None,
        source=TraceSource(
            identity=SourceIdentity(kind="production", source_id=capture.request_id),
            semantic_convention_version=_CHAT_CAPTURE_SEMANTIC_CONVENTION,
        ),
    )


def mine_tasks_from_chat_captures(
    captures: Sequence[ChatCapture],
    mining_spec: MiningSpec | None = None,
    *,
    embedder: DescriptorEmbedder,
) -> ChatCaptureMiningResult:
    """Mine a representative task set from captured chat prompts.

    Args:
        captures: Captured chat prompts with unique request ids.
        mining_spec: Task budgets and leakage-safe mining controls. Defaults
            to the canonical mining defaults.
        embedder: Explicit request-descriptor embedder, exactly as required
            by ``mine_tasks``.

    Returns:
        The mining evidence, each task's resolved capture group key, and the
        run's scalar coverage summary. ``mining`` is ``None`` when no capture
        carried usable task text.

    Raises:
        ValueError: ``captures`` is empty or contains a duplicate request id,
            or mining itself rejects its inputs.
    """
    if not captures:
        raise ValueError("chat-capture mining needs at least one capture")
    request_ids = tuple(capture.request_id for capture in captures)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("chat-capture request ids must be unique")
    spec = mining_spec or MiningSpec()
    group_keys: dict[str, str | None] = {}
    traces: list[Trace] = []
    for capture in captures:
        task = _task_text(capture.messages)
        if task is None:
            continue
        group_keys[capture.request_id] = capture.group_key
        traces.append(_capture_trace(capture, task))
    captures_without_text = len(captures) - len(traces)
    if not traces:
        return ChatCaptureMiningResult(
            mining=None,
            group_keys_by_task_id={},
            summary=ChatCaptureMiningSummary(
                input_capture_count=len(captures),
                captures_without_text=captures_without_text,
                eligible_trace_count=0,
                duplicate_trace_count=0,
                selected_task_count=0,
                fit_task_budget=spec.fit_task_budget,
                held_out_task_budget=spec.held_out_task_budget,
                fit_selected_task_count=0,
                held_out_selected_task_count=0,
                fit_workload_covered=0.0,
                held_out_workload_covered=0.0,
                split_separation_verified=True,
            ),
        )
    result = mine_tasks(
        traces,
        spec,
        embedder=embedder,
        input_trace_count=len(captures),
        invalid_trace_count=captures_without_text,
    )
    coverage = result.coverage
    # The coverage report names each task's representative source trace, so a
    # task's group key comes from the capture the selection actually kept.
    representative_by_task_id = {
        selection.task_id: selection.representative_trace_id for selection in coverage.selections
    }
    group_keys_by_task_id = {
        task.task_id: group_keys[representative_by_task_id[task.task_id]] for task in result.tasks
    }
    summary = ChatCaptureMiningSummary(
        input_capture_count=coverage.input_trace_count,
        captures_without_text=coverage.invalid_trace_count,
        eligible_trace_count=coverage.eligible_trace_count,
        duplicate_trace_count=coverage.duplicate_trace_count,
        selected_task_count=coverage.selected_task_count,
        fit_task_budget=coverage.fit.requested_task_budget,
        held_out_task_budget=coverage.held_out.requested_task_budget,
        fit_selected_task_count=coverage.fit.selected_task_count,
        held_out_selected_task_count=coverage.held_out.selected_task_count,
        fit_workload_covered=coverage.fit.selected_workload_mass
        / max(coverage.fit.eligible_trace_count, 1),
        held_out_workload_covered=coverage.held_out.selected_workload_mass
        / max(coverage.held_out.eligible_trace_count, 1),
        split_separation_verified=coverage.split_separation_verified,
    )
    return ChatCaptureMiningResult(
        mining=result,
        group_keys_by_task_id=group_keys_by_task_id,
        summary=summary,
    )
