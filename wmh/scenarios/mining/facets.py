"""Facet extraction: one compact, embeddable summary per trace (the Clio pattern).

Raw traces are dominated by boilerplate (tool schemas, retrieved content), so embedding them
directly washes out task intent — two traces with identical scaffolding but different tasks land
nearly on top of each other. Instead, a cheap LLM reads a compact digest of each trace and emits a
`TraceFacet`: a short task summary (what the user was trying to get done), the outcome, and a
failure category when the episode failed. The deterministic tool-call signature and the corpus
domain are computed in code, not by the LLM, and join the summary in the embedded text so
clustering groups by capability rather than phrasing. Downstream clustering/selection operates on
facet embeddings only.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum

from pydantic import BaseModel, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.types import ActionKind, Trace
from wmh.providers.base import Message, Provider


class Outcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class TraceFacet(BaseModel):
    """The embeddable summary of one trace; the unit clustering and selection operate on."""

    trace_id: str
    task_summary: str  # <= ~30 words: what the user was trying to get done
    tool_signature: str  # deterministic "tool_a>tool_b>..." with consecutive repeats collapsed
    domain: str | None = None  # from trace metadata when the corpus records one
    outcome: Outcome = Outcome.UNKNOWN
    failure_category: str | None = None  # short label, only when outcome == FAILURE

    def embed_text(self) -> str:
        """The text clustering embeds: domain + task intent + capabilities exercised.

        Embedding the summary alone clusters by phrasing, which splits one capability into
        several clusters ("MMS troubleshooting" vs "International MMS troubleshooting") and lets
        cluster-level allocation double-count it. Domain and the tool signature pull traces that
        exercise the same capability together regardless of how the request was worded.
        """
        parts = []
        if self.domain:
            parts.append(f"[{self.domain}]")
        parts.append(self.task_summary)
        text = " ".join(parts)
        if self.tool_signature:
            text = f"{text} | tools: {self.tool_signature}"
        return text


def trace_domain(trace: Trace) -> str | None:
    """The trace's domain from corpus metadata, when recorded (e.g. tau2's telecom/retail)."""
    value = trace.metadata.get("domain")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def tool_signature(trace: Trace) -> str:
    """Deterministic tool-call sequence signature with consecutive repeats collapsed.

    `search>search>book` becomes `search>book`: the signature captures *which* capabilities the
    episode exercised in what order, not how many retries each took.
    """
    names: list[str] = []
    for step in trace.steps:
        action = step.action
        if action.kind is not ActionKind.TOOL_CALL or not action.name:
            continue
        if not names or names[-1] != action.name:
            names.append(action.name)
    return ">".join(names)


_MAX_DIGEST_STEPS = 30
_MAX_FIELD_CHARS = 300


def trace_digest(trace: Trace, *, max_steps: int = _MAX_DIGEST_STEPS) -> str:
    """A compact plain-text rendering of a trace for the facet-extraction LLM.

    Includes the task, then one line per step (tool name + truncated arguments + truncated
    observation, error-flagged). Long traces keep the first and last steps and elide the middle:
    intent lives at the start, resolution at the end.
    """
    lines: list[str] = []
    task = _trace_task(trace)
    if task:
        lines.append(f"TASK: {_truncate(task)}")
    steps = trace.steps
    if len(steps) > max_steps:
        head = max_steps // 2
        tail = max_steps - head
        shown = list(enumerate(steps))[:head] + list(enumerate(steps))[-tail:]
        elided = len(steps) - max_steps
    else:
        shown = list(enumerate(steps))
        elided = 0
    previous_index = -1
    for index, step in shown:
        if index > previous_index + 1:
            lines.append(f"... ({elided} steps elided) ...")
        previous_index = index
        action = step.action
        if action.kind is ActionKind.TOOL_CALL:
            args = _truncate(str(action.arguments)) if action.arguments else ""
            head_line = f"{index}. CALL {action.name}({args})"
        else:
            head_line = f"{index}. MSG {_truncate(action.content or '')}"
        observation = step.observation
        error_mark = " [ERROR]" if observation.is_error else ""
        lines.append(f"{head_line} -> {_truncate(observation.content)}{error_mark}")
    return "\n".join(lines)


def _truncate(text: str, limit: int = _MAX_FIELD_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _trace_task(trace: Trace) -> str | None:
    for step in trace.steps:
        if step.task and step.task.strip():
            return step.task.strip()
    return None


FACET_SYSTEM = """You summarize one AI-agent episode (a trace of tool calls and messages) into a
compact facet used to organize a large trace corpus.

Respond with ONLY a JSON object, no prose around it:
{"task_summary": "<what the USER was trying to get done, <=30 words, self-contained, no ids>",
 "outcome": "success" | "failure" | "unknown",
 "failure_category": "<short snake_case label, e.g. wrong_tool_arguments; null unless failure>"}

Rules:
- task_summary states the user's goal, not the agent's mechanics ("cancel a flight booking and get
  a refund", NOT "called cancel_reservation").
- outcome is "success" only if the episode visibly achieved the goal; "failure" if it visibly did
  not (errors, refusals, wrong result); otherwise "unknown".
- failure_category names the dominant failure mode in 1-3 words; null when outcome != "failure"."""


class _RawFacet(BaseModel):
    """Lenient view of the extractor's JSON before normalization."""

    task_summary: str
    outcome: str = "unknown"
    failure_category: str | None = None


class FacetExtractor:
    """LLM facet extraction over a trace corpus (one completion per trace)."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def extract(self, trace: Trace) -> TraceFacet:
        """Extract the facet for one trace; falls back to the raw task on an unparseable reply."""
        completion = self._provider.complete(
            FACET_SYSTEM,
            [Message(role="user", content=trace_digest(trace))],
            temperature=0.0,
            max_tokens=512,
        )
        signature = tool_signature(trace)
        domain = trace_domain(trace)
        raw = extract_json_object(completion.text)
        if raw is not None:
            try:
                parsed = _RawFacet.model_validate_json(raw)
            except ValidationError:
                parsed = None
            if parsed is not None and parsed.task_summary.strip():
                outcome = _parse_outcome(parsed.outcome)
                category = parsed.failure_category if outcome is Outcome.FAILURE else None
                return TraceFacet(
                    trace_id=trace.trace_id,
                    task_summary=parsed.task_summary.strip(),
                    tool_signature=signature,
                    domain=domain,
                    outcome=outcome,
                    failure_category=_normalize_category(category),
                )
        # Fallback: the recorded task prompt is still a usable intent summary; flag as UNKNOWN.
        return TraceFacet(
            trace_id=trace.trace_id,
            task_summary=_truncate(_trace_task(trace) or "(no task recorded)", 200),
            tool_signature=signature,
            domain=domain,
            outcome=Outcome.UNKNOWN,
        )

    def extract_all(self, traces: list[Trace], *, concurrency: int = 8) -> list[TraceFacet]:
        """Extract facets for every trace, in order.

        Each facet is one independent LLM call, so they run on a small thread pool
        (`pool.map` preserves input order and propagates exceptions — the replay.py
        precedent); `concurrency=1` keeps the sequential loop.
        """
        if concurrency > 1 and len(traces) > 1:
            with ThreadPoolExecutor(max_workers=min(concurrency, len(traces))) as pool:
                return list(pool.map(self.extract, traces))
        return [self.extract(trace) for trace in traces]


def _parse_outcome(raw: str) -> Outcome:
    try:
        return Outcome(raw.strip().lower())
    except ValueError:
        return Outcome.UNKNOWN


def _normalize_category(category: str | None) -> str | None:
    if category is None:
        return None
    normalized = "_".join(category.strip().lower().split())
    return normalized or None
