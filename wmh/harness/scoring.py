"""Benchmark-neutral score reports consumed by harness search."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wmh.core.text import normalize_durable_text
from wmh.harness.delta import FailureSignature
from wmh.harness.doc import HarnessDoc

ALL_PASS_EVIDENCE = (
    "The harness passed every task on every pass. There are no failures to fix. "
    "Propose a change that should GENERALIZE or ECONOMIZE: a tighter, more transferable "
    "prompt surface; a lower param:max-turns if runs finish early; or a reusable skill "
    "distilled from what worked."
)

MAX_TASK_DESCRIPTION_CHARS = 8_000
MAX_TASK_EVIDENCE_CHARS = 64_000
MAX_MECHANISMS_PER_TASK = 64
MAX_RENDERED_SCORE_EVIDENCE_CHARS = 256_000
_MAX_SCORECARD_CHARS = 32_000
_MAX_FAILURE_HEADER_CHARS = 8_000
_MAX_MECHANISM_SUMMARY_CHARS = 16_000
MechanismLabel = Annotated[str, Field(min_length=1, max_length=4_000)]


class ScoreRunHealth(StrEnum):
    """Whether a scorer report is valid optimizer evidence."""

    VALID = "valid"
    RETRY_REQUIRED = "retry_required"
    UNKNOWN = "unknown"


class ScoreRunHealthError(RuntimeError):
    """A scorer returned a matrix that must not enter search selection."""

    def __init__(self, evaluation_id: str, run_health: ScoreRunHealth) -> None:
        super().__init__(
            f"evaluation {evaluation_id!r} has run_health={run_health.value!r}; "
            "retry or invalidate it before optimizer selection"
        )
        self.evaluation_id = evaluation_id
        self.run_health = run_health


class ScoreCapabilities(BaseModel):
    """Optional score requests a scorer can execute."""

    model_config = ConfigDict(frozen=True)

    task_subsets: bool = False
    attempt_overrides: bool = False


class ScoreRequest(BaseModel):
    """One immutable scoring request issued by the search."""

    model_config = ConfigDict(frozen=True)

    purpose: Literal["seed", "screen", "full", "holdout", "confirmation"]
    task_ids: tuple[str, ...] | None = None
    attempts: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_task_ids(self) -> ScoreRequest:
        if self.task_ids is None:
            return self
        if not self.task_ids:
            raise ValueError("task_ids must be nonempty when requesting a subset")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique")
        return self


class TaskScore(BaseModel):
    """One task's normalized score and proposer-facing evidence."""

    model_config = ConfigDict(allow_inf_nan=False)

    task_id: str = Field(min_length=1, max_length=512)
    score: float = Field(ge=0.0, le=1.0)
    secondary_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    description: str = Field(default="", max_length=MAX_TASK_DESCRIPTION_CHARS)
    mechanisms: tuple[MechanismLabel, ...] = Field(default=(), max_length=MAX_MECHANISMS_PER_TASK)
    evidence: str = Field(default="", max_length=MAX_TASK_EVIDENCE_CHARS)


class HarnessScoreReport(BaseModel):
    """A normalized scorecard over one candidate and one task split."""

    model_config = ConfigDict(allow_inf_nan=False)

    evaluation_id: str = Field(min_length=1, max_length=512, frozen=True)
    label: str = Field(default="", max_length=512)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    secondary_score: float = Field(default=0.0, ge=0.0, le=1.0)
    attempts: int = Field(ge=1)
    run_health: ScoreRunHealth
    per_task: dict[str, TaskScore] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_task_keys(self) -> HarnessScoreReport:
        for task_id, task in self.per_task.items():
            if task_id != task.task_id:
                raise ValueError(
                    f"per_task key {task_id!r} does not match task_id {task.task_id!r}"
                )
        return self


class HarnessScorer(Protocol):
    """Synchronous candidate evaluator used by the harness search loop."""

    capabilities: ScoreCapabilities
    default_attempts: int

    def validate_candidate(self, candidate: HarnessDoc) -> str | None:
        """Return a rejection reason, or None when the candidate is eligible."""

    def before_proposal_batch(self) -> None:
        """Release idle evaluation resources before a potentially long proposal call."""

    def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
        """Evaluate one candidate under the requested task and attempt selection."""


def cluster_score_failures(report: HarnessScoreReport) -> list[FailureSignature]:
    """Group failing tasks by shared mechanism labels, deterministically."""
    mechanisms_by_task = {
        task_id: list(dict.fromkeys(task.mechanisms))
        for task_id, task in report.per_task.items()
        if not task.passed
    }
    clusters: list[FailureSignature] = []
    assigned: set[str] = set()
    for task_id in sorted(mechanisms_by_task):
        if task_id in assigned:
            continue
        member_ids = {task_id}
        mechanisms = set(mechanisms_by_task[task_id])
        grew = True
        while grew:
            grew = False
            for other, other_mechanisms in mechanisms_by_task.items():
                if other in member_ids or not mechanisms.intersection(other_mechanisms):
                    continue
                member_ids.add(other)
                mechanisms.update(other_mechanisms)
                grew = True
        assigned.update(member_ids)
        counts: dict[str, int] = {}
        for member in member_ids:
            for mechanism in mechanisms_by_task[member]:
                counts[mechanism] = counts.get(mechanism, 0) + 1
        label = (
            max(sorted(counts), key=lambda mechanism: counts[mechanism])
            if counts
            else "run failed without mechanism details"
        )
        clusters.append(
            FailureSignature(
                mechanism=label,
                task_ids=sorted(member_ids),
                mechanism_labels=sorted(mechanisms),
            )
        )
    clusters.sort(key=lambda cluster: (-len(cluster.task_ids), cluster.mechanism))
    return clusters


def render_score_evidence(trigger: FailureSignature, report: HarnessScoreReport) -> str:
    """Render one selected failure under a deterministic whole-prompt character budget."""
    if not trigger.task_ids:
        return ALL_PASS_EVIDENCE

    selected = set(trigger.task_ids)
    scorecard = [
        "## Evaluation scorecard",
        "The selected failure is marked TARGET; preserve behavior on the other tasks.",
    ]
    for task_id in sorted(report.per_task):
        task = report.per_task[task_id]
        description = " ".join(normalize_durable_text(task.description).split())
        if len(description) > 240:
            description = f"{description[:237]}..."
        marker = "TARGET" if task_id in selected else "other"
        scorecard.append(
            f"- [{marker}] {task_id}: score={task.score:.2f}, "
            f"secondary_score={task.secondary_score:.2f}: {description}"
        )

    scorecard_text = _bound_score_text(
        "\n".join(scorecard),
        limit=_MAX_SCORECARD_CHARS,
        label="scorecard",
    )
    failure_header = _bound_score_text(
        f"## Selected failure\n\nFailure mechanism: {trigger.mechanism}",
        limit=_MAX_FAILURE_HEADER_CHARS,
        label="failure header",
    )
    mechanism_summary = _bound_score_text(
        "Original trigger mechanisms from the parent (current attempt evidence above is "
        "authoritative):\n"
        + (
            "\n".join(f"- {item}" for item in sorted(trigger.mechanism_labels))
            or "- (none recorded)"
        ),
        limit=_MAX_MECHANISM_SUMMARY_CHARS,
        label="mechanism summary",
    )

    selected_task_ids = sorted(dict.fromkeys(trigger.task_ids))
    separator_chars = 2 * (len(selected_task_ids) + 2)
    task_budget = max(
        0,
        (
            MAX_RENDERED_SCORE_EVIDENCE_CHARS
            - len(scorecard_text)
            - len(failure_header)
            - len(mechanism_summary)
            - separator_chars
        )
        // len(selected_task_ids),
    )
    task_sections: list[str] = []
    for task_id in selected_task_ids:
        task = report.per_task.get(task_id)
        if task is None:
            task_section = (
                f"### Task {task_id}\n\nInstruction: (unknown task)\n\nNo evidence recorded."
            )
        else:
            section = [
                f"### Task {task_id} (score={task.score:.2f}, "
                f"secondary_score={task.secondary_score:.2f})",
                f"Instruction: {normalize_durable_text(task.description) or '(none)'}",
                normalize_durable_text(task.evidence) if task.evidence else "No evidence recorded.",
            ]
            task_section = "\n\n".join(section)
        task_sections.append(
            _bound_score_text(task_section, limit=task_budget, label=f"task {task_id}")
        )

    rendered = "\n\n".join([scorecard_text, failure_header, *task_sections, mechanism_summary])
    return _bound_score_text(
        rendered,
        limit=MAX_RENDERED_SCORE_EVIDENCE_CHARS,
        label="score evidence",
    )


def _bound_score_text(value: str, *, limit: int, label: str) -> str:
    """Bound text deterministically while preserving both its opening and terminal evidence."""
    if len(value) <= limit:
        return value
    if limit <= 0:
        return ""
    marker = f"\n...[{label} truncated; original_chars={len(value)}]...\n"
    if len(marker) >= limit:
        return marker[:limit]
    retained = limit - len(marker)
    head = retained // 2
    tail = retained - head
    return value[:head] + marker + value[-tail:]


def suite_score(report: HarnessScoreReport, suite: Sequence[str]) -> float:
    """Return the mean primary score over a task subset, with missing tasks scored zero."""
    if not suite:
        return 1.0
    scores = [task.score for task_id in suite if (task := report.per_task.get(task_id)) is not None]
    return sum(scores) / len(suite)


def suite_secondary_score(report: HarnessScoreReport, suite: Sequence[str]) -> float:
    """Return the mean secondary score over a subset, with missing tasks scored zero."""
    if not suite:
        return 1.0
    scores = [
        task.secondary_score
        for task_id in suite
        if (task := report.per_task.get(task_id)) is not None
    ]
    return sum(scores) / len(suite)
