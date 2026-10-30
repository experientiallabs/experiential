"""Saved contracts of agent memory learning: its spec, plan, window, lessons and results.

Behavior lives in :mod:`exp.optimize.evaluation.learning.agent_memory`; these are the shapes it
registers, saves and replays.
"""

from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, ContractModel
from exp.common.models import ModelSnapshot
from exp.optimize.evaluation.learning.agent_memory_requests import (
    LessonKind,
    LessonScreening,
    ShownOutcome,
)

Stage = Literal["extract", "merge", "screen"]


class AgentMemorySettings(ContractModel):
    """The spec fields that decide a learning run's requests and results.

    Attributes:
        alias: Catalog alias of the learning model; its provider must use the admitted
            transport.
        batch_bytes: Bound on the rendered content of one request.
        maximum_trace_bytes: Bound on one trace's rendering.
        maximum_message_bytes: Bound on one rendered message.
        threshold: Registered screening threshold a lesson's score must reach.
    """

    alias: str = Field(min_length=1)
    batch_bytes: int = Field(default=150_000, ge=10_000, le=2_000_000, strict=True)
    maximum_trace_bytes: int = Field(default=40_000, ge=2_000, le=1_000_000, strict=True)
    maximum_message_bytes: int = Field(default=2_000, ge=200, le=100_000, strict=True)
    threshold: float = Field(default=0.5, gt=0, lt=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def _bounded(self) -> Self:
        """Require one trace and one message to fit their enclosing bounds."""
        if self.maximum_trace_bytes > self.batch_bytes:
            raise ValueError("maximum_trace_bytes must not exceed batch_bytes")
        if self.maximum_message_bytes > self.maximum_trace_bytes:
            raise ValueError("maximum_message_bytes must not exceed maximum_trace_bytes")
        return self


class AgentMemorySpec(AgentMemorySettings):
    """Learn agent memory from one chat JSON export of outcome-labelled traces.

    Attributes:
        traces: Chat JSON export whose conversations carry ``exp.outcome.status``.
        output: Memory folder to write; absent, empty, or holding this exact memory.
        concurrency: Requests in flight at once within a stage; does not change results.
        hours: Window from the plan's first paid call to its deadline.
        allow_exclusions: Learn from the valid traces when the export holds records that
            cannot be normalized, instead of refusing; the exclusions stay saved with the
            dataset.
    """

    traces: Path
    output: Path
    concurrency: int = Field(default=8, ge=1, le=64, strict=True)
    hours: float = Field(default=5.0, gt=0, le=48, allow_inf_nan=False)
    allow_exclusions: bool = Field(default=False, strict=True)


class AgentMemoryPlan(ArtifactEnvelope):
    """The registered plan every request of one learning run is bound to.

    Attributes:
        plan_id: Identity of the protocol, dataset, settings and model.
        protocol: Request framing and decoding protocol.
        dataset: Imported trace dataset.
        settings: Result-deciding spec fields.
        model: Learning model snapshot.
    """

    plan_id: str
    protocol: str
    dataset: ArtifactInput
    settings: AgentMemorySettings
    model: ModelSnapshot


class AgentMemoryWindow(ArtifactEnvelope):
    """The deadline every request of a plan is fingerprinted with, fixed by its first call.

    Attributes:
        window_id: Identity of the plan it bounds.
        plan: The plan.
        deadline: Latest admission time.
    """

    window_id: str
    plan: ArtifactInput
    deadline: AwareDatetime


class Lesson(ContractModel):
    """One lesson and the traces it was learned from.

    Attributes:
        kind: Lesson kind.
        statement: The lesson.
        applies_when: When a future agent should apply it.
        evidence: What its traces showed, one note per source candidate.
        traces: Cited canonical trace IDs, sorted.
    """

    kind: LessonKind
    statement: str = Field(min_length=1)
    applies_when: str = Field(min_length=1)
    evidence: tuple[str, ...] = Field(min_length=1)
    traces: tuple[str, ...] = Field(min_length=1)


class ScreenedLesson(Lesson):
    """A merged lesson with its screening decision.

    Attributes:
        screening: The three probabilities, absent when the screening call failed.
        score: Screening score, absent when unknown.
        accepted: The score reached the plan's threshold.
    """

    screening: LessonScreening | None
    score: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    accepted: bool


class FailedCall(ContractModel):
    """One request whose answer is not used, and why; never redrawn.

    Attributes:
        stage: Stage that sent it.
        logical_id: Its logical request identity.
        status: ``rejected`` (answer returned and retained, but refused) or ``unknown``
            (dispatched without a saved answer).
        reason: Why.
    """

    stage: Stage
    logical_id: str
    status: Literal["rejected", "unknown"]
    reason: str


class CandidateSet(ArtifactEnvelope):
    """Every candidate the extraction stage proposed.

    Attributes:
        candidates_id: Identity of the plan this set answers.
        plan: The plan.
        candidates: Candidates in batch order, citing canonical trace IDs.
        failed: Extraction calls whose batches proposed nothing usable.
    """

    candidates_id: str
    plan: ArtifactInput
    candidates: tuple[Lesson, ...]
    failed: tuple[FailedCall, ...]


class AgentMemory(ArtifactEnvelope):
    """Merged and screened lessons of one learning run.

    Attributes:
        memory_id: Identity of the candidate set it consolidates.
        plan: The plan.
        candidates: The candidate set.
        outcomes: Trace count per shown outcome (``PASS``, ``FAIL``, ``unknown``).
        lessons: Every merged lesson, accepted or not, by kind and then most cited first.
        failed: Merge and screening calls whose answers are not used.
    """

    memory_id: str
    plan: ArtifactInput
    candidates: ArtifactInput
    outcomes: dict[ShownOutcome, int]
    lessons: tuple[ScreenedLesson, ...]
    failed: tuple[FailedCall, ...]
