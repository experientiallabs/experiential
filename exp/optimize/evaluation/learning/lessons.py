"""Screened lessons: publish a learned checkpoint's reusable misses into the shared library.

Screening (:mod:`exp.optimize.evaluation.learning.screening`) decides which learning misses are
reusable lessons. This module turns one screened checkpoint into a lesson set:

- each screened-in miss becomes a ``rule`` library entry with disposition ``unresolved``,
  ``derived_from`` its source trajectory and ``depends_on`` the record's own dependencies; the
  source trajectory enters the library as ``source`` evidence beside it;
- screened-out, unknown and hypothesis-free misses stay source evidence only: they are not
  lessons, and a frozen run reading the lesson set never serves them or their overview lines;
- raw trajectories are never lessons: they stay ``source`` entries, which only provenance and
  eligibility read.

The lesson set binds the memory checkpoint, the saved threshold fit and the library snapshot.
A frozen held-out audit can read memory restricted to it (``FidelityMemory.lessons``), and
:mod:`exp.optimize.evaluation.learning.lesson_evolution` proposes edits from it. Publication is
content-addressed: rerunning the same screening replays every decision and the same set.
"""

import json
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Discriminator, Field, Tag

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    sha256_bytes,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.common.project.causality import cites
from exp.optimize.evaluation.fidelity.memory_policy import (
    LESSON_SET_TYPE,
    LessonSelection,
    load_lesson_selection,
)
from exp.optimize.evaluation.fidelity.replay import DEADLINE
from exp.optimize.evaluation.learning.repair import LearningRepairSpec, TraceRetirementRequest
from exp.optimize.evaluation.learning.screening import (
    SCREENING_PROTOCOL,
    LearningMiss,
    LessonScreeningSpec,
    MissScreening,
    ThresholdFit,
    delivery_outcomes,
    fit_threshold,
    learning_misses,
    planned_screenings,
    save_fit,
    screen_misses,
    screening_cost,
    screening_dispatched,
)
from exp.optimize.evaluation.learning.traces import (
    LearningPlan,
    Scope,
    learning_plan,
    recorded_plan,
)
from exp.simulation.reconstruction.library import (
    LibraryEntry,
    LibrarySnapshot,
    extend_library,
    resolve_library_entry,
)
from exp.simulation.world_model.experience_memory import MemoryVersion, memory_head

LESSON_TYPE = "learning-lesson"


class ScreeningPreflight(ContractModel):
    """What screening one checkpoint would do, without credentials or writes.

    Attributes:
        lineage: Memory lineage screened.
        version: Memory version screened.
        alias: Native decision alias (the learning plan's recall alias).
        misses: Misses of learned runs in that version.
        no_hypothesis: Misses screened out without a call.
        saved: Decisions already saved, replayed without calls.
        unknown: Dispatched decisions without an answer; never redrawn.
        to_dispatch: New decisions, one physical attempt each.
        estimated_cost_usd: Conservative cost of the new decisions; None when unpriced.
    """

    lineage: str
    version: int
    alias: str
    misses: int
    no_hypothesis: int
    saved: int
    unknown: int
    to_dispatch: int
    estimated_cost_usd: float | None


class ScreenedLesson(ContractModel):
    """One screened miss and where it went.

    Attributes:
        miss: The screened miss.
        screening: Saved decision identity; absent for a miss without hypothesis.
        score: Screening score; absent without an answer.
        lesson: Library entry of a screened-in lesson.
    """

    miss: LearningMiss
    screening: str | None = None
    score: float | None = None
    lesson: str | None = None


class LessonSet(ContractModel):
    """A screened checkpoint: its fit, library snapshot and every miss's disposition.

    Attributes:
        selection: Lineage, version and screened-in record IDs a frozen run reads.
        fit: Saved threshold fit.
        threshold: Fitted threshold.
        library: Library snapshot holding the lessons; absent when none was screened in and
            no prior library was named.
        screened_in: Lessons, in library order.
        screened_out: Answered misses at or below the threshold.
        unknown: Misses whose decision outcome stays unknown.
        no_hypothesis: Misses without an investigation hypothesis.
    """

    selection: LessonSelection
    fit: ArtifactInput
    threshold: float
    library: ArtifactInput | None
    screened_in: tuple[ScreenedLesson, ...]
    screened_out: tuple[ScreenedLesson, ...]
    unknown: tuple[ScreenedLesson, ...]
    no_hypothesis: tuple[ScreenedLesson, ...]

    def summary(self) -> JsonObject:
        """Counts and references for the terminal; the full set stays saved."""
        return {
            "lineage": self.selection.lineage,
            "version": self.selection.version,
            "threshold": self.threshold,
            "screened_in": len(self.screened_in),
            "screened_out": len(self.screened_out),
            "unknown": len(self.unknown),
            "no_hypothesis": len(self.no_hypothesis),
            "fit": self.fit.model_dump(mode="json"),
            "library": None if self.library is None else self.library.model_dump(mode="json"),
        }


class LessonEvolutionRequest(ContractModel):
    """Propose one lesson-library edit from a lesson set and keep it only on held-out gain.

    Attributes:
        resolution: Selects library evolution in the ``repair-learning`` action.
        lessons: Saved lesson set whose screened-in lessons are the training cases.
        library: Library snapshot to extend; defaults to the lesson set's library. Name the
            snapshot an earlier evolution returned to build on its accepted edit.
        run_id: Fresh attempt identity; a registered attempt is never re-run.
        held_out: Fit lineages, outside those of the lessons and the parent rules, the edit is
            selected on.
        protected: Further fit lineages that must not regress.
        minimum_gain: Strict mean exact-reply improvement required on the held-out cases.
    """

    resolution: Literal["evolve-lessons"]
    lessons: ArtifactInput
    library: ArtifactInput | None = Field(default=None, exclude_if=lambda value: value is None)
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    held_out: int = Field(default=3, ge=1, le=32, strict=True)
    protected: int = Field(default=2, ge=1, le=32, strict=True)
    minimum_gain: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)


def _resolution(value: object) -> str | None:
    """The repair resolution of raw JSON or a model; observation retention is the default."""
    if isinstance(value, dict):
        return str(value.get("resolution", "retain-observations"))
    return getattr(value, "resolution", None)


LearningRepair = Annotated[
    Annotated[LearningRepairSpec, Tag("retain-observations")]
    | Annotated[LessonScreeningSpec, Tag("screen-lessons")]
    | Annotated[LessonEvolutionRequest, Tag("evolve-lessons")]
    | Annotated[TraceRetirementRequest, Tag("retire-trace")],
    Discriminator(_resolution),
]


def screening_plan(project: ProjectStore, scope: Scope) -> tuple[LearningPlan, MemoryVersion]:
    """The recorded learning plan of ``scope`` and its memory head.

    Raises:
        ValueError: Nothing of that scope was learned yet.
    """
    if not recorded_plan(project, scope):
        command = "exp eval PROJECT --audit N" if scope == "fit" else "exp build PROJECT"
        raise ValueError(f"no {scope} learning is recorded; run {command} to learn first")
    plan = learning_plan(project, scope=scope)
    head = memory_head(project.artifacts, plan.lineage)
    if head is None:
        raise ValueError(f"{scope} learning has not published memory yet; learn first")
    return plan, head


def accepted_lessons(project: ProjectStore, library: LibrarySnapshot | None) -> list[JsonObject]:
    """Active accepted rule and skill entries, as ``{"entry_id", "rules"}``."""
    if library is None:
        return []
    superseded = {entry.supersedes for entry in library.entries if entry.supersedes}
    rows: list[JsonObject] = []
    for entry in library.entries:
        if (
            entry.kind in ("rule", "skill")
            and entry.disposition == "accepted"
            and entry.entry_id not in superseded
        ):
            payload = resolve_library_entry(project.artifacts, entry)
            knowledge = payload.get("knowledge") if isinstance(payload, dict) else None
            rules = knowledge.get("rules") if isinstance(knowledge, dict) else None
            rows.append({"entry_id": entry.entry_id, "rules": rules})
    return rows


def load_library(project: ProjectStore, reference: ArtifactInput) -> LibrarySnapshot:
    """Read an exact persisted library snapshot."""
    store = project.artifacts
    if artifact_input(store.read(reference.artifact_id).manifest) != reference:
        raise ValueError("library snapshot changed; name its exact saved reference")
    return LibrarySnapshot.model_validate(
        json.loads(store.read_bytes(reference.artifact_id, "library.json"))
    )


def preflight_screening(project: ProjectStore, spec: LessonScreeningSpec) -> ScreeningPreflight:
    """Count misses, saved and unknown decisions and new calls, with a conservative estimate.

    Args:
        project: Project with recorded learning of ``spec.scope``.
        spec: Screening request.

    Returns:
        The preflight. Reads only; credential-free.
    """
    plan, head = screening_plan(project, spec.scope)
    parent = None if spec.library is None else load_library(project, spec.library)
    planned = planned_screenings(
        project,
        learning_misses(project, plan, head),
        accepted_lessons(project, parent),
        alias=plan.recall.jev_alias,
        maximum_reply_bytes=spec.maximum_reply_bytes,
    )
    store = project.artifacts
    pending = [
        request
        for _, request, identity in planned
        if request is not None
        and identity is not None
        and not store.exists(identity)
        and not screening_dispatched(project, identity)
    ]
    unknown = sum(
        identity is not None
        and not store.exists(identity)
        and screening_dispatched(project, identity)
        for _, _, identity in planned
    )
    return ScreeningPreflight(
        lineage=plan.lineage,
        version=head.version,
        alias=plan.recall.jev_alias,
        misses=len(planned),
        no_hypothesis=sum(request is None for _, request, _ in planned),
        saved=sum(identity is not None and store.exists(identity) for _, _, identity in planned),
        unknown=unknown,
        to_dispatch=len(pending),
        estimated_cost_usd=screening_cost(project, plan.recall.jev_alias, pending),
    )


def screen_lessons(
    project: ProjectStore,
    spec: LessonScreeningSpec,
    *,
    code_revision: str,
    clock: Callable[[], datetime],
) -> tuple[ArtifactInput, LessonSet]:
    """Screen a learned checkpoint, fit the threshold and publish its lesson set.

    Args:
        project: Project with recorded learning of ``spec.scope``.
        spec: Screening request.
        code_revision: Recording implementation identity.
        clock: Aware clock.

    Returns:
        The published (or replayed) lesson set and its reference.
    """
    plan, head = screening_plan(project, spec.scope)
    parent = None if spec.library is None else load_library(project, spec.library)
    planned = planned_screenings(
        project,
        learning_misses(project, plan, head),
        accepted_lessons(project, parent),
        alias=plan.recall.jev_alias,
        maximum_reply_bytes=spec.maximum_reply_bytes,
    )
    screenings = screen_misses(
        project,
        planned,
        alias=plan.recall.jev_alias,
        deadline=clock() + DEADLINE,
        clock=clock,
        code_revision=code_revision,
    )
    fit = fit_threshold(
        screenings,
        delivery_outcomes(project, plan),
        thresholds=spec.thresholds,
        lineage=plan.lineage,
        version=head.version,
    )
    identities = [identity for _, _, identity in planned]
    fit_ref = save_fit(
        project,
        fit,
        screenings,
        planned=identities,
        created_at=clock(),
        code_revision=code_revision,
    )
    return _publish(
        project,
        spec,
        fit,
        fit_ref,
        list(zip(screenings, identities, strict=True)),
        parent,
        code_revision=code_revision,
        created_at=clock(),
    )


def _publish(
    project: ProjectStore,
    spec: LessonScreeningSpec,
    fit: ThresholdFit,
    fit_ref: ArtifactInput,
    screened: list[tuple[MissScreening, str | None]],
    parent: LibrarySnapshot | None,
    *,
    code_revision: str,
    created_at: datetime,
) -> tuple[ArtifactInput, LessonSet]:
    """Add screened-in lessons to the library and save the lesson set, or replay it."""
    store = project.artifacts
    identity = stable_id(
        LESSON_SET_TYPE,
        {
            "protocol": SCREENING_PROTOCOL,
            "fit": fit_ref.model_dump(mode="json"),
            "parent": None if spec.library is None else spec.library.model_dump(mode="json"),
        },
    )
    if store.exists(identity):
        return artifact_input(store.read(identity).manifest), LessonSet.model_validate_json(
            store.read_bytes(identity, "lessons.json")
        )
    envelope = ArtifactEnvelope(
        schema_version=1, created_at=created_at, code_revision=code_revision
    )
    # Cited so the new set supersedes them by causal order rather than by creation time.
    superseded = _current_lesson_sets(project, fit.lineage, fit.version)
    known = set() if parent is None else {entry.entry_id for entry in parent.entries}
    additions: list[LibraryEntry] = []
    rows: dict[str, list[ScreenedLesson]] = {"in": [], "out": [], "unknown": [], "none": []}
    for screening, decision in screened:
        miss, score = screening.miss, screening.score
        row = ScreenedLesson(miss=miss, screening=decision, score=score)
        if screening.status == "no-hypothesis":
            rows["none"].append(row)
        elif score is None:
            rows["unknown"].append(row)
        elif score <= fit.selected:
            rows["out"].append(row)
        else:
            assert decision is not None
            if miss.source.artifact_id not in known:
                known.add(miss.source.artifact_id)
                additions.append(_source_entry(project, miss.source))
            entry = _lesson_entry(project, miss, decision, score, envelope)
            if entry.entry_id not in known:
                known.add(entry.entry_id)
                additions.append(entry)
            rows["in"].append(row.model_copy(update={"lesson": entry.entry_id}))
    library = parent
    if additions:
        library = extend_library(
            store, additions, parent=parent, created_at=created_at, code_revision=code_revision
        )
    library_ref = (
        None if library is None else artifact_input(store.read(library.library_id).manifest)
    )
    lessons = LessonSet(
        selection=LessonSelection(
            lineage=fit.lineage,
            version=fit.version,
            lessons=tuple(row.miss.record.artifact_id for row in rows["in"]),
        ),
        fit=fit_ref,
        threshold=fit.selected,
        library=library_ref,
        screened_in=tuple(rows["in"]),
        screened_out=tuple(rows["out"]),
        unknown=tuple(rows["unknown"]),
        no_hypothesis=tuple(rows["none"]),
    )
    store.write_json(
        artifact_id=identity,
        artifact_type=LESSON_SET_TYPE,
        envelope=envelope.model_copy(
            update={
                "inputs": tuple(
                    sorted(
                        {
                            fit_ref,
                            *(() if library_ref is None else (library_ref,)),
                            *superseded,
                        },
                        key=lambda ref: ref.artifact_id,
                    )
                )
            }
        ),
        files={"selection.json": lessons.selection, "lessons.json": lessons},
    )
    return artifact_input(store.read(identity).manifest), lessons


def _source_entry(project: ProjectStore, source: ArtifactInput) -> LibraryEntry:
    """The consumed trajectory as ``source`` evidence; it is never a delivered lesson."""
    store = project.artifacts
    return LibraryEntry(
        entry_id=source.artifact_id,
        kind="trajectory",
        artifact=source,
        path="trajectory.json",
        payload_sha256=sha256_bytes(store.read_bytes(source.artifact_id, "trajectory.json")),
        disposition="source",
        cues=("trajectory:" + source.artifact_id,),
        derived_from=source,
    )


def _lesson_entry(
    project: ProjectStore,
    miss: LearningMiss,
    screening: str,
    score: float,
    envelope: ArtifactEnvelope,
) -> LibraryEntry:
    """Persist one screened-in lesson and reference it as an unresolved rule."""
    store = project.artifacts
    investigation = miss.investigation or {}
    cue = str(investigation.get("observable_cue") or "").strip()
    text = f"{cue}: {miss.hypothesis}" if cue else str(miss.hypothesis)
    identity = stable_id(LESSON_TYPE, {"screening": screening})
    if not store.exists(identity):
        payload: JsonObject = {
            "knowledge": {"rules": text, "source_trace_ids": [miss.trace_id]},
            "miss": miss.record.artifact_id,
            "screening": screening,
            "score": score,
            "authority": "Screened candidate lesson; unresolved until a held-out edit "
            "selection accepts it",
        }
        store.write_json(
            artifact_id=identity,
            artifact_type=LESSON_TYPE,
            envelope=envelope.model_copy(
                update={
                    "inputs": tuple(
                        sorted(
                            {artifact_input(store.read(screening).manifest), miss.source},
                            key=lambda ref: ref.artifact_id,
                        )
                    )
                }
            ),
            files={"lesson.json": payload},
        )
    action = miss.action[0] if isinstance(miss.action, list) and miss.action else miss.action
    name: JsonValue = action.get("name") if isinstance(action, dict) else None
    return LibraryEntry(
        entry_id=identity,
        kind="rule",
        artifact=artifact_input(store.read(identity).manifest),
        path="lesson.json",
        payload_sha256=sha256_bytes(store.read_bytes(identity, "lesson.json")),
        disposition="unresolved",
        cues=tuple(value for value in (str(name or ""), cue) if value),
        derived_from=miss.source,
        depends_on=miss.depends_on,
    )


def load_lesson_set(project: ProjectStore, reference: ArtifactInput) -> LessonSet:
    """Read a saved lesson set and verify it against its selection."""
    load_lesson_selection(project.artifacts, reference)
    return LessonSet.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "lessons.json")
    )


def latest_lesson_set(project: ProjectStore, lineage: str, version: int) -> ArtifactInput:
    """The current lesson set of one memory checkpoint: the one no later set supersedes.

    A lesson set saved for a checkpoint that already has one cites the current set in its
    inputs, so the newest is the one no other set of that checkpoint cites. Creation times are
    never compared: clock skew between the processes that screened cannot pick an older set.

    Args:
        project: Project holding the lesson sets.
        lineage: Memory lineage.
        version: Memory version that must have been screened.

    Returns:
        The lesson set reference.

    Raises:
        ValueError: The checkpoint has no lesson set, or several with no recorded order.
    """
    current = _current_lesson_sets(project, lineage, version)
    if not current:
        raise ValueError(
            f"memory {lineage} v{version} has no screened lesson set; run the repair-learning "
            "continuation with resolution screen-lessons first"
        )
    if len(current) > 1:
        raise ValueError(
            f"memory {lineage} v{version} has {len(current)} screened lesson sets that do not "
            f"supersede one another ({', '.join(item.artifact_id for item in current)}), so "
            "none is the latest; run screen-lessons again to save one that supersedes them"
        )
    return current[0]


def _current_lesson_sets(project: ProjectStore, lineage: str, version: int) -> list[ArtifactInput]:
    """Lesson sets of one checkpoint that no other set of the checkpoint cites, by ID."""
    store = project.artifacts
    found: list[ArtifactInput] = []
    for identity in store.list_ids(artifact_type=LESSON_SET_TYPE):
        reference = artifact_input(store.read(identity).manifest)
        selection = load_lesson_selection(store, reference)
        if (selection.lineage, selection.version) == (lineage, version):
            found.append(reference)
    return sorted(
        (
            item
            for item in found
            if not any(cites(store, other.artifact_id, item) for other in found if other != item)
        ),
        key=lambda item: item.artifact_id,
    )
