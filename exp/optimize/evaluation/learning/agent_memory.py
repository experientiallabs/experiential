"""Agent memory: lessons for a future agent, learned from outcome-labelled traces.

Learned simulator memory predicts how an environment replies. Agent memory instead tells the
next agent what earlier agents learned the hard way: task contracts a verifier enforces, user
preferences, environment rules, procedures that passed and pitfalls that failed. The learning
signal is each trace's recorded outcome, never a replay, so learning needs no world model and
no build: one chat JSON export of outcome-labelled traces and one catalog alias.

Learning runs in two paid stages, each consented separately because the second's size is known
only after the first:

1. ``extract_candidates`` imports the export as an immutable trace dataset, renders every
   trace within registered bounds, packs the renderings into batches and asks each batch for
   candidate lessons citing its traces (:mod:`agent_memory_requests`);
2. ``consolidate`` merges each kind's candidates in bounded rounds, then screens the merged
   lessons in blocks with the screening owner's three probabilities and score
   (:func:`exp.optimize.evaluation.learning.screening.screening_score`), comparing each
   against lessons accepted before it. A lesson is accepted at the registered threshold.

``prepare_agent_memory`` registers the plan before any call: the dataset, the bounds and the
model snapshot. The first paid stage registers the plan's window, whose deadline every request
is fingerprinted with, so a rerun finds the same plan and window and replays saved answers
without calls. Every call goes through the structured completion
owner over the admitted transport and the plan's request budget: its intent is saved before
dispatch, its answer after. A returned answer that is refused, or a dispatch without a saved
answer, is recorded and never redrawn; its batch adds no candidates, its merge leaves its
candidates unmerged and its screening leaves its lessons unknown and unexported.

:mod:`agent_memory_export` writes accepted lessons as a markdown folder: ``MEMORY.md`` indexes
one file per kind. Citations stay in the saved memory artifact, which names every trace a lesson
came from.
"""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    canonical_json_bytes,
    sha256_json,
    stable_id,
)
from exp.common.models import ModelCatalog, load_model_catalog
from exp.common.models.unknown_outcome import DispatchOutcomeUnknown
from exp.common.project import ProjectStore, artifact_input
from exp.common.traces.ingest.dataset import persist_trace_dataset
from exp.common.traces.ingest.sources import load_trace_source
from exp.optimize.evaluation.executable.model_costs import CostComponent, model_cost, total_cost
from exp.optimize.evaluation.learning.agent_memory_contracts import (
    AgentMemory,
    AgentMemoryPlan,
    AgentMemorySettings,
    AgentMemorySpec,
    AgentMemoryWindow,
    CandidateSet,
    FailedCall,
    Lesson,
    ScreenedLesson,
    Stage,
)
from exp.optimize.evaluation.learning.agent_memory_requests import (
    KINDS,
    SHOWN_OUTCOMES,
    LessonKind,
    RenderedTrace,
    ShownOutcome,
    batches,
    decode_extraction,
    decode_merge,
    decode_screening,
    extract_request,
    merge_request,
    outcome_counts,
    render_trace,
    screen_request,
)
from exp.optimize.evaluation.learning.screening import screening_score
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.admitted_transport import (
    AdmittedJsonTransport,
    require_admitted_transport,
)
from exp.runtime.models.providers.transport import HttpxJsonTransport, RetryPolicy
from exp.runtime.models.structured_completion import (
    CompletionPlan,
    ReturnedOutputRejected,
    StructuredCompletion,
)
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.engines.transition import Completion

# Identity of the code that decides this artifact's behavior, in place of its source text.
# Bump it when behavior changes. exp/tests/code_protocols_test.py fails when the covered
# source changes until it is bumped or, for a pure refactor, its digest is updated.
AGENT_MEMORY_PROTOCOL = "agent-memory-v2"
PLAN_TYPE = "agent-memory-plan"
WINDOW_TYPE = "agent-memory-window"
CANDIDATES_TYPE = "agent-memory-candidates"
MEMORY_TYPE = "agent-memory"
MERGE_ROUNDS = 4
SCREEN_BLOCK = 20
MAXIMUM_LESSONS = 40
# Instructions, schema and provider wire framing around the bounded content of every request.
REQUEST_MARGIN_BYTES = 16_000
OUTPUT_TOKENS: dict[Stage, int] = {"extract": 32_000, "merge": 32_000, "screen": 8_000}


@dataclass(frozen=True)
class PreparedMemory:
    """A registered plan and the renderings its requests show.

    Attributes:
        plan: Registered plan.
        reference: The plan's exact artifact reference.
        traces: Renderings in dataset order.
        batches: Trace positions per extraction batch.
        alias_sha256: Digest of the alias's catalog record and connection as planned and
            priced; dispatch refuses a catalog whose alias no longer matches it.
    """

    plan: AgentMemoryPlan
    reference: ArtifactInput
    traces: tuple[RenderedTrace, ...]
    batches: tuple[tuple[int, ...], ...]
    alias_sha256: str


@dataclass(frozen=True)
class Completions:
    """One structured completion per stage, sharing the plan's budget and transport.

    Attributes:
        extract: Extraction calls.
        merge: Merge calls.
        screen: Screening calls.
        concurrency: Requests in flight at once within a stage.
    """

    extract: Completion
    merge: Completion
    screen: Completion
    concurrency: int


def prepare_agent_memory(
    project: ProjectStore,
    spec: AgentMemorySpec,
    *,
    code_revision: str,
    clock: Callable[[], datetime],
) -> PreparedMemory:
    """Import the traces and register the plan, credential-free and without provider calls.

    Both artifacts are content-addressed, so registering again replays them unchanged.

    Args:
        project: Project that owns the dataset, plan and every later artifact.
        spec: Learning specification.
        code_revision: Recording implementation identity.
        clock: Aware clock.

    Returns:
        The registered plan and the renderings its requests show.

    Raises:
        ValueError: The export holds records that cannot be normalized and exclusions are not
            allowed, the alias does not use the admitted transport, or the learning model has
            no input or output price.
    """
    catalog = load_model_catalog(project.model_catalog_path)
    require_admitted_transport(catalog, spec.alias, consumer="agent memory")
    model, _ = RuntimeModelCatalog(catalog, environment={}).snapshot(spec.alias)
    if model_cost(project, spec.alias, 1, OUTPUT_TOKENS["extract"]).usd is None:
        raise ValueError(
            f"model alias {spec.alias!r} has no input or output price or no context window; add "
            "them to the model catalog so every request has a known reservation"
        )
    result = load_trace_source("chat-json", spec.traces)
    if result.issues and not spec.allow_exclusions:
        raise ValueError(
            f"{len(result.issues)} records in {spec.traces} cannot be normalized (first: "
            f"{result.issues[0].message}); fix the export, or set allow_exclusions to learn from "
            "the valid traces"
        )
    dataset = persist_trace_dataset(
        result, project.artifacts, created_at=clock(), code_revision=code_revision
    )
    settings = AgentMemorySettings.model_validate(
        spec.model_dump(include=set(AgentMemorySettings.model_fields))
    )
    reference = artifact_input(dataset.manifest)
    plan_id = stable_id(
        PLAN_TYPE,
        {
            "protocol": AGENT_MEMORY_PROTOCOL,
            "dataset": reference.model_dump(mode="json"),
            "settings": settings.model_dump(mode="json"),
            "model": model.model_dump(mode="json"),
        },
    )
    plan_reference, plan = _write(
        project,
        plan_id,
        PLAN_TYPE,
        "plan.json",
        AgentMemoryPlan(
            schema_version=1,
            created_at=clock(),
            inputs=(reference,),
            code_revision=code_revision,
            plan_id=plan_id,
            protocol=AGENT_MEMORY_PROTOCOL,
            dataset=reference,
            settings=settings,
            model=model,
        ),
    )
    traces = tuple(
        render_trace(
            trace,
            f"T{index + 1}",
            maximum_message_bytes=spec.maximum_message_bytes,
            maximum_trace_bytes=spec.maximum_trace_bytes,
        )
        for index, trace in enumerate(dataset.traces)
    )
    return PreparedMemory(
        plan=plan,
        reference=plan_reference,
        traces=traces,
        batches=batches(traces, spec.batch_bytes),
        alias_sha256=_alias_sha256(catalog, spec.alias),
    )


def extraction_cost(project: ProjectStore, prepared: PreparedMemory) -> CostComponent:
    """Conservative cost of every extraction request, priced by the shared cost owner.

    Args:
        project: Project whose catalog prices the learning model.
        prepared: Registered plan.

    Returns:
        One component: every extraction request at the largest one's input and the whole
        generation bound the spend ceiling reserves.
    """
    sizes = [_json_bytes(_extract_body(prepared, batch)) for batch in prepared.batches]
    return _priced(
        project, prepared, len(sizes), "extract", max(sizes, default=0) + REQUEST_MARGIN_BYTES
    )


def consolidation_cost(
    project: ProjectStore, prepared: PreparedMemory, candidates: CandidateSet
) -> tuple[CostComponent, CostComponent]:
    """Conservative cost of merging and screening a candidate set.

    The first merge round is charged its exact chunks. A merge answer can rewrite lessons
    longer than their sources, so a later round may pack into more chunks; but a merge never
    returns more lessons than it was shown, so a later round sends at most one request per two
    of the kind's candidates, each within the chunk limit. A kind that fits one chunk has no
    later round. Every screening block is charged its own candidates, as if none had merged,
    plus the most accepted lessons it can show.

    Args:
        project: Project whose catalog prices the learning model.
        prepared: Registered plan.
        candidates: Extraction stage result.

    Returns:
        The merge and screening components.
    """
    limit = prepared.plan.settings.batch_bytes
    merge: list[int] = []
    screen: list[int] = []
    for kind in KINDS:
        rows = [row for row in candidates.candidates if row.kind == kind]
        shown = [_json_bytes(_shown_candidate(row)) for row in rows]
        # The same packer as execution, so each chunk is one merge request of the first round.
        chunks = _chunks(rows, limit, _shown_candidate)
        merge.extend(
            sum(_json_bytes(_shown_candidate(row)) for row in chunk) + REQUEST_MARGIN_BYTES
            for chunk in chunks
            if len(chunk) > 1
        )
        if len(chunks) > 1:
            later = (MERGE_ROUNDS - 1) * (len(rows) // 2)
            merge.extend([limit + REQUEST_MARGIN_BYTES] * later)
        accepted = min(sum(shown), limit)
        screen.extend(
            sum(shown[start : start + SCREEN_BLOCK]) + accepted + REQUEST_MARGIN_BYTES
            for start in range(0, len(shown), SCREEN_BLOCK)
        )
    return (
        _priced(project, prepared, len(merge), "merge", max(merge, default=0)),
        _priced(project, prepared, len(screen), "screen", max(screen, default=0)),
    )


def request_reservation(project: ProjectStore, prepared: PreparedMemory) -> float:
    """What one in-flight request reserves under the spend ceiling, at the plan's largest size.

    An explicit ``spend.ceiling_usd`` must leave this much headroom for every concurrent request.

    Args:
        project: Project whose catalog prices the learning model.
        prepared: Registered plan.

    Returns:
        USD reserved by the largest request of any stage.
    """
    size = prepared.plan.settings.batch_bytes + REQUEST_MARGIN_BYTES
    return max(known_cost(_priced(project, prepared, 1, stage, size)) for stage in OUTPUT_TOKENS)


def known_cost(*components: CostComponent) -> float:
    """The known total of a priced plan's components.

    Args:
        components: Components priced by this module.

    Returns:
        Their total in USD.

    Raises:
        ValueError: A component is unpriced, which planning refuses unless the catalog changed.
    """
    total = total_cost(components)
    if total is None:
        raise ValueError(
            "the learning model lost its prices after planning; restore them in the model catalog"
        )
    return total


def stage_completions(
    project: ProjectStore,
    prepared: PreparedMemory,
    *,
    hours: float,
    concurrency: int,
    code_revision: str,
    clock: Callable[[], datetime],
) -> Completions:
    """Build the plan's admitted, budgeted structured completions, one per stage.

    The first call registers the plan's window, ``hours`` from now; later calls reuse it, so
    every request keeps its fingerprint and saved answers replay. Credentials are read here,
    so callers build these only after spend consent, and building refuses unless that consent
    installed the operation's spend ceiling: the transport reserves against the ceiling it
    finds here, so completions built earlier would dispatch unreserved.

    Args:
        project: Project owning the window, budget ledger and call artifacts.
        prepared: Registered plan.
        hours: Window length, used only when the window is first registered.
        concurrency: Requests in flight at once within a stage.
        code_revision: Recording implementation identity.
        clock: Aware clock.

    Returns:
        The three stage completions.

    Raises:
        ValueError: No operation spend ceiling is installed, or the alias's catalog record or
            connection changed since planning, including its capabilities and prices.
    """
    plan = prepared.plan
    identity = plan.plan_id
    deadline = _window(project, prepared, hours, code_revision, clock).deadline
    transport = AdmittedJsonTransport(
        HttpxJsonTransport(),
        ledger=DispatchLedger(
            project.artifacts.project_directory / (identity + ".sqlite"),
            RetryPolicy().maximum_attempts * _request_bound(prepared),
            sha256_json(identity),
        ),
        artifacts=project.artifacts,
        code_revision=code_revision,
        deadline=deadline,
        clock=clock,
    )
    if transport.spend_ceiling is None:
        raise ValueError(
            "agent memory dispatches only under the operation spend ceiling; build its stage "
            "completions after the stage's spend consent"
        )
    catalog = load_model_catalog(project.model_catalog_path)
    resolved = RuntimeModelCatalog(catalog, transport_factory=lambda: transport).resolve(
        plan.settings.alias
    )
    if (
        resolved.snapshot != plan.model
        or _alias_sha256(catalog, plan.settings.alias) != prepared.alias_sha256
    ):
        raise ValueError(
            f"model alias {plan.settings.alias!r} changed in the model catalog after the plan "
            f"was registered with {plan.model.model_id!r}; restore it, or rerun to plan anew"
        )
    budget = RequestBudget(project, identity=identity)
    limit = resolved.capabilities.maximum_output_tokens

    def stage(name: Stage) -> StructuredCompletion:
        """One stage's completion with its own output reservation."""
        tokens = OUTPUT_TOKENS[name] if limit is None else min(limit, OUTPUT_TOKENS[name])
        return StructuredCompletion(
            client=resolved.client,
            plan=CompletionPlan(
                model=resolved.snapshot,
                capabilities=resolved.capabilities,
                deadline=deadline,
                output_tokens=tokens,
                maximum_cost_per_call=known_cost(
                    _priced(
                        project,
                        prepared,
                        1,
                        name,
                        prepared.plan.settings.batch_bytes + REQUEST_MARGIN_BYTES,
                    )
                ),
                code_revision=code_revision,
                output_protocol="tool",
            ),
            budget=budget,
            artifacts=project.artifacts,
            clock=clock,
        )

    return Completions(
        extract=stage("extract"),
        merge=stage("merge"),
        screen=stage("screen"),
        concurrency=concurrency,
    )


def candidates_identity(prepared: PreparedMemory) -> str:
    """Identity of the plan's candidate set, saved once extraction completes."""
    return stable_id(CANDIDATES_TYPE, {"plan": prepared.reference.model_dump(mode="json")})


def memory_identity(candidates: ArtifactInput) -> str:
    """Identity of the agent memory consolidated from one saved candidate set."""
    return stable_id(MEMORY_TYPE, {"candidates": candidates.model_dump(mode="json")})


def extract_candidates(
    project: ProjectStore,
    prepared: PreparedMemory,
    completions: Callable[[], Completions],
    *,
    code_revision: str,
    clock: Callable[[], datetime],
) -> tuple[ArtifactInput, CandidateSet]:
    """Ask every batch for candidate lessons and save them; a saved set replays without calls.

    Args:
        project: Project owning the candidate set.
        prepared: Registered plan.
        completions: Builds the stage completions, called only when the stage dispatches.
        code_revision: Recording implementation identity.
        clock: Aware clock.

    Returns:
        The saved candidate set and its reference.
    """
    identity = candidates_identity(prepared)
    if project.artifacts.exists(identity):
        return _read(project, identity, "candidates.json", CandidateSet)
    stages = completions()
    failed: list[FailedCall] = []

    def ask(position: int) -> tuple[Lesson, ...]:
        """One batch's candidates, or none when its answer is not usable."""
        shown = [prepared.traces[index] for index in prepared.batches[position]]
        logical_id = stable_id(
            "agent-memory-extract", {"plan": prepared.plan.plan_id, "batch": position}
        )
        body = _extract_body(prepared, prepared.batches[position])
        labels = [trace.label for trace in shown]
        lessons = _ask(
            stages.extract,
            body,
            logical_id,
            "extract",
            failed,
            partial(decode_extraction, labels=labels),
        )
        ids = {trace.label: trace.trace_id for trace in shown}
        return tuple(
            Lesson(
                kind=row.kind,
                statement=row.statement,
                applies_when=row.applies_when,
                evidence=(row.evidence,),
                traces=tuple(sorted({ids[label] for label in row.traces})),
            )
            for row in lessons or ()
        )

    found = _in_parallel(ask, range(len(prepared.batches)), stages.concurrency)
    candidates = CandidateSet(
        schema_version=1,
        created_at=clock(),
        inputs=(prepared.reference,),
        code_revision=code_revision,
        candidates_id=identity,
        plan=prepared.reference,
        candidates=tuple(row for rows in found for row in rows),
        failed=_distinct(failed),
    )
    return _write(project, identity, CANDIDATES_TYPE, "candidates.json", candidates)


def consolidate(
    project: ProjectStore,
    prepared: PreparedMemory,
    candidates: tuple[ArtifactInput, CandidateSet],
    completions: Callable[[], Completions],
    *,
    code_revision: str,
    clock: Callable[[], datetime],
) -> tuple[ArtifactInput, AgentMemory]:
    """Merge and screen a candidate set into agent memory; saved memory replays without calls.

    Args:
        project: Project owning the memory.
        prepared: Registered plan.
        candidates: The extraction stage's saved set and its reference.
        completions: Builds the stage completions, called only when the stage dispatches.
        code_revision: Recording implementation identity.
        clock: Aware clock.

    Returns:
        The saved agent memory and its reference.
    """
    reference, found = candidates
    identity = memory_identity(reference)
    if project.artifacts.exists(identity):
        return _read(project, identity, "memory.json", AgentMemory)
    stages = completions()
    failed: list[FailedCall] = []
    merged = [
        lesson
        for kind in KINDS
        for lesson in _merge_kind(
            prepared, [row for row in found.candidates if row.kind == kind], stages, failed
        )
    ]
    merged.sort(key=lambda row: (-len(row.traces), KINDS.index(row.kind), row.statement))
    outcomes = {trace.trace_id: trace.outcome for trace in prepared.traces}
    screened = _screen(prepared, merged, outcomes, stages, failed)
    memory = AgentMemory(
        schema_version=1,
        created_at=clock(),
        inputs=tuple(sorted((prepared.reference, reference), key=lambda ref: ref.artifact_id)),
        code_revision=code_revision,
        memory_id=identity,
        plan=prepared.reference,
        candidates=reference,
        outcomes=outcome_counts(prepared.traces),
        lessons=tuple(screened),
        failed=_distinct(failed),
    )
    return _write(project, identity, MEMORY_TYPE, "memory.json", memory)


def _window(
    project: ProjectStore,
    prepared: PreparedMemory,
    hours: float,
    code_revision: str,
    clock: Callable[[], datetime],
) -> AgentMemoryWindow:
    """The plan's saved window, registering it ``hours`` from now on first use."""
    identity = stable_id(WINDOW_TYPE, {"plan": prepared.reference.model_dump(mode="json")})
    if project.artifacts.exists(identity):
        return _read(project, identity, "window.json", AgentMemoryWindow)[1]
    now = clock()
    window = AgentMemoryWindow(
        schema_version=1,
        created_at=now,
        inputs=(prepared.reference,),
        code_revision=code_revision,
        window_id=identity,
        plan=prepared.reference,
        deadline=now + timedelta(hours=hours),
    )
    return _write(project, identity, WINDOW_TYPE, "window.json", window)[1]


def _merge_kind(
    prepared: PreparedMemory,
    rows: list[Lesson],
    completions: Completions,
    failed: list[FailedCall],
) -> list[Lesson]:
    """Merge one kind's candidates in rounds until one request holds them all, at most four.

    Chunks of one round are independent and run concurrently. A failed merge leaves its
    chunk's candidates unmerged and out of every later round, so no other request asks about
    them again; they follow the merged lessons in round and chunk order.
    """
    unmerged: list[Lesson] = []
    for _ in range(MERGE_ROUNDS):
        chunks = _chunks(rows, prepared.plan.settings.batch_bytes, _shown_candidate)
        results = _in_parallel(
            lambda chunk: _merge_chunk(chunk, completions, failed, prepared.plan.plan_id),
            chunks,
            completions.concurrency,
        )
        rows = []
        for chunk, merged in zip(chunks, results, strict=True):
            if merged is None:
                unmerged.extend(chunk)
            else:
                rows.extend(merged)
        if len(chunks) <= 1:
            break
    return rows + unmerged


def _merge_chunk(
    chunk: Sequence[Lesson], completions: Completions, failed: list[FailedCall], plan_id: str
) -> list[Lesson] | None:
    """One merge call: each group becomes one lesson citing all its sources' traces.

    A lone candidate needs no call and is returned as it is; a failed merge returns None.
    """
    if len(chunk) < 2:
        return list(chunk)
    kind = chunk[0].kind
    body = merge_request(kind, [_shown_candidate(row) for row in chunk])
    logical_id = stable_id(
        "agent-memory-merge", {"plan": plan_id, "request_sha256": sha256_json(body)}
    )
    groups = _ask(
        completions.merge,
        body,
        logical_id,
        "merge",
        failed,
        partial(decode_merge, candidates=len(chunk)),
    )
    if groups is None:
        return None
    return [
        Lesson(
            kind=kind,
            statement=group.statement,
            applies_when=group.applies_when,
            evidence=tuple(
                dict.fromkeys(note for i in group.sources for note in chunk[i].evidence)
            ),
            traces=tuple(sorted({trace for i in group.sources for trace in chunk[i].traces})),
        )
        for group in groups
    ]


def _screen(
    prepared: PreparedMemory,
    lessons: Sequence[Lesson],
    outcomes: Mapping[str, ShownOutcome],
    completions: Completions,
    failed: list[FailedCall],
) -> list[ScreenedLesson]:
    """Screen lessons in blocks of one kind, in order; each block sees lessons accepted before.

    Blocks run in sequence because coverage depends on what earlier blocks accepted.
    """
    screened: list[ScreenedLesson] = []
    for kind in KINDS:
        rows = [lesson for lesson in lessons if lesson.kind == kind]
        for start in range(0, len(rows), SCREEN_BLOCK):
            block = rows[start : start + SCREEN_BLOCK]
            accepted = _accepted_shown(screened, kind, prepared.plan.settings.batch_bytes)
            body = screen_request([_shown_lesson(row, outcomes) for row in block], accepted)
            logical_id = stable_id(
                "agent-memory-screen",
                {"plan": prepared.plan.plan_id, "request_sha256": sha256_json(body)},
            )
            decisions = _ask(
                completions.screen,
                body,
                logical_id,
                "screen",
                failed,
                partial(decode_screening, lessons=len(block)),
            ) or (None,) * len(block)
            for lesson, decision in zip(block, decisions, strict=True):
                score = (
                    None
                    if decision is None
                    else screening_score(decision.general, decision.covered, decision.supported)
                )
                screened.append(
                    ScreenedLesson(
                        **lesson.model_dump(),
                        screening=decision,
                        score=score,
                        accepted=score is not None and score >= prepared.plan.settings.threshold,
                    )
                )
    return screened


def _ask[DecodedT](
    complete: Completion,
    body: JsonObject,
    logical_id: str,
    stage: Stage,
    failed: list[FailedCall],
    decode: Callable[[JsonObject], DecodedT],
) -> DecodedT | None:
    """Send or replay one request and decode its answer; an unusable answer returns None.

    A returned answer that the completion owner or ``decode`` refuses is recorded as
    ``rejected``, a dispatch without a saved answer as ``unknown``; neither is redrawn. Every
    other failure, including a spend stop or the deadline, propagates: the run stops with its
    completed calls saved and resumes in place.
    """
    try:
        answer = complete(body, logical_id)
    except ReturnedOutputRejected as error:
        status, reason = "rejected", str(error)
    except DispatchOutcomeUnknown:
        status, reason = "unknown", "dispatched without a saved answer; never redrawn"
    else:
        try:
            return decode(answer)
        except ValueError as error:
            status, reason = "rejected", f"answer refused: {error}"
    failed.append(FailedCall(stage=stage, logical_id=logical_id, status=status, reason=reason))
    return None


def _distinct(failed: Sequence[FailedCall]) -> tuple[FailedCall, ...]:
    """Each failed request once, in stage and identity order.

    Requests with identical bodies share one logical identity and are listed once.
    """
    unique = {row.logical_id: row for row in failed}
    return tuple(sorted(unique.values(), key=lambda row: (row.stage, row.logical_id)))


def _in_parallel[ItemT, ResultT](
    work: Callable[[ItemT], ResultT], items: Iterable[ItemT], concurrency: int
) -> list[ResultT]:
    """Results of ``work`` over ``items`` in input order, at most ``concurrency`` at once.

    The first failure cancels every item not yet started and is raised once running ones end.
    """
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, item) for item in items]
        try:
            return [future.result() for future in futures]
        except BaseException:
            # A stop dispatches nothing more: requests not yet started are cancelled.
            for future in futures:
                future.cancel()
            raise


def _extract_body(prepared: PreparedMemory, batch: Sequence[int]) -> JsonObject:
    """The extraction request of one batch."""
    return extract_request(
        [prepared.traces[index] for index in batch], maximum_lessons=MAXIMUM_LESSONS
    )


def _shown_candidate(lesson: Lesson) -> JsonObject:
    """A candidate as a merge request shows it: statement, condition and first evidence."""
    return {
        "statement": lesson.statement,
        "applies_when": lesson.applies_when,
        "evidence": lesson.evidence[0],
    }


def _shown_lesson(lesson: Lesson, outcomes: Mapping[str, ShownOutcome]) -> JsonObject:
    """A lesson as a screening request shows it, with its cited traces' outcomes."""
    cited = [outcomes[trace] for trace in lesson.traces]
    return {
        "kind": lesson.kind,
        "statement": lesson.statement,
        "applies_when": lesson.applies_when,
        "evidence": list(lesson.evidence[:3]),
        "cited_outcomes": {label: cited.count(label) for label in SHOWN_OUTCOMES},
    }


def _accepted_shown(
    screened: Sequence[ScreenedLesson], kind: LessonKind, limit: int
) -> list[JsonObject]:
    """Accepted lessons of one kind, most supported first, within ``limit`` bytes."""
    shown: list[JsonObject] = []
    size = 0
    for lesson in screened:
        if not lesson.accepted or lesson.kind != kind:
            continue
        row: JsonObject = {"kind": lesson.kind, "statement": lesson.statement}
        size += _json_bytes(row)
        if size > limit:
            break
        shown.append(row)
    return shown


def _chunks[ItemT](
    rows: Sequence[ItemT], limit: int, shown: Callable[[ItemT], JsonObject]
) -> list[list[ItemT]]:
    """Consecutive rows packed into chunks of at most ``limit`` bytes as requests show them."""
    chunks: list[list[ItemT]] = []
    size = 0
    for row in rows:
        length = _json_bytes(shown(row))
        if not chunks or size + length > limit:
            chunks.append([])
            size = 0
        chunks[-1].append(row)
        size += length
    return chunks


def _alias_sha256(catalog: ModelCatalog, alias: str) -> str:
    """Digest of one alias's whole catalog record and its connection."""
    record = catalog.models[alias]
    return sha256_json(
        {
            "record": record.model_dump(mode="json"),
            "connection": catalog.connections[record.connection].model_dump(mode="json"),
        }
    )


def _json_bytes(value: JsonObject) -> int:
    """UTF-8 size of the value's JSON text."""
    return len(json.dumps(value, ensure_ascii=False).encode())


def _priced(
    project: ProjectStore, prepared: PreparedMemory, attempts: int, stage: Stage, size: int
) -> CostComponent:
    """``attempts`` requests of one stage at ``size`` input bytes, one byte per token."""
    if not attempts:
        return CostComponent(prepared.plan.settings.alias, 0, 0.0)
    return model_cost(
        project, prepared.plan.settings.alias, attempts, OUTPUT_TOKENS[stage], input_tokens=size
    )


def _request_bound(prepared: PreparedMemory) -> int:
    """Upper bound on the requests one plan sends across its three stages."""
    extraction = len(prepared.batches)
    candidates = extraction * MAXIMUM_LESSONS
    return extraction + MERGE_ROUNDS * candidates + candidates


def _write[EnvelopeT: ArtifactEnvelope](
    project: ProjectStore, identity: str, kind: str, path: str, envelope: EnvelopeT
) -> tuple[ArtifactInput, EnvelopeT]:
    """Save a typed envelope as one immutable artifact and return its reference."""
    saved, manifest = project.artifacts.write_or_replay(
        artifact_id=identity,
        artifact_type=kind,
        envelope=envelope,
        envelope_path=path,
        envelope_type=type(envelope),
        files={path: canonical_json_bytes(envelope)},
    )
    return artifact_input(manifest), saved


def _read[EnvelopeT: ArtifactEnvelope](
    project: ProjectStore, identity: str, path: str, kind: type[EnvelopeT]
) -> tuple[ArtifactInput, EnvelopeT]:
    """Read a saved typed envelope and its reference."""
    stored = project.artifacts.read(identity)
    return artifact_input(stored.manifest), kind.model_validate_json(
        project.artifacts.read_bytes(identity, path)
    )
