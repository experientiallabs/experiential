"""Screen learning misses into reusable lessons before they reach later simulators.

Predict-then-learn learning records a miss on every step whose prediction differs from the
recorded reply, and learned memory serves all of them. Most misses describe one instance's data
(contents the simulator could not see), not how the environment behaves, so serving every miss
dilutes memory. Screening asks one cheap typed native decision per miss, with three
probabilities and nothing else:

- ``behavioural``: the miss is about how the environment behaves, not this instance's data;
- ``covered``: accepted library lessons already state it;
- ``supported``: the recorded reply supports the miss's hypothesis.

A miss's score is ``behavioural * supported * (1 - covered)``. The threshold is a registered
grid, fitted on the fit split from saved learning evidence without a provider call. Every later
fit prediction of the same tool that was delivered a miss's record or its overview line (alone
or inside a summary) counts +1 when it was exact and -1 when it was not; a lesson is about how
one tool replies, so a delivery to another tool's prediction is not evidence about it. The grid
value with the greatest total over retained misses is selected (ties keep the higher value, so
without delivery evidence the highest value is kept). The fit is saved and binds the lesson set.
Deliveries are observational: several records reach one prediction together, so this credit
is an association on the fit split, not a causal effect; the held-out audit measures that.

Each decision is dispatched once through the admitted transport with a one-attempt ledger: its
intent is saved before dispatch and its answer after. An intent without an answer, a failed
dispatch or a malformed answer is saved as ``unknown``, is never redrawn, and is screened out.
A miss without an investigation hypothesis is screened out without a call.
"""

import json
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelSnapshot, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.fidelity.decision_access import decision_caller
from exp.optimize.evaluation.learning.traces import LearningPlan, Scope, learned_runs
from exp.runtime.gateway.decisions_contracts import (
    ChoiceQuestion,
    DecisionRequest,
    NoulCriteria,
    NoulQuestion,
    ScoreQuestion,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.call_evidence import canonical_call
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.registry import ResolvedModel
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.runtime.sessions.store import SessionStore
from exp.simulation.reconstruction.library import LibrarySnapshot, resolve_library_entry
from exp.simulation.world_model.experience_memory import MemoryVersion
from exp.simulation.world_model.memory_tree import read_line

# Identity of the code that decides this artifact's behavior, in place of its source text.
# Bump it when behavior changes. exp/tests/code_protocols_test.py fails when the covered
# source changes until it is bumped or, for a pure refactor, its digest is updated.
SCREENING_PROTOCOL = "learning-miss-screening-v1"
SCREENING_TYPE = "learning-miss-screening"
FIT_TYPE = "learning-lesson-threshold-fit"
QUESTIONS = ("behavioural", "covered", "supported")
Status = Literal["answered", "unknown", "no-hypothesis"]


class LessonScreeningSpec(ContractModel):
    """Screen one learned memory checkpoint's misses and publish its lesson set.

    Attributes:
        resolution: Selects screening in the ``repair-learning`` action.
        scope: Learning plan whose memory head is screened; ``fit`` keeps the build's held-out
            lineages out, so an independent audit can compare the lessons.
        thresholds: Registered score grid the threshold is fitted on, strictly increasing.
        maximum_reply_bytes: Bound on each reply shown to a decision; longer replies are cut
            and labelled.
        library: Prior lesson library whose accepted lessons count as already covered.
    """

    resolution: Literal["screen-lessons"]
    scope: Scope = "fit"
    thresholds: tuple[float, ...] = Field(
        default=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), min_length=1
    )
    maximum_reply_bytes: int = Field(default=4_000, ge=256, le=32_000, strict=True)
    library: ArtifactInput | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def ordered(self) -> "LessonScreeningSpec":
        """Require a strictly increasing grid inside the unit interval."""
        grid = self.thresholds
        if any(not 0 <= value < 1 for value in grid) or any(
            b <= a for a, b in zip(grid, grid[1:], strict=False)
        ):
            raise ValueError("screening thresholds must increase strictly within [0, 1)")
        return self


class LearningMiss(ContractModel):
    """One learned prediction miss, as its memory library recorded it.

    Attributes:
        record: Session research record of the miss.
        run_id: Learning run (and session) that predicted it.
        trace_id: Recorded trace the run replayed.
        task_lineage: Task lineage of that trace.
        source: Consumed trajectory the record was derived from.
        depends_on: Other sources supplied to the work that produced the record.
        action: Public action whose reply was mispredicted.
        predicted_reply: Simulator reply.
        observed_reply: Recorded reply disclosed after the prediction.
        investigation: Investigator output, absent when none ran.
    """

    record: ArtifactInput
    run_id: str
    trace_id: str
    task_lineage: str
    source: ArtifactInput
    depends_on: tuple[ArtifactInput, ...] = ()
    action: JsonValue
    predicted_reply: JsonValue
    observed_reply: JsonValue
    investigation: JsonObject | None = None

    @property
    def tool(self) -> str | None:
        """Name of the mispredicted action's tool."""
        return _tool(self.action)

    @property
    def hypothesis(self) -> str | None:
        """The investigation's hypothesis, if it stated one."""
        text = (self.investigation or {}).get("hypothesis")
        return text.strip() or None if isinstance(text, str) else None


class MissScreening(ContractModel):
    """The saved screening decision of one miss: probabilities, or why there are none.

    Attributes:
        miss: Screened miss.
        status: ``answered``; ``unknown`` (dispatched without a usable answer, never redrawn);
            or ``no-hypothesis`` (screened out without a call).
        request_sha256: Decision request, absent without a call.
        behavioural: Probability the miss is about environment behaviour.
        covered: Probability accepted lessons already cover it.
        supported: Probability the recorded reply supports its hypothesis.
        reason: Why the outcome is unknown, when it is.
    """

    miss: LearningMiss
    status: Status
    request_sha256: Sha256 | None = None
    behavioural: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    covered: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    supported: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    reason: str | None = None

    @property
    def score(self) -> float | None:
        """Reusable, uncovered and supported: the product of the three probabilities."""
        if self.behavioural is None or self.covered is None or self.supported is None:
            return None
        return screening_score(self.behavioural, self.covered, self.supported)


def screening_score(reusable: float, covered: float, supported: float) -> float:
    """Screening score of one candidate lesson: reusable, uncovered and supported.

    Args:
        reusable: Probability the lesson holds beyond the instance it came from.
        covered: Probability accepted lessons already state it.
        supported: Probability its evidence supports it.

    Returns:
        ``reusable * supported * (1 - covered)``, compared against a registered threshold.
    """
    return reusable * supported * (1 - covered)


class ThresholdFit(ContractModel):
    """The threshold fitted on the fit split, with the objective at every grid value.

    Attributes:
        lineage: Memory lineage screened.
        version: Memory version screened.
        thresholds: Registered grid.
        objective: Net delivered outcome of retained misses at each grid value.
        retained: Misses scoring above each grid value.
        selected: Fitted threshold; a miss is screened in when its score exceeds it.
        status: ``fitted``, or ``no-delivery-evidence`` when no screened record was delivered.
        deliveries: Later same-tool fit predictions delivered a screened record.
    """

    lineage: str
    version: int = Field(ge=1)
    thresholds: tuple[float, ...]
    objective: tuple[int, ...]
    retained: tuple[int, ...]
    selected: float
    status: Literal["fitted", "no-delivery-evidence"]
    deliveries: int = Field(ge=0)


def learning_misses(
    project: ProjectStore, plan: LearningPlan, version: MemoryVersion
) -> tuple[LearningMiss, ...]:
    """Every consumed miss of a memory version's learned runs, in library order.

    Args:
        project: Project with a recorded learning plan.
        plan: That plan.
        version: Memory version whose library holds the miss records.

    Returns:
        Misses of learned runs whose records the version admitted, with typed provenance.
    """
    store = project.artifacts
    traces = {prepared.id: case for case, prepared, _ in learned_runs(project, plan)}
    if artifact_input(store.read(version.library.artifact_id).manifest) != version.library:
        raise ValueError("memory library manifest changed")
    library = LibrarySnapshot.model_validate(
        json.loads(store.read_bytes(version.library.artifact_id, "library.json"))
    )
    misses: list[LearningMiss] = []
    for entry in library.entries:
        if entry.kind != "research" or entry.derived_from is None:
            continue
        payload = resolve_library_entry(store, entry)
        if not isinstance(payload, dict) or payload.get("reply_matched") is not False:
            continue
        case = traces.get(str(payload.get("session_id")))
        if case is None:
            continue
        investigation = payload.get("investigation")
        misses.append(
            LearningMiss(
                record=entry.artifact,
                run_id=str(payload["session_id"]),
                trace_id=case.trace_id,
                task_lineage=case.task.lineage_group_id,
                source=entry.derived_from,
                depends_on=entry.depends_on,
                action=payload.get("actions"),
                predicted_reply=payload.get("predicted_reply"),
                observed_reply=payload.get("observed_reply"),
                investigation=investigation if isinstance(investigation, dict) else None,
            )
        )
    return tuple(misses)


def screening_request(
    miss: LearningMiss, accepted: Sequence[JsonObject], *, maximum_reply_bytes: int
) -> DecisionRequest:
    """Frame one miss and the accepted lessons as three probability questions.

    Args:
        miss: Miss with a hypothesis.
        accepted: Accepted lesson rules, each ``{"entry_id", "rules"}``.
        maximum_reply_bytes: Bound on each shown reply.

    Returns:
        The native decision request.
    """
    investigation = miss.investigation or {}
    state: JsonObject = {
        "miss": {
            "action": miss.action,
            "predicted_reply": _bounded(miss.predicted_reply, maximum_reply_bytes),
            "recorded_reply": _bounded(miss.observed_reply, maximum_reply_bytes),
            "investigation": {
                key: investigation[key]
                for key in ("discrepancy", "hypothesis", "scope", "uncertainty", "observable_cue")
                if key in investigation
            },
        },
        "accepted_lessons": list(accepted),
        "authority": "One past prediction miss and its tentative hypothesis; recorded values "
        "describe that instance only.",
    }
    questions: dict[str, NoulQuestion | ChoiceQuestion | ScoreQuestion] = {
        "behavioural": NoulQuestion(
            instructions="Is this miss about how the environment behaves (how a tool replies "
            "to its arguments and visible state), so that the hypothesis would apply to other "
            "instances? Answer false when it is about this instance's data: record contents, "
            "values or state the simulator could not see.",
            criteria=NoulCriteria(true="reusable behaviour", false="instance-specific data"),
        ),
        "covered": NoulQuestion(
            instructions="Do the accepted_lessons already state this hypothesis, so a new "
            "lesson would add nothing? With no accepted lessons, answer false.",
            criteria=NoulCriteria(true="already covered", false="new lesson"),
        ),
        "supported": NoulQuestion(
            instructions="Does the recorded_reply support the hypothesis, rather than leave "
            "it a guess? A cut reply supports only what its shown part shows.",
            criteria=NoulCriteria(true="supported", false="unsupported or contradicted"),
        ),
    }
    return DecisionRequest(state=state, questions=questions)


def _bounded(value: JsonValue, limit: int) -> JsonValue:
    """The value itself when its JSON fits ``limit`` bytes, else a labelled prefix."""
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    encoded = text.encode()
    if len(encoded) <= limit:
        return value
    return {
        "cut_json_prefix": encoded[:limit].decode(errors="ignore"),
        "total_bytes": len(encoded),
        "authority": "Cut for the screening budget; the rest was not shown",
    }


def screening_identity(
    request: DecisionRequest, record: ArtifactInput, model: ModelSnapshot
) -> str:
    """One decision per exact request, miss record and decision model."""
    return stable_id(
        SCREENING_TYPE,
        {
            "protocol": SCREENING_PROTOCOL,
            "record": record.model_dump(mode="json"),
            "request_sha256": sha256_json(request),
            "model": model.model_dump(mode="json"),
        },
    )


def planned_screenings(
    project: ProjectStore,
    misses: Sequence[LearningMiss],
    accepted: Sequence[JsonObject],
    *,
    alias: str,
    maximum_reply_bytes: int,
) -> list[tuple[LearningMiss, DecisionRequest | None, str | None]]:
    """Every miss with its request and decision identity; none for a miss without hypothesis.

    Args:
        project: Project whose catalog names the decision model.
        misses: Misses to screen.
        accepted: Accepted lessons shown as coverage.
        alias: Native decision alias (the learning plan's recall alias).
        maximum_reply_bytes: Bound on each shown reply.

    Returns:
        ``(miss, request, identity)`` per miss, in order. Credential-free.
    """
    model = RuntimeModelCatalog(
        load_model_catalog(project.model_catalog_path), environment={}
    ).snapshot(alias)[0]
    rows: list[tuple[LearningMiss, DecisionRequest | None, str | None]] = []
    for miss in misses:
        if miss.hypothesis is None:
            rows.append((miss, None, None))
            continue
        request = screening_request(miss, accepted, maximum_reply_bytes=maximum_reply_bytes)
        rows.append((miss, request, screening_identity(request, miss.record, model)))
    return rows


def screening_cost(
    project: ProjectStore, alias: str, requests: Sequence[DecisionRequest]
) -> float | None:
    """Conservative credential-free cost of the requests, or None when a price is unknown."""
    capabilities = RuntimeModelCatalog(
        load_model_catalog(project.model_catalog_path), environment={}
    ).snapshot(alias)[1]
    prices = (
        capabilities.input_cost_per_million_tokens_usd,
        capabilities.output_cost_per_million_tokens_usd,
    )
    if prices[0] is None or prices[1] is None:
        return None if requests else 0.0
    tokens = [(r.input_token_reservation, r.output_token_reservation) for r in requests]
    return sum(i * prices[0] + o * prices[1] for i, o in tokens) / 1_000_000


def screen_misses(
    project: ProjectStore,
    planned: Sequence[tuple[LearningMiss, DecisionRequest | None, str | None]],
    *,
    alias: str,
    deadline: datetime,
    clock: Callable[[], datetime],
    code_revision: str,
) -> tuple[MissScreening, ...]:
    """Screen every planned miss once; saved decisions replay and unknowns stay unknown.

    Args:
        project: Project owning the saved decisions.
        planned: Output of :func:`planned_screenings`.
        alias: Native decision alias.
        deadline: Aware dispatch cutoff.
        clock: Aware clock.
        code_revision: Recording implementation identity.

    Returns:
        One screening per miss, in order.
    """
    store = project.artifacts
    outcomes: list[MissScreening] = []
    resolved: ResolvedModel | None = None
    for miss, request, identity in planned:
        if request is None or identity is None:
            outcomes.append(MissScreening(miss=miss, status="no-hypothesis"))
            continue
        with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
            if not store.exists(identity):
                envelope = ArtifactEnvelope(
                    schema_version=1, created_at=clock(), code_revision=code_revision
                )
                intent = identity + "-intent"
                if screening_dispatched(project, identity):
                    screening = MissScreening(
                        miss=miss,
                        status="unknown",
                        request_sha256=sha256_json(request),
                        reason="dispatched without a saved answer; never redrawn",
                    )
                    response: JsonObject | None = None
                else:
                    # Resolve credentials and the decision wire before the intent: a refusal
                    # here dispatched nothing, so the miss must stay retryable rather than
                    # become an unknown outcome.
                    if resolved is None:
                        resolved = RuntimeModelCatalog(
                            load_model_catalog(project.model_catalog_path)
                        ).resolve(alias)
                    decide = _decider(project, identity, resolved, deadline, clock, code_revision)
                    if not store.exists(intent):
                        store.write_json(
                            artifact_id=intent,
                            artifact_type=SCREENING_TYPE + "-intent",
                            envelope=envelope.model_copy(update={"inputs": (miss.record,)}),
                            files={"request.json": request},
                        )
                    screening, response = _dispatch(project, miss, request, identity, decide)
                cited = sorted(
                    {artifact_input(store.read(intent).manifest), miss.record, miss.source},
                    key=lambda ref: ref.artifact_id,
                )
                store.write_json(
                    artifact_id=identity,
                    artifact_type=SCREENING_TYPE,
                    envelope=envelope.model_copy(update={"inputs": tuple(cited)}),
                    files={
                        "screening.json": {
                            "screening": screening.model_dump(mode="json"),
                            "response": response,
                        }
                    },
                )
            saved = json.loads(store.read_bytes(identity, "screening.json"))
            outcomes.append(MissScreening.model_validate(saved["screening"]))
    return tuple(outcomes)


def screening_dispatched(project: ProjectStore, identity: str) -> bool:
    """Whether a screening may have reached the provider: its one ledger slot was claimed.

    The admitted transport claims the slot only after every local refusal (deadline, spend
    reservation, host admission, request retention) and immediately before sending, so an intent
    with an unclaimed ledger provably sent nothing and is dispatched again. The ledger is created
    before the intent is written, so an intent without its ledger has lost that evidence and
    counts as dispatched.
    """
    path = project.artifacts.project_directory / (identity + ".sqlite")
    return project.artifacts.exists(identity + "-intent") and (
        not path.exists() or _ledger(path, identity).used() > 0
    )


def _ledger(path: Path, identity: str) -> DispatchLedger:
    """The one-attempt dispatch ledger of one screening."""
    return DispatchLedger(path, 1, sha256_json(identity))


def _decider(
    project: ProjectStore,
    identity: str,
    resolved: ResolvedModel,
    deadline: datetime,
    clock: Callable[[], datetime],
    code_revision: str,
) -> Callable[[DecisionRequest], JsonObject]:
    """The decision caller on a one-attempt ledger; refuses a non-native alias before sending."""
    store = project.artifacts
    transport = AdmittedJsonTransport(
        HttpxJsonTransport(),
        ledger=_ledger(store.project_directory / (identity + ".sqlite"), identity),
        artifacts=store,
        code_revision=code_revision,
        deadline=deadline,
        clock=clock,
    )
    return decision_caller(resolved, transport, deadline, clock)


def _dispatch(
    project: ProjectStore,
    miss: LearningMiss,
    request: DecisionRequest,
    identity: str,
    decide: Callable[[DecisionRequest], JsonObject],
) -> tuple[MissScreening, JsonObject | None]:
    """Send one decision under its saved intent and decode its probabilities."""
    store = project.artifacts
    base = MissScreening(miss=miss, status="unknown", request_sha256=sha256_json(request))
    try:
        with canonical_call(artifact_input(store.read(identity + "-intent").manifest)):
            response = decide(request)
    except (OSError, RuntimeError, ValueError) as error:
        if not screening_dispatched(project, identity):
            # Refused locally before the slot claim: nothing was sent, so a later run retries.
            raise
        return base.model_copy(update={"reason": f"dispatch failed: {type(error).__name__}"}), None
    probabilities = _probabilities(response)
    if probabilities is None:
        return base.model_copy(update={"reason": "malformed answer; never redrawn"}), response
    return base.model_copy(update={"status": "answered", **probabilities}), response


def _probabilities(response: JsonObject) -> dict[str, float] | None:
    """Exactly the three noul answers in [0, 1], or None for any other shape."""
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(QUESTIONS):
        return None
    found: dict[str, float] = {}
    for key in QUESTIONS:
        answer = answers[key]
        value = answer.get("noul") if isinstance(answer, dict) else None
        if (
            not isinstance(answer, dict)
            or answer.get("type") != "noul"
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
        ):
            return None
        found[key] = float(value)
    return found


def _tool(actions: JsonValue) -> str | None:
    """The tool name of a single action, or of the first of a list of actions."""
    action = actions[0] if isinstance(actions, list) and actions else actions
    name = action.get("name") if isinstance(action, dict) else None
    return name if isinstance(name, str) else None


Deliveries = dict[str, list[tuple[str | None, bool]]]


def delivery_outcomes(project: ProjectStore, plan: LearningPlan) -> Deliveries:
    """For each record, the tool and exactness of every learned prediction it was delivered to.

    Read from saved learning sessions and feedback receipts only, without provider calls.

    Args:
        project: Project with a recorded learning plan.
        plan: That plan.

    Returns:
        Record ID to ``(tool, exact)`` per delivering prediction.
    """
    found: Deliveries = {}
    for _, prepared, _ in learned_runs(project, plan):
        path = project.artifacts.project_directory / (prepared.id + "-sessions.sqlite")
        evidence_id = prepared.id + "-report-intent"
        if not path.exists() or not project.artifacts.exists(evidence_id):
            continue
        evidence = json.loads(project.artifacts.read_bytes(evidence_id, "evidence.json"))
        feedback = evidence.get("sources", {}).get("online_feedback", {})
        matched = {
            str(row.get("operation_id")): row.get("reply_matched")
            for row in feedback.get("receipts", [])
            if isinstance(row, dict)
        }
        transitions = SessionStore(path).read_session(prepared.id).state.get("transitions")
        for transition in transitions if isinstance(transitions, list) else []:
            if not isinstance(transition, dict):
                continue
            exact = matched.get(str(transition.get("operation_id")))
            evidence = transition.get("evidence") or {}
            delivered = evidence.get("session_research_ids")
            if not isinstance(exact, bool) or not isinstance(delivered, list):
                continue
            tool = _tool(transition.get("actions"))
            origins = _overview_origins(project, plan.lineage, evidence.get("memory_overview_ids"))
            for identity in {*map(str, delivered), *origins}:
                found.setdefault(identity, []).append((tool, exact))
    return found


def _overview_origins(project: ProjectStore, lineage: str, rows: JsonValue) -> set[str]:
    """Record origins of the overview rows a prediction received; a summary covers its lines."""
    origins: set[str] = set()
    for row in rows if isinstance(rows, list) else []:
        span = str(row).removeprefix("#").split("-")
        if not all(part.isdigit() for part in span) or len(span) > 2:
            continue
        for index in range(int(span[0]), int(span[-1]) + 1):
            origins.add(read_line(project.artifacts, lineage, index).origin)
    return origins


def fit_threshold(
    screenings: Sequence[MissScreening],
    deliveries: Deliveries,
    *,
    thresholds: Sequence[float],
    lineage: str,
    version: int,
) -> ThresholdFit:
    """Select the grid threshold with the greatest net delivered outcome of retained misses.

    Args:
        screenings: Answered and unanswered screenings; only answered ones are scored.
        deliveries: Output of :func:`delivery_outcomes`.
        thresholds: Registered increasing grid.
        lineage: Memory lineage screened.
        version: Memory version screened.

    Returns:
        The fit; ties, including the no-evidence case, keep the highest threshold.
    """
    scored: list[tuple[float, int]] = []
    total = 0
    for row in screenings:
        if row.score is None:
            continue
        evidence = [
            exact
            for tool, exact in deliveries.get(row.miss.record.artifact_id, [])
            if tool is not None and tool == row.miss.tool
        ]
        total += len(evidence)
        scored.append((row.score, sum(1 if exact else -1 for exact in evidence)))
    objective = tuple(sum(net for score, net in scored if score > t) for t in thresholds)
    retained = tuple(sum(score > t for score, _ in scored) for t in thresholds)
    best = max(objective)
    selected = max(t for t, value in zip(thresholds, objective, strict=True) if value == best)
    return ThresholdFit(
        lineage=lineage,
        version=version,
        thresholds=tuple(thresholds),
        objective=objective,
        retained=retained,
        selected=selected,
        status="fitted" if total else "no-delivery-evidence",
        deliveries=total,
    )


def save_fit(
    project: ProjectStore,
    fit: ThresholdFit,
    screenings: Sequence[MissScreening],
    *,
    planned: Sequence[str | None],
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Persist the fit, citing every saved screening decision it scored.

    Args:
        project: Project owning the fit.
        fit: Fitted threshold.
        screenings: Screenings the fit read, in planned order.
        planned: Decision identity of each screening, None for a miss without hypothesis.
        created_at: Aware creation time.
        code_revision: Recording implementation identity.

    Returns:
        The saved fit reference; an identical fit replays.
    """
    store = project.artifacts
    cited = tuple(
        sorted(
            {artifact_input(store.read(identity).manifest) for identity in planned if identity},
            key=lambda ref: ref.artifact_id,
        )
    )
    identity = stable_id(
        FIT_TYPE,
        {
            "protocol": SCREENING_PROTOCOL,
            "fit": fit.model_dump(mode="json"),
            "screenings": [ref.model_dump(mode="json") for ref in cited],
            "misses": [row.miss.record.artifact_id for row in screenings],
        },
    )
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type=FIT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=created_at, code_revision=code_revision, inputs=cited
            ),
            files={"fit.json": fit},
        )
    return artifact_input(store.read(identity).manifest)
