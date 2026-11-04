"""The whole decision-variance pipeline offline, on the grounded ticket world.

Every stage runs through its owner on the grounded ticket world of
:mod:`~exp.optimize.evaluation.variance.tests.grounded_world_test`; only the wire is scripted.
Two stand-ins are explicit: the variance build is replaced by a fixture executor that saves the
exposure plan and a world-learning trace dataset (the grounded world is not an ``exp build``
project), and the rules owner's extraction is replaced by the fixture's saved ticket rule book.
Everything after that (binding the rule book into the mining runtime, TraceML labels, two-tier
error mining, the coalition, the packet, the agentic world and rubric authors, the freeze, the
judges, the families, admission, records, Harbor export and cross-talk) is real.

The scripted judge answers from an explicit synthetic policy: an episode passes exactly when its
last recorded ticket outcome is ``right``, except that the first replicate of every group fails,
so group advantages carry signal. This proves plumbing, not model quality.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    SourceIdentity,
    sha256_json,
)
from exp.common.models import (
    BillingSource,
    ModelCapabilities,
    ModelRecord,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.project import ProjectStore, artifact_input
from exp.common.traces import Trace, TraceSource
from exp.common.traces.ingest import TraceNormalizationResult, persist_trace_dataset
from exp.common.traces.trace import TraceOutcome
from exp.optimize.evaluation.rules.stage import REPORT_TYPE, RulesStageReport, RulesStageSpec
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport
from exp.optimize.evaluation.variance.authoring_packet import (
    AuthoringPacket,
    SourceRef,
    build_authoring_packet,
    save_packet,
)
from exp.optimize.evaluation.variance.authoring_packet_test import save_book, tool_reply
from exp.optimize.evaluation.variance.authors import DECISION_STATE_TOOL, SCENARIO_TOOL
from exp.optimize.evaluation.variance.build_records import EXPOSURE_TYPE, VarianceBuildSpec
from exp.optimize.evaluation.variance.evidence_tools import EvidenceToolProfile
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.family import FamilyReport
from exp.optimize.evaluation.variance.judge_prompt import ASSESS_TOOL, JUDGE_PROMPT_VERSION
from exp.optimize.evaluation.variance.judge_test import _call
from exp.optimize.evaluation.variance.judged_test import state_judge
from exp.optimize.evaluation.variance.pipeline import (
    SpendConsentRequired,
    StageExecutor,
    load_stage_record,
    plan_decision_variance,
    run_decision_variance,
    stage_outputs,
)
from exp.optimize.evaluation.variance.pipeline_records import RECORDS_STAGE
from exp.optimize.evaluation.variance.pipeline_spec import (
    AuthoringSettings,
    BuildOutputs,
    BuildSettings,
    CostLine,
    DecisionVarianceSpec,
    FamiliesOutputs,
    MiningOutputs,
    MiningSettings,
    PipelineApproval,
    PipelineReport,
    RecordsOutputs,
    RecordsSettings,
    RulesOutputs,
    RulesSettings,
    StageContext,
    StageDependencies,
    StageName,
    StageRun,
    TraceMLOutputs,
    TraceMLSettings,
)
from exp.optimize.evaluation.variance.records import AuditDisposition
from exp.optimize.evaluation.variance.rubric_agent import (
    CRITIQUE_TOOL,
    RubricAuthoringProfile,
)
from exp.optimize.evaluation.variance.rubric_agent import (
    PLAN_TOOL as RUBRIC_PLAN_TOOL,
)
from exp.optimize.evaluation.variance.rubric_author import RUBRIC_TOOL, author_rubric, save_rubric
from exp.optimize.evaluation.variance.rubric_author_test import covering_answer, criterion
from exp.optimize.evaluation.variance.rubric_revision import REVISION_TOOL, load_revision
from exp.optimize.evaluation.variance.tests.grounded_world_test import (
    ACTIONS,
    INITIAL,
    GroundedFixture,
    TicketActor,
    grounded_fixture,
    tier_decision,
    tier_scenario,
)
from exp.optimize.evaluation.variance.world_agent import PLAN_TOOL as WORLD_PLAN_TOOL
from exp.optimize.evaluation.variance.world_author import WORLD_TOOL
from exp.optimize.evaluation.variance.world_author_test import world_answer
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models.providers.transport import JsonHttpResponse, JsonHttpTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.traceml import ACTION_LABELS, TRACEML_BANK_ID
from exp.simulation.rules.contracts_test import call, result
from exp.simulation.rules.model import RuleBook

REVISION = "test"
"""Producing source identity of every saved artifact."""

TIER_B = "actor-b"
"""The second actor tier's catalog alias."""

FIRST_REPLICATES = ("-cell-0", "-branch-0")
"""Episode id suffixes of each group's first replicate, which the policy judges a failure."""

Answer = Callable[[JsonObject], JsonObject]
"""Maps a request payload to the forced tool's arguments."""


class Interrupted(BaseException):
    """Simulates the process stopping in the middle of a judge request."""


def _user(payload: JsonObject) -> JsonObject:
    """The parsed JSON user content of a request."""
    messages = payload["messages"]
    assert isinstance(messages, list) and isinstance(messages[-1], dict)
    content = json.loads(str(messages[-1]["content"]))
    assert isinstance(content, dict)
    return content


def _packet(payload: JsonObject) -> AuthoringPacket:
    """The authoring packet an author request carries."""
    return AuthoringPacket.model_validate(_user(payload)["authoring_packet"])


def semantic(request: JsonObject) -> str:
    """Pass a complete episode whose last ticket outcome is right; first replicates fail."""
    evidence = request["episode_evidence"]
    assert isinstance(evidence, dict)
    verdict = state_judge(("tickets", "outcome"), "right")(evidence)
    binding = request["judge_binding"]
    assert isinstance(binding, dict)
    if verdict == "pass" and str(binding["episode_id"]).endswith(FIRST_REPLICATES):
        return "fail"
    return verdict


def assessment(
    request: JsonObject, verdict: str, problems: tuple[str, ...] = ()
) -> JsonHttpResponse:
    """Answer ``assess_episode`` for one verdict with a finding on every rubric criterion."""
    evidence = request["episode_evidence"]
    criteria = request["rubric_criteria"]
    assert isinstance(evidence, dict) and isinstance(criteria, list)
    events = evidence["events"]
    assert isinstance(events, list)
    scored = [e for e in events if isinstance(e, dict) and e["scope"] == "scored"]
    decisive = (scored or events)[0]
    assert isinstance(decisive, dict)
    status = {"pass": "met", "fail": "not_met", "unassessable": "insufficient_evidence"}[verdict]
    findings: list[JsonValue] = []
    for row in criteria:
        assert isinstance(row, dict)
        found = status if row["role"] == "required_correctness" else "met"
        findings.append(
            {
                "criterion_id": row["criterion_id"],
                "status": found,
                "event_refs": [] if found == "insufficient_evidence" else [decisive["event_id"]],
                "reason": f"{row['criterion_id']} is {found}.",
            }
        )
    return _call(
        ASSESS_TOOL,
        {
            "verdict": verdict,
            "criterion_findings": findings,
            "reason": f"Overall {verdict}.",
            "decisive_refs": [] if verdict == "unassessable" else [decisive["event_id"]],
            "rubric_problems": list(problems),
        },
    )


def decision_answer(step: int) -> JsonObject:
    """The decision-state author's answer, grounded in the before-state and the tier rule."""
    return {
        "state": tier_decision(step).model_dump(mode="json", exclude_defaults=True),
        "grounding": [
            {
                "path": "/tickets/severity",
                "revealed_by": ["check_sla"],
                "refs": ["state:/tickets/severity", "rule:tickets.tier"],
            }
        ],
        "limitations": ["Simulated tickets only."],
    }


def scenario_answer() -> JsonObject:
    """The scenario author's answer: the urgent and routine tiers."""
    scenarios = (tier_scenario("urgent", "high"), tier_scenario("routine", "low"))
    return {
        "scenarios": [s.model_dump(mode="json", exclude_defaults=True) for s in scenarios],
        "limitations": ["Tier variants."],
    }


def selector_answer() -> JsonObject:
    """One point selector choosing the start, before the actor has read the tier."""
    return {
        "points": [
            {
                "step": 0,
                "decision": "Escalate or close the ticket",
                "alternatives": ["Check the service tier first"],
                "deciding_state": [],
                "rationale": "Attempts diverge on the ticket's handling.",
                "evidence_refs": ["point:0", "group"],
            }
        ],
        "audit": [
            {
                "category": "guess_required",
                "explanation": "Some attempts never read the tier before handling the ticket.",
                "evidence_refs": ["group"],
            }
        ],
        "limitations": ["Simulated tickets only."],
    }


def critic_answer() -> JsonObject:
    """One critic estimating the chosen point."""
    return {
        "estimates": [
            {"step": 0, "probability": 0.6, "rationale": "Base rate.", "evidence_refs": ["group"]}
        ],
        "limitations": ["Simulated tickets only."],
    }


WORLD_PLAN: JsonObject = {
    "state_machine": "tickets: open -> escalated | closed; close refuses unless lock is none",
    "programs": ["read_ticket", "check_sla", "escalate", "close"],
    "failure_modes": ["held ticket refused", "missing lock refused"],
    "fixtures": ["held close", "escalate high"],
    "supported_by": ["rule:tickets.lock"],
}
"""The world author's plan."""

RUBRIC_PLAN: JsonObject = {
    "success_condition": "The ticket is handled as its tier requires and reported truthfully.",
    "requirements": [
        {
            "statement": "A high tier ticket is escalated and a low tier ticket closed.",
            "supported_by": ["rule:tickets.tier"],
            "scenario_ids": ["urgent", "routine"],
            "accounts_for": [],
        }
    ],
    "alternatives": ["Reading the ticket before checking the tier."],
}
"""The rubric author's plan."""

RUBRIC_CRITIQUE: JsonObject = {
    "findings": [
        {"check": 16, "status": "satisfied", "criteria": ["handled"], "note": "Alternatives."}
    ],
    "revisions": [],
}
"""The rubric author's self-critique."""

AMBIGUOUS = "criterion handled does not say whether the tier must be read before handling"
"""The rubric problem the troubled judge reports."""

CHECKED = criterion("tier.checked", question="Was the ticket's tier read before it was handled?")
"""The criterion a revision adds, which resolves the problem."""


def revision_answer(payload: JsonObject) -> JsonObject:
    """A revised rubric adding the tier-check criterion and resolving every cited problem."""
    cited = _user(payload)["problems"]
    assert isinstance(cited, list)
    return {
        **covering_answer(CHECKED),
        "resolutions": [
            {"problem_id": row["problem_id"], "change": "Required reading the tier first."}
            for row in cited
            if isinstance(row, dict)
        ],
    }


def unrevised(request: JsonObject) -> bool:
    """Whether a judge request applies a rubric without the revision's criterion."""
    criteria = request["rubric_criteria"]
    assert isinstance(criteria, list)
    return all(isinstance(row, dict) and row["criterion_id"] != "tier.checked" for row in criteria)


ANSWERS: dict[str, Answer] = {
    REVISION_TOOL: revision_answer,
    "choose_points": lambda _: selector_answer(),
    "estimate_points": lambda _: critic_answer(),
    DECISION_STATE_TOOL: lambda _: decision_answer(0),
    SCENARIO_TOOL: lambda _: scenario_answer(),
    WORLD_PLAN_TOOL: lambda _: WORLD_PLAN,
    WORLD_TOOL: lambda payload: world_answer(_packet(payload)),
    RUBRIC_PLAN_TOOL: lambda _: RUBRIC_PLAN,
    RUBRIC_TOOL: lambda _: covering_answer(),
    CRITIQUE_TOOL: lambda _: RUBRIC_CRITIQUE,
}
"""Every non-judge role's answer, by the forced tool it must call."""


class Wire(JsonHttpTransport):
    """The scripted author, role and judge wire, answering by the request's forced tool.

    Attributes:
        requests: Every non-judge request, as ``(tool, payload)``.
        judged: Every judge request's ``<judge definition>/<episode id>``, in order.
        interrupt_at: Judge request number (1-based, counted from now) that raises
            :class:`Interrupted` once, if set.
        troubled: Whether the judge reports a rubric problem for a judge request, if set.
    """

    def __init__(self) -> None:
        """Start with no request served."""
        self.requests: list[tuple[str, JsonObject]] = []
        self.judged: list[str] = []
        self.interrupt_at: int | None = None
        self.troubled: Callable[[JsonObject], bool] | None = None

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> JsonHttpResponse:
        """Refuse metadata reads.

        Raises:
            AssertionError: Always.
        """
        del url, headers, timeout_seconds
        raise AssertionError("the wire received a metadata request")

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Judge an episode from the policy, or answer the forced role tool.

        Raises:
            Interrupted: At the configured judge request.
        """
        del url, headers, timeout_seconds
        choice = payload.get("tool_choice")
        function = choice.get("function") if isinstance(choice, dict) else None
        tool = str(function.get("name")) if isinstance(function, dict) else "required"
        if tool != ASSESS_TOOL:
            self.requests.append((tool, payload))
            return tool_reply(tool, ANSWERS[tool](payload))
        messages = payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        request = json.loads(str(messages[1]["content"]))
        binding = request["judge_binding"]
        self.judged.append(f"{binding['judge_definition']}/{binding['episode_id']}")
        if self.interrupt_at is not None:
            self.interrupt_at -= 1
            if self.interrupt_at == 0:
                self.interrupt_at = None
                raise Interrupted
        problems = (AMBIGUOUS,) if self.troubled is not None and self.troubled(request) else ()
        return assessment(request, semantic(request), problems)

    def calls(self) -> tuple[int, int]:
        """Role and judge requests so far."""
        return len(self.requests), len(self.judged)


@dataclass
class TieredActor(JsonHttpTransport):
    """The fixture's scripted actor policy, answering as whichever actor model was requested.

    Attributes:
        policy: The fixture's actor transport, which records every request.
    """

    policy: TicketActor

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> JsonHttpResponse:
        """Delegate a metadata read, which the policy refuses."""
        return self.policy.get(url, headers=headers, timeout_seconds=timeout_seconds)

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer from the policy under the requested model's identity."""
        reply = self.policy.post(
            url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
        )
        assert isinstance(reply.body, dict)
        return JsonHttpResponse(
            status_code=reply.status_code, body={**reply.body, "model": payload["model"]}
        )


def _ticket_trace(trace_id: str) -> Trace:
    """A world-learning trajectory: check the tier, then escalate the high ticket."""
    return Trace(
        trace_id=trace_id,
        task="Resolve the support ticket with the tools you have.",
        tools=(),
        spans=(
            call("c1", 0, "check_sla", {}, "k1"),
            result("r1", 1, "check_sla", {"severity": "high", "lock": "none"}, "k1"),
            call("c2", 2, "escalate", {}, "k2"),
            result("r2", 3, "escalate", {"status": "escalated"}, "k2"),
        ),
        outcome=TraceOutcome(status="success"),
        source=TraceSource(
            identity=SourceIdentity(kind="otlp", source_id="tickets.otlp", sha256="a" * 64),
            semantic_convention_version="1.37.0",
        ),
    )


def exposure_plan() -> ExposurePlan:
    """Mining probes, world learning and final evaluation, fixed before anything is tagged."""
    return ExposurePlan(
        bindings=ArtifactInput(artifact_id="bindings", sha256="0" * 64),
        seed=0,
        probe_fraction=0.5,
        roles={"tickets": "mining_probes", "sources": "world_learning", "held": "final_evaluation"},
        traces={"trace-1": "sources", "trace-2": "sources", "trace-final": "held"},
    )


@dataclass(frozen=True)
class FixtureBuild:
    """The variance-build stand-in: the exposure plan and the world-learning trace dataset.

    Attributes:
        fixture: The grounded fixture whose world model the stage names.
    """

    fixture: GroundedFixture

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """Key the stand-in by its fixed content."""
        del project, spec
        return {"fixture": "grounded-tickets-v1"}

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[()], tuple[str, ...]]:
        """The stand-in makes no call."""
        del project, spec, code_revision
        return (), ("Fixture build: grounded ticket world; exposure fixed before tagging.",)

    def run(self, context: StageContext) -> StageRun:
        """Save the exposure plan and the trace dataset, or adopt the saved ones."""
        store = context.project.artifacts
        if not store.exists("fixture-exposure"):
            store.write_json(
                artifact_id="fixture-exposure",
                artifact_type=EXPOSURE_TYPE,
                envelope=ArtifactEnvelope(
                    schema_version=1, created_at=datetime.now(UTC), code_revision=REVISION
                ),
                files={"exposure.json": exposure_plan()},
            )
        exposure = artifact_input(store.read("fixture-exposure").manifest)
        if not store.exists("fixture-traces"):
            persist_trace_dataset(
                TraceNormalizationResult(
                    traces=(_ticket_trace("trace-1"), _ticket_trace("trace-2")), issues=()
                ),
                store,
                created_at=datetime.now(UTC),
                code_revision=REVISION,
                dataset_id="fixture-traces",
            )
        dataset = artifact_input(store.read("fixture-traces").manifest)
        outputs = BuildOutputs(
            report=exposure,
            exposure=exposure,
            world_model=self.fixture.definition.world_model,
            trace_dataset=dataset,
            rules_traces=("trace-1", "trace-2"),
        )
        return StageRun(outputs, (exposure, dataset), {"fixture": True})


def fixture_rules_stage(
    project: ProjectStore,
    spec: RulesStageSpec,
    *,
    exposure: ExposurePlan,
    traces: tuple[Trace, ...],
    code_revision: str,
) -> RulesStageReport:
    """The rules owner's stand-in: the saved ticket rule book, after the owner's exposure check."""
    exposure.require((trace.trace_id for trace in traces), "world_learning")
    rulebook, book = save_book(project)
    report = _report(spec, book, rulebook)
    if not project.artifacts.exists(report.identity + "-result"):
        project.artifacts.write_json(
            artifact_id=report.identity + "-result",
            artifact_type=REPORT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=(rulebook,),
            ),
            files={"report.json": report},
        )
    return report


def _report(spec: RulesStageSpec, book: RuleBook, rulebook: ArtifactInput) -> RulesStageReport:
    """A rules-stage report naming the saved ticket rule book."""
    return RulesStageReport(
        spec=spec,
        identity="pipeline-ticket-rules-stage",
        extraction_identity="pipeline-ticket-rules-extraction",
        contracts=dict(book.contracts),
        draws=(),
        failures=(),
        rulebook=rulebook,
        rules=len(book.rules),
        rule_counts={"supported": len(book.rules), "contradicted": 0, "unknown": 0},
        open_questions=len(book.questions),
        diagnostic=(),
        requires_witness=(),
        sessions=(),
        session_ids=(),
        properties=None,
        workloads=(),
        triage=(),
        families_must_stay_diagnostic=(),
    )


def write_bank(root: Path) -> Path:
    """Write the local ``traceml-action-v1`` bank with every action column."""
    banks = root / "banks"
    banks.mkdir(exist_ok=True)
    questions = "".join(
        f'\n[[questions]]\nquestion_id = "action.{label}"\nfamily = "action"\n'
        f'text = "Is the step {label}?"\n'
        for label in ACTION_LABELS
    )
    (banks / f"{TRACEML_BANK_ID}.toml").write_text(
        f'bank_id = "{TRACEML_BANK_ID}"\nversion = 1\n{questions}'
    )
    return banks


def step_tagger(project: ProjectStore) -> DecisionsTraceTagger:
    """The real decisions tagger over a scripted boundary labelling every step a read."""
    catalog = load_model_catalog(project.model_catalog_path)
    model = RuntimeModelCatalog(catalog, environment={}).snapshot("judge")[0]

    def decide(request: DecisionRequest) -> JsonObject:
        """Answer every proposition: the step is a read."""
        questions = request.questions
        return {
            "answers": {
                question: {"type": "noul", "noul": 0.9 if question == "action.read" else 0.05}
                for question in questions
            },
            "usage": {"cost": 0.0},
        }

    return DecisionsTraceTagger(decide, model, maximum_concurrency=2)


def add_tier(project: ProjectStore, alias: str) -> None:
    """Add a second actor alias on the fixture's connection."""
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["actor"]
    write_model_catalog(
        project.model_catalog_path,
        catalog.model_copy(
            update={
                "models": {
                    **catalog.models,
                    alias: ModelRecord(
                        connection=record.connection,
                        model=f"{alias}-v1",
                        billing_source=BillingSource.CUSTOMER_MANAGED,
                        capabilities=ModelCapabilities(
                            supports_tools=True,
                            supports_structured_output=True,
                            context_window_tokens=200_000,
                            maximum_output_tokens=8_192,
                        ),
                    ),
                }
            }
        ),
    )


def rebound(fixture: GroundedFixture, alias: str) -> GroundedFixture:
    """Add an actor tier and rebind every frozen setting to the changed catalog."""
    add_tier(fixture.project, alias)
    catalog = load_model_catalog(fixture.project.model_catalog_path)
    digest = sha256_json(catalog.model_dump(mode="json"))
    definition = fixture.definition.model_copy(update={"catalog_sha256": digest})
    job = fixture.job.model_copy(
        update={"catalog_sha256": digest, "configuration": definition.model_dump(mode="json")}
    )
    settings = fixture.settings.model_copy(update={"catalog_sha256": digest})
    return replace(fixture, job=job, definition=definition, settings=settings)


def mining_rubric(fixture: GroundedFixture) -> ArtifactInput:
    """Author and save the mining-probe task's rubric before the run, as an operator would."""
    project = fixture.project
    rulebook, book = save_book(project)
    start = load_continuation_checkpoint(project.artifacts, fixture.job.study.checkpoint)
    packet = build_authoring_packet(
        family_id="ticket-mining",
        task=fixture.definition.task,
        checkpoint=fixture.job.study.checkpoint,
        before_state=INITIAL,
        public_history=start.actor.messages,
        remaining_calls=5,
        sources=(
            SourceRef(
                source_id="trace-1",
                kind="trace",
                trace_id="trace-1",
                summary="An agent checked the tier and escalated the ticket.",
            ),
        ),
        exposure=exposure_plan(),
        rule_book=rulebook,
        book=book,
        decision=tier_decision(0),
        scenarios=(tier_scenario("urgent", "high"), tier_scenario("routine", "low")),
    )
    reference = save_packet(project, packet, code_revision=REVISION)
    draft = author_rubric(
        project, fixture.settings, reference, identity="mining-rubric", code_revision=REVISION
    )
    assert draft.status == "ready", draft.failure
    return save_rubric(project, draft, code_revision=REVISION)


def make_spec(
    fixture: GroundedFixture,
    rubric: ArtifactInput,
    bank: Path,
    *,
    actors: tuple[str, ...] = ("actor", TIER_B),
    dispositions: tuple[AuditDisposition, ...] = (),
    crosstalk: bool = True,
) -> DecisionVarianceSpec:
    """A small run: G=3, J=1, C=1, K=2, M=3, forced selectors and critics, single-pass judge."""
    settings = fixture.settings
    forced = EvidenceToolProfile()
    return DecisionVarianceSpec(
        build=BuildSettings(spec=VarianceBuildSpec()),
        rules=RulesSettings(spec=RulesStageSpec(author=settings), maximum_traces=4),
        traceml=TraceMLSettings(decision_alias="judge", bank_root=bank, steps_per_call=2),
        mining=MiningSettings(
            job=fixture.job,
            rubric=rubric,
            judge=settings,
            judge_prompt_set=JUDGE_PROMPT_VERSION,
            judge_evidence_tool_budget=0,
            selector=settings,
            critic=settings,
            actors=actors,
            attempts=3,
            selector_draws=1,
            critic_draws=1,
            points=2,
            continuations=3,
            group_physical_attempts=40,
            selector_profile=forced,
            critic_profile=forced,
        ),
        authoring=AuthoringSettings(
            decision_author=settings,
            scenario_author=settings,
            world_author=settings,
            rubric_author=settings,
            actions=ACTIONS,
            world_prompt_sets=("failure-mode-first",),
            rubric_profile=RubricAuthoringProfile(variants=("failure-patterns-first",)),
            repeats=6,
            confirmation_repeats=6,
            dispositions=dispositions,
        ),
        records=RecordsSettings(
            intent="Escalate or close a support ticket as its service tier requires.",
            crosstalk=crosstalk,
            crosstalk_attempts=3,
            crosstalk_top_k=1,
            crosstalk_repeats=2,
        ),
    )


@dataclass
class Run:
    """One fixture project wired for the pipeline.

    Attributes:
        fixture: The grounded world and its transports.
        wire: The author, role and judge wire.
        rubric: The mining rubric.
        bank: The TraceML bank directory.
        executors: The build stand-in.
        dependencies: The injected TraceML tagger and Harbor root.
    """

    fixture: GroundedFixture
    wire: Wire
    rubric: ArtifactInput
    bank: Path
    executors: dict[StageName, StageExecutor]
    dependencies: StageDependencies

    @property
    def project(self) -> ProjectStore:
        """The project."""
        return self.fixture.project

    def calls(self) -> tuple[int, int, int, int]:
        """Actor, predictor, role and judge requests so far."""
        return (
            len(self.fixture.actor.requests),
            len(self.fixture.predictor.requests),
            *self.wire.calls(),
        )

    def run(self, spec: DecisionVarianceSpec, *, approve: bool = True) -> PipelineReport:
        """Plan, consent once and run or replay the pipeline."""
        plan = plan_decision_variance(
            self.project, spec, code_revision=REVISION, executors=self.executors
        )
        approval = PipelineApproval.for_plan(plan) if approve else None
        return run_decision_variance(
            self.project,
            spec,
            code_revision=REVISION,
            approval=approval,
            executors=self.executors,
            dependencies=self.dependencies,
        )


def wired(root: Path, monkeypatch: pytest.MonkeyPatch) -> Run:
    """A grounded fixture with two actor tiers, the scripted wire and the stand-ins installed."""
    fixture = rebound(grounded_fixture(root, monkeypatch), TIER_B)
    wire = Wire()
    actor = TieredActor(fixture.actor)
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actor
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: wire
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.pipeline_stages.run_rules_stage", fixture_rules_stage
    )
    rubric = mining_rubric(fixture)
    return Run(
        fixture=fixture,
        wire=wire,
        rubric=rubric,
        bank=write_bank(root),
        executors={"variance-build": FixtureBuild(fixture)},
        dependencies=StageDependencies(
            traceml_tagger=step_tagger(fixture.project), harbor_root=root / "harbor"
        ),
    )


def dispositions_for(project: ProjectStore, report: PipelineReport) -> tuple[AuditDisposition, ...]:
    """A reviewer disposition for every audit finding of every family of a run."""
    record = load_stage_record(project, report.stages[4].identity)
    assert record is not None
    found: dict[str, AuditDisposition] = {}
    for row in stage_outputs(record, FamiliesOutputs).families:
        assert row.report is not None
        family = FamilyReport.model_validate_json(
            project.artifacts.read_bytes(row.report.artifact_id, "report.json")
        )
        for finding in family.findings:
            found[finding.finding_id] = AuditDisposition(
                finding_id=finding.finding_id,
                resolution="accepted_with_reason",
                rationale="Reviewed against the saved grounded trajectories.",
                resolver="pipeline-reviewer",
            )
    return tuple(found[key] for key in sorted(found))


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Run, PipelineReport]]:
    """Run the whole pipeline once over a fresh two-tier grounded project."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        run = wired(tmp_path_factory.mktemp("pipeline"), monkeypatch)
        spec = make_spec(run.fixture, run.rubric, run.bank)
        yield run, run.run(spec)


def _outputs[Model: ContractModel](
    run: Run, report: PipelineReport, index: int, model: type[Model]
) -> Model:
    """One stage's typed outputs."""
    record = load_stage_record(run.project, report.stages[index].identity)
    assert record is not None
    return stage_outputs(record, model)


def test_every_stage_runs_through_its_owner_on_two_tiers(
    pipeline: tuple[Run, PipelineReport],
) -> None:
    """Stages run in order, each saved, with the rule book bound before any probe."""
    run, report = pipeline
    project = run.project
    assert report.complete and [row.replayed for row in report.stages] == [False] * 6
    assert [row.stage for row in report.stages] == [
        "variance-build",
        "rules",
        "traceml",
        "error-mining",
        "families",
        "records",
    ]
    rules = _outputs(run, report, 1, RulesOutputs)
    bound = rules.job.configuration
    assert bound["rule_book"] == rules.rulebook.model_dump(mode="json")
    assert rules.job.study.checkpoint != run.fixture.job.study.checkpoint
    labels = _outputs(run, report, 2, TraceMLOutputs).labels
    assert set(labels) == {"actor", TIER_B}
    mining = _outputs(run, report, 3, MiningOutputs)
    assert mining.profile.prompt_version == JUDGE_PROMPT_VERSION
    assert [tier.tier for tier in mining.tiers] == ["actor", TIER_B]
    for tier in mining.tiers:
        saved = ActorCriticReport.model_validate_json(
            project.artifacts.read_bytes(tier.report.artifact_id, "report.json")
        )
        assert saved.group_tier is not None and saved.group_tier.label == tier.tier
        assert saved.spec.initial.study.checkpoint == rules.job.study.checkpoint
        assert [ref.artifact for ref in saved.spec.references] == [labels[tier.tier]]
        assert saved.spec.initial.actor_alias == tier.tier
    families = _outputs(run, report, 4, FamiliesOutputs).families
    assert [(row.tier, row.point) for row in families] == [("actor", 0), (TIER_B, 0)]
    for row in families:
        assert row.failure is None, row.failure
        assert None not in (row.packet, row.world, row.rubric, row.freeze)
        assert set(row.judging) == {"urgent", "routine"}
        for binding in row.judging.values():
            assert binding.world_freeze == row.freeze
            assert binding.judge.definition.rubric == row.rubric
            assert binding.judge.definition.profile == mining.profile
    usage = [row.usage.physical_attempts for row in report.stages]
    assert usage[0] == 0 and all(count > 0 for count in usage[2:5])


def test_dispositions_rekey_only_families_and_records_with_no_new_call(
    pipeline: tuple[Run, PipelineReport],
) -> None:
    """A changed authoring field replays every earlier stage and admits the reviewed families."""
    run, first = pipeline
    project = run.project
    dispositions = dispositions_for(project, first)
    spec = make_spec(run.fixture, run.rubric, run.bank, dispositions=dispositions)
    before = run.calls()
    reviewed = run.run(spec)
    assert reviewed.complete
    assert [row.replayed for row in reviewed.stages] == [True] * 4 + [False] * 2
    assert [row.identity for row in reviewed.stages[:4]] == [
        row.identity for row in first.stages[:4]
    ]
    assert reviewed.stages[4].identity != first.stages[4].identity
    families = _outputs(run, reviewed, 4, FamiliesOutputs).families
    assert all(row.admitted for row in families), [row.reasons for row in families]
    records = _outputs(run, reviewed, 5, RecordsOutputs)
    assert set(records.training) == {"actor/0", f"{TIER_B}/0"}
    assert set(records.selectors) == {"actor", TIER_B}
    assert records.harbor and all(Path(row.path).is_dir() for row in records.harbor)
    assert all(row.diagnostic for row in records.harbor)
    assert records.crosstalk is not None or records.crosstalk_skipped is not None
    after_review = run.calls()
    # Only the cross-talk probe may buy new episodes; every family draw and judgment replays.
    assert after_review[2] == before[2]
    replay = run.run(spec, approve=False)
    assert [row.replayed for row in replay.stages] == [True] * 6
    assert run.calls() == after_review


def test_a_changed_records_field_rekeys_only_the_records_stage(
    pipeline: tuple[Run, PipelineReport],
) -> None:
    """Disabling cross-talk keeps every earlier stage and makes no call."""
    run, first = pipeline
    dispositions = dispositions_for(run.project, first)
    spec = make_spec(run.fixture, run.rubric, run.bank, dispositions=dispositions, crosstalk=False)
    before = run.calls()
    report = run.run(spec)
    assert [row.replayed for row in report.stages] == [True] * 5 + [False]
    assert _outputs(run, report, 5, RecordsOutputs).crosstalk is None
    assert run.calls() == before


def test_unsaved_stages_need_consent_to_the_exact_plan(
    pipeline: tuple[Run, PipelineReport],
) -> None:
    """No approval, or an approval of another plan, is refused before any call."""
    run, _ = pipeline
    spec = make_spec(run.fixture, run.rubric, run.bank, crosstalk=False).model_copy(
        update={"records": RecordsSettings(intent="Another intent.", crosstalk=False)}
    )
    before = run.calls()
    with pytest.raises(SpendConsentRequired, match="unsaved"):
        run.run(spec, approve=False)
    stale = PipelineApproval(plan_sha256="0" * 64, ceiling_usd=10.0, unpriced_allowed=True)
    with pytest.raises(SpendConsentRequired, match="another plan"):
        run_decision_variance(
            run.project,
            spec,
            code_revision=REVISION,
            approval=stale,
            executors=run.executors,
            dependencies=run.dependencies,
        )
    assert run.calls() == before


@dataclass(frozen=True)
class PricedRecords:
    """The records executor with one extra priced line, to exercise the ceiling."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """The records executor's keys."""
        return RECORDS_STAGE.keys(project, spec)

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """The records plan plus a one-dollar line."""
        costs, lines = RECORDS_STAGE.plan(project, spec, code_revision)
        return (*costs, CostLine(name="priced-probe", attempts=1, usd=1.0)), lines

    def run(self, context: StageContext) -> StageRun:
        """The records executor's run."""
        return RECORDS_STAGE.run(context)


def test_the_plan_prices_every_unsaved_stage_and_a_low_ceiling_stops_before_it(
    pipeline: tuple[Run, PipelineReport],
) -> None:
    """Saved stages carry no cost; a stage whose bound exceeds the ceiling never starts."""
    run, _ = pipeline
    spec = make_spec(run.fixture, run.rubric, run.bank).model_copy(
        update={
            "records": RecordsSettings(
                intent="A priced intent.",
                crosstalk=True,
                crosstalk_attempts=3,
                crosstalk_top_k=1,
                crosstalk_repeats=2,
            )
        }
    )
    executors: dict[StageName, StageExecutor] = {**run.executors, "records": PricedRecords()}
    plan = plan_decision_variance(run.project, spec, code_revision=REVISION, executors=executors)
    assert [row.saved for row in plan.stages] == [True] * 5 + [False]
    assert all(not row.costs for row in plan.stages[:5])
    records = plan.stages[5]
    assert records.lines and records.known_usd == 1.0
    assert {line.name for line in records.unpriced} >= {"actor", "judge"}
    approval = PipelineApproval.for_plan(plan, ceiling_usd=0.5)
    before = run.calls()
    report = run_decision_variance(
        run.project,
        spec,
        code_revision=REVISION,
        approval=approval,
        executors=executors,
        dependencies=run.dependencies,
    )
    assert report.stopped is not None and "records" in report.stopped
    assert len(report.stages) == 5 and not report.complete
    assert run.calls() == before


def test_rubric_problems_from_judging_drive_a_revision_and_a_new_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirmation judging reports a rubric problem; the revised judges own the final family."""
    run = wired(tmp_path, monkeypatch)
    run.wire.troubled = unrevised
    spec = make_spec(run.fixture, run.rubric, run.bank, actors=("actor",), crosstalk=False)
    report = run.run(spec)
    assert report.complete
    (family,) = _outputs(run, report, 4, FamiliesOutputs).families
    assert family.failure is None, family.failure
    assert set(family.revisions) == {"urgent", "routine"}
    for variant, reference in family.revisions.items():
        revision = load_revision(run.project, reference)
        assert revision.status == "revised" and revision.problems
        assert revision.initial.rubric == family.rubric
        binding = family.judging[variant]
        assert binding.judge.definition == revision.final
        assert binding.judge.definition.rubric != family.rubric
    assert family.report is not None
    final = FamilyReport.model_validate_json(
        run.project.artifacts.read_bytes(family.report.artifact_id, "report.json")
    )
    assert final.coverage and all(not row.rubric_problems for row in final.coverage.values())
    before = run.calls()
    assert [row.replayed for row in run.run(spec, approve=False).stages] == [True] * 6
    assert run.calls() == before


def test_an_interrupted_stage_resumes_without_recalling_finished_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A judge interruption mid-mining resumes: earlier stages replay, nothing is asked twice."""
    run = wired(tmp_path, monkeypatch)
    spec = make_spec(run.fixture, run.rubric, run.bank, actors=("actor",), crosstalk=False)
    run.wire.interrupt_at = 4
    with pytest.raises(Interrupted):
        run.run(spec)
    finished = [
        row.stage
        for row in plan_decision_variance(
            run.project, spec, code_revision=REVISION, executors=run.executors
        ).stages
        if row.saved
    ]
    assert finished == ["variance-build", "rules", "traceml"]
    before = run.calls()
    report = run.run(spec)
    assert report.complete
    assert [row.replayed for row in report.stages] == [True] * 3 + [False] * 3
    # The interrupted request's intent was recorded before dispatch, so its outcome stays unknown
    # and is never sent again; every other saved judgment replays.
    judged = run.wire.judged
    assert len(judged) == len(set(judged)), "no judgment is asked twice"
    assert run.calls()[0] >= before[0]
