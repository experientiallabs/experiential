"""An offline proof that one saved LLM verdict owns every label on the decision-variance path.

The whole path runs through its real owners on the grounded ticket world of
:mod:`~exp.optimize.evaluation.variance.tests.grounded_world_test`; only the wire is scripted:

1. **Authoring.** One authoring packet over the world's start, then independent world and rubric
   author draws, a world freeze with restored variant starts and executed rule challenges, and a
   compatibility review. The world's validated close program is served in every variant runtime.
2. **Judge.** One :class:`~exp.optimize.evaluation.variance.episodes.JudgeProfile` fixed before
   any rubric, then a :class:`~exp.optimize.evaluation.variance.episodes.JudgeDefinition` per
   variant bound to the authored rubric and to the digests a run from that frozen variant has.
3. **Error mining.** A small G/J/C/K/M run from the frozen urgent start, judged by that judge.
   Its point selectors and critic are the default agentic investigators: each draw plans, reads
   evidence, drafts, self checks and answers, and every phase is saved on its transcript. The
   selector draws become group-relative selector training records. One judge-flip run forces
   single-response selectors and critics instead.
4. **Family.** Decision state, scenarios, frozen-variant witnesses, judged selection and fresh
   confirmation contrasts, the admission policy and training records.
5. **Harbor.** A qualified executable task whose trial ledger the same judge rewards.

The scripted judge (:class:`ProofWire`) answers from an explicit synthetic policy over the rendered
evidence: an episode passes exactly when its last recorded ticket outcome is ``right``, except that
the first replicate of every group is judged a failure so group advantages carry signal. This proves
plumbing, not model quality: labels, rates, advantages, critic rewards, admission and Harbor
rewards must all follow it, a replay must make no call, and each control of the implementation
brief's step 10 that is decidable offline is exercised here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, JsonValue
from exp.common.models import ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
)
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.executable.judged_acceptance import (
    accept_judged_task,
    load_judged_acceptance,
)
from exp.optimize.evaluation.executable.task import persist_executable_task
from exp.optimize.evaluation.executable.trial import open_executable_harbor_trial
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.challenges import ChallengeReport
from exp.optimize.evaluation.rules.stage import (
    REPORT_TYPE,
    RulesStageReport,
    RulesStageSpec,
    load_rulebook,
)
from exp.optimize.evaluation.variance.actor_critic import (
    ActorCriticReport,
    ActorCriticSpec,
    run_actor_critic,
)
from exp.optimize.evaluation.variance.admission import (
    FamilyAdmissionPolicy,
    VariantJudging,
    coverage_problems,
    frozen_run_digests,
    judging_problems,
    variant_problems,
)
from exp.optimize.evaluation.variance.authoring_packet import (
    AuthoringPacket,
    SourceRef,
    build_authoring_packet,
    load_packet,
    save_packet,
)
from exp.optimize.evaluation.variance.authoring_packet_test import (
    observed_episode,
    save_book,
    tool_reply,
)
from exp.optimize.evaluation.variance.authors import DECISION_STATE_TOOL, SCENARIO_TOOL
from exp.optimize.evaluation.variance.contracts import (
    ActorView,
    FreshBriefing,
    InstructionAxis,
    InstructionValue,
    Scenario,
    ScenarioLimits,
    ScenarioSet,
    check_scenarios,
)
from exp.optimize.evaluation.variance.episodes import JudgeProfile, episode_reward
from exp.optimize.evaluation.variance.evidence_tools import (
    AGENTIC_PROFILE,
    AnalysisTranscript,
    EvidenceToolProfile,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.family import (
    DecisionFamilySpec,
    FamilyReport,
    family_executable_task,
    run_decision_family,
)
from exp.optimize.evaluation.variance.judge import load_assessment
from exp.optimize.evaluation.variance.judge_prompt import ASSESS_TOOL, judge_profile
from exp.optimize.evaluation.variance.judge_test import _call
from exp.optimize.evaluation.variance.judged import (
    EpisodeJudge,
    JudgedStudy,
    judge_study,
    judged_episode,
    slot_coverage,
    slot_rewards,
)
from exp.optimize.evaluation.variance.judged_test import JudgeRouter, always, state_judge
from exp.optimize.evaluation.variance.records import (
    AuditDisposition,
    TrainingRecords,
    selector_records,
    training_eligibility,
    training_records,
)
from exp.optimize.evaluation.variance.rewards import SelectorObjective, group_advantages
from exp.optimize.evaluation.variance.role_prompts import (
    DRAFT_TOOL_NAME,
    PLAN_TOOL,
    SELF_CHECK_TOOL,
)
from exp.optimize.evaluation.variance.rubric_author import (
    RUBRIC_TOOL,
    author_rubric,
    judge_definition,
    load_rubric,
    save_rubric,
)
from exp.optimize.evaluation.variance.rubric_author_test import rubric_answer
from exp.optimize.evaluation.variance.tests.grounded_world_test import (
    ACTIONS,
    INITIAL,
    RULE,
    GroundedFixture,
    grounded_fixture,
    tier_decision,
    tier_scenario,
)
from exp.optimize.evaluation.variance.world_author import (
    WORLD_TOOL,
    WorldSpec,
    author_world,
    load_world_draft,
    world_draft_reference,
    world_runtime_definition,
)
from exp.optimize.evaluation.variance.world_author_test import CARELESS_CLOSE, CLOSE, world_answer
from exp.optimize.evaluation.variance.world_freeze import (
    CompatibilityReview,
    WorldFreeze,
    freeze_world,
    review_compatibility,
)
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.induction.sandbox import ProgramSandbox

Policy = Callable[[JsonObject], str]
"""Maps one rendered judge request (binding, rubric, scope and evidence) to a verdict."""

INTENT = "Escalate or close a support ticket as its service tier requires."
"""The capability the exported Harbor task measures."""

REVISION = "test"
"""Producing source identity of every saved artifact."""

FIRST_REPLICATES = ("-cell-0", "-branch-0")
"""Episode id suffixes of each group's first replicate, which the policy judges a failure."""

FREE_CHOICE = "required"
"""The tool choice of an agentic turn that may read evidence or close its phase."""

FORCED_PROFILE = EvidenceToolProfile()
"""The single forced-response profile the judge-flip run gives its selectors and critic."""


class Interrupted(BaseException):
    """Simulates the process stopping in the middle of a judge request."""


def episode_id(request: JsonObject) -> str:
    """Return the judged episode identity named by a rendered judge request."""
    binding = request["judge_binding"]
    assert isinstance(binding, dict)
    return str(binding["episode_id"])


def events_of(request: JsonObject) -> list[JsonObject]:
    """Return the rendered evidence events of one judge request."""
    evidence = request["episode_evidence"]
    assert isinstance(evidence, dict) and isinstance(evidence["events"], list)
    return [event for event in evidence["events"] if isinstance(event, dict)]


def semantic(request: JsonObject) -> str:
    """The synthetic task policy: pass a complete episode whose last ticket outcome is right.

    The first replicate of each contrast arm and continuation group is judged a failure, so every
    downstream group carries a non-constant label and its advantages are non-zero.

    Args:
        request: The rendered judge request.

    Returns:
        ``pass``, ``fail`` or ``unassessable``.
    """
    evidence = request["episode_evidence"]
    assert isinstance(evidence, dict)
    verdict = state_judge(("tickets", "outcome"), "right")(evidence)
    if verdict == "pass" and episode_id(request).endswith(FIRST_REPLICATES):
        return "fail"
    return verdict


def inverted(policy: Policy) -> Policy:
    """The same policy with pass and fail swapped, for the same evidence."""
    return lambda request: {"pass": "fail", "fail": "pass"}.get(policy(request), policy(request))


def malformed_for(suffix: str, policy: Policy) -> Policy:
    """The policy, except that episodes whose id ends with ``suffix`` get a malformed answer."""
    return lambda request: "malformed" if episode_id(request).endswith(suffix) else policy(request)


def assessment(request: JsonObject, verdict: str) -> JsonHttpResponse:
    """Answer ``assess_episode`` for one verdict, with a finding on every rubric criterion.

    Args:
        request: The rendered judge request.
        verdict: ``pass``, ``fail``, ``unassessable`` or ``malformed``.

    Returns:
        The scripted provider response citing the first scored event.
    """
    if verdict == "malformed":
        return _call(ASSESS_TOOL, {"verdict": "pass", "reason": "No findings."})
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
                "reason": f"{row['criterion_id']} is {found} at {decisive['event_id']}.",
            }
        )
    return _call(
        ASSESS_TOOL,
        {
            "verdict": verdict,
            "criterion_findings": findings,
            "reason": f"Overall {verdict}.",
            "decisive_refs": [] if verdict == "unassessable" else [decisive["event_id"]],
        },
    )


def priced_reply(tool: str, arguments: JsonObject) -> JsonHttpResponse:
    """One scripted reply calling ``tool``, reporting token usage the priced catalog can price."""
    reply = tool_reply(tool, arguments)
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
    return JsonHttpResponse(status_code=200, body={**reply.body, "usage": usage})


class ProofWire(JudgeRouter):
    """The scripted author wire: queued role replies, and judge answers from per-judge policies.

    Attributes:
        roles: Queued ``(tool, reply)`` pairs for every non-judge draw, served in order.
        role_requests: Every non-judge request, as ``(tool, payload)``.
        policies: Verdict policy per judge definition identity; others use ``verdict``.
        requests_judged: Every rendered judge request, in order.
        interrupt_at: Judge request number (1-based, counted from now) that raises
            :class:`Interrupted` once, if set.
    """

    def __init__(self, policy: Policy) -> None:
        """Start with no queued roles and one default verdict policy."""
        super().__init__(ScriptedJsonTransport(()), always("pass"))
        self.policy = policy
        self.roles: list[tuple[str, JsonHttpResponse]] = []
        self.role_requests: list[tuple[str, JsonObject]] = []
        self.policies: dict[str, Policy] = {}
        self.requests_judged: list[JsonObject] = []
        self.interrupt_at: int | None = None

    def queue(self, *replies: tuple[str, JsonObject]) -> None:
        """Queue forced role answers by tool name."""
        self.roles.extend((tool, tool_reply(tool, arguments)) for tool, arguments in replies)

    def queue_agentic(self, answer_tool: str, answer: JsonObject) -> None:
        """Queue one agentic draw: plan, one evidence read, draft, self check and final answer.

        Args:
            answer_tool: The role's answer tool, forced in the final phase.
            answer: The answer given both as the draft and as the final answer.
        """
        plan: JsonObject = {
            "hypotheses": ["The tier decides the handling."],
            "planned_reads": ["group"],
        }
        self.roles.extend(
            (
                (PLAN_TOOL.name, priced_reply(PLAN_TOOL.name, plan)),
                (FREE_CHOICE, priced_reply("list_divergence_points", {})),
                (FREE_CHOICE, priced_reply(DRAFT_TOOL_NAME, answer)),
                (
                    FREE_CHOICE,
                    priced_reply(SELF_CHECK_TOOL.name, {"issues": [], "revise": False}),
                ),
                (answer_tool, priced_reply(answer_tool, answer)),
            )
        )

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Serve a queued role answer, or judge the rendered evidence under its judge's policy.

        Args:
            url: Endpoint.
            headers: Headers.
            payload: The provider request.
            timeout_seconds: Timeout.

        Returns:
            The scripted response.

        Raises:
            AssertionError: A role request arrives unscripted or for another tool.
            Interrupted: The configured judge request was reached.
        """
        del url, headers, timeout_seconds
        choice = payload.get("tool_choice")
        function = choice.get("function") if isinstance(choice, dict) else None
        tool = str(function.get("name")) if isinstance(function, dict) else str(choice)
        if tool != ASSESS_TOOL:
            self.role_requests.append((tool, payload))
            assert self.roles, f"unscripted role request for {tool}"
            expected, reply = self.roles.pop(0)
            assert expected == tool, f"scripted {expected}, but the draw asked for {tool}"
            return reply
        messages = payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        request = json.loads(str(messages[1]["content"]))
        assert isinstance(request, dict)
        if self.interrupt_at is not None:
            self.interrupt_at -= 1
            if self.interrupt_at == 0:
                self.interrupt_at = None
                raise Interrupted
        self.requests_judged.append(request)
        evidence = request["episode_evidence"]
        assert isinstance(evidence, dict)
        self.judged.append(evidence)
        binding = request["judge_binding"]
        assert isinstance(binding, dict)
        policy = self.policies.get(str(binding["judge_definition"]), self.policy)
        return assessment(request, policy(request))


def exposure_plan() -> ExposurePlan:
    """One frozen plan for the whole path: mining probes, world learning and final evaluation."""
    return ExposurePlan(
        bindings=ArtifactInput(artifact_id="bindings", sha256="0" * 64),
        seed=0,
        probe_fraction=0.5,
        roles={"tickets": "mining_probes", "sources": "world_learning", "held": "final_evaluation"},
        traces={"trace-1": "sources", "trace-final": "held"},
    )


def decision_answer(step: int) -> JsonObject:
    """The decision-state author's answer at one step, grounded in the state and the tier rule."""
    decision = tier_decision(step).model_dump(mode="json", exclude_defaults=True)
    return {
        "state": decision,
        "grounding": [
            {
                "path": "/tickets/severity",
                "revealed_by": ["check_sla"],
                "refs": ["state:/tickets/severity", f"rule:{RULE.rule_id}"],
            }
        ],
        "limitations": ["Simulated tickets only."],
    }


def scenario_answer(*extra: Scenario) -> JsonObject:
    """The scenario author's answer: the urgent and routine tiers, plus any extra scenarios."""
    scenarios = (tier_scenario("urgent", "high"), tier_scenario("routine", "low"), *extra)
    return {
        "scenarios": [s.model_dump(mode="json", exclude_defaults=True) for s in scenarios],
        "limitations": ["Tier variants."],
    }


def selector_answer() -> JsonObject:
    """One point selector choosing the start and the step after the first read."""
    return {
        "points": [
            {
                "step": step,
                "decision": "Escalate or close the ticket",
                "alternatives": ["Check the service tier first"],
                "deciding_state": [],
                "rationale": "Attempts diverge on the ticket's handling.",
                "evidence_refs": [f"point:{step}", "group"],
            }
            for step in (0, 1)
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
    """One critic estimating both chosen points."""
    return {
        "estimates": [
            {
                "step": step,
                "probability": 0.6,
                "rationale": "Base rate.",
                "evidence_refs": ["group"],
            }
            for step in (0, 1)
        ],
        "limitations": ["Simulated tickets only."],
    }


def rules_stage(
    project: ProjectStore, settings: AuthorSettings, rulebook: ArtifactInput
) -> ArtifactInput:
    """Save a rules-stage report naming the packet's own rule book.

    Args:
        project: Project receiving the report.
        settings: Author settings recorded on the stage spec.
        rulebook: The rule book the authoring packet was built from.

    Returns:
        The ``rules-stage-report`` reference.
    """
    book = load_rulebook(project, rulebook)
    report = RulesStageReport(
        spec=RulesStageSpec(author=settings),
        identity="ticket-proof-rules-stage",
        extraction_identity="ticket-proof-rules-extraction",
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
    manifest = project.artifacts.write_json(
        artifact_id="ticket-proof-rules-stage-result",
        artifact_type=REPORT_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=REVISION,
            inputs=(rulebook,),
        ),
        files={"report.json": report},
    )
    return artifact_input(manifest)


def freeze_reference(project: ProjectStore, freeze: WorldFreeze) -> ArtifactInput:
    """Return the saved reference of a world-freeze report."""
    return artifact_input(project.artifacts.read(freeze.identity + "-result").manifest)


@dataclass
class Authored:
    """One authored, frozen and reviewed world with its rubric.

    Attributes:
        world: The ready world.
        world_ref: The saved world draft.
        freeze: The freeze report.
        freeze_ref: Its saved reference.
        review: The world/rubric compatibility review.
    """

    world: WorldSpec
    world_ref: ArtifactInput
    freeze: WorldFreeze
    freeze_ref: ArtifactInput
    review: CompatibilityReview


@dataclass
class Proof:
    """The whole offline path over one project, and the scripted wire it ran on.

    Attributes:
        fixture: The grounded world, its job and its actor and predictor transports.
        wire: The author/judge wire.
        packet: The authoring packet.
        packet_ref: Its saved reference.
        rubric: The saved rubric artifact.
        authored: The frozen world and its review.
        profile: The judge profile fixed before the rubric existed.
        judging: The frozen judge binding per variant.
        exposure: The frozen data roles.
        job: The error-mining job at the frozen urgent start.
        mining: The error-mining registration.
        report: The error-mining report.
        family: The eligible family report.
        family_spec: Its registration, dispositions included.
        trials: Harbor trial results by trial id.
        qualification: The executable qualification of the urgent task.
        task: The saved urgent executable task.
    """

    fixture: GroundedFixture
    wire: ProofWire
    packet: AuthoringPacket
    packet_ref: ArtifactInput
    rubric: ArtifactInput
    authored: Authored
    profile: JudgeProfile
    judging: dict[str, VariantJudging]
    exposure: ExposurePlan
    job: ContinuationJob
    mining: ActorCriticSpec
    report: ActorCriticReport
    family: FamilyReport
    family_spec: DecisionFamilySpec
    trials: dict[str, JsonObject] = field(default_factory=dict)
    qualification: ArtifactInput | None = None
    task: ArtifactInput | None = None

    @property
    def project(self) -> ProjectStore:
        """The proof project."""
        return self.fixture.project

    def calls(self) -> tuple[int, int, int, int]:
        """Actor, predictor, role and judge requests made so far."""
        return (
            len(self.fixture.actor.requests),
            len(self.fixture.predictor.requests),
            len(self.wire.role_requests),
            len(self.wire.requests_judged),
        )


def author_and_freeze(
    fixture: GroundedFixture,
    wire: ProofWire,
    packet_ref: ArtifactInput,
    rubric: ArtifactInput,
    *,
    close: str = CLOSE,
    identity: str = "proof-world",
) -> Authored:
    """Author one world from the packet, freeze it and review it against the saved rubric.

    Args:
        fixture: The grounded fixture.
        wire: The author wire, already queued with the world answer.
        packet_ref: The saved packet.
        rubric: The saved rubric.
        close: Source of the close program the world author returns.
        identity: World draft identity.

    Returns:
        The authored world, its freeze and its review.
    """
    project = fixture.project
    packet = load_packet(project, packet_ref)
    wire.queue((WORLD_TOOL, world_answer(packet, close=close)))
    with ProgramSandbox() as sandbox:
        draft = author_world(
            project,
            fixture.settings,
            packet_ref,
            sandbox=sandbox,
            identity=identity,
            code_revision=REVISION,
        )
        assert draft.status == "ready" and draft.world is not None, draft.failure
        world_ref = world_draft_reference(project, draft)
        freeze = freeze_world(
            project,
            packet_ref,
            world_ref,
            sandbox=sandbox,
            code_revision=REVISION,
            base=fixture.definition,
            actor=ChatAgentRuntime(maximum_model_calls=6),
        )
    world = load_world_draft(project, world_ref).world
    assert world is not None
    review = review_compatibility(packet, world, load_rubric(project, rubric))
    return Authored(world, world_ref, freeze, freeze_reference(project, freeze), review)


def variant_judging(
    proof_project: ProjectStore,
    packet: AuthoringPacket,
    authored: Authored,
    rubric: ArtifactInput,
    profile: JudgeProfile,
    settings: AuthorSettings,
    base: GroundedFixture,
    variant: str,
) -> VariantJudging:
    """Bind the profile and rubric to the digests of a run from one frozen variant.

    Args:
        proof_project: Project holding the freeze and rubric.
        packet: The authoring packet.
        authored: The frozen world.
        rubric: The saved rubric.
        profile: The judge profile.
        settings: Admitted judge request settings matching the profile.
        base: The fixture whose runtime definition the variant's runtime extends.
        variant: The variant identity.

    Returns:
        The variant's judge binding with its freeze, runtime and review.
    """
    task, world = frozen_run_digests(proof_project, authored.freeze_ref, variant)
    definition = judge_definition(
        proof_project, profile, rubric, task_sha256=task, world_sha256=world
    )
    runtime = world_runtime_definition(packet, authored.world, variant, base=base.definition)
    return VariantJudging(
        judge=EpisodeJudge(definition=definition, settings=settings),
        world_freeze=authored.freeze_ref,
        freeze_variant=variant,
        runtime=runtime.model_dump(mode="json"),
        compatibility=authored.review,
    )


def frozen_job(fixture: GroundedFixture, binding: VariantJudging) -> ContinuationJob:
    """The fixture job rebound to one frozen variant's start and runtime definition."""
    assert binding.world_freeze is not None and binding.runtime is not None
    freeze = WorldFreeze.model_validate_json(
        fixture.project.artifacts.read_bytes(binding.world_freeze.artifact_id, "freeze.json")
    )
    row = next(r for r in freeze.variants if r.variant_id == binding.freeze_variant)
    assert row.checkpoint is not None
    probe = world_definition_probe(binding.runtime)
    study = fixture.job.study.model_copy(
        update={"checkpoint": row.checkpoint, "probe_sha256": probe}
    )
    return ContinuationJob.model_validate_json(
        fixture.job.model_copy(
            update={"study": study, "configuration": binding.runtime}
        ).model_dump_json()
    )


def world_definition_probe(runtime: JsonObject) -> str:
    """Return the probe digest of a grounded runtime definition given as configuration."""
    return GroundedRuntimeDefinition.model_validate(runtime).probe_sha256


def qualify(proof: Proof, scenario_id: str) -> tuple[ArtifactInput, ArtifactInput]:
    """Qualify one variant's exact executable task with a feasible run and witnessed controls.

    Args:
        proof: The proof whose family exports the task.
        scenario_id: The variant.

    Returns:
        The saved executable task and its qualification.
    """
    project = proof.project
    task = family_executable_task(proof.family, scenario_id, intent=INTENT)
    reference = persist_executable_task(
        project, task, created_at=datetime.now(UTC), code_revision=REVISION
    )
    qualification = accept_judged_task(
        project,
        reference,
        admission=None,
        admitted=True,
        created_at=datetime.now(UTC),
        code_revision=REVISION,
    )
    saved = load_judged_acceptance(project, qualification)
    assert saved.qualified, saved
    return reference, qualification


HARBOR_EPISODES: dict[str, tuple[tuple[str, ...], str]] = {
    "check-then-escalate": (("check_sla", "escalate"), "The tier is high, so I escalated it."),
    "read-check-escalate": (
        ("read_ticket", "check_sla", "escalate"),
        "I read the ticket, saw the high tier and escalated it.",
    ),
    "close-without-checking": (("close",), "I closed the ticket."),
}
"""Harbor episodes: two different valid solutions and one mistake, as tool calls and a message."""


def run_trial(
    project: ProjectStore, task: ArtifactInput, qualification: ArtifactInput, trial_id: str
) -> JsonObject:
    """Run, or replay, one Harbor trial of a scripted episode and return its result."""
    tools, message = HARBOR_EPISODES[trial_id]
    with open_executable_harbor_trial(
        project, task, trial_id=trial_id, code_revision=REVISION, qualification=qualification
    ) as trial:
        if trial.result()["judge"] is not None and trial.result().get("reward") is not None:
            return trial.result()
        for index, name in enumerate(tools):
            trial.call(ToolCall(name=name, call_id=f"{trial_id}-{index}", arguments={}))
        trial.message(message)
        trial.submit({"answer": message})
        return trial.result()


def disposed(spec: DecisionFamilySpec, report: FamilyReport) -> DecisionFamilySpec:
    """Return the spec with a reviewer disposition for every finding of a report."""
    return spec.model_copy(
        update={
            "dispositions": tuple(
                AuditDisposition(
                    finding_id=finding.finding_id,
                    resolution="accepted_with_reason",
                    rationale="Reviewed against the saved grounded trajectories.",
                    resolver="proof-reviewer",
                )
                for finding in report.findings
            )
        }
    )


def run_family(
    proof_project: ProjectStore,
    spec: DecisionFamilySpec,
    report: ActorCriticReport,
    exposure: ExposurePlan,
) -> FamilyReport:
    """Run or replay the family at the frozen start, point 0."""
    return run_decision_family(
        proof_project, spec, exposure=exposure, report=report, point=0, code_revision=REVISION
    )


def build(root: Path, monkeypatch: pytest.MonkeyPatch) -> Proof:
    """Run the whole offline path once over a fresh grounded project.

    Args:
        root: Project root.
        monkeypatch: Installs the scripted transports.

    Returns:
        The proof with every stage's saved result.
    """
    fixture = grounded_fixture(root, monkeypatch, priced=True)
    project, settings = fixture.project, fixture.settings
    wire = ProofWire(semantic)
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: wire
    )
    exposure = exposure_plan()
    # The judge profile is configured before any rubric exists.
    profile = judge_profile(
        model_alias=settings.model_alias,
        model=settings.model,
        maximum_output_tokens=settings.output_tokens,
    )
    rulebook, book = save_book(project)
    start = load_continuation_checkpoint(project.artifacts, fixture.job.study.checkpoint)
    packet = build_authoring_packet(
        family_id="ticket-tier",
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
                summary="An agent closed a low tier ticket.",
            ),
        ),
        exposure=exposure,
        rule_book=rulebook,
        book=book,
        decision=tier_decision(0),
        scenarios=(tier_scenario("urgent", "high"), tier_scenario("routine", "low")),
        observed_episodes=(observed_episode(),),
    )
    packet_ref = save_packet(project, packet, code_revision=REVISION)
    wire.queue((RUBRIC_TOOL, rubric_answer()))
    rubric_draft = author_rubric(
        project, settings, packet_ref, identity="proof-rubric", code_revision=REVISION
    )
    assert rubric_draft.status == "ready", rubric_draft.failure
    rubric = save_rubric(project, rubric_draft, code_revision=REVISION)
    authored = author_and_freeze(fixture, wire, packet_ref, rubric)
    assert authored.freeze.trainable and authored.review.compatible, authored.freeze.problems
    judging = {
        variant: variant_judging(
            project, packet, authored, rubric, profile, settings, fixture, variant
        )
        for variant in ("urgent", "routine")
    }
    job = frozen_job(fixture, judging["urgent"])
    mining = ActorCriticSpec.error_mining(
        job,
        episode_judge=judging["urgent"].judge,
        selector=settings,
        critic=settings,
        attempts=3,
        selector_draws=2,
        critic_draws=1,
        points=2,
        continuations=3,
        group_physical_attempts=40,
    )
    assert mining.selector_profile == mining.critic_profile == AGENTIC_PROFILE
    for _ in mining.judges:
        wire.queue_agentic("choose_points", selector_answer())
    wire.queue_agentic("estimate_points", critic_answer())
    report = run_actor_critic(project, mining, exposure=exposure, code_revision=REVISION)
    assert all(row.failure is None for row in report.points), report.points
    spec = DecisionFamilySpec(
        decision_author=settings,
        scenario_author=settings,
        rules_stage=rules_stage(project, settings, rulebook),
        actions=ACTIONS,
        judging=judging,
        repeats=6,
        confirmation_repeats=6,
        tiers={job.study.actor_model.model_id: "strong"},
        measure_decisions=False,
    )
    wire.queue((DECISION_STATE_TOOL, decision_answer(0)), (SCENARIO_TOOL, scenario_answer()))
    first = run_family(project, spec, report, exposure)
    assert first.failure is None, first.failure
    spec = disposed(spec, first)
    family = run_family(project, spec, report, exposure)
    assert family.eligibility.eligible, family.eligibility.reasons
    proof = Proof(
        fixture=fixture,
        wire=wire,
        packet=packet,
        packet_ref=packet_ref,
        rubric=rubric,
        authored=authored,
        profile=profile,
        judging=judging,
        exposure=exposure,
        job=job,
        mining=mining,
        report=report,
        family=family,
        family_spec=spec,
    )
    proof.task, proof.qualification = qualify(proof, "urgent")
    for trial_id in HARBOR_EPISODES:
        proof.trials[trial_id] = run_trial(project, proof.task, proof.qualification, trial_id)
    return proof


@pytest.fixture(scope="module")
def proof(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Proof]:
    """Build the offline path once; every proof below adds its own stages to the same project."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        # Module scope outlives the per-test XDG isolation: keep the host's admission limits
        # (and their concurrency) out of the scripted, order-sensitive run.
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path_factory.mktemp("proof-xdg")))
        yield build(tmp_path_factory.mktemp("proof"), monkeypatch)


def judged_study(project: ProjectStore, reference: ArtifactInput | None) -> JudgedStudy:
    """Load a saved judged continuation study."""
    assert reference is not None
    return JudgedStudy.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "judged.json")
    )


def point_labels(
    project: ProjectStore, report: ActorCriticReport
) -> dict[int, tuple[float | None, ...]]:
    """Every error-mining point's continuation labels, read back from saved assessments."""
    labels = {}
    for row in report.points:
        judged = judged_study(project, row.judged)
        labels[row.point] = slot_rewards(project, judged.branches, judged.judge)
    return labels


def option_labels(
    project: ProjectStore, family: FamilyReport, stage: str
) -> dict[tuple[str, str], tuple[float | None, ...]]:
    """Every family contrast's per-option labels, read back from saved assessments."""
    labels = {}
    for row in family.variants:
        contrast = row.selection if stage == "selection" else row.confirmation
        if contrast is None:
            continue
        for option in contrast.options:
            slots = tuple(cell.slot for cell in option.cells)
            labels[(row.scenario_id, option.option_id)] = slot_rewards(
                project, slots, contrast.judge
            )
    return labels


def role_transcript(project: ProjectStore, reference: ArtifactInput | None) -> AnalysisTranscript:
    """Load one selector or critic draw's saved analysis transcript."""
    assert reference is not None
    return AnalysisTranscript.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "transcript.json")
    )


def trial_reward(project: ProjectStore, result: JsonObject) -> float | None:
    """Re-encode a Harbor result's saved assessment, the only source its reward may have."""
    binding = result["judge"]
    assert isinstance(binding, dict) and isinstance(binding["assessment"], dict)
    return episode_reward(
        load_assessment(project, ArtifactInput.model_validate(binding["assessment"]))
    )


def test_every_label_rate_advantage_and_reward_traces_to_a_saved_assessment(proof: Proof) -> None:
    """Error mining, family records and Harbor rewards are all encodings of saved verdicts."""
    project, report, family = proof.project, proof.report, proof.family
    urgent = proof.judging["urgent"].judge.identity
    assert report.counts is not None
    assert (report.counts.continuations_planned, report.counts.continuations_graded) == (6, 6)
    for row in report.points:
        judged = judged_study(project, row.judged)
        assert judged.judge == urgent and row.meaning.judge_sha256 == urgent
        labels = slot_rewards(project, judged.branches, urgent)
        assert labels == (0.0, 1.0, 1.0)
        assert (row.outcome.successes, row.outcome.failures, row.outcome.unknown) == (2, 1, 0)
        assert row.group.advantages == group_advantages(row.group.point_id, labels).advantages
        for slot in judged.branches:
            found = judged_episode(project, slot, urgent)
            assert found is not None and found.status == "judged"
    (critic,) = report.critics
    assert critic.score is not None and critic.score.errors == {
        f"point-{step}": pytest.approx((0.6 - 2 / 3) ** 2) for step in (0, 1)
    }
    assert family.eligibility.eligible and family.records is not None
    records = family.records
    assert isinstance(records, TrainingRecords)
    groups = records.groups["strong"]
    assert sorted((g.scenario_id, g.option_id) for g in groups) == [
        ("routine", "close"),
        ("routine", "escalate"),
        ("urgent", "close"),
        ("urgent", "escalate"),
    ]
    confirmed = option_labels(project, family, "confirmation")
    for group in groups:
        assert group.judge == proof.judging[group.scenario_id].judge.identity
        labels = slot_rewards(project, group.episodes, group.judge)
        assert labels == confirmed[(group.scenario_id, group.option_id)]
        assert None not in labels and all(slot.episode for slot in group.episodes)
        expected = group_advantages(f"{group.scenario_id}:{group.option_id}", labels)
        assert group.advantages == expected.advantages
    assert confirmed[("urgent", "escalate")] == (0.0, *(1.0,) * 5)
    assert {group.advantages for group in groups if group.option_id == "close"} == {(0.0,) * 6}
    # The error-mining judge is not the per-variant frozen judges, so discovery critics
    # scored by it are refused for training, with the reason recorded.
    assert records.critics == () and records.critics_refused
    rewards = {trial: trial_reward(project, result) for trial, result in proof.trials.items()}
    assert rewards == {
        "check-then-escalate": 1.0,
        "read-check-escalate": 1.0,
        "close-without-checking": 0.0,
    }
    for trial, result in proof.trials.items():
        assert result["reward"] == rewards[trial] and result["status"] == "completed"
        assert (
            result["label_authority"] == "frozen LLM episode judge" and "mechanical" not in result
        )


def test_agentic_roles_record_every_phase_and_selector_draws_become_training_records(
    proof: Proof,
) -> None:
    """Each selector and critic draw plans, reads, drafts, self checks and answers on disk."""
    project, report = proof.project, proof.report
    assert len(report.judges) == 2 and len(report.critics) == 1
    for draw in (*report.judges, *report.critics):
        transcript = role_transcript(project, draw.transcript)
        assert transcript.stop == "answered" and transcript.skipped == ()
        assert [call.phase for call in transcript.calls] == [
            "plan",
            "investigate",
            "investigate",
            "self_check",
            "final",
        ]
        assert [row.phase for row in transcript.outputs] == ["plan", "draft", "self_check"]
        assert transcript.outputs[1].host_validation == "valid"
        assert [(row.name, row.status) for row in transcript.tools] == [
            ("list_divergence_points", "ok")
        ]
        assert transcript.calls[-1].forced_answer and transcript.usage.unpriced_calls == 0
    urgent = proof.judging["urgent"].judge.identity
    # Each point has three judged continuations, so the objective scores points with three.
    objective = SelectorObjective(minimum_scored=3)
    selectors = selector_records(project, report, judge=urgent, objective=objective)
    assert selectors.refused == () and len(selectors.draws) == 2
    assert all(row.trainable and row.points == (0, 1) for row in selectors.draws)
    # Both draws chose the same points, each judged 0, 1, 1: their rewards tie at the mean outcome
    # variance and the advantages are zero.
    for row in selectors.draws:
        assert row.score is not None and row.score.reward == pytest.approx(8 / 9)
    assert [row.advantage for row in selectors.draws] == [0.0, 0.0]
    routine = proof.judging["routine"].judge.identity
    other = selector_records(project, report, judge=routine, objective=objective)
    assert not any(row.trainable for row in other.draws) and other.refused


def test_the_frozen_world_serves_its_programs_and_records_its_rule_challenges(
    proof: Proof,
) -> None:
    """Every variant runtime binds the validated close program; the freeze executed challenges."""
    for binding in proof.judging.values():
        assert binding.runtime is not None
        runtime = GroundedRuntimeDefinition.model_validate(binding.runtime)
        assert runtime.programs is not None and "close" in runtime.programs.induced_tools
    freeze = proof.authored.freeze
    assert freeze.challenges is not None
    saved = ChallengeReport.model_validate_json(
        proof.project.artifacts.read_bytes(freeze.challenges.artifact_id, "report.json")
    )
    # The ticket book's equality rules have no numeric or membership boundary to challenge, so
    # the served world executed no workload and no rule changed status or became diagnostic.
    assert saved.rulebook == proof.packet.rule_book and saved.runs == freeze.workloads == ()
    assert saved.rules == freeze.challenge_rules
    assert (saved.diagnostic, saved.world_defects) == ((), ())


def test_the_mechanical_verifier_disagreeing_never_changes_a_label(proof: Proof) -> None:
    """Every continuation's verifier grade passes; the judge's failed first replicates stay 0."""
    project = proof.project
    for row in proof.report.points:
        assert row.study is not None
        mechanical = [b.assessment for b in load_study(project.artifacts, row.study).branches]
        assert all(grade is not None and grade.status == "pass" for grade in mechanical)
    assert all(labels[0] == 0.0 for labels in point_labels(project, proof.report).values())


def test_replaying_every_stage_makes_no_new_call(proof: Proof) -> None:
    """Authors, freeze, error mining, family, qualification and Harbor all replay from disk."""
    project, settings, before = proof.project, proof.fixture.settings, proof.calls()
    with ProgramSandbox() as sandbox:
        draft = author_world(
            project,
            settings,
            proof.packet_ref,
            sandbox=sandbox,
            identity="proof-world",
            code_revision=REVISION,
        )
        assert world_draft_reference(project, draft) == proof.authored.world_ref
        assert (
            freeze_world(
                project,
                proof.packet_ref,
                proof.authored.world_ref,
                sandbox=sandbox,
                code_revision=REVISION,
                base=proof.fixture.definition,
                actor=ChatAgentRuntime(maximum_model_calls=6),
            )
            == proof.authored.freeze
        )
    rubric = author_rubric(
        project, settings, proof.packet_ref, identity="proof-rubric", code_revision=REVISION
    )
    assert save_rubric(project, rubric, code_revision=REVISION) == proof.rubric
    assert (
        run_actor_critic(project, proof.mining, exposure=proof.exposure, code_revision=REVISION)
        == proof.report
    )
    assert run_family(project, proof.family_spec, proof.report, proof.exposure) == proof.family
    assert proof.task is not None and proof.qualification is not None
    assert qualify(proof, "urgent") == (proof.task, proof.qualification)
    for trial_id, result in proof.trials.items():
        assert run_trial(project, proof.task, proof.qualification, trial_id) == result
    assert proof.calls() == before and proof.wire.roles == []


def test_imposed_actions_are_the_experiments_and_inherited_prefixes_are_context(
    proof: Proof,
) -> None:
    """A forced contrast action is scored and owned by the experiment; a prefix is context."""
    cells = [r for r in proof.wire.requests_judged if "action-contrast" in episode_id(r)]
    assert cells
    for request in cells:
        scope = request["assessment_scope"]
        evidence = request["episode_evidence"]
        assert isinstance(scope, dict) and isinstance(evidence, dict)
        assert scope["experiment_imposed_event_ids"] == ["imposed.call"]
        events = {e["event_id"]: e for e in evidence["events"] if isinstance(e, dict)}
        imposed = events["imposed.call"]
        assert (imposed["owner"], imposed["scope"], imposed["kind"]) == (
            "experiment",
            "scored",
            "tool_call",
        )
        assert scope["scored_start"] == "imposed.call"
        actor_ids = scope["actor_scored_event_ids"]
        assert isinstance(actor_ids, list) and "imposed.call" not in actor_ids
    inherited = [
        [e for e in events_of(r) if str(e["event_id"]).startswith("prefix.")]
        for r in proof.wire.requests_judged
    ]
    point_one = [rows for rows in inherited if any(e["kind"] == "tool_call" for e in rows)]
    assert point_one, "continuations from point 1 carry the inherited read_ticket call"
    for rows in point_one:
        assert {(e["owner"], e["scope"]) for e in rows if e["kind"] == "tool_call"} == {
            ("actor", "context")
        }
        assert {e["scope"] for e in rows} == {"context"}


def test_private_judge_material_never_reaches_actors_or_the_other_author(proof: Proof) -> None:
    """Rubric text, judge prompts and calibration cases stay out of actor and world-author input."""
    project = proof.project
    rubric = load_rubric(project, proof.rubric)
    secret = rubric.body.task_success
    actor_wire = json.dumps([request for request in proof.fixture.actor.requests])
    for marker in (secret, ASSESS_TOOL, "criterion_id", "calibration_cases", "rubric"):
        assert marker not in actor_wire
    world_requests = [payload for tool, payload in proof.wire.role_requests if tool == WORLD_TOOL]
    assert world_requests and secret not in json.dumps(world_requests)
    for request in proof.wire.requests_judged:
        shown = request["rubric"]
        assert isinstance(shown, dict) and isinstance(shown["body"], dict)
        assert "calibration_cases" not in shown["body"]
    for result in proof.trials.values():
        assert secret not in json.dumps(result) and "criterion_id" not in json.dumps(result)
    rulebook, book = proof.packet.rule_book, load_rulebook(project, proof.packet.rule_book)
    with pytest.raises(ValueError, match="reserved for final evaluation"):
        build_authoring_packet(
            family_id="ticket-tier",
            task=proof.packet.task,
            checkpoint=proof.packet.checkpoint,
            before_state=INITIAL,
            public_history=(),
            remaining_calls=5,
            sources=(
                SourceRef(source_id="held", kind="trace", trace_id="trace-final", summary="Held."),
            ),
            exposure=proof.exposure,
            rule_book=rulebook,
            book=book,
            decision=tier_decision(0),
            scenarios=(tier_scenario("urgent", "high"), tier_scenario("routine", "low")),
        )


def flipped_judging(
    proof: Proof, profile: JudgeProfile, policy: Policy
) -> dict[str, VariantJudging]:
    """Bind a second judge profile to the same rubric and frozen variants under its own policy."""
    judging = {
        variant: variant_judging(
            proof.project,
            proof.packet,
            proof.authored,
            proof.rubric,
            profile,
            proof.fixture.settings,
            proof.fixture,
            variant,
        )
        for variant in ("urgent", "routine")
    }
    for binding in judging.values():
        proof.wire.policies[binding.judge.identity] = policy
    return judging


def test_a_judge_flip_moves_every_downstream_number_over_the_same_evidence(proof: Proof) -> None:
    """Inverted verdicts over the same saved episodes flip labels, rates, advantages and rewards."""
    project, wire = proof.project, proof.wire
    profile = judge_profile(
        model_alias=proof.profile.model_alias,
        model=proof.profile.model,
        maximum_output_tokens=proof.profile.maximum_output_tokens,
        temperature=1.0,
    )
    judging = flipped_judging(proof, profile, inverted(semantic))
    # This run forces one response per selector and critic draw instead of the agentic loop.
    wire.queue(*[("choose_points", selector_answer())] * 2, ("estimate_points", critic_answer()))
    forced = proof.mining.model_copy(
        update={
            "episode_judge": judging["urgent"].judge,
            "selector_profile": FORCED_PROFILE,
            "critic_profile": FORCED_PROFILE,
        }
    )
    mining = ActorCriticSpec.model_validate(forced.model_dump())
    report = run_actor_critic(project, mining, exposure=proof.exposure, code_revision=REVISION)
    for draw in (*report.judges, *report.critics):
        transcript = role_transcript(project, draw.transcript)
        assert [call.phase for call in transcript.calls] == ["final"] and not transcript.outputs
    # The group replays; the start's continuations are the same saved episodes. The pick seed
    # derives from the run identity, so later points may come from another identical attempt.
    assert report.group == proof.report.group
    assert report.points[0].study == proof.report.points[0].study
    base, flipped = point_labels(project, proof.report), point_labels(project, report)
    assert flipped == {
        point: tuple(1.0 - x for x in labels if x is not None) for point, labels in base.items()
    }
    for old, new in zip(proof.report.points, report.points, strict=True):
        negated = [-x for x in old.group.advantages if x is not None]
        assert list(new.group.advantages) == pytest.approx(negated)
    (critic,) = report.critics
    assert critic.score is not None
    assert critic.score.errors == {f"point-{s}": pytest.approx((0.6 - 1 / 3) ** 2) for s in (0, 1)}
    spec = proof.family_spec.model_copy(update={"judging": judging})
    wire.queue((DECISION_STATE_TOOL, decision_answer(0)), (SCENARIO_TOOL, scenario_answer()))
    actor_before = len(proof.fixture.actor.requests)
    family = run_family(project, spec, report, proof.exposure)
    assert len(proof.fixture.actor.requests) == actor_before
    assert [r.selection.contrasts for r in family.variants if r.selection] == [
        r.selection.contrasts for r in proof.family.variants if r.selection
    ]
    base_cells, flipped_cells = (
        option_labels(project, proof.family, "selection"),
        option_labels(project, family, "selection"),
    )
    assert flipped_cells == {
        key: tuple(1.0 - x for x in labels if x is not None) for key, labels in base_cells.items()
    }
    assert family.selection is not None and not family.selection.qualifies
    assert set(family.selection.triage) == {"urgent", "routine"}
    assert not family.eligibility.eligible and family.records is None
    flipped_proof = replace(proof, family=family, judging=judging, trials={})
    task, qualification = qualify(flipped_proof, "urgent")
    rewards = {
        trial: run_trial(project, task, qualification, trial)["reward"] for trial in HARBOR_EPISODES
    }
    assert rewards == {
        "check-then-escalate": 0,
        "read-check-escalate": 0,
        "close-without-checking": 1,
    }
    assert wire.roles == []


def test_malformed_judgments_leave_coverage_gaps_that_block_training(proof: Proof) -> None:
    """A malformed answer is an unscored gap in the denominator, and admission refuses records."""
    project, wire = proof.project, proof.wire
    for binding in proof.judging.values():
        wire.policies[binding.judge.identity] = malformed_for("-cell-19", semantic)
    try:
        spec = proof.family_spec.model_copy(
            update={"confirmation_seed": 1, "confirmation_repeats": 10}
        )
        wire.queue((DECISION_STATE_TOOL, decision_answer(0)), (SCENARIO_TOOL, scenario_answer()))
        family = run_family(project, spec, proof.report, proof.exposure)
    finally:
        for binding in proof.judging.values():
            wire.policies.pop(binding.judge.identity)
    assert family.confirmation is not None and family.confirmation.qualifies
    for scenario, coverage in family.coverage.items():
        assert (coverage.planned, coverage.malformed, coverage.unscored) == (20, 1, 1), scenario
    labels = option_labels(project, family, "confirmation")
    for scenario in ("urgent", "routine"):
        unscored = [x for (s, _), row in labels.items() if s == scenario for x in row if x is None]
        assert len(unscored) == 1, "the malformed cell stays in the denominator as unscored"
    failed = [c for c in family.eligibility.conditions if not c.holds]
    assert [c.name for c in failed] == ["judge_coverage"]
    assert "1 episodes are operationally unscored" in failed[0].detail
    assert family.records is None
    lenient = FamilyAdmissionPolicy(allow_operational_gaps=True, maximum_unscored_fraction=0.1)
    assert all(coverage_problems(c, lenient) == ((), ()) for c in family.coverage.values())


def test_an_unsupported_authorization_boundary_blocks_training(proof: Proof) -> None:
    """A world that silently closes a ticket with no lock state freezes as rejected; no records."""
    project, wire = proof.project, proof.wire
    careless = author_and_freeze(
        proof.fixture,
        wire,
        proof.packet_ref,
        proof.rubric,
        close=CARELESS_CLOSE,
        identity="proof-careless",
    )
    assert not careless.freeze.trainable
    assert {row.status for row in careless.freeze.variants} == {"rejected"}
    allowed = {p.case for p in careless.freeze.probes if p.resolution == "silently_allowed"}
    assert allowed == {"missing:/tickets/lock"}
    bindings = {
        variant: binding.model_copy(update={"world_freeze": careless.freeze_ref})
        for variant, binding in proof.judging.items()
    }
    _, frozen = variant_problems(project, bindings["urgent"])
    assert frozen == ("the world freeze is not trainable", "freeze variant urgent is rejected")
    family = proof.family
    starts = {
        row.scenario_id: tuple(c.checkpoint for c in (row.selection, row.confirmation) if c)
        for row in family.variants
    }
    problems = judging_problems(project, bindings, family.coverage, family.spec.admission, starts)
    assert set(problems["world_freeze"]) == {"urgent", "routine"}
    eligibility = training_eligibility(
        selection=family.selection,
        confirmation=family.confirmation,
        findings=family.findings,
        dispositions=proof.family_spec.dispositions,
        critics=proof.report.critics,
        uncovered_tools=(),
        relied_rules=(),
        contradicted_rules=(),
        unwitnessed_rules=(),
        meanings=tuple(r.meaning for r in family.variants if r.meaning),
        judging=problems,
    )
    assert not eligibility.eligible
    built = training_records(project, eligibility, proof.report, {})
    assert isinstance(built, tuple) and any("world_freeze" in reason for reason in built)
    before = proof.calls()
    spec = proof.family_spec.model_copy(update={"judging": bindings, "confirmation_seed": 3})
    wire.queue((DECISION_STATE_TOOL, decision_answer(0)), (SCENARIO_TOOL, scenario_answer()))
    refused = run_family(project, spec, proof.report, proof.exposure)
    assert refused.failure is not None and refused.failure.startswith("frozen start:")
    assert refused.records is None and not refused.eligibility.eligible
    actor, predictor, _, judged = proof.calls()
    assert (actor, predictor, judged) == (before[0], before[1], before[3])


def test_state_only_rewording_is_refused_before_any_contrast(proof: Proof) -> None:
    """Wording alone never varies a state-only decision, and a reworded duplicate is no variant."""
    project, wire, before = proof.project, proof.wire, proof.calls()
    reworded = tier_scenario("urgent-reworded", "high").model_copy(
        update={"instruction": "Please resolve this support ticket with the tools available."}
    )
    spec = proof.family_spec.model_copy(update={"confirmation_seed": 2})
    wire.queue(
        (DECISION_STATE_TOOL, decision_answer(0)), (SCENARIO_TOOL, scenario_answer(reworded))
    )
    family = run_family(project, spec, proof.report, proof.exposure)
    assert family.failure is not None and family.failure.startswith("scenarios:")
    assert family.scenarios is not None
    reasons = [p.reason for p in family.scenarios.problems if p.scenario_id == "urgent-reworded"]
    assert any("wording alone cannot vary it" in reason for reason in reasons)
    assert family.records is None and family.variants == ()
    actor, predictor, _, judged = proof.calls()
    assert (actor, predictor, judged) == (before[0], before[1], before[3])
    task = proof.packet.task
    told = "Kindly resolve the support ticket using whichever tools you have."
    decision = tier_decision(0).model_copy(
        update={
            "instruction_axis": InstructionAxis(
                values=(
                    InstructionValue(
                        value_id="original",
                        instruction=task.instruction,
                        kind="original",
                        obligations="Resolve the ticket.",
                    ),
                    InstructionValue(
                        value_id="polite",
                        instruction=told,
                        kind="reworded",
                        obligations="Resolve the ticket.",
                        equivalent_to="original",
                    ),
                ),
                effect="A polite request carries the same obligations.",
            )
        }
    )
    polite = tier_scenario("polite", "high").model_copy(
        update={
            "instruction": told,
            "initialization": "fresh_briefing",
            "briefing": FreshBriefing(text="A customer asks you to look at their support ticket."),
        }
    )
    found = check_scenarios(
        INITIAL,
        decision,
        ScenarioSet(
            scenarios=(tier_scenario("urgent", "high"), tier_scenario("routine", "low"), polite),
            limitations=("Tier variants.",),
        ),
        ScenarioLimits(tools=frozenset(t.name for t in task.tools), remaining_calls=5),
        ActorView(instruction=task.instruction, observed=(task.instruction,)),
    )
    assert [p.scenario_id for p in found] == ["polite"]
    assert "only by an equivalent rewording" in found[0].reason


def test_judging_resumes_after_interruption_and_a_new_profile_invalidates_labels(
    proof: Proof,
) -> None:
    """A stopped judging resumes without redrawing saved assessments; a new profile relabels all."""
    project, wire = proof.project, proof.wire
    row = proof.report.points[0]
    assert row.study is not None and row.judged is not None
    profile = judge_profile(
        model_alias=proof.profile.model_alias,
        model=proof.profile.model,
        maximum_output_tokens=proof.profile.maximum_output_tokens,
        temperature=0.5,
    )
    (judge,) = (flipped_judging(proof, profile, semantic)["urgent"].judge,)
    assert judge.identity != proof.judging["urgent"].judge.identity
    start = len(wire.requests_judged)
    wire.interrupt_at = 2
    with pytest.raises(Interrupted):
        judge_study(
            project, row.study, judge, configuration=proof.job.configuration, code_revision=REVISION
        )
    assert len(wire.requests_judged) == start + 1
    _, judged = judge_study(
        project, row.study, judge, configuration=proof.job.configuration, code_revision=REVISION
    )
    # Resuming draws only the episode never started: the saved assessment is reused, and the
    # request in flight at the stop has an unknown outcome, so it is never redrawn.
    assert len(wire.requests_judged) == start + 2
    assert len({episode_id(r) for r in wire.requests_judged[start:]}) == 2
    assert slot_rewards(project, judged.branches, judge.identity) == (0.0, None, 1.0)
    resumed = slot_coverage(project, judged.branches, judge.identity)
    assert (resumed.planned, resumed.scored, resumed.unavailable) == (3, 2, 1)
    old = judged_study(project, row.judged)
    with pytest.raises(ValueError, match="different judge"):
        slot_rewards(project, judged.branches, old.judge)
    coverage = slot_coverage(project, old.branches, old.judge)
    assert (coverage.planned, coverage.scored) == (3, 3)
