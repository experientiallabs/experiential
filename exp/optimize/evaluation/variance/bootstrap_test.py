"""Offline tests for the start bootstrap over a real grounded world and scripted role draws.

The family's rules are learned by the real rules stage, every bootstrap role runs the real
agentic loop, starts are captured by the real capture owner in the grounded ticket world of
:mod:`~exp.optimize.evaluation.variance.tests.grounded_world_test`, and the registered job runs
through the real actor-critic stage. Only the wire is scripted.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    sha256_bytes,
)
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase
from exp.common.traces import Trace
from exp.optimize.evaluation.continuations.job import preflight_continuation_job
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    RuleMemoryBinding,
)
from exp.optimize.evaluation.rules.research_test import FINAL_TOOLS, ROLE_MARKERS, SMALL
from exp.optimize.evaluation.rules.stage import RulesStageSpec
from exp.optimize.evaluation.variance import bootstrap as bootstrap_module
from exp.optimize.evaluation.variance.actor_critic import run_actor_critic
from exp.optimize.evaluation.variance.actor_critic_test import _exposure, _spec
from exp.optimize.evaluation.variance.bootstrap import (
    freeze_batch,
    frozen_report,
    load_batch_manifest,
    load_outcome,
    registered_job,
    resolve_bootstrap,
    run_bootstrap,
)
from exp.optimize.evaluation.variance.bootstrap_agents import FamilyKnowledge
from exp.optimize.evaluation.variance.bootstrap_capture import start_definition
from exp.optimize.evaluation.variance.bootstrap_contracts import (
    PROVENANCE_STATEMENTS,
    BatchManifest,
    BootstrapBinding,
    BootstrapFamily,
    BootstrapRefusal,
    BootstrapRegistration,
    BootstrapSpec,
    BootstrapTask,
    FamilyInput,
    JobTemplate,
    ManifestEntry,
    RuntimeTemplate,
    StartDraft,
    family_from_simulator,
)
from exp.optimize.evaluation.variance.bootstrap_stage import PLAN_LINE
from exp.optimize.evaluation.variance.evidence_tools_test import _reply
from exp.optimize.evaluation.variance.exposure import ExposurePlan, Role
from exp.optimize.evaluation.variance.judged_test import JudgeRouter, state_judge
from exp.optimize.evaluation.variance.pipeline_spec import (
    DecisionVarianceSpec,
    MiningSettings,
    RulesOutputs,
    StageContext,
)
from exp.optimize.evaluation.variance.pipeline_stages import RULES_STAGE
from exp.optimize.evaluation.variance.pipeline_test import make_spec
from exp.optimize.evaluation.variance.tests.family_end_to_end_test import _roles
from exp.optimize.evaluation.variance.tests.grounded_world_test import (
    TOOLS,
    GroundedFixture,
    grounded_fixture,
    grounded_rules_stage,
)
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    RecordedRequest,
    ScriptedJsonTransport,
)
from exp.simulation.retrieval.scope import CorpusScope
from exp.simulation.rules.contracts_test import call, result, trace
from exp.simulation.rules.model import RuleBook, ToolContract
from exp.simulation.rules.recall import RecallPolicy

MARKERS = {
    "You author the starting state": "start_author",
    "You review the starting state": "start_reviewer",
    "You research one missing fact": "start_researcher",
    "You check the first reveal": "reveal_checker",
    **ROLE_MARKERS,
}
FINALS = {
    **FINAL_TOOLS,
    "start_author": "submit_start",
    "start_reviewer": "submit_review",
    "start_researcher": "submit_finding",
    "reveal_checker": "submit_reveal_check",
}
TASKS = ("obs", "recon", "auth", "gap", "bad")
"""Observed, reconstructed and authored starts, one needing research, one refused."""

RECORDED: JsonObject = {
    "tickets": {"severity": "low", "lock": "none", "status": "open", "outcome": "pending"}
}
HIGH: JsonObject = {
    "tickets": {"severity": "high", "lock": "none", "status": "open", "outcome": "pending"}
}


def _ticket_trace(trace_id: str, severity: str = "high") -> Trace:
    """A recorded read, tier check and escalation."""
    return trace(
        trace_id,
        (
            call("c1", 0, "read_ticket", {}, "k1"),
            result("r1", 1, "read_ticket", {"ticket": "T-1", "status": "open"}, "k1"),
            call("c2", 2, "check_sla", {}, "k2"),
            result("r2", 3, "check_sla", {"severity": severity, "lock": "none"}, "k2"),
            call("c3", 4, "escalate", {}, "k3"),
            result("r3", 5, "escalate", {"status": "escalated"}, "k3"),
            # A reconstructed start leaves at least the last 3 steps for the agent.
            *(
                row
                for index in range(4, 7 if trace_id == "src-recon" else 4)
                for row in (
                    call(f"c{index}", 2 * index - 2, "read_ticket", {}, f"k{index}"),
                    result(
                        f"r{index}",
                        2 * index - 1,
                        "read_ticket",
                        {"ticket": "T-1", "status": "escalated"},
                        f"k{index}",
                    ),
                )
            ),
        ),
        declare=False,
    )


def _task(task_id: str) -> TaskCase:
    """A selected development task of the ticket family."""
    return TaskCase(
        task_id=task_id,
        lineage_group_id=f"l-{task_id}",
        origin="authored",
        partition="fit",
        instruction="Handle the support ticket.",
        tools=TOOLS,
        workload_weight=1,
    )


def _fact(fact_id: str, status: str, refs: list[str], deciding: bool = True) -> JsonObject:
    """One fact about the ticket tier."""
    return {
        "fact_id": fact_id,
        "statement": "The ticket's service tier decides escalation or closing.",
        "status": status,
        "deciding": deciding,
        "state_paths": ["/tickets/severity"],
        "references": refs,
    }


def _draft(task_id: str, opening: JsonObject) -> JsonObject:
    """The author's answer for each fixture task."""
    findings = opening["findings"]
    assert isinstance(findings, list)
    state, provenance, extra = HIGH, "authored", {}
    facts = [_fact("tier", "inferred", ["trace:w1:span:r2"])]
    if task_id == "obs":
        state, provenance = RECORDED, "observed"
        facts = [_fact("tier", "observed", ["recorded-state"])]
    elif task_id == "recon":
        provenance = "reconstructed"
        # The locator form of the trace, as the inventory names it, means the same trace.
        extra = {"replay_trace": "trace:src-recon", "prefix_steps": 2}
        facts = [_fact("tier", "observed", ["trace:src-recon:span:r2"])]
    elif task_id == "gap" and not findings:
        facts = [_fact("tier", "unknown", [])]
    elif task_id == "gap":
        facts = [_fact("tier", "inferred", ["finding:r0-0"])]
    if provenance == "authored":
        extra = {"reveal_probes": [{"name": "check_sla", "arguments": {}}]}
    return {
        "instruction": "Handle the support ticket the way its service tier requires.",
        "tools": [tool.model_dump(mode="json") for tool in TOOLS],
        "manifest": {
            "domains": [{"name": "tickets", "owner": "simulated"}],
            "tools": [
                {
                    "name": tool.name,
                    "reads": ["tickets"],
                    "writes": ["tickets"] if tool.name in ("escalate", "close") else [],
                }
                for tool in TOOLS
            ],
        },
        "initial_state": state,
        "probe_paths": {"tickets": "/tickets"},
        "verifier": {
            "description": "The ticket is handled the way its tier requires.",
            "outcome_schema": {},
            "required_paths": ["/final_state/tickets/outcome"],
            "rules": [
                {"path": "/final_state/tickets/outcome", "operator": "equals", "value": "right"}
            ],
            "preserved_paths": ["/tickets/severity"],
        },
        "provenance": provenance,
        "facts": facts,
        "limitations": [],
        **extra,
    }


def _research_answer(trace_id: str) -> JsonObject:
    """A research answer citing the trajectory's own tier check."""
    return {
        "trace_id": trace_id,
        "rules": [
            {
                "tool": "escalate",
                "kind": "effect",
                "statement": "Escalating hands the ticket to the second line.",
                "evidence": [f"trace:{trace_id}:span:r3"],
                "counter_evidence": [],
                "confidence": 0.7,
            }
        ],
        "questions": [],
        "hypotheses": [],
        "questions_asked": [],
        "limitations": ["Two recorded tickets."],
    }


def answers(role: str, opening: JsonObject) -> JsonObject:
    """Every role's answer for the fixture family."""
    if role == "research":
        trajectory = opening["trajectory"]
        assert isinstance(trajectory, dict)
        return _research_answer(str(trajectory["trace_id"]))
    if role == "weighing":
        rule = opening["rule"]
        assert isinstance(rule, dict) and isinstance(rule["evidence"], list)
        return {
            "rule_id": rule["rule_id"],
            "status": "supported",
            "rationale": "Both recorded escalations show the handover.",
            "supporting": rule["evidence"][:1],
        }
    if role == "condensation":
        return {"summary": "Tickets are escalated or closed by tier.", "entries": []}
    if role == "reveal_checker":
        return {"consistent": True, "summary": "check_sla shows the authored high tier."}
    task = _task_id(opening)
    if role == "start_author":
        return _draft(task, opening)
    if role == "start_researcher":
        return {
            "status": "resolved",
            "answer": "Recorded tier checks show a high tier.",
            "fact_status": "inferred",
            "evidence": ["trace:w1:span:r2"],
        }
    if task == "bad":
        return {
            "verdict": "refuse",
            "summary": "Unsupported.",
            "refusal": "Nothing recorded decides how this ticket should be handled.",
        }
    return {"verdict": "accept", "summary": "Coherent, specified and supports the task."}


def _task_id(opening: JsonObject) -> str:
    """The task of a bootstrap role's opening."""
    task = opening["task"]
    assert isinstance(task, dict)
    return str(task["task_id"])


def _reads(role: str, opening: JsonObject) -> tuple[str, JsonObject]:
    """One read per role."""
    if role == "research":
        return "search_related_steps", {"tool_name": "escalate", "top_k": 2}
    if role == "weighing":
        rule = opening["rule"]
        assert isinstance(rule, dict) and isinstance(rule["evidence"], list)
        return "read_evidence", {"locator": rule["evidence"][0]}
    if role in ("condensation", "reveal_checker"):
        return ("list_rules", {}) if role == "condensation" else ("read_findings", {})
    return "read_trace", {"trace_id": "w1"}


class BootScript(ScriptedJsonTransport):
    """Thread-safe scripted agent for the rules and bootstrap roles: plan, read, draft, check.

    Attributes:
        fail: Requests that raise a provider error instead of answering.
        seen: ``(role, task)`` of every request, in order.
        draws: ``(role, task)`` of every draw's first request, in order.
    """

    def __init__(
        self,
        fail: Callable[[str, JsonObject], bool] | None = None,
        answer: Callable[[str, JsonObject], JsonObject] | None = None,
    ) -> None:
        """Bind the failure rule and the answer rule (default :func:`answers`)."""
        super().__init__()
        self.fail = fail or (lambda role, opening: False)
        self.answer = answer or answers
        self.lock = threading.Lock()
        self.seen: list[tuple[str, str | None]] = []
        self.draws: list[tuple[str, str | None]] = []

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer one request by its role and phase."""
        del timeout_seconds
        messages = payload["messages"]
        assert isinstance(messages, list)
        system = str(messages[0]["content"])
        role = next(name for marker, name in MARKERS.items() if system.startswith(marker))
        opening = json.loads(str(messages[1]["content"]))
        task = (
            _task_id(opening) if "task" in opening and isinstance(opening["task"], dict) else None
        )
        turns = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
        with self.lock:
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            self.seen.append((role, task))
            if turns == 0:
                self.draws.append((role, task))
        if self.fail(role, opening):
            raise ProviderTransportError("scripted provider outage")
        plan: JsonObject = {"hypotheses": ["the tier decides"], "planned_reads": ["traces"]}
        script: list[tuple[str, JsonObject]] = [
            ("record_plan", plan),
            _reads(role, opening),
            ("submit_draft", self.answer(role, opening)),
            ("record_self_check", {"issues": [], "revise": False}),
            (FINALS[role], self.answer(role, opening)),
        ]
        return _reply([script[min(turns, len(script) - 1)]])

    def tasks(self, role: str) -> list[str | None]:
        """Tasks of every draw of one role."""
        return [task for name, task in self.draws if name == role]


@dataclass
class Setup:
    """A grounded project, its traces, exposure and bootstrap registration."""

    fixture: GroundedFixture
    spec: BootstrapSpec
    exposure: ExposurePlan
    traces: dict[str, Trace]

    @property
    def project(self) -> ProjectStore:
        """The project."""
        return self.fixture.project

    def run(
        self,
        script: BootScript,
        monkeypatch: pytest.MonkeyPatch,
        tasks: Collection[str] | None = None,
    ) -> BatchManifest:
        """Run the bootstrap with the author wire routed to ``script``."""
        monkeypatch.setattr(
            "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: script
        )
        return run_bootstrap(
            self.project,
            self.spec,
            exposure=self.exposure,
            traces=self.traces,
            code_revision="test",
            tasks=tasks,
        )


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Setup:
    """Freeze a ticket family of five selected tasks over the grounded world."""
    fixture = grounded_fixture(tmp_path, monkeypatch, priced=True)
    project, settings = fixture.project, fixture.settings
    traces = {name: _ticket_trace(name) for name in ("w1", "w2", "src-recon")}
    roles: dict[str, Role] = {"l-w1": "world_learning", "l-w2": "world_learning"}
    roles.update(dict.fromkeys((f"l-{task}" for task in TASKS), "mining_probes"))
    exposure = ExposurePlan(
        bindings=ArtifactInput(artifact_id="bindings", sha256="0" * 64),
        seed=0,
        probe_fraction=0.5,
        roles=roles,
        traces={"w1": "l-w1", "w2": "l-w2", "src-recon": "l-recon"},
    )
    recorded = project.artifacts.write_json(
        artifact_id="recorded-obs",
        artifact_type="recorded-environment-state",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={"state.json": RECORDED},
    )
    runtime_file = tmp_path / "bootstrap_runtime.py"
    runtime_file.write_text(
        "from exp.optimize.evaluation.executable.grounded import continuation_runtime\n"
    )
    definition = fixture.definition
    tasks = [
        BootstrapTask(task=_task("obs"), recorded_state=artifact_input(recorded)),
        BootstrapTask(task=_task("recon"), source_traces=("src-recon",)),
        *(BootstrapTask(task=_task(name)) for name in ("auth", "gap", "bad")),
    ]
    spec = BootstrapSpec(
        families=(
            BootstrapFamily(
                family=FamilyInput(
                    family_id="tickets",
                    member_traces=("w1", "w2"),
                    state_domains=("tickets",),
                ),
                tasks=tuple(tasks),
            ),
        ),
        rules=RulesStageSpec(
            author=settings,
            concurrency=2,
            profile="agentic_research",
            research=SMALL,
            weighing=SMALL,
            condensation=SMALL,
        ),
        author=settings,
        author_profile=SMALL,
        review_profile=SMALL,
        research_profile=SMALL,
        research_rounds=1,
        runtime=RuntimeTemplate(
            runtime_file=runtime_file,
            runtime_sha256=sha256_bytes(runtime_file.read_bytes()),
            world_model=definition.world_model,
            catalog_sha256=definition.catalog_sha256,
            maximum_physical_attempts=20_000,
            deadline=definition.deadline,
        ),
        job=JobTemplate(
            actor_alias="actor",
            maximum_model_calls=6,
            maximum_physical_attempts=40,
            deadline=definition.deadline,
            repeats=2,
        ),
    )
    return Setup(fixture, spec, exposure, traces)


def _row(manifest: BatchManifest, task: str) -> ManifestEntry:
    """One task's manifest row."""
    entry = manifest.entry(task)
    assert entry is not None
    return entry


def _registration(setup: Setup, manifest: BatchManifest, task: str) -> BootstrapRegistration:
    """The saved registration of one task."""
    entry = manifest.entry(task)
    assert entry is not None
    outcome = load_outcome(setup.project, entry.identity)
    assert isinstance(outcome, BootstrapRegistration)
    return outcome


def test_a_family_bootstraps_into_validated_registrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each provenance is captured and validated its own way; only observed claims the original."""
    setup = _setup(tmp_path, monkeypatch)
    script = BootScript()
    manifest = setup.run(script, monkeypatch)

    assert manifest.by_status("registered") == ("obs", "recon", "auth", "gap")
    assert manifest.by_status("refused") == ("bad",)
    assert {"research", "weighing", "condensation"} <= {role for role, _ in script.seen}
    obs, recon, auth = (_registration(setup, manifest, task) for task in ("obs", "recon", "auth"))
    assert (obs.provenance, obs.validation.method, obs.claims_original) == (
        "observed",
        "exact_restore",
        True,
    )
    assert (recon.provenance, recon.validation.method, recon.claims_original) == (
        "reconstructed",
        "prefix_replay",
        False,
    )
    assert [step.agrees for step in recon.validation.replay] == [True, True]
    assert [step.rule for step in recon.validation.replay] == ["exact", "exact"]
    assert recon.validation.replay[1].replayed == {"severity": "high", "lock": "none"}
    assert (auth.provenance, auth.validation.method, auth.claims_original) == (
        "authored",
        "coherence_and_first_reveal",
        False,
    )
    assert auth.statement == PROVENANCE_STATEMENTS["authored"]
    assert "never claims to be" in auth.statement
    assert auth.validation.reveals[0].content == {"severity": "high", "lock": "none"}
    assert len(auth.validation.checks) == len(setup.spec.reveal_variants)
    assert all(row.passed for row in (obs.validation, recon.validation, auth.validation))
    assert len(auth.rounds[0].reviews) == len(setup.spec.review_variants)
    assert len({draw.variant for draw in auth.rounds[0].reviews}) == 3
    for registration in (obs, recon, auth):
        preflight_continuation_job(setup.project, registration.job)
        # Qualification needs an explicit tool list: the start's own, none invented or dropped.
        assert registration.draft.verifier.allowed_tools == tuple(tool.name for tool in TOOLS)
        assert registration.job.study.verifier.allowed_tools == tuple(tool.name for tool in TOOLS)
    assert load_batch_manifest(setup.project, manifest.batch) == manifest

    replayed = setup.run(BootScript(), monkeypatch)
    assert [row.status for row in replayed.entries] == [row.status for row in manifest.entries]
    assert [row.attempts for row in replayed.entries] == [1] * 5


def test_a_missing_deciding_fact_becomes_research_and_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown deciding fact asks the research step, and the next draft cites its finding."""
    setup = _setup(tmp_path, monkeypatch)
    script = BootScript()
    gap = _registration(setup, setup.run(script, monkeypatch, tasks={"gap"}), "gap")

    assert [row.verdict for row in gap.rounds] == ["research", "accept"]
    assert gap.rounds[0].items[0].fact_id == "tier"
    assert [(row.finding_id, row.status) for row in gap.findings] == [("r0-0", "resolved")]
    assert gap.findings[0].evidence == ("trace:w1:span:r2",)
    assert gap.draft.facts[0].references == ("finding:r0-0",)
    assert script.tasks("start_researcher") == ["gap"]
    assert script.tasks("start_author") == ["gap", "gap"]


def _broken_author(role: str, opening: JsonObject) -> JsonObject:
    """The live failure shape: domains as a map and no tool policies, repaired once shown."""
    answer = answers(role, opening)
    if role != "start_author" or "validation_errors" in opening:
        return answer
    return {**answer, "manifest": {"domains": {"tickets": "simulated"}}}


def test_a_draft_failing_validation_is_repaired_and_the_task_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A schema-broken draft gets a repair draw shown its errors; the repaired start is captured."""
    setup = _setup(tmp_path, monkeypatch)
    script = BootScript(answer=_broken_author)
    auth = _registration(setup, setup.run(script, monkeypatch, tasks={"auth"}), "auth")

    author = auth.rounds[0].author
    assert author.failure is None and author.invalid is not None
    assert "manifest.domains" in author.invalid and "manifest.tools" in author.invalid
    assert [(row.round, row.failure) for row in author.repairs] == [(0, None)]
    assert author.answer == author.repairs[0].answer
    assert auth.draft.manifest.domains[0].name == "tickets"
    assert auth.provenance == "authored"
    assert script.tasks("start_author") == ["auth", "auth"]
    repair = next(
        json.loads(str(request.payload["messages"][1]["content"]))  # type: ignore[index]
        for request in script.requests
        if "validation_errors" in str(request.payload["messages"][1]["content"])  # type: ignore[index]
    )
    errors = repair["validation_errors"]
    assert repair["previous_answer"]["manifest"] == {"domains": {"tickets": "simulated"}}
    assert any(row.startswith("manifest.domains") for row in errors)
    assert any(row.startswith("manifest.tools") for row in errors)
    assert len(errors) <= 40 and all(len(row) < 300 for row in errors)
    inputs = bootstrap_module._inputs(auth.rounds, (), None, None, ())
    assert {author.transcript, author.repairs[0].transcript} <= set(inputs)

    replayed = BootScript(answer=_broken_author)
    setup.run(replayed, monkeypatch, tasks={"auth"})
    assert replayed.requests == []


def test_a_draft_valid_first_time_triggers_no_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid first draft is the draft; no repair draw is bought."""
    setup = _setup(tmp_path, monkeypatch)
    script = BootScript()
    auth = _registration(setup, setup.run(script, monkeypatch, tasks={"auth"}), "auth")

    author = auth.rounds[0].author
    assert (author.failure, author.invalid, author.repairs) == (None, None, ())
    assert script.tasks("start_author") == ["auth"]
    assert not any(
        "validation_errors" in str(request.payload["messages"][1]["content"])  # type: ignore[index]
        for request in script.requests
    )


def test_unrepairable_drafts_refuse_after_the_bounded_repairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A draft still broken after every repair is refused as author_unusable, repairs recorded."""
    setup = _setup(tmp_path, monkeypatch)

    def always_broken(role: str, opening: JsonObject) -> JsonObject:
        answer = answers(role, opening)
        return {**answer, "manifest": {"domains": {}}} if role == "start_author" else answer

    script = BootScript(answer=always_broken)
    manifest = setup.run(script, monkeypatch, tasks={"auth"})
    refusal = load_outcome(setup.project, _row(manifest, "auth").identity)
    assert isinstance(refusal, BootstrapRefusal) and refusal.kind == "author_unusable"
    author = refusal.rounds[0].author
    assert len(author.repairs) == setup.spec.author_repairs == 4
    assert "4 repair draw(s) failed" in (author.failure or "")
    assert script.tasks("start_author") == ["auth"] * 5


@pytest.mark.parametrize("trace_id", ["src-recon", "w1"])
@pytest.mark.parametrize("change", ["content", "missing", "held_out"])
def test_replay_rechecks_current_trace_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trace_id: str, change: str
) -> None:
    """Saved starts refuse changed, absent or excluded evidence before any provider call."""
    setup = _setup(tmp_path, monkeypatch)
    first = setup.run(BootScript(), monkeypatch, tasks={"recon", "auth"})
    assert first.by_status("registered") == ("recon", "auth")
    if change == "content":
        setup.traces[trace_id] = _ticket_trace(trace_id, severity="low")
    elif change == "missing":
        del setup.traces[trace_id]
    else:
        lineage = setup.exposure.traces[trace_id]
        setup.exposure = setup.exposure.model_copy(
            update={"roles": setup.exposure.roles | {lineage: "final_evaluation"}}
        )
    silent = BootScript(lambda role, opening: True)
    replay = setup.run(silent, monkeypatch, tasks={"recon", "auth"})
    assert _row(replay, "recon").status == "failed"
    assert _row(replay, "recon").outcome is None
    assert _row(replay, "auth").status == ("registered" if trace_id == "src-recon" else "failed")
    assert silent.requests == []


def test_replay_ignores_unrelated_traces_and_keeps_exact_saved_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evidence outside a task's family and sources never invalidates its saved outcome."""
    setup = _setup(tmp_path, monkeypatch)
    first = setup.run(BootScript(), monkeypatch, tasks={"auth", "bad"})
    setup.traces["src-recon"] = _ticket_trace("src-recon", severity="low")
    silent = BootScript(lambda role, opening: True)
    replay = setup.run(silent, monkeypatch, tasks={"auth", "bad"})
    assert [(row.identity, row.outcome, row.attempts) for row in replay.entries] == [
        (row.identity, row.outcome, row.attempts) for row in first.entries
    ]
    assert silent.requests == []


def test_replay_refuses_a_saved_outcome_without_its_evidence_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved outcome cannot adopt caller trace bytes after its validation was recorded."""
    setup = _setup(tmp_path, monkeypatch)
    first = setup.run(BootScript(), monkeypatch, tasks={"recon"})
    evidence_id = _row(first, "recon").identity + "-evidence"
    store = setup.project.artifacts
    exists = store.exists

    def without_binding(artifact_id: str) -> bool:
        """Represent a persisted outcome that has no trace binding."""
        return artifact_id != evidence_id and exists(artifact_id)

    monkeypatch.setattr(store, "exists", without_binding)
    silent = BootScript(lambda role, opening: True)
    replay = setup.run(silent, monkeypatch, tasks={"recon"})
    assert _row(replay, "recon").status == "failed"
    assert "no trace evidence binding" in (_row(replay, "recon").failure or "")
    assert silent.requests == []


def test_invalid_task_evidence_is_checked_before_family_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task with absent source evidence spends nothing learning an unfrozen family."""
    setup = _setup(tmp_path, monkeypatch)
    del setup.traces["src-recon"]
    silent = BootScript(lambda role, opening: True)
    manifest = setup.run(silent, monkeypatch, tasks={"recon"})
    assert manifest.by_status("failed") == ("recon",)
    assert "were not supplied" in (_row(manifest, "recon").failure or "")
    assert silent.requests == []


def test_unresolved_research_and_refusing_reviews_end_in_diagnostic_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusing review and research beyond the round bound each save a diagnosis."""
    setup = _setup(tmp_path, monkeypatch)
    setup.spec = setup.spec.model_copy(update={"research_rounds": 0})
    manifest = setup.run(BootScript(), monkeypatch, tasks={"gap", "bad"})

    bad = load_outcome(setup.project, _row(manifest, "bad").identity)
    gap = load_outcome(setup.project, _row(manifest, "gap").identity)
    assert isinstance(bad, BootstrapRefusal) and bad.kind == "review_refused"
    assert "Nothing recorded decides" in bad.diagnosis
    assert isinstance(gap, BootstrapRefusal) and gap.kind == "research_exhausted"
    assert [item.fact_id for item in gap.unresolved] == ["tier"]
    assert manifest.by_status("refused") == ("gap", "bad")


def test_the_batch_resumes_after_an_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selected tasks run first; an interrupted capture fails alone and resumes on the rerun."""
    setup = _setup(tmp_path, monkeypatch)
    first = setup.run(BootScript(), monkeypatch, tasks={"obs"})
    assert [row.task_id for row in first.entries] == ["obs"]

    real = bootstrap_module.capture_start
    interrupted = {"recon": 1}

    def flaky(project, spec, template, definition, sources, *, code_revision):  # noqa: ANN001, ANN202
        if interrupted.get(definition.task.task_id):
            interrupted[definition.task.task_id] -= 1
            raise RuntimeError("process interrupted during capture")
        return real(project, spec, template, definition, sources, code_revision=code_revision)

    monkeypatch.setattr(bootstrap_module, "capture_start", flaky)
    script = BootScript()
    second = setup.run(script, monkeypatch)
    assert second.by_status("failed") == ("recon",)
    assert "interrupted" in (_row(second, "recon").failure or "")
    assert "obs" not in script.tasks("start_author")
    assert _row(second, "obs").attempts == 1

    resume = BootScript()
    third = setup.run(resume, monkeypatch)
    assert third.by_status("registered") == ("obs", "recon", "auth", "gap")
    assert _row(third, "recon").attempts == 2
    assert resume.seen == []


def test_one_failing_task_never_blocks_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider outage on one task's author is that task's failure only."""
    setup = _setup(tmp_path, monkeypatch)
    outage = BootScript(
        lambda role, opening: role == "start_author" and _task_id(opening) == "auth"
    )
    manifest = setup.run(outage, monkeypatch)

    assert manifest.by_status("failed") == ("auth",)
    assert "ProviderTransportError" in (_row(manifest, "auth").failure or "")
    assert manifest.by_status("registered") == ("obs", "recon", "gap")
    assert manifest.by_status("refused") == ("bad",)


def test_the_registered_job_runs_in_the_actor_critic_stage_and_the_pipeline_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frozen batch hands its job to the pipeline entry and the actor-critic stage runs it."""
    setup = _setup(tmp_path, monkeypatch)
    manifest = setup.run(BootScript(), monkeypatch, tasks={"recon"})
    frozen = freeze_batch(setup.project, manifest)
    assert freeze_batch(setup.project, manifest) == frozen
    job = registered_job(setup.project, BootstrapBinding(batch=frozen, task_id="recon"))
    assert job == _registration(setup, manifest, "recon").job
    with pytest.raises(ValueError, match="no registered start"):
        registered_job(setup.project, BootstrapBinding(batch=frozen, task_id="bad"))

    project = setup.project
    roles = JudgeRouter(_roles(held=False), state_judge(("tickets", "outcome"), "right"))
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: roles
    )
    exposure = _exposure(project, job, "mining_probes")
    report = run_actor_critic(project, _spec(project, job), exposure=exposure, code_revision="test")
    assert all(row.failure is None for row in report.points)


def test_a_frozen_family_reaches_the_pipeline_with_no_rules_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frozen family rules are reused by the bootstrap and the pipeline rules stage; none run."""
    setup = _setup(tmp_path, monkeypatch)
    project = setup.project
    report = grounded_rules_stage(project, setup.fixture.settings)
    rulebook = frozen_report(project, report).rulebook
    family = setup.spec.families[0].family.model_copy(
        update={"rule_book": rulebook, "rules_report": report}
    )
    setup.spec = BootstrapSpec.model_validate(
        {
            **setup.spec.model_dump(mode="json"),
            "rules": None,
            "families": [
                {
                    "family": family.model_dump(mode="json"),
                    "tasks": [row.model_dump(mode="json") for row in setup.spec.families[0].tasks],
                }
            ],
        }
    )
    script = BootScript()
    manifest = setup.run(script, monkeypatch, tasks={"recon"})
    recon = _registration(setup, manifest, "recon")
    assert not {"research", "weighing", "condensation"} & {role for role, _ in script.seen}
    assert (recon.rules_report, recon.rulebook) == (report, rulebook)
    assert recon.job.configuration["rule_book"] == rulebook.model_dump(mode="json")

    binding = BootstrapBinding(batch=freeze_batch(project, manifest), task_id="recon")
    registered = make_spec(setup.fixture, None, tmp_path)
    mining = registered.mining.model_copy(update={"job": None, "bootstrap": binding})
    bootstrapped = DecisionVarianceSpec.model_validate(
        {**registered.model_dump(mode="json"), "mining": mining.model_dump(mode="json")}
    )
    with pytest.raises(ValueError, match="unresolved"):
        _ = bootstrapped.mining.registered
    unbound = MiningSettings.model_validate(
        {**mining.model_dump(mode="json"), "bootstrap": None, "job": None}
    )
    with pytest.raises(ValueError, match="register a mining job"):
        resolve_bootstrap(project, bootstrapped.model_copy(update={"mining": unbound}))
    resolved = resolve_bootstrap(project, bootstrapped)
    assert resolved.mining.job == recon.job
    # Mining runs the registered job on a simulator ledger of its own.
    ceiling = resolved.mining.simulator_ceiling(recon.job)
    assert resolved.mining.registered == recon.job.model_copy(
        update={"configuration": {**recon.job.configuration, "maximum_physical_attempts": ceiling}}
    )
    assert RULES_STAGE.plan(project, resolved, "test") == ((), (PLAN_LINE,))
    silent = BootScript(lambda role, opening: True)
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: silent
    )
    run = RULES_STAGE.run(StageContext(project, resolved, "rules", "test"))
    outputs = RulesOutputs.model_validate(run.outputs.model_dump(mode="json"))
    assert (outputs.report, outputs.rulebook, outputs.job) == (
        report,
        rulebook,
        resolved.mining.registered,
    )
    assert outputs.challenges == 0 and silent.seen == []


@dataclass(frozen=True)
class _Reviewed:
    """A reviewed, descriptive state domain as the families owner names it."""

    name: str
    description: str


@dataclass(frozen=True)
class _Simulator:
    """The FamilySimulator fields the adapter reads, plus one it must ignore."""

    family_id: str
    members: tuple[str, ...]
    rules_traces: tuple[str, ...]
    tool_contracts: dict[str, ToolContract]
    state_domains: tuple[_Reviewed, ...]
    corpus: CorpusScope | None
    rule_book: ArtifactInput | None
    rule_memory: RuleMemoryBinding | None
    rules_report: ArtifactInput | None
    runtime: GroundedRuntimeDefinition | None
    world_semantics: str = "ignored"


def test_the_family_adapter_keeps_the_simulator_frozen_knowledge_exactly() -> None:
    """Members, rule traces, domain names, rule book and rule memory cross unchanged."""
    book = ArtifactInput(artifact_id="family-rule-book", sha256="1" * 64)
    memory = RuleMemoryBinding(
        memory_set=ArtifactInput(artifact_id="family-memory", sha256="2" * 64),
        recall=RecallPolicy(jev_alias="jev"),
    )
    simulator = _Simulator(
        "tickets",
        ("w1", "w2"),
        ("w1",),
        {"escalate": ToolContract(name="escalate")},
        (_Reviewed("tickets", "Support tickets"),),
        None,
        book,
        memory,
        None,
        None,
    )
    family = family_from_simulator(simulator)
    assert family.member_traces == ("w1", "w2") and family.learning_traces == ("w1",)
    assert family.state_domains == ("tickets",)
    assert (family.rule_book, family.rule_memory, family.rules_report) == (book, memory, None)
    assert family.frozen and family.tool_contracts == simulator.tool_contracts
    with pytest.raises(ValueError, match="frozen rule book"):
        family_from_simulator(replace(simulator, rule_book=None))


def test_a_frozen_family_runtime_is_kept_and_only_the_start_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """World model, limits and rules stay the family's; the task, state and projection change."""
    setup = _setup(tmp_path, monkeypatch)
    frozen = setup.fixture.definition.model_copy(update={"maximum_reasks": 2})
    family = FamilyInput(family_id="tickets", member_traces=("w1",), runtime=frozen)
    draft = StartDraft.model_validate(_draft("recon", {"findings": []}))
    knowledge = FamilyKnowledge(
        family_id="tickets",
        report=None,
        rulebook=ArtifactInput(artifact_id="unused", sha256="0" * 64),
        book=RuleBook(contracts={}),
        rule_memory=None,
        traces={},
        contracts={},
    )
    task = BootstrapTask(task=_task("recon"))
    definition = start_definition(setup.spec.runtime, family, knowledge, task, draft)
    assert definition.model_dump(
        exclude={"task", "manifest", "initial_state", "probe_paths"}
    ) == frozen.model_dump(exclude={"task", "manifest", "initial_state", "probe_paths", "facts"})
    assert (definition.task.task_id, definition.initial_state) == ("recon", HIGH)
    assert definition.facts is None and definition.maximum_reasks == 2


def test_an_admitted_verifier_meets_the_qualification_shape_preconditions() -> None:
    """Qualification refuses a verifier without explicit tools or a state-retention check, so
    authoring refuses it first instead of after mining (executable/qualification.py)."""
    from exp.optimize.evaluation.variance.bootstrap_agents import (
        check_qualification_preconditions,
    )
    from exp.simulation.evaluation.tasks import TaskVerifier

    base: JsonObject = {
        "description": "d",
        "outcome_schema": {},
        "rules": [{"path": "/final_state/tickets/outcome", "operator": "equals", "value": "x"}],
    }
    with pytest.raises(ValueError, match="no permitted tools"):
        check_qualification_preconditions(
            TaskVerifier.model_validate({**base, "preserved_paths": ["/tickets/severity"]})
        )
    with pytest.raises(ValueError, match="preserved_paths"):
        check_qualification_preconditions(
            TaskVerifier.model_validate({**base, "allowed_tools": ["close"]})
        )
    check_qualification_preconditions(
        TaskVerifier.model_validate(
            {**base, "allowed_tools": ["close"], "preserved_paths": ["/tickets/severity"]}
        )
    )
