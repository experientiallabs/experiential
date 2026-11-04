"""A real grounded world behind a continuation job, driven offline by scripted transports.

The fixture builds a trace-grounded world model artifact, binds it through the built-in executable
factory (:func:`~exp.optimize.evaluation.executable.grounded.continuation_runtime` over a
:class:`~exp.optimize.evaluation.executable.grounded.GroundedRuntimeDefinition`), and freezes a
:class:`~exp.optimize.evaluation.continuations.job.ContinuationJob` whose runtime file selects that
factory. Every provider call goes through the admitted HTTP transports; only the wire is scripted:

- :class:`TicketPredictor` answers each grounded prediction request from the call and the
  simulated state the request delivers, so restored forks predict the same way.
- :class:`TicketActor` answers each actor request from the conversation: call the first tool,
  then escalate, then finish.

The world is one support ticket whose tier (``/tickets/severity``) decides the right handling:
escalating a ``high`` ticket or closing a ``low`` one resolves it, and the verifier checks
``/final_state/tickets/outcome``. ``read_ticket`` shows only the handling status, ``check_sla``
shows the tier and the ticket lock, and a held lock makes ``check_sla`` fail.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    JsonValue,
    sha256_bytes,
    sha256_json,
)
from exp.common.models import (
    BillingSource,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ToolCall,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase, ToolSchema
from exp.optimize.evaluation.continuations.job import ContinuationJob, execute_continuation_job
from exp.optimize.evaluation.continuations.study import ContinuationStudySpec
from exp.optimize.evaluation.executable import grounded as grounded_module
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    continuation_runtime,
)
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.optimize.evaluation.rules.stage import (
    REPORT_TYPE,
    RULE_BOOK_TYPE,
    RulesStageReport,
    RulesStageSpec,
    rulebook_files,
)
from exp.optimize.evaluation.variance.contracts import (
    Criterion,
    DecisionOption,
    DecisionState,
    Scenario,
    StateEdit,
    VariationAxis,
    WitnessStep,
)
from exp.optimize.evaluation.variance.contrast import run_observation_witness, run_option_contrast
from exp.optimize.evaluation.variance.judged_test import (
    episode_judge,
    route_judge,
    run_digests,
    state_judge,
)
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest
from exp.runtime.models.providers.transport import JsonHttpResponse, JsonHttpTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import ContinuationCheckpoint, persist_continuation_checkpoint
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier
from exp.simulation.rules.model import Rule, RuleBook, ToolContract

TOOLS = (
    ToolSchema(
        name="read_ticket",
        description="Read the ticket's handling status.",
        input_schema={"type": "object"},
    ),
    ToolSchema(
        name="check_sla",
        description="Read the ticket's service tier and whether another agent holds it.",
        input_schema={"type": "object"},
    ),
    ToolSchema(
        name="escalate",
        description="Hand the ticket to the second line.",
        input_schema={"type": "object"},
    ),
    ToolSchema(
        name="close",
        description="Close the ticket as handled.",
        input_schema={"type": "object"},
    ),
)
"""The tools the actor may call."""

INITIAL: JsonObject = {
    "tickets": {"severity": "high", "lock": "none", "status": "open", "outcome": "pending"}
}
"""Simulated state the world starts in."""

ACTIONS: dict[str, ToolCall] = {
    "escalate": ToolCall(name="escalate", call_id="force-escalate", arguments={}),
    "close": ToolCall(name="close", call_id="force-close", arguments={}),
}
"""Concrete forced first action per decision option."""

RULE = Rule(
    rule_id="tickets.tier",
    tool="escalate",
    kind="effect",
    statement="Escalating resolves a high tier ticket; closing resolves a low tier one.",
    status="supported",
    support=2,
    contradiction=0,
)
"""The learned rule the decision state is grounded in."""

_RIGHT = {"escalate": "high", "close": "low"}


def _completion(model: str, message: JsonObject, finish: str) -> JsonHttpResponse:
    """Return one Chat Completions wire response.

    Args:
        model: Served model identity.
        message: The assistant message.
        finish: Finish reason.

    Returns:
        The response.
    """
    return JsonHttpResponse(
        status_code=200,
        body={
            "id": "reply",
            "model": model,
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        },
    )


def _tool_message(
    model: str, name: str, call_id: str, arguments: JsonObject, content: str | None = None
) -> JsonHttpResponse:
    """Return one wire response making a single tool call.

    Args:
        model: Served model identity.
        name: Tool name.
        call_id: Call identity.
        arguments: Call arguments.
        content: Assistant text sent with the call, if any.

    Returns:
        The response.
    """
    return _completion(
        model,
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "type": "function",
                    "id": call_id,
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        "tool_calls",
    )


@dataclass
class TicketPredictor(JsonHttpTransport):
    """Scripted grounded predictor: the ticket world's dynamics, computed from each request.

    Attributes:
        requests: Every prediction request served.
    """

    requests: list[JsonObject] = field(default_factory=list)

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> JsonHttpResponse:
        """Refuse metadata reads, which the predictor never needs.

        Args:
            url: Endpoint.
            headers: Headers.
            timeout_seconds: Timeout.

        Raises:
            AssertionError: Always.
        """
        del url, headers, timeout_seconds
        raise AssertionError("the grounded predictor made a metadata request")

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Predict one tool call's reply and state effect from the delivered state.

        Args:
            url: Endpoint.
            headers: Headers.
            payload: The prediction request.
            timeout_seconds: Timeout.

        Returns:
            One ``submit_transition`` proposal.
        """
        del url, headers, timeout_seconds
        self.requests.append(payload)
        messages = payload["messages"]
        assert isinstance(messages, list)
        content = next(
            str(row["content"])
            for row in reversed(messages)
            if isinstance(row, dict) and row.get("role") == "user"
        )
        delivered = json.loads(content)
        call = delivered["candidate_response"]["tool_calls"][0]
        ticket = delivered["environment_state"]["domains"]["tickets"]
        reply, edits, error = _transition(call["name"], ticket)
        proposal = {
            "observations": [
                {"call_id": call["call_id"], "reply": reply, "encoding": "json", "is_error": error}
            ],
            "effect": {"kind": "edits", "edits": edits},
        }
        return _tool_message("fixture-v1", "submit_transition", "proposal", proposal)


def _transition(name: str, ticket: JsonObject) -> tuple[JsonValue, list[JsonValue], bool]:
    """Return the reply, state edits and error flag of one call on the ticket.

    Args:
        name: Tool name.
        ticket: The ticket domain before the call.

    Returns:
        ``(reply, edits, is_error)``.
    """
    if name == "read_ticket":
        return {"ticket": "T-1", "status": ticket["status"]}, [], False
    if name == "check_sla":
        if ticket["lock"] == "held":
            return {"error": "ticket is held by another agent"}, [], True
        return {"severity": ticket["severity"], "lock": ticket["lock"]}, [], False
    status = "escalated" if name == "escalate" else "closed"
    outcome = "right" if ticket["severity"] == _RIGHT[name] else "wrong"
    edits: list[JsonValue] = [
        {"op": "set", "path": ["domains", "tickets", "status"], "value": status},
        {"op": "set", "path": ["domains", "tickets", "outcome"], "value": outcome},
    ]
    return {"status": status}, edits, False


@dataclass
class TicketActor(JsonHttpTransport):
    """Scripted actor policy: call ``first``, then escalate, and finish once the ticket is handled.

    Attributes:
        first: The first tool the actor calls.
        narrate: Whether the escalation message restates the tier the actor last saw.
        requests: Every actor request served.
    """

    first: str = "read_ticket"
    narrate: bool = False
    requests: list[JsonObject] = field(default_factory=list)

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> JsonHttpResponse:
        """Refuse metadata reads, which the actor never needs.

        Args:
            url: Endpoint.
            headers: Headers.
            timeout_seconds: Timeout.

        Raises:
            AssertionError: Always.
        """
        del url, headers, timeout_seconds
        raise AssertionError("the actor made a metadata request")

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer from the tool replies the conversation already holds.

        Args:
            url: Endpoint.
            headers: Headers.
            payload: The actor request.
            timeout_seconds: Timeout.

        Returns:
            The next action of the policy.
        """
        del url, headers, timeout_seconds
        self.requests.append(payload)
        messages = payload["messages"]
        assert isinstance(messages, list)
        replies = [row for row in messages if isinstance(row, dict) and row.get("role") == "tool"]
        seen = [json.loads(str(row["content"])) for row in replies]
        if any(row.get("status") in ("escalated", "closed") for row in seen):
            return _completion("actor-v1", {"role": "assistant", "content": "done"}, "stop")
        if not replies:
            return _tool_message("actor-v1", self.first, "actor-0", {})
        tier = next((row["severity"] for row in seen if "severity" in row), None)
        said = f"The tier is {tier}, so I escalate." if self.narrate else None
        return _tool_message("actor-v1", "escalate", f"actor-{len(replies)}", {}, said)


@dataclass
class GroundedFixture:
    """A project with a grounded world, its frozen job and the scripted transports.

    Attributes:
        project: The project.
        job: The frozen job starting at the world's initial checkpoint.
        definition: The grounded runtime definition the job's runtime file loads.
        actor: The actor transport.
        predictor: The predictor transport.
        settings: Frozen role settings for judges, critics and authors.
    """

    project: ProjectStore
    job: ContinuationJob
    definition: GroundedRuntimeDefinition
    actor: TicketActor
    predictor: TicketPredictor
    settings: AuthorSettings


def grounded_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    first: str = "read_ticket",
    narrate: bool = False,
    priced: bool = False,
) -> GroundedFixture:
    """Freeze a continuation job over a real grounded world, wired to scripted transports.

    Args:
        tmp_path: Project root.
        monkeypatch: Installs the scripted transports.
        first: The first tool the scripted actor calls.
        narrate: Whether the actor restates the tier it saw when escalating.
        priced: Whether the judge alias has per-million token rates, which agentic role budgets
            need to verify their dollar allowance.

    Returns:
        The fixture.
    """
    project, fidelity = source_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["model"]
    roles = {
        alias: ModelRecord(
            connection=record.connection,
            model=f"{alias}-v1",
            billing_source=BillingSource.CUSTOMER_MANAGED,
            capabilities=ModelCapabilities(
                supports_tools=True,
                supports_structured_output=True,
                context_window_tokens=200_000,
                maximum_output_tokens=8_192,
                input_cost_per_million_tokens_usd=1.0 if priced and alias == "judge" else None,
                output_cost_per_million_tokens_usd=2.0 if priced and alias == "judge" else None,
            ),
        )
        for alias in ("actor", "judge")
    }
    catalog = ModelCatalog(connections=catalog.connections, models={**catalog.models, **roles})
    write_model_catalog(project.model_catalog_path, catalog)
    catalog_sha256 = sha256_json(catalog.model_dump(mode="json"))
    actor, predictor = TicketActor(first=first, narrate=narrate), TicketPredictor()
    monkeypatch.setattr(grounded_module, "HttpxJsonTransport", lambda: predictor)
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actor
    )
    task = TaskCase(
        task_id="ticket",
        lineage_group_id="tickets",
        origin="authored",
        partition="fit",
        instruction="Resolve the support ticket with the tools you have.",
        tools=TOOLS,
        workload_weight=1,
    )
    deadline = datetime.now(UTC) + timedelta(hours=1)
    definition = GroundedRuntimeDefinition(
        task=task,
        world_model=fidelity.world_model,
        catalog_sha256=catalog_sha256,
        manifest=WorldManifest(
            domains=(StateDomain(name="tickets", owner="simulated"),),
            tools=tuple(
                ToolEffectPolicy(
                    name=tool.name,
                    reads=("tickets",),
                    writes=("tickets",) if tool.name in _RIGHT else (),
                )
                for tool in TOOLS
            ),
        ),
        initial_state=INITIAL,
        probe_paths={"tickets": "/tickets"},
        maximum_physical_attempts=20_000,
        deadline=deadline,
    )
    source = project.artifacts.write_json(
        artifact_id="ticket-source",
        artifact_type="authored-task",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={"task.json": {"origin": "authored grounded ticket fixture"}},
    )
    policy = ChatAgentRuntime(maximum_model_calls=6)
    with (
        continuation_runtime(project, definition.model_dump(mode="json")) as runtime,
        runtime.world.open(task) as session,
    ):
        start = ContinuationCheckpoint(
            task=task,
            actor=policy.initial_checkpoint(task),
            world=session.checkpoint(),
            sources=(artifact_input(source),),
        )
    checkpoint = persist_continuation_checkpoint(
        project.artifacts, start, created_at=datetime.now(UTC), code_revision="test"
    )
    runtime_file = tmp_path / "grounded_runtime.py"
    runtime_file.write_text(
        "from exp.optimize.evaluation.executable.grounded import continuation_runtime\n"
    )
    registry = RuntimeModelCatalog(catalog, environment={})
    job = ContinuationJob(
        study=ContinuationStudySpec(
            checkpoint=checkpoint,
            actor_model=registry.snapshot("actor")[0],
            verifier=TaskVerifier(
                description="The ticket is handled the way its tier requires.",
                outcome_schema={},
                required_paths=("/final_state/tickets/outcome",),
                rules=(
                    OutcomeRule(
                        path="/final_state/tickets/outcome", operator="equals", value="right"
                    ),
                ),
                allowed_tools=tuple(tool.name for tool in TOOLS),
                preserved_paths=("/tickets/severity", "/tickets/lock"),
            ),
            probe_sha256=definition.probe_sha256,
            repeats=2,
            selector="grounded ticket fixture; the world's initial checkpoint",
        ),
        runtime_file=runtime_file,
        runtime_sha256=sha256_bytes(runtime_file.read_bytes()),
        configuration=definition.model_dump(mode="json"),
        actor_alias="actor",
        catalog_sha256=catalog_sha256,
        maximum_model_calls=6,
        maximum_physical_attempts=40,
        deadline=deadline,
    )
    settings = AuthorSettings(
        model_alias="judge",
        model=registry.snapshot("judge")[0],
        catalog_sha256=catalog_sha256,
        output_tokens=1024,
        deadline=deadline,
    )
    return GroundedFixture(project, job, definition, actor, predictor, settings)


def grounded_rules_stage(project: ProjectStore, settings: AuthorSettings) -> ArtifactInput:
    """Save a rule book covering every ticket tool and a stage report naming it.

    Args:
        project: Project receiving the artifacts.
        settings: Author settings recorded on the stage spec.

    Returns:
        The ``rules-stage-report`` artifact.
    """
    book = RuleBook(
        contracts={tool.name: ToolContract(name=tool.name) for tool in TOOLS}, rules=(RULE,)
    )
    envelope = ArtifactEnvelope(
        schema_version=1, created_at=datetime.now(UTC), code_revision="test"
    )
    rulebook = artifact_input(
        project.artifacts.write(
            artifact_id="ticket-rule-book",
            artifact_type=RULE_BOOK_TYPE,
            envelope=envelope,
            files=rulebook_files(book),
        )
    )
    report = RulesStageReport(
        spec=RulesStageSpec(author=settings),
        identity="ticket-rules-stage",
        extraction_identity="ticket-rules-extraction",
        contracts=dict(book.contracts),
        draws=(),
        failures=(),
        rulebook=rulebook,
        rules=1,
        rule_counts={"supported": 1, "contradicted": 0, "unknown": 0},
        open_questions=0,
        diagnostic=(),
        requires_witness=(),
        sessions=(),
        session_ids=(),
        properties=None,
        workloads=(),
        triage=(),
        families_must_stay_diagnostic=(),
    )
    return artifact_input(
        project.artifacts.write_json(
            artifact_id="ticket-rules-stage-result",
            artifact_type=REPORT_TYPE,
            envelope=envelope.model_copy(update={"inputs": (rulebook,)}),
            files={"report.json": report},
        )
    )


def tier_decision(step: int) -> DecisionState:
    """The escalate-or-close decision at one step, varying the ticket tier.

    Args:
        step: Decision boundary.

    Returns:
        The decision state.
    """
    return DecisionState(
        step=step,
        decision="Escalate or close the ticket.",
        options=(
            DecisionOption(option_id="escalate", action="Hand the ticket to the second line"),
            DecisionOption(option_id="close", action="Close the ticket as handled"),
        ),
        relevant_paths=("/tickets",),
        axes=(
            VariationAxis(
                path="/tickets/severity", values=("high", "low"), effect="The tier decides."
            ),
        ),
        criteria=(Criterion(criterion_id="handled", text="The ticket was handled correctly."),),
        evidence_refs=("point", f"rule:{RULE.rule_id}"),
    )


def tier_scenario(scenario_id: str, severity: str, *, lock: str | None = None) -> Scenario:
    """One scenario setting the tier, and optionally the lock, revealed by ``check_sla``.

    Args:
        scenario_id: Scenario identity.
        severity: The tier.
        lock: The lock value to set, when the scenario edits it.

    Returns:
        The scenario.
    """
    edits = [StateEdit(path="/tickets/severity", value=severity)]
    if lock is not None:
        edits.append(StateEdit(path="/tickets/lock", value=lock))
    return Scenario(
        scenario_id=scenario_id,
        edits=tuple(edits),
        correct_option="escalate" if severity == "high" else "close",
        deciding_paths=("/tickets/severity",),
        witness=(WitnessStep(tool="check_sla", arguments={}, reveals=("/tickets/severity",)),),
        rationale=f"A {severity} tier ticket is resolved by the matching handling.",
        criteria=("handled",),
    )


def test_grounded_job_continues_offline_and_replays_without_its_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actor reads, escalates and finishes in the grounded world; the high tier passes."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    study = execute_continuation_job(fixture.project, fixture.job, code_revision="test")

    assert [branch.status for branch in study.branches] == ["completed", "completed"]
    assert all(
        branch.assessment and branch.assessment.status == "pass" for branch in study.branches
    )
    assert [len(branch.checkpoints) for branch in study.branches] == [2, 2]
    assert len(fixture.actor.requests) == 6 and len(fixture.predictor.requests) == 4
    fixture.job.runtime_file.unlink()
    assert execute_continuation_job(fixture.project, fixture.job, code_revision="test") == study
    assert len(fixture.actor.requests) == 6


def test_witness_and_option_contrast_run_on_the_grounded_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``check_sla`` shows the tier, and forcing escalation beats closing a high ticket."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    start = fixture.job.study.checkpoint
    witness = run_observation_witness(
        fixture.project,
        checkpoint=start,
        scenario=tier_scenario("urgent", "high"),
        decision=tier_decision(0),
        job=fixture.job,
        code_revision="test",
    )
    assert witness.holds, witness.problems
    assert witness.state == INITIAL
    route_judge(monkeypatch, verdict=state_judge(("tickets", "outcome"), "right"))

    contrast = run_option_contrast(
        fixture.project,
        checkpoint=start,
        decision=tier_decision(0),
        actions=ACTIONS,
        job=fixture.job,
        repeats=2,
        judge=episode_judge(
            fixture.project,
            fixture.settings,
            **run_digests(fixture.project, start, fixture.job.configuration),
        ),
        code_revision="test",
    )

    outcomes = {row.option_id: (row.successes, row.planned) for row in contrast.options}
    assert outcomes == {"escalate": (2, 2), "close": (0, 2)}
