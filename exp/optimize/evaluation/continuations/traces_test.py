"""Saved trajectories become trace imports whose learning authority is the world owner.

Every fixture runs a real continuation study offline: scripted actor responses against a native
XLSX artifact world, a native and a simulated SQLite record world, and a mixed world whose
refunds are simulated. Nothing opens a provider.
"""

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    JsonValue,
    sha256_json,
    stable_id,
)
from exp.common.models import AssistantAction, ModelMessage, ToolCall
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.common.tasks import TaskCase, ToolSchema, load_task_set
from exp.common.traces import Trace
from exp.common.traces.ingest.otlp import TraceNormalizationResult
from exp.common.traces.ingest.persistence import read_ingested_traces
from exp.common.traces.sqlite import SQLiteTraceStore
from exp.common.traces.sqlite_schema import trace_database_path
from exp.optimize.evaluation.audit.panel_test import (
    USERS,
    SimulatedProvider,
    build_project,
    episode,
)
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import (
    ContinuationStudySpec,
    WorldRuntime,
    run_continuation_study,
)
from exp.optimize.evaluation.continuations.study_test import probe, workbook_runtime
from exp.optimize.evaluation.continuations.traces import (
    TrajectoryTraceExport,
    TrajectoryTraceSummary,
    export_trajectory_traces,
    trajectory_studies,
)
from exp.optimize.evaluation.equivalence import policy as policy_module
from exp.optimize.evaluation.equivalence.policy_test import fixture_policy, registered_policy
from exp.optimize.evaluation.fidelity.replay import recorded_exchanges
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.agents.chat_test import _Model, _response
from exp.runtime.environments.mixed import MixedEvent
from exp.runtime.environments.mixed_test import _order, _refund, _world_runtime
from exp.runtime.environments.record_world import RecordWorldFactory
from exp.runtime.environments.tests.world_conformance_test import (
    TICKET_SCHEMA,
    TICKET_TOOLS,
    ticket_case,
    workbook_case,
)
from exp.runtime.environments.world_artifacts import WorldCheckpoint
from exp.runtime.environments.world_contracts import (
    ExecutableCheckpoint,
    StateDomain,
    ToolEffectPolicy,
    WorldManifest,
)
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.build import build_task_set
from exp.simulation.continuations import ContinuationCheckpoint, persist_continuation_checkpoint
from exp.simulation.continuations_test import capsule
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier

REVISION = "test"
PROJECT = "case"
ACTOR = ChatAgentRuntime(maximum_model_calls=6)
MODEL = _response(AssistantAction(content="done")).model
PROBE = sha256_json({"probe": "none"})
ANSWERED = TaskVerifier(
    description="The actor answers done.",
    outcome_schema={},
    required_paths=("/final_answer",),
    rules=(OutcomeRule(path="/final_answer", operator="equals", value="done"),),
)
SIMULATED_TICKETS = WorldManifest(
    domains=(
        StateDomain(name="tickets", owner="simulated"),
        StateDomain(name="outbox", owner="simulated"),
    ),
    tools=(
        ToolEffectPolicy(name="open_ticket", reads=("tickets",), writes=("tickets",)),
        ToolEffectPolicy(name="escalate_ticket", writes=("tickets",)),
        ToolEffectPolicy(name="close_ticket", reads=("tickets",), writes=("tickets",)),
    ),
)


def empty_probe(checkpoint: WorldCheckpoint) -> JsonObject:
    """Project no private state; these studies grade only the actor's answer."""
    del checkpoint
    return {}


def tool(name: str) -> ToolSchema:
    """Declare one fixture tool visible to the actor."""
    return ToolSchema(name=name, description="Fixture tool.", input_schema={"type": "object"})


def ticket_task(index: int) -> TaskCase:
    """One authored ticket task with its own lineage."""
    return TaskCase(
        task_id=f"ticket-task-{index}",
        lineage_group_id=f"ticket-lineage-{index}",
        partition="fit",
        instruction=f"Open a ticket for printer {index} and close it.",
        workload_weight=1,
        origin="authored",
        tools=(tool("open_ticket"), tool("close_ticket")),
    )


def ticket_fork(index: int, fork: int) -> tuple[AssistantAction, ...]:
    """Open and close one ticket; forks differ only in the title the actor writes."""
    title = f"Printer {index} jam, attempt {fork}"
    return (
        AssistantAction(
            tool_calls=(
                ToolCall(
                    call_id="call-1",
                    name="open_ticket",
                    arguments={"title": title, "priority": "low"},
                ),
            )
        ),
        AssistantAction(
            tool_calls=(ToolCall(call_id="call-1", name="close_ticket", arguments={"ticket": 2}),)
        ),
        AssistantAction(content="done"),
    )


def simulated_tickets() -> ExecutableWorldRuntime:
    """The same ticket service, run as a simulated owner."""
    tools = {
        name: TICKET_TOOLS[name] for name in ("open_ticket", "escalate_ticket", "close_ticket")
    }
    return ExecutableWorldRuntime(
        SIMULATED_TICKETS,
        RecordWorldFactory(service={"name": "ticket-simulator"}, schema=TICKET_SCHEMA, tools=tools),
        owner="simulated",
        seed=7,
    )


def capture(
    store: ArtifactStore, runtime: WorldRuntime, task: TaskCase
) -> tuple[ArtifactInput, ContinuationCheckpoint]:
    """Capture a world at the actor's initial turn boundary from an authored source."""
    source = store.write_json(
        artifact_id=f"authored-{task.task_id}",
        artifact_type="authored-task",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision=REVISION
        ),
        files={"task.json": {"origin": "authored fixture, not benchmark evidence"}},
    )
    points: list[ContinuationCheckpoint] = []
    with runtime.open(task) as session:
        ACTOR.run(
            task,
            model=_Model((_response(AssistantAction(content="stop")),)),
            environment=session,
            on_checkpoint=lambda point: points.append(
                ContinuationCheckpoint(
                    task=task,
                    actor=point,
                    world=session.checkpoint(),
                    sources=(artifact_input(source),),
                )
            ),
        )
    reference = persist_continuation_checkpoint(
        store, points[0], created_at=datetime.now(UTC), code_revision=REVISION
    )
    return reference, points[0]


def study(
    store: ArtifactStore,
    checkpoint: ArtifactInput,
    world: WorldRuntime,
    forks: Sequence[Sequence[AssistantAction]],
    *,
    actor: ChatAgentRuntime = ACTOR,
    verifier: TaskVerifier = ANSWERED,
    study_probe: Callable[[WorldCheckpoint], JsonObject] = empty_probe,
    probe_sha256: str = PROBE,
    policy: ArtifactInput | None = None,
) -> ArtifactInput:
    """Run one saved study, one branch per scripted fork, and return its result reference.

    The study pins ``policy``, by default the fixture admission policy, before any branch runs.
    """
    spec = ContinuationStudySpec(
        checkpoint=checkpoint,
        actor_model=MODEL,
        verifier=verifier,
        probe_sha256=probe_sha256,
        repeats=len(forks),
        selector="fixture forks",
        admission_policy=policy or registered_policy(store),
    )
    run_continuation_study(
        store,
        spec,
        actor=actor,
        model=_Model(tuple(_response(action) for fork in forks for action in fork)),
        world=world,
        probe=study_probe,
        probe_sha256=probe_sha256,
        code_revision=REVISION,
    )
    identity = stable_id("continuation-study", {"spec": sha256_json(spec), "code": REVISION})
    return artifact_input(store.read(identity + "-result").manifest)


def panel(store: ArtifactStore, name: str, studies: Sequence[ArtifactInput]) -> ArtifactInput:
    """Save a generator shaped like every producer: a result whose cells bind study results."""
    cells = []
    for index, result in enumerate(studies):
        cells.append(
            artifact_input(
                store.write_json(
                    artifact_id=f"{name}-cell-{index}",
                    artifact_type="actor-panel-cell",
                    envelope=ArtifactEnvelope(
                        schema_version=1,
                        created_at=datetime.now(UTC),
                        code_revision=REVISION,
                        inputs=(result,),
                    ),
                    files={"cell.json": {"index": index}},
                )
            )
        )
    return artifact_input(
        store.write_json(
            artifact_id=f"{name}-result",
            artifact_type="actor-panel-report",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=REVISION,
                inputs=tuple(sorted(cells, key=lambda row: row.artifact_id)),
            ),
            files={"report.json": {"cells": len(cells)}},
        )
    )


def ticket_studies(store: ArtifactStore, tasks: int, forks: int = 2) -> list[ArtifactInput]:
    """Capture each ticket task natively and continue it with distinct forks."""
    results = []
    for index in range(tasks):
        reference, _ = capture(store, ticket_case().runtime, ticket_task(index))
        results.append(
            study(
                store,
                reference,
                ticket_case().runtime,
                [ticket_fork(index, fork) for fork in range(forks)],
            )
        )
    return results


def imported(root: Path, summary: TrajectoryTraceSummary, authority: str) -> tuple[Trace, ...]:
    """Read one authority's committed import back from the shared trace database."""
    row = next(item for item in summary.imports if item.authority == authority)
    assert row.import_id is not None
    return read_ingested_traces(root, row.import_id).traces


def metadata(trace: Trace) -> JsonObject:
    """The provenance every span of an exported trace carries."""
    value = trace.spans[0].attributes["exp.trace.metadata"]
    assert isinstance(value, dict)
    assert all(span.attributes["exp.trace.metadata"] == value for span in trace.spans)
    return value


def steps(trace: Trace) -> list[JsonObject]:
    """The per-call authority ledger of one exported trace."""
    value = metadata(trace)["steps"]
    assert isinstance(value, list)
    return [item for item in value if isinstance(item, dict)]


def test_native_workbook_branches_export_with_provenance_and_collapse_identical_runs(
    tmp_path: Path,
) -> None:
    """Two identical native XLSX branches become one native trace listing both occurrences."""
    project = ProjectStore(tmp_path, PROJECT)
    policy = registered_policy(project.artifacts)
    checkpoint, actor = capsule(project.artifacts)
    reference = persist_continuation_checkpoint(
        project.artifacts, checkpoint, created_at=datetime.now(UTC), code_revision=REVISION
    )
    edit = workbook_case().first
    result = study(
        project.artifacts,
        reference,
        workbook_runtime(checkpoint.world),
        [(AssistantAction(tool_calls=(edit,)), AssistantAction(content="done"))] * 2,
        actor=actor,
        verifier=TaskVerifier(
            description="Delete one data row.",
            outcome_schema={},
            required_paths=("/final_state/rows",),
            rules=(OutcomeRule(path="/final_state/rows", operator="equals", value=6),),
        ),
        study_probe=probe,
        probe_sha256=sha256_json({"probe": "xlsx-row-count-v1"}),
    )
    spec = TrajectoryTraceExport(policy=policy, generators=(result,))
    dry = export_trajectory_traces(project, spec, dry_run=True)
    assert (dry.branches, dry.native, dry.mixed, dry.simulated, dry.duplicates) == (2, 1, 0, 0, 1)
    assert [row.import_id for row in dry.imports] == [None]
    assert not SQLiteTraceStore(trace_database_path(tmp_path)).list_imports(PROJECT)

    summary = export_trajectory_traces(project, spec)
    assert summary.model_dump(exclude={"imports"}) == dry.model_dump(exclude={"imports"})
    (trace,) = imported(tmp_path, summary, "generated-native")
    assert trace.source.identity.kind == "native-world"
    assert trace.source.identity.source_id == f"generated-native:{result.artifact_id}"
    assert trace.conversation_id == checkpoint.task.lineage_group_id
    assert trace.initial_context == {"task_lineage": "world-conformance", "replay_from": 0}
    assert trace.outcome is not None and trace.outcome.status == "success"
    provenance = metadata(trace)
    assert provenance["authority"] == "generated-native"
    assert isinstance(checkpoint.world, ExecutableCheckpoint)
    assert provenance["world"] == {
        "kind": "executable",
        "manifest_sha256": checkpoint.world.manifest_sha256,
        "backend_sha256": checkpoint.world.backend_sha256,
        "owner": "native",
    }
    assert provenance["actor_model"] == MODEL.model_dump(mode="json")
    verifier = provenance["verifier"]
    assert isinstance(verifier, dict) and verifier["status"] == "pass"
    occurrences = provenance["occurrences"]
    assert isinstance(occurrences, list) and len(occurrences) == 2
    # Each branch carries its own verdict under the policy, outside the trace identity.
    occurrences = [dict(row) for row in occurrences if isinstance(row, dict)]
    verdicts = [row.pop("admission") for row in occurrences]
    assert all(
        isinstance(row, dict)
        and row["status"] == "admitted"
        and row["policy"] == policy.model_dump(mode="json")
        for row in verdicts
    )
    assert "admission" not in provenance
    assert summary.admission == {"admitted": 2, "refused": 0, "unassessable": 0}
    assert {json.dumps(row, sort_keys=True) for row in occurrences} == {
        json.dumps(
            {
                "generators": [result.artifact_id],
                "study": result.artifact_id,
                "branch": index,
                "branch_status": "completed",
                "checkpoint": reference.artifact_id,
                "fork_lineage": [reference.artifact_id],
                "prefix": "executed",
                "code_revision": REVISION,
            },
            sort_keys=True,
        )
        for index in range(2)
    }
    assert steps(trace) == [
        {
            "call_id": edit.call_id,
            "tool": "edit_workbook",
            "phase": "branch",
            "owner": "native",
            "reply": "recorded",
        }
    ]
    assert [exchange.name for exchange in recorded_exchanges(trace, "lineage")] == ["edit_workbook"]
    # An unchanged export replays the same content-addressed import.
    again = export_trajectory_traces(project, spec)
    assert again.imports == summary.imports


def test_record_world_authority_splits_native_from_simulated_and_build_refuses_simulated(
    tmp_path: Path,
) -> None:
    """A simulated owner's trajectories are labelled and refused as learning source."""
    project = ProjectStore(tmp_path, PROJECT)
    policy = registered_policy(project.artifacts)
    store = project.artifacts
    native = ticket_studies(store, tasks=1)
    reference, _ = capture(store, simulated_tickets(), ticket_task(9))
    simulated = study(store, reference, simulated_tickets(), [ticket_fork(9, 0)])
    generator = panel(store, "actor-panel-fixture", [*native, simulated])
    assert trajectory_studies(store, generator) == tuple(
        sorted((*native, simulated), key=lambda row: row.artifact_id)
    )
    summary = export_trajectory_traces(
        project, TrajectoryTraceExport(policy=policy, generators=(generator,), project="learner")
    )
    assert (summary.project, summary.native, summary.simulated, summary.mixed) == (
        "learner",
        2,
        1,
        0,
    )
    natives = imported(tmp_path, summary, "generated-native")
    assert all(trace.source.identity.kind == "native-world" for trace in natives)
    assert all(row["owner"] == "native" for trace in natives for row in steps(trace))
    (labelled,) = imported(tmp_path, summary, "generated-simulated")
    assert labelled.source.identity.kind == "simulation"
    assert metadata(labelled)["authority"] == "generated-simulated"
    assert [(row["owner"], row["reply"]) for row in steps(labelled)] == [
        ("simulated", "recorded"),
        ("simulated", "recorded"),
    ]
    with pytest.raises(ValueError, match="carry simulation provenance"):
        build_task_set(
            TraceNormalizationResult(traces=(labelled,), issues=()),
            ProjectStore(tmp_path, "learner").artifacts,
            created_at=datetime.now(UTC),
            code_revision=REVISION,
        )


def mixed_task() -> TaskCase:
    """A public task over native orders and simulated refunds."""
    return TaskCase(
        task_id="mixed-task",
        lineage_group_id="mixed-lineage",
        partition="fit",
        instruction="Refund the customer, then order a lamp.",
        workload_weight=1,
        origin="authored",
        tools=(tool("place_order"), tool("refund")),
    )


def test_mixed_episodes_with_a_simulated_reply_are_simulated_including_through_fork_lineage(
    tmp_path: Path,
) -> None:
    """A simulated refund decides the mixed grade, so the trace is simulated, also as a prefix."""
    project = ProjectStore(tmp_path, PROJECT)
    policy = registered_policy(project.artifacts)
    store = project.artifacts
    events: list[MixedEvent] = []
    world = _world_runtime(events)
    root, _ = capture(store, world, mixed_task())
    parent = study(
        store,
        root,
        world,
        [
            (
                AssistantAction(tool_calls=(_refund("r1"),)),
                AssistantAction(tool_calls=(_order("o1"),)),
                AssistantAction(content="done"),
            )
        ],
    )
    summary = export_trajectory_traces(
        project, TrajectoryTraceExport(policy=policy, generators=(parent,))
    )
    assert (summary.native, summary.mixed, summary.simulated) == (0, 0, 1)
    (trace,) = imported(tmp_path, summary, "generated-simulated")
    assert trace.source.identity.kind == "simulation"
    assert [(row["tool"], row["owner"], row["reply"]) for row in steps(trace)] == [
        ("refund", "simulated", "recorded"),
        ("place_order", "native", "recorded"),
    ]
    world_identity = metadata(trace)["world"]
    assert isinstance(world_identity, dict) and world_identity["kind"] == "mixed"
    with pytest.raises(ValueError, match="carry simulation provenance"):
        build_task_set(
            TraceNormalizationResult(traces=(trace,), issues=()),
            ProjectStore(tmp_path, "learner").artifacts,
            created_at=datetime.now(UTC),
            code_revision=REVISION,
        )

    # A fork resumed after the simulated refund inherits it as prefix and stays simulated.
    (branch,) = load_study(store, parent).branches
    fork_point = branch.checkpoints[0]
    child = study(
        store,
        fork_point,
        world,
        [(AssistantAction(tool_calls=(_order("o2", "desk"),)), AssistantAction(content="done"))],
    )
    summary = export_trajectory_traces(
        project, TrajectoryTraceExport(policy=policy, generators=(child,))
    )
    assert (summary.native, summary.simulated) == (0, 1)
    (forked,) = imported(tmp_path, summary, "generated-simulated")
    assert [(row["phase"], row["owner"], row["reply"]) for row in steps(forked)] == [
        ("prefix", "simulated", "recorded"),
        ("branch", "native", "recorded"),
    ]
    occurrences = metadata(forked)["occurrences"]
    assert isinstance(occurrences, list)
    (occurrence,) = occurrences
    assert isinstance(occurrence, dict)
    assert occurrence["fork_lineage"] == [fork_point.artifact_id, root.artifact_id]
    assert occurrence["prefix"] == "executed"
    assert forked.conversation_id == trace.conversation_id == "mixed-lineage"


def test_branches_without_world_evidence_are_skipped_and_generators_must_reach_studies(
    tmp_path: Path,
) -> None:
    """An answer-only branch has no reply to learn from; a foreign artifact is refused."""
    project = ProjectStore(tmp_path, PROJECT)
    policy = registered_policy(project.artifacts)
    store = project.artifacts
    reference, _ = capture(store, ticket_case().runtime, ticket_task(0))
    result = study(store, reference, ticket_case().runtime, [(AssistantAction(content="done"),)])
    summary = export_trajectory_traces(
        project, TrajectoryTraceExport(policy=policy, generators=(result,))
    )
    assert summary.imports == () and summary.native == summary.simulated == 0
    (skip,) = summary.skipped
    assert skip.branch == 0 and "no world evidence" in skip.reason
    with pytest.raises(ValueError, match="reaches no saved continuation study"):
        trajectory_studies(store, reference)
    stale = result.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="differs from its saved artifact"):
        trajectory_studies(store, stale)


def test_export_refuses_a_policy_the_study_did_not_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reaching a policy through a study's cited evidence never lets it judge the study."""
    project = ProjectStore(tmp_path, PROJECT)
    [result] = ticket_studies(project.artifacts, 1)
    other = registered_policy(project.artifacts, fixture_policy(margin=0.4))
    # As if a comparison cited by the study's training evidence were registered under ``other``.
    monkeypatch.setattr(policy_module, "require_cites", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match=f"not {other.artifact_id}; only the policy a study pins"):
        export_trajectory_traces(
            project, TrajectoryTraceExport(policy=other, generators=(result,)), dry_run=True
        )


def exchanges(traces: Sequence[Trace]) -> list[tuple[str, JsonValue, JsonValue]]:
    """Every recorded exchange of the exported traces, for a scripted simulator's truth."""
    return [
        (exchange.name, exchange.arguments, exchange.reply)
        for trace in traces
        for exchange in recorded_exchanges(trace, "lineage")
    ]


def authored_prefix(
    store: ArtifactStore, point: ContinuationCheckpoint, *messages: ModelMessage
) -> ArtifactInput:
    """Persist a checkpoint whose actor context gained conversation no world executed."""
    actor = point.actor.model_copy(update={"messages": (*point.actor.messages, *messages)})
    return persist_continuation_checkpoint(
        store,
        point.model_copy(update={"actor": actor}),
        created_at=datetime.now(UTC),
        code_revision=REVISION,
    )


def test_authored_prefix_replies_are_never_proven_native(tmp_path: Path) -> None:
    """Context no world executed is excluded as unknown; an unpaired reply skips the branch."""
    project = ProjectStore(tmp_path, PROJECT)
    policy = registered_policy(project.artifacts)
    store = project.artifacts
    _, point = capture(store, ticket_case().runtime, ticket_task(0))
    claimed = ToolCall(
        call_id="a1", name="open_ticket", arguments={"title": "x", "priority": "low"}
    )
    authored = authored_prefix(
        store,
        point,
        ModelMessage(role="assistant", assistant_action=AssistantAction(tool_calls=(claimed,))),
        ModelMessage(role="tool", content='{"ticket": 7}', tool_call_id="a1"),
    )
    orphan = authored_prefix(
        store, point, ModelMessage(role="tool", content='{"ticket": 7}', tool_call_id="none")
    )
    results = [
        study(store, reference, ticket_case().runtime, [ticket_fork(0, 0)])
        for reference in (authored, orphan)
    ]
    summary = export_trajectory_traces(
        project,
        TrajectoryTraceExport(policy=policy, generators=(panel(store, "authored", results),)),
    )
    assert (summary.native, summary.mixed) == (1, 1)
    (skip,) = summary.skipped
    assert skip.study == results[1].artifact_id and "answers no earlier call" in skip.reason
    (trace,) = imported(tmp_path, summary, "generated-native")
    assert [(row["phase"], row["owner"], row["reply"]) for row in steps(trace)] == [
        ("prefix", "unknown", "excluded"),
        ("branch", "native", "recorded"),
        ("branch", "native", "recorded"),
    ]
    assert '{"ticket": 7}' not in trace.model_dump_json()
    occurrences = metadata(trace)["occurrences"]
    assert isinstance(occurrences, list) and isinstance(occurrences[0], dict)
    assert occurrences[0]["prefix"] == "unverified"
    assert occurrences[0]["fork_lineage"] == [authored.artifact_id]
    # Exported call IDs stay unique even though the actor reused call-1 across turns.
    call_ids = [row["call_id"] for row in steps(trace)]
    assert call_ids == ["a1", "call-1", "call-1~1"]


def test_export_refuses_a_study_whose_lineage_a_rebuild_reserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lineage fit when its study ran but held out by a later build is never exported.

    A build of one lineage keeps it fit; a build of two holds out whichever ranks second, so a
    rebuild with one more user can move the studied lineage to held out.
    """

    def partitions(project: ProjectStore) -> dict[str, str]:
        """Each task lineage of the project's current build, with its partition."""
        build = project.load_project().build
        assert build is not None
        tasks = load_task_set(project.artifacts, build.task_set.artifact_id).tasks
        return {task.lineage_group_id: task.partition for task in tasks}

    first = [episode(0)]
    _, project, _ = build_project(
        tmp_path, monkeypatch, SimulatedProvider(first), episodes=first, relative_traces=True
    )
    store = project.artifacts
    policy = registered_policy(store)
    ((lineage, partition),) = partitions(project).items()
    assert partition == "fit"
    task = ticket_task(0).model_copy(update={"lineage_group_id": lineage})
    reference, _ = capture(store, ticket_case().runtime, task)
    result = study(store, reference, ticket_case().runtime, [ticket_fork(0, 0)], policy=policy)
    spec = TrajectoryTraceExport(policy=policy, generators=(result,))
    assert export_trajectory_traces(project, spec, dry_run=True).native == 1
    for user in range(1, USERS):
        later = [episode(0), episode(user)]
        project = build_project(
            tmp_path, monkeypatch, SimulatedProvider(later), episodes=later, relative_traces=True
        )[1]
        if partitions(project)[lineage] == "held_out":
            break
    else:
        pytest.fail("no second user ranked before the studied lineage")
    with pytest.raises(ValueError, match="lineage is reserved by the frozen build"):
        export_trajectory_traces(project, spec)
