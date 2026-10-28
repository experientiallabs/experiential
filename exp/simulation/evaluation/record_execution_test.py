"""Native receipts prove actual state effects rather than trusting successful-looking replies."""

import sqlite3
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pytest

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    JsonValue,
    sha256_json,
)
from exp.common.models import ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase, ToolSchema
from exp.runtime.environments.record_world import RecordTool, RecordToolError, RecordWorldFactory
from exp.runtime.environments.world_contracts import (
    ExecutableCheckpoint,
    StateDomain,
    ToolEffectPolicy,
    WorldAccess,
    WorldEvent,
    WorldManifest,
)
from exp.runtime.environments.worlds import ExecutableWorldRuntime, WorldContext
from exp.simulation.evaluation.admission import (
    EvaluationSpecification,
    persist_specification,
    record_label_witness,
    verify_label_witness,
)
from exp.simulation.evaluation.record_execution import (
    RecordReplayRuntime,
    execute_record_script,
    load_record_execution,
    record_state,
    verify_record_execution,
)
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier, VerifierChallenge, grade

NOW = datetime(2026, 10, 18, tzinfo=UTC)
SCHEMA = """
CREATE TABLE orders(id TEXT PRIMARY KEY, address TEXT, status TEXT);
INSERT INTO orders VALUES ('a', 'old', 'pending'), ('b', 'other', 'pending');
CREATE TABLE outbox(message TEXT);
"""


def access(arguments: JsonObject) -> WorldAccess:
    """Declare both business state and the sandboxed effect sink."""
    del arguments
    return WorldAccess(writes=frozenset({"orders", "outbox"}))


def change(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Apply a change, lie about applying it, or emit an unwanted effect when requested."""
    del context
    order = str(arguments.get("id", "a"))
    row = db.execute("SELECT status FROM orders WHERE id=?", (order,)).fetchone()
    if row is None or row[0] != "pending":
        raise RecordToolError("order is unavailable for address changes")
    if not arguments.get("noop"):
        db.execute(
            "UPDATE orders SET address=? WHERE id=?", (str(arguments.get("address", "new")), order)
        )
    if arguments.get("email"):
        db.execute("INSERT INTO outbox VALUES ('unwanted email')")
    return {"success": True}


def fixture() -> tuple[RecordReplayRuntime, TaskCase]:
    """Use a real isolated SQLite service with an evaluator-only projection."""
    factory = RecordWorldFactory(
        service={"name": "authored-retail-fixture"},
        schema=SCHEMA,
        tools={"change": RecordTool(access, change)},
    )
    manifest = WorldManifest(
        domains=(
            StateDomain(name="orders", owner="native"),
            StateDomain(name="outbox", owner="native", kind="effect", sandboxed=True),
        ),
        tools=(ToolEffectPolicy(name="change", writes=("orders", "outbox")),),
    )
    world = ExecutableWorldRuntime(manifest, factory, seed=7)
    replay = RecordReplayRuntime(
        world, partial(record_state, world), sha256_json({"probe": "all-tables-v1"})
    )
    task = TaskCase(
        task_id="retail",
        lineage_group_id="retail",
        origin="authored",
        partition="fit",
        workload_weight=1,
        instruction="Change order a to new; preserve other records and send no email.",
        tools=(
            ToolSchema(
                name="change", description="Change an address.", input_schema={"type": "object"}
            ),
        ),
    )
    return replay, task


def source(project: ProjectStore) -> ArtifactInput:
    """Save explicitly authored intent; this is not a native-production observation."""
    if not project.artifacts.exists("authored-intent"):
        project.artifacts.write_json(
            artifact_id="authored-intent",
            artifact_type="authored-intent",
            envelope=ArtifactEnvelope(schema_version=1, created_at=NOW, code_revision="test"),
            files={"intent.json": {"scope": "authored local record behavior"}},
        )
    return artifact_input(project.artifacts.read("authored-intent").manifest)


def receipt(
    project: ProjectStore, arguments: JsonObject
) -> tuple[ArtifactInput, RecordReplayRuntime]:
    """Execute one actual native call twice before saving its complete receipt."""
    replay, task = fixture()
    with replay.world.open(task) as world:
        baseline = world.checkpoint()
    reference = execute_record_script(
        project.artifacts,
        replay,
        task=task,
        baseline=baseline,
        actions=(ToolCall(call_id="change-1", name="change", arguments=arguments),),
        finish_tick=1,
        sources=(source(project),),
        created_at=NOW,
        code_revision="test",
    )
    return reference, replay


def acceptance() -> TaskVerifier:
    """Inspect actual requested record and preserve unrelated records and effect sinks."""
    return TaskVerifier(
        description="Independent requested-address and preservation specification.",
        outcome_schema={"type": "object"},
        required_paths=("/final_state/tables/orders/rows/0/1",),
        rules=(
            OutcomeRule(path="/final_state/tables/orders/rows/0/1", operator="equals", value="new"),
        ),
        preserved_paths=("/tables/orders/rows/1", "/tables/outbox"),
        allowed_tools=("change",),
    )


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ({}, "pass"),
        ({"noop": True}, "fail"),
        ({"id": "b"}, "fail"),
        ({"address": "partial"}, "fail"),
        ({"email": True}, "fail"),
    ],
)
def test_native_state_not_success_reply_decides_goal_and_effects(
    tmp_path: Path, arguments: JsonObject, expected: str
) -> None:
    """Identical success replies cannot hide no-op, wrong-order, partial or collateral writes."""
    project = ProjectStore(tmp_path, "records")
    ref, replay = receipt(project, arguments)
    result = verify_record_execution(project.artifacts, ref, replay)
    assert result.steps[0].reply.content == '{"success":true}'
    assert grade(acceptance(), result.outcome).status == expected
    assert len(result.steps) == 1 and result.finish_tick == 1
    assert load_record_execution(project.artifacts, ref) == result


def test_native_witness_requires_explicit_replay_and_exact_labeled_trajectory(
    tmp_path: Path,
) -> None:
    """No receipt may load code on its own, approve other actions or claim a copied predicate."""
    project = ProjectStore(tmp_path, "records")
    ref, replay = receipt(project, {})
    result = load_record_execution(project.artifacts, ref)
    spec = persist_specification(
        project.artifacts,
        EvaluationSpecification(
            description="Independent authored address contract.",
            predicate=acceptance(),
            subject="native_observation",
        ),
        sources=(source(project),),
        created_at=NOW,
        code_revision="test",
    )
    with pytest.raises(ValueError, match="explicitly supplied"):
        record_label_witness(
            project.artifacts,
            specification=spec,
            outcome=result.outcome,
            receipt=ref,
            created_at=NOW,
            code_revision="test",
        )
    witness = record_label_witness(
        project.artifacts,
        specification=spec,
        outcome=result.outcome,
        receipt=ref,
        created_at=NOW,
        code_revision="test",
        record_runtime=replay,
    )
    challenge = VerifierChallenge(
        id="native",
        outcome=result.outcome,
        expected="pass",
        authority="native_execution",
        rationale="Requested record changed without collateral writes.",
        witness=witness,
    )
    assert (
        verify_label_witness(
            project.artifacts, challenge, label="pass", compared=(), record_runtime=replay
        )
        is None
    )
    refusal = verify_label_witness(project.artifacts, challenge, label="pass", compared=())
    assert refusal is not None and refusal.reason == "unverifiable_witness"
    refusal = verify_label_witness(
        project.artifacts, challenge, label="pass", compared=(acceptance(),), record_runtime=replay
    )
    assert refusal is not None and refusal.reason == "self_approval"
    for key, value in (("steps", []), ("final_answer", "The address is updated.")):
        with pytest.raises(ValueError, match=key):
            record_label_witness(
                project.artifacts,
                specification=spec,
                outcome={**result.outcome, key: value},
                receipt=ref,
                created_at=NOW,
                code_revision="test",
                record_runtime=replay,
            )


def test_changed_native_semantics_and_missing_evidence_refuse(tmp_path: Path) -> None:
    """A different projector cannot reuse a receipt, and unavailable state is not a pass."""
    project = ProjectStore(tmp_path, "records")
    ref, replay = receipt(project, {})
    other = RecordReplayRuntime(replay.world, replay.probe, "0" * 64)
    with pytest.raises(ValueError, match="backend definition"):
        verify_record_execution(project.artifacts, ref, other)
    result = load_record_execution(project.artifacts, ref)
    assert grade(acceptance(), {**result.outcome, "final_state": None}).status == "unassessable"
    # An observed but emptied state removed protected records: a violation, not missing evidence.
    assert grade(acceptance(), {**result.outcome, "final_state": {}}).status == "fail"


def test_receipt_binds_scheduled_outage_and_refusal_without_reroll(tmp_path: Path) -> None:
    """A scheduled outage returns an error with unchanged state and survives replay."""
    project = ProjectStore(tmp_path, "records")
    replay, task = fixture()
    world = ExecutableWorldRuntime(
        replay.world.manifest,
        replay.world.factory,
        seed=7,
        events=(
            WorldEvent(event_id="outage", kind="outage", due=0, tools=("change",), duration=3),
        ),
    )
    replay = RecordReplayRuntime(world, partial(record_state, world), replay.probe_sha256)
    with world.open(task) as session:
        baseline = session.checkpoint()
    ref = execute_record_script(
        project.artifacts,
        replay,
        task=task,
        baseline=baseline,
        actions=(ToolCall(call_id="outage-call", name="change", arguments={}),),
        finish_tick=1,
        sources=(source(project),),
        created_at=NOW,
        code_revision="test",
    )
    result = verify_record_execution(project.artifacts, ref, replay)
    assert result.steps[0].reply.is_error
    assert result.outcome["initial_state"] == result.outcome["final_state"]
    assert grade(acceptance(), result.outcome).status == "fail"


def test_unavailable_tool_and_impossible_cutoff_fail_before_execution(tmp_path: Path) -> None:
    """Setup cannot use undeclared capabilities, repeated call IDs or a backwards clock."""
    project = ProjectStore(tmp_path, "records")
    replay, task = fixture()
    with replay.world.open(task) as world:
        baseline = world.checkpoint()
    call = ToolCall(call_id="one", name="change", arguments={})
    for actions, tick, match in (
        ((call, call), 2, "unique"),
        ((call,), 0, "finish_tick"),
        ((call.model_copy(update={"name": "private"}),), 1, "capability"),
    ):
        with pytest.raises(ValueError, match=match):
            execute_record_script(
                project.artifacts,
                replay,
                task=task,
                baseline=baseline,
                actions=actions,
                finish_tick=tick,
                sources=(source(project),),
                created_at=NOW,
                code_revision="test",
            )
    assert not project.artifacts.list_ids(artifact_type="native-record-execution")


def test_replay_distinguishes_json_numbers_from_booleans(tmp_path: Path) -> None:
    """Python equality must not certify a projection changing from numeric one to true."""
    project = ProjectStore(tmp_path, "records")
    replay, task = fixture()
    calls = 0

    def unstable_probe(checkpoint: ExecutableCheckpoint) -> JsonObject:
        """Return distinct JSON types on the second execution of the same world."""
        nonlocal calls
        del checkpoint
        calls += 1
        return {"available": 1 if calls <= 3 else True}

    replay = RecordReplayRuntime(replay.world, unstable_probe, replay.probe_sha256)
    with replay.world.open(task) as world:
        baseline = world.checkpoint()
    with pytest.raises(ValueError, match="does not replay exactly"):
        execute_record_script(
            project.artifacts,
            replay,
            task=task,
            baseline=baseline,
            actions=(ToolCall(call_id="one", name="change", arguments={}),),
            finish_tick=1,
            sources=(source(project),),
            created_at=NOW,
            code_revision="test",
        )


def test_record_witness_refuses_no_goal_evidence(tmp_path: Path) -> None:
    """Empty detector sets and process-completed predicates cannot certify a native goal."""
    project = ProjectStore(tmp_path, "records")
    ref, replay = receipt(project, {"noop": True})
    result = load_record_execution(project.artifacts, ref)
    for rules in ((), (OutcomeRule(path="/stop_reason", operator="equals", value="completed"),)):
        spec = persist_specification(
            project.artifacts,
            EvaluationSpecification(
                description="Insufficient goal evidence.",
                subject="native_observation",
                predicate=TaskVerifier(
                    description="Process completed with no detected errors.",
                    outcome_schema={"type": "object"},
                    rules=rules,
                ),
            ),
            created_at=NOW,
            code_revision="test",
        )
        with pytest.raises(ValueError, match="observable goal"):
            record_label_witness(
                project.artifacts,
                specification=spec,
                outcome=result.outcome,
                receipt=ref,
                record_runtime=replay,
                created_at=NOW,
                code_revision="test",
            )


@pytest.mark.parametrize("change_reply", [True, False])
def test_verification_executes_the_native_backend_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change_reply: bool
) -> None:
    """Valid saved digests cannot substitute for reproducing actual replies and state changes."""
    from exp.runtime.environments.interface import Observation
    from exp.runtime.environments.record_world import RecordWorld

    project = ProjectStore(tmp_path, "records")
    ref, replay = receipt(project, {})
    original = RecordWorld.execute

    def changed(self: RecordWorld, action: ToolCall, context: WorldContext) -> Observation:
        """Either change only the reply or return the old reply without doing the write."""
        if change_reply:
            original(self, action, context)
            return Observation(content='{"success":false}')
        return Observation(content='{"success":true}')

    monkeypatch.setattr(RecordWorld, "execute", changed)
    with pytest.raises(ValueError, match="does not reproduce"):
        verify_record_execution(project.artifacts, ref, replay)


def test_native_scripts_cannot_bypass_build_owned_source_lineage(tmp_path: Path) -> None:
    """A local fit label alone does not authorize consuming trace-backed development evidence."""
    project = ProjectStore(tmp_path, "records")
    replay, task = fixture()
    with replay.world.open(task) as world:
        baseline = world.checkpoint()
    task = task.model_copy(update={"origin": "mined", "source_trace_ids": ("unresolved-trace",)})
    with pytest.raises(ValueError, match="frozen build"):
        execute_record_script(
            project.artifacts,
            replay,
            task=task,
            baseline=baseline,
            actions=(),
            finish_tick=0,
            sources=(source(project),),
            created_at=NOW,
            code_revision="test",
        )
    assert not project.artifacts.list_ids(artifact_type="native-record-execution")
