"""Committed candidate checkpoints rewind and fork without generating observations."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, SourceIdentity
from exp.common.models import (
    AssistantAction,
    ModelRequest,
    ModelResponse,
    OperationEconomics,
    ToolCall,
    load_model_catalog,
)
from exp.common.project import (
    ProjectProviderFreeStage,
    ProjectTracePreparationSettings,
    artifact_input,
    export_project_bundle,
    restore_project_bundle,
)
from exp.common.project.project import write_project_config
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.sessions.errors import StateUnavailableError
from exp.runtime.sessions.store import SessionStore
from exp.simulation.world_model.checkpoints import (
    capture_stored_world_checkpoint,
    load_world_checkpoint,
)
from exp.simulation.world_model.durable import SessionPolicy
from exp.simulation.world_model.grounded_session import open_grounded_session
from exp.simulation.world_model.runtime import load_grounded_world_model


class _UnusedModel:
    """Reject any unintended provider execution while projecting candidate checkpoints."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Checkpoint and fork creation must not invoke a model."""
        raise AssertionError("unexpected model call")


def test_committed_prefix_forks_two_independent_worlds_without_calls(tmp_path: Path) -> None:
    """A prefix is read from committed session state, never synthesized by a model."""
    project, spec = source_project(tmp_path)
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path))
    _, capabilities = catalog.snapshot("model")
    runtime = load_grounded_world_model(
        project.artifacts,
        spec.world_model.artifact_id,
        client=_UnusedModel(),
        capabilities=capabilities,
    )
    now = datetime.now(UTC)
    policy = SessionPolicy(
        deadline=now + timedelta(minutes=5),
        maximum_operations=9,
        inference_identity="0" * 64,
        code_revision="fixture",
    )
    calls: list[str] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Return only caller-authored fixture transitions through the actual commit owner."""
        calls.append(identity)
        proposal: JsonObject = {
            "observations": [{"call_id": "call", "reply": "ok", "encoding": "json"}],
            "effect": {
                "kind": "edits",
                "edits": [{"op": "set", "path": ["count"], "value": len(calls)}],
            },
        }
        return ModelResponse(
            output=AssistantAction(
                tool_calls=(
                    ToolCall(call_id="proposal", name="submit_transition", arguments=proposal),
                )
            ),
            model=runtime.artifact.model,
            economics=OperationEconomics(),
        ).model_dump(mode="json")

    store = SessionStore(tmp_path / "source.sqlite")
    source = open_grounded_session(
        runtime=runtime,
        task=spec.task,
        store=store,
        artifacts=project.artifacts,
        session_id="source",
        policy=policy,
        complete=complete,
        lock_path=tmp_path / "source.lock",
        maximum_output_tokens=4096,
        output_protocol="json-tool",
    )
    for i in range(3):
        source.step(
            (ToolCall(call_id="call", name="inspect", arguments={"key": "current"}),),
            operation_id=str(i),
        )
    checkpoint = capture_stored_world_checkpoint(
        store,
        "source",
        spec.world_model,
        project.artifacts,
        public_context=spec.task.initial_context,
        prefix_position=1,
        source_actor=None,
        created_at=now,
        code_revision="fixture",
    )
    ref = artifact_input(project.artifacts.read(checkpoint.artifact_id).manifest)
    assert load_world_checkpoint(project.artifacts, ref) == checkpoint
    assert checkpoint.current.business_state == {"count": 1}
    assert len(checkpoint.current.transitions) == 1
    before = len(calls)
    forks = [
        open_grounded_session(
            runtime=runtime,
            task=spec.task,
            store=SessionStore(tmp_path / f"fork-{i}.sqlite"),
            artifacts=project.artifacts,
            session_id=f"fork-{i}",
            policy=policy,
            complete=complete,
            lock_path=tmp_path / f"fork-{i}.lock",
            maximum_output_tokens=4096,
            output_protocol="json-tool",
            checkpoint=checkpoint,
        )
        for i in range(2)
    ]
    assert len(calls) == before
    assert all(fork.state == {"count": 1} and fork.transitions == () for fork in forks)
    assert source.state == {"count": 3} and len(source.transitions) == 3
    # The checkpoint owns revisions 0..1 of the source; collection keeps them, reclaims 2.
    assert store.collect_garbage().complete
    assert store.read_state(store.ref_at("source", 1), ("business_state",)) == {"count": 1}
    with pytest.raises(StateUnavailableError):
        store.read_state(store.ref_at("source", 2))
    assert store.release_pin(checkpoint.artifact_id, "source")
    store.collect_garbage()
    with pytest.raises(StateUnavailableError):
        store.read_state(store.ref_at("source", 1))
    assert load_world_checkpoint(project.artifacts, ref) == checkpoint
    forks[0].step(
        (ToolCall(call_id="call", name="inspect", arguments={"key": "current"}),),
        operation_id="branch",
    )
    assert forks[1].state == {"count": 1} and source.state == {"count": 3}
    branch_checkpoint = capture_stored_world_checkpoint(
        SessionStore(tmp_path / "fork-0.sqlite"),
        "fork-0",
        spec.world_model,
        project.artifacts,
        public_context=spec.task.initial_context,
        prefix_position=0,
        source_actor=None,
        parent_checkpoint=ref,
        created_at=now,
        code_revision="fixture",
    )
    assert branch_checkpoint.initial.business_state == checkpoint.current.business_state
    assert branch_checkpoint.current.history == checkpoint.current.history
    grandchild = open_grounded_session(
        runtime=runtime,
        task=spec.task,
        store=SessionStore(tmp_path / "grandchild.sqlite"),
        artifacts=project.artifacts,
        session_id="grandchild",
        policy=policy,
        complete=complete,
        lock_path=tmp_path / "grandchild.lock",
        maximum_output_tokens=4096,
        output_protocol="json-tool",
        checkpoint=branch_checkpoint,
    )
    assert grandchild.state == {"count": 1} and grandchild.transitions == ()
    with pytest.raises(ValueError, match="initialization"):
        capture_stored_world_checkpoint(
            SessionStore(tmp_path / "fork-0.sqlite"),
            "fork-0",
            spec.world_model,
            project.artifacts,
            public_context=spec.task.initial_context,
            prefix_position=0,
            source_actor=None,
            created_at=now,
            code_revision="fixture",
        )
    with pytest.raises(ValueError, match="context"):
        open_grounded_session(
            runtime=runtime,
            task=spec.task.model_copy(update={"initial_context": {}}),
            store=SessionStore(tmp_path / "bad.sqlite"),
            artifacts=project.artifacts,
            session_id="bad",
            policy=policy,
            complete=complete,
            lock_path=tmp_path / "bad.lock",
            maximum_output_tokens=4096,
            checkpoint=checkpoint,
        )
    portable_source = project.artifacts.write_json(
        artifact_id="portable-stage-source",
        artifact_type="trace-dataset",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=now,
            code_revision="fixture",
            source=SourceIdentity(kind="file", source_id="portable-fixture", sha256="a" * 64),
        ),
        files={"trace.json": {"trace_id": "portable"}},
    )
    source_ref = artifact_input(portable_source)
    tasks = project.artifacts.write_json(
        artifact_id="checkpoint-task-set",
        artifact_type="task-set",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=now, inputs=(source_ref,), code_revision="fixture"
        ),
        files={"tasks.json": {"task_ids": ["checkpoint"]}},
    )
    write_project_config(
        project.paths,
        project.load_project().model_copy(
            update={
                "trace_preparation": ProjectTracePreparationSettings(source_kind="otlp"),
                "provider_free_stage": ProjectProviderFreeStage(
                    trace_dataset=source_ref, task_set=artifact_input(tasks)
                ),
            }
        ),
    )
    project.select_evaluation_task(ref)
    archive = export_project_bundle(project, tmp_path / "task.zip", producer_revision="fixture")
    restored = restore_project_bundle(
        archive.path, root=tmp_path / "restored", expected_sha256=archive.sha256
    )
    assert restored.load_project().evaluation_tasks == (ref,)
    assert load_world_checkpoint(restored.artifacts, ref) == checkpoint
    bad = ArtifactInput(artifact_id=ref.artifact_id, sha256="0" * 64)
    with pytest.raises(ValueError, match="digest"):
        load_world_checkpoint(restored.artifacts, bad)
