"""Real checkpoint handoffs preserve provenance, replay and candidate/native disagreement."""

import json
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, sha256_json
from exp.common.models import (
    AssistantAction,
    ModelResponse,
    OperationEconomics,
    ToolCall,
    load_model_catalog,
)
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.optimize.evaluation.harbor.bridge import serve_harbor_trial
from exp.optimize.evaluation.harbor.export import export_harbor_task
from exp.optimize.evaluation.harbor.trial import HarborTrial
from exp.optimize.evaluation.harbor.trial_test import UnusedModel
from exp.optimize.evaluation.native_artifacts import (
    NativeMaterializationSpec,
    export_native_artifacts,
    materialize_native_artifacts,
)
from exp.runtime.environments.artifact_edits import (
    NativeArtifactBinding,
    NativeArtifactOutput,
    NativeArtifactPlan,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.sessions.store import SessionStore
from exp.simulation.evaluation.tasks import EvaluationTask, TaskVerifier, persist_evaluation_task
from exp.simulation.world_model.checkpoints import capture_world_checkpoint, load_world_checkpoint
from exp.simulation.world_model.durable import DurableWorldModelSession, SessionPolicy
from exp.simulation.world_model.grounded_session import open_grounded_session
from exp.simulation.world_model.runtime import load_grounded_world_model


class TextAdapter:
    """Second, non-workbook adapter proving the core handoff has no spreadsheet assumptions."""

    def __init__(self) -> None:
        """Count executions so saved-receipt replay cannot conceal reexecution."""
        self.calls = 0

    @property
    def definition(self) -> JsonObject:
        """Describe the deliberately bounded text fixture."""
        return {"adapter": "append-text-fixture-v1"}

    def apply(
        self,
        files: Mapping[str, bytes],
        operations: tuple[ToolCall, ...],
        output_paths: tuple[str, ...],
    ) -> NativeArtifactOutput:
        """Apply a typed append operation to detached input bytes."""
        self.calls += 1
        assert operations[0].name == "append"
        assert output_paths == ("output.txt",)
        content = files["input.txt"] + b" changed"
        return NativeArtifactOutput({"output.txt": content}, {"text": content.decode()})


def native_fixture(
    tmp_path: Path, *, predicted: str = "initial changed"
) -> tuple[ProjectStore, NativeMaterializationSpec, TextAdapter, DurableWorldModelSession]:
    """Commit a typed edit through the actual simulator proposal and checkpoint owners."""
    project, base = source_project(tmp_path)
    now = datetime.now(UTC)
    initial = artifact_input(
        project.artifacts.write(
            artifact_id="native-input",
            artifact_type="native-input",
            envelope=ArtifactEnvelope(schema_version=1, created_at=now, code_revision="fixture"),
            files={"input.txt": b"initial"},
        )
    )
    adapter = TextAdapter()
    binding = NativeArtifactBinding(
        input_artifact=initial,
        adapter_sha256=sha256_json(adapter.definition),
        output_paths=("output.txt",),
    )
    plan = NativeArtifactPlan(
        input_sha256=initial.sha256,
        operations=(ToolCall(call_id="append-1", name="append", arguments={}),),
        expected_projection={"text": predicted},
    )
    task = base.task.model_copy(
        update={"initial_context": {"native_artifact_binding": binding.model_dump(mode="json")}}
    )
    registry = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path))
    _, capabilities = registry.snapshot("model")
    runtime = load_grounded_world_model(
        project.artifacts,
        base.world_model.artifact_id,
        client=UnusedModel(),
        capabilities=capabilities,
    )

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Save a deterministic fixture proposal through the real commit boundary."""
        return ModelResponse(
            output=AssistantAction(
                tool_calls=(
                    ToolCall(
                        call_id="proposal",
                        name="submit_transition",
                        arguments={
                            "observations": [
                                {"call_id": "call", "reply": "saved", "encoding": "json"}
                            ],
                            "effect": {
                                "kind": "edits",
                                "edits": [
                                    {
                                        "op": "set",
                                        "path": [binding.plan_key],
                                        "value": plan.model_dump(mode="json"),
                                    }
                                ],
                            },
                        },
                    ),
                )
            ),
            model=runtime.artifact.model,
            economics=OperationEconomics(),
        ).model_dump(mode="json")

    session = open_grounded_session(
        runtime=runtime,
        task=task,
        store=SessionStore(tmp_path / "source.sqlite"),
        artifacts=project.artifacts,
        session_id="source",
        policy=SessionPolicy(
            deadline=now + timedelta(minutes=5),
            maximum_operations=3,
            inference_identity="0" * 64,
            code_revision="fixture",
        ),
        complete=complete,
        lock_path=tmp_path / "source.lock",
        maximum_output_tokens=4096,
        output_protocol="json-tool",
    )
    session.step(
        (ToolCall(call_id="call", name="inspect", arguments={"key": "current"}),),
        operation_id="step-1",
    )
    checkpoint = capture_world_checkpoint(
        session, base.world_model, project.artifacts, created_at=now, code_revision="fixture"
    )
    spec = NativeMaterializationSpec(
        checkpoint=artifact_input(project.artifacts.read(checkpoint.artifact_id).manifest),
        binding=binding,
    )
    return project, spec, adapter, session


def test_native_handoff_replays_saved_bytes_and_preserves_candidate(tmp_path: Path) -> None:
    """Native output is published once and never overwrites candidate evidence."""
    project, spec, adapter, session = native_fixture(tmp_path)
    before = session.committed_record()
    assert (
        materialize_native_artifacts(
            project.artifacts, spec, adapter, code_revision="fixture", dry_run=True
        )
        is None
    )
    assert adapter.calls == 0
    first = materialize_native_artifacts(project.artifacts, spec, adapter, code_revision="fixture")
    assert first is not None
    assert (
        materialize_native_artifacts(project.artifacts, spec, adapter, code_revision="fixture")
        == first
    )
    assert adapter.calls == 1
    assert session.committed_record() == before
    export_native_artifacts(project.artifacts, first, tmp_path / "delivered")
    assert (tmp_path / "delivered/output.txt").read_bytes() == b"initial changed"
    report = json.loads(project.artifacts.read_bytes(first.artifact_id, "receipt.json"))
    assert report["status"] == "matched"
    assert report["qualification"] == "diagnostic_only"
    assert report["checkpoint"] == spec.checkpoint.model_dump(mode="json")


def test_divergent_native_output_is_retained_but_not_delivered(tmp_path: Path) -> None:
    """A native helper cannot silently turn a wrong prediction into a passing submission."""
    project, spec, adapter, session = native_fixture(tmp_path, predicted="wrong prediction")
    before = session.state
    reference = materialize_native_artifacts(
        project.artifacts, spec, adapter, code_revision="fixture"
    )
    assert reference is not None
    assert (
        json.loads(project.artifacts.read_bytes(reference.artifact_id, "receipt.json"))["status"]
        == "diverged"
    )
    with pytest.raises(ValueError, match="contradicts"):
        export_native_artifacts(project.artifacts, reference, tmp_path / "delivered")
    assert not (tmp_path / "delivered").exists()
    assert session.state == before


def test_binding_changes_are_refused_before_adapter_execution(tmp_path: Path) -> None:
    """A caller cannot retrofit a different native input into an already predicted task."""
    project, spec, adapter, _ = native_fixture(tmp_path)
    changed = spec.model_copy(
        update={"binding": spec.binding.model_copy(update={"output_paths": ("other.txt",)})}
    )
    with pytest.raises(ValueError, match="not frozen"):
        materialize_native_artifacts(project.artifacts, changed, adapter, code_revision="fixture")
    assert adapter.calls == 0


def test_harbor_delivers_only_native_declared_bytes(tmp_path: Path) -> None:
    """An actual frozen fork exposes native outputs through the shipped standalone client."""

    project, spec, adapter, source = native_fixture(tmp_path)
    saved = load_world_checkpoint(project.artifacts, spec.checkpoint)
    now = datetime.now(UTC)
    start = capture_world_checkpoint(
        source,
        saved.world_model,
        project.artifacts,
        prefix_position=0,
        created_at=now,
        code_revision="fixture",
    )
    start_ref = artifact_input(project.artifacts.read(start.artifact_id).manifest)
    # A public task is constructed from the same registered context and tool contract.
    _, original = source_project(tmp_path / "tool-contract")
    task = TaskCase.model_validate(
        {**original.task.model_dump(mode="json"), "initial_context": saved.public_context}
    )
    task_ref = persist_evaluation_task(
        project.artifacts,
        EvaluationTask(
            intent="Produce output.txt",
            task=task,
            world_model=saved.world_model,
            checkpoint=start_ref,
            verifier=TaskVerifier(
                description="Diagnostic software fixture", outcome_schema={"type": "object"}
            ),
            challenges=(),
        ),
        created_at=now,
        code_revision="fixture",
    )
    registry = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path))
    _, capabilities = registry.snapshot("model")
    runtime = load_grounded_world_model(
        project.artifacts,
        saved.world_model.artifact_id,
        client=UnusedModel(),
        capabilities=capabilities,
    )

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Return the fixture's explicit native plan through a fresh committed fork."""
        return ModelResponse(
            output=AssistantAction(
                tool_calls=(
                    ToolCall(
                        call_id="proposal",
                        name="submit_transition",
                        arguments={
                            "observations": [
                                {"call_id": "call", "reply": "saved", "encoding": "json"}
                            ],
                            "effect": {
                                "kind": "edits",
                                "edits": [
                                    {"op": "set", "path": [key], "value": value}
                                    for key, value in source.state.items()
                                ],
                            },
                        },
                    ),
                )
            ),
            model=runtime.artifact.model,
            economics=OperationEconomics(),
        ).model_dump(mode="json")

    fork = open_grounded_session(
        runtime=runtime,
        task=task,
        store=SessionStore(tmp_path / "fork.sqlite"),
        artifacts=project.artifacts,
        session_id="fork",
        policy=SessionPolicy(
            deadline=now + timedelta(minutes=5),
            maximum_operations=3,
            inference_identity="0" * 64,
            code_revision="fixture",
        ),
        complete=complete,
        lock_path=tmp_path / "fork.lock",
        maximum_output_tokens=4096,
        output_protocol="json-tool",
        checkpoint=start,
    )
    trial = HarborTrial(project.artifacts, task_ref, fork, "fixture", native_adapter=adapter)
    with pytest.raises(ValueError, match="submit"):
        trial.output_files()
    trial.call(ToolCall(call_id="call", name="inspect", arguments={"key": "current"}))
    trial.submit({"completed": True})
    with serve_harbor_trial(trial) as endpoint:
        client = Path(__file__).parent / "harbor" / "client.py"
        for index in range(2):
            destination = tmp_path / f"actor-files-{index}"
            subprocess.run(
                [sys.executable, str(client), "outputs", "--directory", str(destination)],
                env={
                    "EXP_HARBOR_ENDPOINT": endpoint.url,
                    "EXP_HARBOR_ACTOR_TOKEN": endpoint.actor_token,
                },
                check=True,
                capture_output=True,
                timeout=30,
            )
            assert (destination / "output.txt").read_bytes() == b"initial changed"
    assert adapter.calls == 1
    assert trial.result()["qualification"] == "diagnostic_only"
    package = export_harbor_task(
        project, task_ref, tmp_path / "harbor", name="test/native", diagnostic=True
    )
    config = tomllib.loads((package / "task.toml").read_text())
    assert config["artifacts"] == [
        {"source": "/app/outputs/output.txt", "destination": "output.txt"}
    ]
    assert config["metadata"]["diagnostic_only"] is True


def test_changed_adapter_is_refused_even_when_a_saved_receipt_exists(tmp_path: Path) -> None:
    """Replay must still check the frozen adapter rather than trust a completed artifact ID."""
    project, spec, adapter, _ = native_fixture(tmp_path)
    materialize_native_artifacts(project.artifacts, spec, adapter, code_revision="fixture")

    class ChangedAdapter(TextAdapter):
        """A later implementation whose output must never inherit the original receipt."""

        @property
        def definition(self) -> JsonObject:
            """Name a distinct operation contract."""
            return {"adapter": "changed"}

    changed = ChangedAdapter()
    with pytest.raises(ValueError, match="adapter implementation"):
        materialize_native_artifacts(project.artifacts, spec, changed, code_revision="fixture")
    assert changed.calls == 0
