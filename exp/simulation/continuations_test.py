"""Pairing and immutable source checks for actor/world continuation capsules."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, sha256_json
from exp.common.models import AssistantAction
from exp.common.project import ArtifactManifest, ArtifactStore, ProjectPaths, artifact_input
from exp.runtime.agents.chat import ChatAgentRuntime, ChatCheckpoint
from exp.runtime.agents.chat_test import _Model, _response
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)


def capsule(store: ArtifactStore) -> tuple[ContinuationCheckpoint, ChatAgentRuntime]:
    """Capture a real XLSX world at the actor's initial turn boundary."""
    source = store.write_json(
        artifact_id="authored-source",
        artifact_type="authored-task",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={"task.json": {"origin": "authored conformance task, not benchmark evidence"}},
    )
    case = workbook_case()
    actor = ChatAgentRuntime(maximum_model_calls=4)
    checkpoints: list[ContinuationCheckpoint] = []
    with case.runtime.open(case.task) as session:

        def capture(checkpoint: ChatCheckpoint) -> None:
            """Capture both halves synchronously at a completed actor turn."""
            checkpoints.append(
                ContinuationCheckpoint(
                    task=case.task,
                    actor=checkpoint,
                    world=session.checkpoint(),
                    sources=(artifact_input(source),),
                )
            )

        actor.run(
            case.task,
            model=_Model((_response(AssistantAction(content="stop")),)),
            environment=session,
            on_checkpoint=capture,
        )
    return checkpoints[0], actor


def test_checkpoint_round_trip_and_foreign_task_refusal(tmp_path: Path) -> None:
    """A saved capsule binds private world bytes and the public actor prefix separately."""
    store = ArtifactStore(ProjectPaths(tmp_path, "case"))
    checkpoint, _ = capsule(store)
    ref = persist_continuation_checkpoint(
        store, checkpoint, created_at=datetime.now(UTC), code_revision="test"
    )
    assert load_continuation_checkpoint(store, ref) == checkpoint
    assert (
        persist_continuation_checkpoint(
            store, checkpoint, created_at=datetime.now(UTC), code_revision="test"
        )
        == ref
    )
    assert checkpoint.actor.messages[0].role == "user"
    assert "base64" not in checkpoint.actor.messages[0].model_dump_json()
    value = checkpoint.model_dump(mode="json")
    value["task"]["instruction"] = "another task"
    with pytest.raises(ValueError, match="another public task"):
        ContinuationCheckpoint.model_validate(value)


def test_private_answers_never_enter_a_continuation_capsule(tmp_path: Path) -> None:
    """Reference answers cannot be smuggled through a correctly rehashed public task."""
    checkpoint, _ = capsule(ArtifactStore(ProjectPaths(tmp_path, "case")))
    task = checkpoint.task.model_copy(update={"reference_answer": "private answer"})
    with pytest.raises(ValueError, match="private answers"):
        ContinuationCheckpoint(
            task=task,
            actor=checkpoint.actor.model_copy(update={"task_sha256": sha256_json(task)}),
            world=checkpoint.world,
            sources=checkpoint.sources,
        )


@pytest.mark.parametrize("raced_type", ["executable-world-image", "continuation-checkpoint"])
def test_publication_adopts_identical_concurrent_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raced_type: str
) -> None:
    """A second publisher winning after the existence check must not fail a valid capture."""
    store = ArtifactStore(ProjectPaths(tmp_path, "case"))
    checkpoint, _ = capsule(store)
    write = store.write
    raced = False

    def write_after_winner(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, bytes],
    ) -> ArtifactManifest:
        """Publish the competing writer immediately before this writer's atomic insert."""
        nonlocal raced
        if artifact_type == raced_type and not raced:
            raced = True
            write(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                envelope=envelope,
                files=files,
            )
        return write(
            artifact_id=artifact_id, artifact_type=artifact_type, envelope=envelope, files=files
        )

    monkeypatch.setattr(store, "write", write_after_winner)
    now = datetime.now(UTC)
    reference = persist_continuation_checkpoint(
        store, checkpoint, created_at=now, code_revision="test"
    )
    assert raced
    assert load_continuation_checkpoint(store, reference) == checkpoint
    assert (
        persist_continuation_checkpoint(
            store, checkpoint, created_at=now + timedelta(seconds=1), code_revision="test"
        )
        == reference
    )


def test_distinct_environment_task_cannot_change_tools_or_source_partition(tmp_path: Path) -> None:
    """Separating actor wording from restore inputs cannot hide reserved data or extra tools."""
    checkpoint, actor = capsule(ArtifactStore(ProjectPaths(tmp_path, "case")))
    task = checkpoint.task.model_copy(
        update={"task_id": "variant", "instruction": "A new wording."}
    )
    value = ContinuationCheckpoint(
        task=task,
        environment_task=checkpoint.task,
        actor=actor.initial_checkpoint(task),
        world=checkpoint.world,
        sources=checkpoint.sources,
    )
    assert value.world_task == checkpoint.task
    for changed in (
        {"partition": "held_out"},
        {"tools": ()},
        {"source_trace_ids": ("secret",), "origin": "mined"},
    ):
        raw = value.model_dump(mode="json")
        raw["environment_task"].update(changed)
        with pytest.raises(ValueError, match="share tools and source partition"):
            ContinuationCheckpoint.model_validate(raw)


def test_reworded_actor_restores_mixed_world_under_its_original_task(tmp_path: Path) -> None:
    """A mixed checkpoint's original task identity survives a new actor prompt and artifact ID."""
    from exp.runtime.environments.mixed_test import _order, _task, _world_runtime

    store = ArtifactStore(ProjectPaths(tmp_path, "case"))
    source, _ = capsule(store)
    original = _task("place_order", "refund").model_copy(update={"reference_answer": None})
    actor_task = original.model_copy(
        update={"task_id": "drill-task", "instruction": "Place an order."}
    )
    actor = ChatAgentRuntime(maximum_model_calls=4)
    world = _world_runtime([])
    with world.open(original) as session:
        image = session.checkpoint()
    checkpoint = ContinuationCheckpoint(
        task=actor_task,
        environment_task=original,
        actor=actor.initial_checkpoint(actor_task),
        world=image,
        sources=source.sources,
    )
    ref = persist_continuation_checkpoint(
        store, checkpoint, created_at=datetime.now(UTC), code_revision="test"
    )
    saved = load_continuation_checkpoint(store, ref)
    with world.restore(saved.world_task, image) as session:
        reply = session.execute(_order("o-variant"))
        assert reply.metadata["authority"] == "native"
        assert session.checkpoint().sequence == image.sequence + 1
