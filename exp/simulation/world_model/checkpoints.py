"""Immutable checkpoints projected from committed candidate sessions, without native rescue."""

import copy
import json
from collections.abc import Callable
from datetime import datetime
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    sha256_json,
    stable_id,
    strict_equal,
)
from exp.common.evaluations.reconstruction import WorkflowEvidence
from exp.common.project import ArtifactStore, artifact_input
from exp.runtime.sessions.store import SessionRecord, SessionStore
from exp.simulation.world_model.durable import DurableWorldModelSession, SessionSnapshot
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact


class WorldCheckpoint(ArtifactEnvelope):
    """A source-audited prefix of the simulator's own modeled world.

    Attributes:
        artifact_id: Immutable checkpoint artifact identity.
        world_model: Exact knowledge, serving model and retrieval binding.
        source_session: Immutable copy of the authoritative committed session record.
        source_actor: Optional saved actor chronology associated with the session.
        parent_checkpoint: Exact checkpoint used to initialize a forked source session.
        prefix_position: Number of committed transitions retained.
        initial: Source instance before its first committed transition.
        current: Projected source state and history including the retained transition prefix.
        public_context: Original public instance context, never inferred from private state.
        source_origin: Candidate state authority, never native environment truth.
    """

    artifact_id: str
    world_model: ArtifactInput
    source_session: ArtifactInput
    source_actor: ArtifactInput | None = None
    parent_checkpoint: ArtifactInput | None = None
    prefix_position: int = Field(ge=0, strict=True)
    initial: SessionSnapshot
    current: SessionSnapshot
    public_context: JsonObject
    source_origin: Literal["candidate_checkpoint"] = "candidate_checkpoint"


def state_at(record: SessionRecord, prefix_position: int) -> SessionSnapshot:
    """Project one committed prefix while validating the complete source state chain.

    Args:
        record: Detached authoritative runtime session record.
        prefix_position: Number of committed transitions to keep.

    Returns:
        Candidate-owned state and history at that prefix, including source audit transitions.

    Raises:
        ValueError: The position, history, revision or transition continuity is inconsistent.
    """
    snapshot = SessionSnapshot.model_validate(record.state)
    transitions = snapshot.transitions
    if not 0 <= prefix_position <= len(transitions) or record.version != len(transitions):
        raise ValueError("checkpoint prefix is outside the committed session revisions")
    if len(snapshot.history) < len(transitions):
        raise ValueError("committed checkpoint history is incomplete")
    initial_history = snapshot.history[: len(snapshot.history) - len(transitions)]
    state = transitions[0].get("before") if transitions else snapshot.business_state
    if not isinstance(state, dict):
        raise ValueError("committed checkpoint has no initial candidate state")
    states = [state]
    history = list(initial_history)
    histories = [tuple(history)]
    for transition in transitions:
        before, after = transition.get("before"), transition.get("after")
        if not strict_equal(before, state) or not isinstance(after, dict):
            raise ValueError("committed checkpoint transition states are discontinuous")
        history.append(
            {"actions": transition.get("actions"), "observations": transition.get("observations")}
        )
        state = after
        states.append(state)
        histories.append(tuple(history))
    if not strict_equal(state, snapshot.business_state) or not strict_equal(
        history, list(snapshot.history)
    ):
        raise ValueError(
            "checkpoint source state or history differs from its committed transitions"
        )
    return SessionSnapshot(
        business_state=copy.deepcopy(states[prefix_position]),
        history=copy.deepcopy(histories[prefix_position]),
        transitions=copy.deepcopy(transitions[:prefix_position]),
    )


def capture_stored_world_checkpoint(
    session_store: SessionStore,
    session_id: str,
    world_model: ArtifactInput,
    artifacts: ArtifactStore,
    *,
    public_context: JsonObject,
    prefix_position: int | None = None,
    source_actor: ArtifactInput | None = None,
    parent_checkpoint: ArtifactInput | None = None,
    created_at: datetime,
    code_revision: str,
) -> WorldCheckpoint:
    """Capture a completed source session without constructing a provider or engine.

    Args:
        session_store: Runtime owner of the source committed session.
        session_id: Existing source session identity.
        world_model: Exact grounded simulator artifact used by the source.
        artifacts: Owner of immutable operation proposals and the resulting checkpoint.
        public_context: Original actor-independent initialization context.
        prefix_position: Committed prefix to retain; absent captures the current head.
        source_actor: Optional immutable actor evidence reference.
        parent_checkpoint: Required when the source session was initialized from a checkpoint.
        created_at: Aware creation time.
        code_revision: Producing source identity.

    Returns:
        Persisted, source-verified candidate checkpoint. The source store retains the
        checkpointed revisions under the checkpoint's artifact identity until
        ``SessionStore.release_pin`` releases them.
    """

    def proposals(operation_id: str) -> JsonObject | None:
        """Read one operation's authoritative saved proposal."""
        return session_store.proposal(session_id, operation_id)

    checkpoint = _capture_record(
        session_store.read_session(session_id),
        proposals,
        world_model,
        artifacts,
        public_context=public_context,
        prefix_position=prefix_position,
        source_actor=source_actor,
        parent_checkpoint=parent_checkpoint,
        created_at=created_at,
        code_revision=code_revision,
    )
    session_store.pin_revisions(
        checkpoint.artifact_id, session_id, 0, checkpoint.prefix_position + 1
    )
    return checkpoint


def _capture_record(
    record: SessionRecord,
    proposals: Callable[[str], JsonObject | None],
    world_model: ArtifactInput,
    artifacts: ArtifactStore,
    *,
    public_context: JsonObject,
    prefix_position: int | None,
    source_actor: ArtifactInput | None,
    parent_checkpoint: ArtifactInput | None,
    created_at: datetime,
    code_revision: str,
) -> WorldCheckpoint:
    """Verify a committed record against its runtime proposal owner, then persist its prefix."""
    artifact = load_grounded_world_model_artifact(artifacts, world_model)
    if record.binding.knowledge_sha256 != sha256_json(artifact):
        raise ValueError("checkpoint session belongs to a different simulator artifact")
    snapshot = SessionSnapshot.model_validate(record.state)
    position = len(snapshot.transitions) if prefix_position is None else prefix_position
    initial, current = state_at(record, 0), state_at(record, position)
    _validate_initialization(
        artifacts, record, initial, world_model, public_context, parent_checkpoint
    )
    dependencies = {world_model}
    if parent_checkpoint is not None:
        dependencies.add(parent_checkpoint)
    for transition in snapshot.transitions:
        proposal = ArtifactInput.model_validate(transition.get("proposal"))
        stored = artifacts.read(proposal.artifact_id)
        if artifact_input(stored.manifest) != proposal:
            raise ValueError("checkpoint operation proposal manifest digest differs")
        raw = json.loads(artifacts.read_bytes(proposal.artifact_id, "record.json"))
        if not strict_equal(raw.get("proposal"), proposals(str(transition.get("operation_id")))):
            raise ValueError("checkpoint proposal differs from its authoritative runtime record")
        dependencies.add(proposal)
    if source_actor is not None:
        stored_actor = artifacts.read(source_actor.artifact_id)
        if artifact_input(stored_actor.manifest) != source_actor:
            raise ValueError("checkpoint actor evidence manifest digest differs")
        if stored_actor.manifest.artifact_type == "actor-workflow":
            payload = json.loads(artifacts.read_bytes(source_actor.artifact_id, "actor.json"))
        elif stored_actor.manifest.artifact_type == "council-trace":
            payload = json.loads(artifacts.read_bytes(source_actor.artifact_id, "trace.json")).get(
                "workflow"
            )
        else:
            raise ValueError("checkpoint source actor is not a saved actor chronology")
        if not isinstance(payload, dict):
            raise ValueError("checkpoint source actor has no saved chronology")
        evidence = WorkflowEvidence.model_validate(payload.get("evidence"))
        if evidence.workflow_id != record.session_id or len(evidence.steps) < len(
            snapshot.transitions
        ):
            raise ValueError("checkpoint actor chronology belongs to another session")
        for step, transition in zip(
            evidence.steps[: len(snapshot.transitions)], snapshot.transitions, strict=True
        ):
            actions = transition.get("actions")
            if (
                not isinstance(actions, list)
                or len(actions) != 1
                or not isinstance(actions[0], dict)
            ):
                raise ValueError(
                    "checkpoint actor chronology requires single-tool source operations"
                )
            action = {"name": actions[0].get("name"), "arguments": actions[0].get("arguments")}
            if (
                step.status != "valid"
                or not strict_equal(step.action, action)
                or not strict_equal(step.before, transition.get("before"))
                or not strict_equal(step.after, transition.get("after"))
                or not strict_equal([step.reply], transition.get("replies"))
            ):
                raise ValueError(
                    "checkpoint actor evidence differs from committed session transitions"
                )
        dependencies.add(source_actor)
    source_id = stable_id(
        "world-session-source",
        {
            "record": record.model_dump(mode="json"),
            "context": public_context,
            "actor": None if source_actor is None else source_actor.model_dump(mode="json"),
            "parent_checkpoint": None
            if parent_checkpoint is None
            else parent_checkpoint.model_dump(mode="json"),
        },
    )
    if not artifacts.exists(source_id):
        artifacts.write_json(
            artifact_id=source_id,
            artifact_type="world-session-source",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=created_at,
                inputs=tuple(sorted(dependencies, key=lambda ref: ref.artifact_id)),
                code_revision=code_revision,
            ),
            files={
                "record.json": {
                    "session": record.model_dump(mode="json"),
                    "public_context": public_context,
                    "parent_checkpoint": None
                    if parent_checkpoint is None
                    else parent_checkpoint.model_dump(mode="json"),
                }
            },
        )
    source_ref = artifact_input(artifacts.read(source_id).manifest)
    identity = stable_id(
        "world-checkpoint", {"source": source_ref.model_dump(mode="json"), "prefix": position}
    )
    if artifacts.exists(identity):
        return load_world_checkpoint(artifacts, artifact_input(artifacts.read(identity).manifest))
    checkpoint = WorldCheckpoint(
        schema_version=1,
        created_at=created_at,
        inputs=tuple(sorted({source_ref, world_model}, key=lambda ref: ref.artifact_id)),
        code_revision=code_revision,
        artifact_id=identity,
        world_model=world_model,
        source_session=source_ref,
        source_actor=source_actor,
        parent_checkpoint=parent_checkpoint,
        prefix_position=position,
        initial=initial,
        current=current,
        public_context=copy.deepcopy(public_context),
    )
    artifacts.write_json(
        artifact_id=identity,
        artifact_type="world-checkpoint",
        envelope=checkpoint,
        files={"checkpoint.json": checkpoint.model_dump(mode="json")},
    )
    return checkpoint


def capture_world_checkpoint(
    session: DurableWorldModelSession,
    world_model: ArtifactInput,
    artifacts: ArtifactStore,
    *,
    prefix_position: int | None = None,
    source_actor: ArtifactInput | None = None,
    parent_checkpoint: ArtifactInput | None = None,
    created_at: datetime,
    code_revision: str,
) -> WorldCheckpoint:
    """Capture a live session through its authoritative stored-record owner."""
    context = session.evaluation_context.get("task_context", {})
    if not isinstance(context, dict):
        raise ValueError("checkpoint session has no public initialization context")
    checkpoint = _capture_record(
        session.committed_record(),
        session.recorded_proposal,
        world_model,
        artifacts,
        public_context=context,
        prefix_position=prefix_position,
        source_actor=source_actor,
        parent_checkpoint=parent_checkpoint,
        created_at=created_at,
        code_revision=code_revision,
    )
    session.retain_history(checkpoint.artifact_id, checkpoint.prefix_position + 1)
    return checkpoint


def load_world_checkpoint(artifacts: ArtifactStore, reference: ArtifactInput) -> WorldCheckpoint:
    """Load a checksum-verified checkpoint and independently recompute its source projection.

    Args:
        artifacts: Immutable project artifact owner.
        reference: Exact checkpoint manifest reference.

    Returns:
        Validated source-bound candidate checkpoint.
    """
    stored = artifacts.read(reference.artifact_id)
    if (
        artifact_input(stored.manifest) != reference
        or stored.manifest.artifact_type != "world-checkpoint"
    ):
        raise ValueError("checkpoint manifest digest or artifact type differs")
    checkpoint = WorldCheckpoint.model_validate_json(
        artifacts.read_bytes(reference.artifact_id, "checkpoint.json")
    )
    source = artifacts.read(checkpoint.source_session.artifact_id)
    if artifact_input(source.manifest) != checkpoint.source_session:
        raise ValueError("checkpoint source manifest digest differs")
    payload = json.loads(artifacts.read_bytes(checkpoint.source_session.artifact_id, "record.json"))
    record = SessionRecord.model_validate(payload["session"])
    expected_parent = (
        None
        if checkpoint.parent_checkpoint is None
        else checkpoint.parent_checkpoint.model_dump(mode="json")
    )
    if payload.get("parent_checkpoint") != expected_parent:
        raise ValueError("checkpoint parent differs from its committed source")
    _validate_initialization(
        artifacts,
        record,
        state_at(record, 0),
        checkpoint.world_model,
        checkpoint.public_context,
        checkpoint.parent_checkpoint,
    )
    artifact = load_grounded_world_model_artifact(artifacts, checkpoint.world_model)
    if (
        checkpoint.artifact_id != reference.artifact_id
        or record.binding.knowledge_sha256 != sha256_json(artifact)
        or payload["public_context"] != checkpoint.public_context
        or checkpoint.initial != state_at(record, 0)
        or checkpoint.current != state_at(record, checkpoint.prefix_position)
    ):
        raise ValueError("checkpoint differs from its committed source projection")
    return checkpoint


def _validate_initialization(
    artifacts: ArtifactStore,
    record: SessionRecord,
    initial: SessionSnapshot,
    world_model: ArtifactInput,
    public_context: JsonObject,
    parent_checkpoint: ArtifactInput | None,
) -> None:
    """Check the exact initialization receipt, including an independently loaded parent fork."""
    expected = sha256_json(public_context)
    if parent_checkpoint is not None:
        parent = load_world_checkpoint(artifacts, parent_checkpoint)
        if parent.world_model != world_model or parent.public_context != public_context:
            raise ValueError(
                "parent checkpoint world or public context differs from initialization"
            )
        if (
            initial.business_state != parent.current.business_state
            or initial.history != parent.current.history
        ):
            raise ValueError(
                "fork source initial state or history differs from its parent checkpoint"
            )
        expected = sha256_json(
            {
                "public_context": public_context,
                "checkpoint": parent.artifact_id,
                "state": parent.current.model_dump(mode="json"),
            }
        )
    if record.binding.initialization_receipt_sha256 != expected:
        raise ValueError(
            "checkpoint public context or parent differs from the source initialization"
        )
