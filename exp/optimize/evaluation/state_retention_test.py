"""Checkpoint retirement releases session history pins only for unreferenced checkpoints."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject
from exp.common.models import (
    AssistantAction,
    ModelRequest,
    ModelResponse,
    OperationEconomics,
    ToolCall,
    load_model_catalog,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.optimize.evaluation.state_retention import (
    RETIREMENT_TYPE,
    CheckpointRetirement,
    RetirementDecision,
    StateCollection,
    collect_state_garbage,
    retire_checkpoint,
)
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

NOW = datetime.now(UTC)


class _UnusedModel:
    """Reject provider execution: checkpoints and collection never call a model."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Fail any unintended call."""
        raise AssertionError("unexpected model call")


def _checkpointed(tmp_path: Path) -> tuple[ProjectStore, SessionStore, ArtifactInput]:
    """Commit three source operations and checkpoint the first; the pin holds revisions 0-1."""
    project, spec = source_project(tmp_path)
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path))
    _, capabilities = catalog.snapshot("model")
    runtime = load_grounded_world_model(
        project.artifacts,
        spec.world_model.artifact_id,
        client=_UnusedModel(),
        capabilities=capabilities,
    )
    count = [0]

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Return a fixture transition that replaces the whole state each time."""
        del body, identity
        count[0] += 1
        proposal: JsonObject = {
            "observations": [{"call_id": "call", "reply": "ok", "encoding": "json"}],
            "effect": {
                "kind": "edits",
                "edits": [
                    {"op": "set", "path": ["count"], "value": count[0]},
                    {"op": "set", "path": ["rows"], "value": {"n": {"v": count[0]}}},
                ],
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

    store = SessionStore(project.artifacts.project_directory / "source.sqlite")
    source = open_grounded_session(
        runtime=runtime,
        task=spec.task,
        store=store,
        artifacts=project.artifacts,
        session_id="source",
        policy=SessionPolicy(
            deadline=datetime.now(UTC) + timedelta(minutes=5),
            maximum_operations=9,
            inference_identity="0" * 64,
            code_revision="fixture",
        ),
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
        created_at=NOW,
        code_revision="fixture",
    )
    return project, store, artifact_input(project.artifacts.read(checkpoint.artifact_id).manifest)


def _save_referrer(project: ProjectStore, checkpoint: ArtifactInput, *, as_input: bool) -> str:
    """Save a study-like artifact naming the checkpoint as an input or only inside its data."""
    identity = "study-" + ("input" if as_input else "payload")
    project.artifacts.write_json(
        artifact_id=identity,
        artifact_type="continuation-study",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=NOW,
            code_revision="fixture",
            inputs=(checkpoint,) if as_input else (),
        ),
        files={"study.json": {"checkpoint": checkpoint.artifact_id}},
    )
    return identity


def _retire(
    project: ProjectStore, checkpoint: ArtifactInput, *, dry_run: bool = False
) -> RetirementDecision:
    """Retire through the production owner."""
    return retire_checkpoint(
        project,
        CheckpointRetirement(checkpoint=checkpoint, reason="superseded by a newer capture"),
        created_at=NOW,
        code_revision="fixture",
        dry_run=dry_run,
    )


def test_a_dry_run_reports_pins_and_reclaimable_state_and_changes_nothing(
    tmp_path: Path,
) -> None:
    """Both dry runs write no artifact, release no pin and delete no node."""
    project, store, checkpoint = _checkpointed(tmp_path)
    artifacts = project.artifacts.list_ids()
    before = store.preview_collection()
    inventory = collect_state_garbage(project, StateCollection(), dry_run=True)
    (row,) = inventory.stores
    assert row.store == "source.sqlite" and row.report is None
    assert row.preview == before and inventory.reclaimable_nodes == before.reclaimable_nodes > 0
    (pin,) = inventory.pins
    assert (pin.pin.owner_id, pin.pin.start_revision, pin.pin.end_revision) == (
        checkpoint.artifact_id,
        0,
        2,
    )
    assert pin.owner == "retirable" and pin.exclusive_nodes > 0 and pin.referrers == ()
    decision = _retire(project, checkpoint, dry_run=True)
    assert decision.record is None and [row.pin for row in decision.released] == [pin.pin]
    assert store.pins() == (pin.pin,)
    assert store.preview_collection() == before
    assert project.artifacts.list_ids() == artifacts
    assert store.read_state(store.ref_at("source", 2), ("business_state", "count")) == 2


def test_a_retired_checkpoint_releases_its_pin_and_collection_reclaims_its_history(
    tmp_path: Path,
) -> None:
    """Retirement is recorded, the pin goes, and the history it alone held is reclaimed."""
    project, store, checkpoint = _checkpointed(tmp_path)
    first = collect_state_garbage(project, StateCollection(batch_size=3), dry_run=False)
    (row,) = first.stores
    assert row.report is not None and row.report.reclaimed_nodes == row.preview.reclaimable_nodes
    assert store.read_state(store.ref_at("source", 1), ("business_state", "count")) == 1
    held = first.pins[0].exclusive_nodes
    decision = _retire(project, checkpoint)
    assert decision.record is not None and len(decision.released) == 1
    assert project.artifacts.list_ids(artifact_type=RETIREMENT_TYPE) == (
        decision.record.artifact_id,
    )
    assert store.pins() == ()
    second = collect_state_garbage(project, StateCollection(vacuum=True), dry_run=False)
    assert second.pins == ()
    report = second.stores[0].report
    assert report is not None and report.complete and report.reclaimed_nodes == held
    assert second.stores[0].vacuumed_bytes is not None
    with pytest.raises(StateUnavailableError):
        store.read_state(store.ref_at("source", 1))
    assert load_world_checkpoint(project.artifacts, checkpoint).current.business_state["count"] == 1
    again = _retire(project, checkpoint)
    assert again.record == decision.record and again.released == ()


@pytest.mark.parametrize("as_input", [True, False])
def test_a_referenced_checkpoint_refuses_retirement_and_keeps_its_history(
    tmp_path: Path, as_input: bool
) -> None:
    """A saved study naming the checkpoint, in its inputs or only in its data, keeps the pin."""
    project, store, checkpoint = _checkpointed(tmp_path)
    referrer = _save_referrer(project, checkpoint, as_input=as_input)
    with pytest.raises(ValueError, match=f"still referenced by 1 saved artifact.*{referrer}"):
        _retire(project, checkpoint)
    with pytest.raises(ValueError, match="still referenced"):
        _retire(project, checkpoint, dry_run=True)
    assert project.artifacts.list_ids(artifact_type=RETIREMENT_TYPE) == ()
    inventory = collect_state_garbage(project, StateCollection(), dry_run=False)
    (pin,) = inventory.pins
    assert pin.owner == "referenced" and pin.referrers == (referrer,) and pin.referrer_count == 1
    assert store.read_state(store.ref_at("source", 1), ("business_state", "count")) == 1
    with pytest.raises(StateUnavailableError):
        store.read_state(store.ref_at("source", 2))


def test_pins_not_owned_by_a_checkpoint_are_reported_and_never_released(tmp_path: Path) -> None:
    """An unknown owner is listed for the operator; retirement cannot reach it."""
    project, store, checkpoint = _checkpointed(tmp_path)
    store.pin_revisions("frozen-evaluation", "source", 2, 3)
    _retire(project, checkpoint)
    inventory = collect_state_garbage(project, StateCollection(), dry_run=False)
    (pin,) = inventory.pins
    assert (pin.pin.owner_id, pin.owner) == ("frozen-evaluation", "not_a_checkpoint")
    assert store.read_state(store.ref_at("source", 2), ("business_state", "count")) == 2
    other = _save_referrer(project, checkpoint, as_input=False)
    with pytest.raises(ValueError, match="artifact type differs"):
        _retire(project, artifact_input(project.artifacts.read(other).manifest))
