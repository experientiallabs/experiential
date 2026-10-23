"""Verify complete continuation records independently of training admission policy."""

import json

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.project import ArtifactStore, artifact_input
from exp.optimize.evaluation.continuations.critic import require_prediction_binding
from exp.optimize.evaluation.continuations.study import (
    ContinuationBranch,
    ContinuationStudy,
    saved_actor_call_evidence,
    study_identity,
)
from exp.simulation.continuations import load_continuation_checkpoint


def load_study(store: ArtifactStore, reference: ArtifactInput) -> ContinuationStudy:
    """Read a complete result and verify its registered branch inventory and dependencies."""
    manifest = store.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != "continuation-study":
        raise ValueError("study reference differs from its immutable result")
    study = ContinuationStudy.model_validate_json(
        store.read_bytes(reference.artifact_id, "record.json")
    )
    identity = study_identity(study.spec, manifest.code_revision)
    if reference.artifact_id != identity + "-result":
        raise ValueError("study result is not bound to its registered configuration")
    if tuple(row.index for row in study.branches) != tuple(range(study.spec.repeats)):
        raise ValueError("study lost or reordered planned branches; retain every attempt")
    registration = store.read(identity).manifest
    if json.loads(store.read_bytes(identity, "record.json")) != study.spec.model_dump(mode="json"):
        raise ValueError("study configuration differs from its pre-outcome registration")
    expected_registration = {study.spec.checkpoint}
    if study.spec.selection_plan is not None:
        from exp.optimize.evaluation.continuations.selection import require_selection_binding

        expected_registration.add(study.spec.selection_plan)
        require_selection_binding(store, study.spec.selection_plan, study.spec)
    if study.spec.critic_prediction is not None:
        expected_registration.add(study.spec.critic_prediction)
        require_prediction_binding(
            store,
            study.spec.critic_prediction,
            checkpoint=study.spec.checkpoint,
            actor_model=study.spec.actor_model,
            temperature=study.spec.temperature,
            output_tokens=study.spec.maximum_output_tokens,
            reasoning_effort=study.spec.reasoning_effort,
            probability=study.spec.predicted_success,
            verifier_sha256=sha256_json(study.spec.verifier),
            probe_sha256=study.spec.probe_sha256,
        )
    if study.spec.training_specification is not None:
        expected_registration.add(study.spec.training_specification)
    if study.spec.training_evidence is not None:
        expected_registration.update(study.spec.training_evidence.sources())
    if study.spec.admission_policy is not None:
        expected_registration.add(study.spec.admission_policy)
    if set(registration.inputs) != expected_registration:
        raise ValueError("study registration lost its frozen checkpoint or training specification")
    inputs = [artifact_input(registration)]
    for branch in study.branches:
        branch_id = f"{identity}-branch-{branch.index}"
        saved = store.read(branch_id).manifest
        if (
            ContinuationBranch.model_validate_json(store.read_bytes(branch_id, "record.json"))
            != branch
        ):
            raise ValueError("study result differs from its committed branch")
        expected_branch = {artifact_input(registration), *branch.checkpoints, *branch.actor_calls}
        if branch.actor_calls != saved_actor_call_evidence(store, branch_id):
            raise ValueError("study branch lost its retained actor call inventory")
        parent = load_continuation_checkpoint(store, study.spec.checkpoint)
        previous_call = parent.actor.calls
        for point_ref in branch.checkpoints:
            point = load_continuation_checkpoint(store, point_ref)
            intent = artifact_input(store.read(branch_id + "-intent").manifest)
            if (
                point.task != parent.task
                or point.environment_task != parent.environment_task
                or point.actor.runtime_sha256 != parent.actor.runtime_sha256
                or point.actor.calls <= previous_call
                or point.actor.messages[: len(parent.actor.messages)] != parent.actor.messages
                or set(point.sources) != {study.spec.checkpoint, intent}
            ):
                raise ValueError("study checkpoint lost its exact prefix or branch provenance")
            previous_call = point.actor.calls
        if branch.final_world is not None:
            expected_branch.add(branch.final_world)
            if (
                artifact_input(store.read(branch.final_world.artifact_id).manifest)
                != branch.final_world
            ):
                raise ValueError("study branch world image changed")
        if set(saved.inputs) != expected_branch:
            raise ValueError("study branch lost its registration or world image")
        inputs.append(artifact_input(saved))
    if set(manifest.inputs) != set(inputs):
        raise ValueError("study dependencies differ from its complete branch inventory")
    return study
