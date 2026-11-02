"""Tests for completed-build verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import canonical_json_bytes, stable_id
from exp.common.project import ProjectConfig, ProjectStore, artifact_input
from exp.common.project.manifests import file_digest
from exp.common.project.testing import RawArtifact
from exp.common.traces.ingest.otlp import TraceNormalizationResult
from exp.optimize.router.composition_test import (
    _TAGGED_MINING_SPEC,
    _TIME,
    _bind_completed_build,
    _compact_normalized_traces,
)
from exp.optimize.router.errors import RouterCompositionError
from exp.optimize.router.evaluation.build import reconstruct_completed_project_build
from exp.simulation.build import BuildReviewReadiness, select_completed_build
from exp.simulation.mining.bindings import (
    LINEAGE_BINDINGS_PATH,
    TaskSetLineageBindings,
    load_task_set_lineage_bindings,
    task_set_lineage_binding_id,
)
from exp.simulation.mining.features import DecisionTagSelectionSpec
from exp.simulation.mining.service_test import _ProfileTagger
from exp.simulation.retrieval.tests.retrieval_test import _message_trace as _trace


def test_a_tagged_task_set_is_verified_from_storage_without_mining_anything(
    tmp_path: Path,
) -> None:
    """A task set mined by asking a decision model verifies like any other stored task set.

    Verification loads the task set, its coverage and its complete lineage bindings and lets
    their content-addressed identities prove the stored evidence is unaltered. None of that
    reads the selection geometry, so this path needs no decision model, alias or bank root, and
    the build command's default mining route is optimizable. Writing no new artifact is the
    evidence that nothing was derived again: a second mining run would persist its own task set.

    Args:
        tmp_path: Isolated project root for the tagged build.
    """
    project = ProjectStore(tmp_path, "router-a")
    project.initialize(ProjectConfig(project_id="router-a"))
    normalized = _compact_normalized_traces()
    completed = _bind_completed_build(
        project,
        normalized,
        revision="test-revision",
        mining_spec=_TAGGED_MINING_SPEC,
        tagger=_ProfileTagger(),
    )
    stored_ids = set(project.artifacts.list_ids())

    verified = reconstruct_completed_project_build(project, normalized, created_at=_TIME)

    assert isinstance(verified.review.mining_spec.selection, DecisionTagSelectionSpec)
    assert verified.completed == completed
    assert verified.task_set.task_set_id == completed.task_set.artifact_id
    assert verified.trace_dataset.dataset.dataset_id == completed.trace_dataset.artifact_id
    assert verified.tasks
    assert all(task.origin == "mined" for task in verified.tasks)
    assert verified.lineage.task_set_id == completed.task_set.artifact_id
    assert tuple(binding.trace_id for binding in verified.lineage.bindings) == tuple(
        sorted(trace.trace_id for trace in normalized.traces)
    )
    assert set(project.artifacts.list_ids()) == stored_ids


def test_verification_refuses_a_task_set_whose_stored_tasks_no_longer_match_their_binding(
    tmp_path: Path,
) -> None:
    """Editing a stored task record breaks the digest the completed build points at.

    The task set is immutable evidence: its manifest binds every payload digest and the completed
    build binds that manifest. Proving the stored task set is the one that was stored is the whole
    guarantee verification has left, so an altered payload must refuse here and say what to do.

    Args:
        tmp_path: Isolated project root for the build whose task payload is rewritten.
    """
    project = ProjectStore(tmp_path, "router-a")
    project.initialize(ProjectConfig(project_id="router-a"))
    normalized = _compact_normalized_traces()
    completed = _bind_completed_build(
        project,
        normalized,
        revision="test-revision",
        mining_spec=_TAGGED_MINING_SPEC,
        tagger=_ProfileTagger(),
    )
    payload = RawArtifact(project.paths, completed.task_set.artifact_id) / "tasks.jsonl"
    records = payload.read_bytes().decode("utf-8").split("\n")
    first = json.loads(records[0])
    first["instruction"] = f"{first['instruction']} (edited outside the store)"
    payload.write_bytes("\n".join([json.dumps(first), *records[1:]]).encode("utf-8"))

    with pytest.raises(RouterCompositionError, match="no longer load against their stored") as bad:
        reconstruct_completed_project_build(project, normalized, created_at=_TIME)

    assert "rebuild the project" in str(bad.value)


def test_verification_refuses_traces_that_do_not_normalize_to_the_stored_dataset(
    tmp_path: Path,
) -> None:
    """The supplied canonical traces still have to be the build's own evidence.

    Mining is out of verification, but normalization persistence is deterministic and
    provider-free, so the one thing these traces can still prove is that the stored trace dataset
    holds exactly them. The replay runs under the stored dataset's own identity, so a divergent
    corpus refuses instead of landing a second dataset beside the completed build.

    Args:
        tmp_path: Isolated project root for the build and the divergent corpus.
    """
    project = ProjectStore(tmp_path, "router-a")
    project.initialize(ProjectConfig(project_id="router-a"))
    normalized = _compact_normalized_traces()
    _bind_completed_build(
        project,
        normalized,
        revision="test-revision",
        mining_spec=_TAGGED_MINING_SPEC,
        tagger=_ProfileTagger(),
    )
    stored_ids = set(project.artifacts.list_ids())
    divergent = TraceNormalizationResult(
        traces=tuple(_trace(index) for index in range(len(normalized.traces) - 1)),
        issues=(),
    )

    with pytest.raises(RouterCompositionError, match="traces differ from the completed"):
        reconstruct_completed_project_build(project, divergent, created_at=_TIME)

    assert set(project.artifacts.list_ids()) == stored_ids


def test_verification_refuses_a_re_signed_task_set_that_no_longer_binds_its_own_identity(
    tmp_path: Path,
) -> None:
    """A tamper that repairs every digest it broke still fails the content-addressed identity.

    This is the check the whole verification now rests on, so it is pinned against the strongest
    tamper available: one partition assignment is moved, the lineage payload is re-signed with a
    valid binding-set identity, the artifact manifest is re-signed over the new payload, and the
    selected build and its review are re-pointed at the re-signed manifest. Nothing is left
    inconsistent, and the task-set identity still cannot follow from the stored tasks, coverage
    and bindings, so verification refuses.

    Args:
        tmp_path: Isolated project root for the build whose lineage evidence is rewritten.
    """
    project = ProjectStore(tmp_path, "router-a")
    project.initialize(ProjectConfig(project_id="router-a"))
    normalized = _compact_normalized_traces()
    completed = _bind_completed_build(
        project,
        normalized,
        revision="test-revision",
        mining_spec=_TAGGED_MINING_SPEC,
        tagger=_ProfileTagger(),
    )
    lineage = load_task_set_lineage_bindings(project.artifacts, completed.task_set.artifact_id)
    moved = tuple(
        binding.model_copy(
            update={"partition": "held_out" if binding.partition == "fit" else "fit"}
        )
        if index == 0
        else binding
        for index, binding in enumerate(lineage.bindings)
    )
    payload = canonical_json_bytes(
        TaskSetLineageBindings(
            binding_set_id=task_set_lineage_binding_id(
                lineage.task_set_id, lineage.trace_dataset, moved
            ),
            task_set_id=lineage.task_set_id,
            trace_dataset=lineage.trace_dataset,
            bindings=moved,
        )
    )
    raw = RawArtifact(project.paths, completed.task_set.artifact_id)
    manifest = project.artifacts.read(completed.task_set.artifact_id).manifest
    (raw / LINEAGE_BINDINGS_PATH).write_bytes(payload)
    resigned = manifest.model_copy(
        update={
            "files": tuple(
                file_digest(entry.path, payload) if entry.path == LINEAGE_BINDINGS_PATH else entry
                for entry in manifest.files
            )
        }
    )
    (raw / "manifest.json").write_bytes(canonical_json_bytes(resigned))
    task_input = artifact_input(resigned)
    review_state = project.read_review()
    assert isinstance(review_state, dict)
    # Every geometry field of this spec names a choice, so the review identity is the plain
    # content digest of its binding.
    binding = BuildReviewReadiness.model_validate(review_state["build_review"]).model_dump(
        mode="json", exclude={"readiness_id"}
    )
    binding["task_set"] = task_input.model_dump(mode="json")
    select_completed_build(
        project,
        completed.model_copy(update={"task_set": task_input}),
        BuildReviewReadiness.model_validate(
            {**binding, "readiness_id": stable_id("build-review", binding)}
        ),
    )

    with pytest.raises(RouterCompositionError, match="no longer load against their stored") as bad:
        reconstruct_completed_project_build(project, normalized, created_at=_TIME)

    assert "identity does not bind its complete lineage evidence" in str(bad.value)
