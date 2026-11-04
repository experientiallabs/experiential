"""Saved variance-build artifacts are content-addressed, adopted when identical and pinned."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.common.project import ProjectConfig, ProjectStore
from exp.optimize.evaluation.variance.build_records import (
    EXPOSURE_TYPE,
    MATRIX_TYPE,
    ScopeRule,
    corpus_scope,
    load_exposure_plan,
    load_tag_matrix,
    probe_selection,
    save_artifact,
    trace_admissions,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.simulation.mining.tag_clusters import (
    AdmittedTrace,
    ClusterAdmission,
    ClusterAdmissionSpec,
)
from exp.simulation.mining.tag_selection import TagRule


def _project(root: Path) -> ProjectStore:
    """Initialize an empty project."""
    project = ProjectStore(root, "records")
    project.initialize(ProjectConfig(project_id="records"))
    return project


def _plan(seed: int) -> ExposurePlan:
    """A two-lineage exposure plan."""
    return ExposurePlan(
        bindings=ArtifactInput(artifact_id="bindings", sha256="0" * 64),
        seed=seed,
        probe_fraction=0.5,
        roles={"a": "world_learning", "b": "mining_probes"},
        traces={"t-a": "a", "t-b": "b"},
    )


def test_an_identical_save_adopts_the_saved_artifact(tmp_path: Path) -> None:
    """Saving the same content twice returns one reference and loads it back."""
    project = _project(tmp_path)
    files = {"exposure.json": _plan(0)}

    first = save_artifact(project.artifacts, EXPOSURE_TYPE, files, (), "test")

    assert save_artifact(project.artifacts, EXPOSURE_TYPE, files, (), "test") == first
    assert load_exposure_plan(project, first) == _plan(0)
    other = save_artifact(project.artifacts, EXPOSURE_TYPE, {"exposure.json": _plan(1)}, (), "t")
    assert other != first
    assert len(project.artifacts.list_ids(artifact_type=EXPOSURE_TYPE)) == 2


def test_a_reader_refuses_another_artifact_type(tmp_path: Path) -> None:
    """A reference is read only as the artifact type it was saved as."""
    project = _project(tmp_path)
    saved = save_artifact(project.artifacts, EXPOSURE_TYPE, {"exposure.json": _plan(0)}, (), "t")

    with pytest.raises(ValueError, match=f"pinned {MATRIX_TYPE}"):
        load_tag_matrix(project, saved)


def _admission(traces: tuple[str, ...]) -> ClusterAdmission:
    """An admission of the given traces from one cluster."""
    return ClusterAdmission(
        spec=ClusterAdmissionSpec(),
        clustering="tag-clusters-x",
        admitted=tuple(
            AdmittedTrace(
                trace_id=trace, cluster_id="c00", confidence=0.9, rank=index + 1, reason="c00 why"
            )
            for index, trace in enumerate(traces)
        ),
        quotas=(),
    )


def test_corpora_come_from_a_rule_else_admission_else_the_role() -> None:
    """Admission narrows a corpus; an admitted trace outside world learning is refused."""
    plan = _plan(0).model_copy(
        update={
            "roles": {"a": "world_learning", "b": "mining_probes", "c": "world_learning"},
            "traces": {"t-a": "a", "t-b": "b", "t-c": "c"},
        }
    )
    assert corpus_scope(plan, "construction", None, None).ids == ("t-a", "t-c")
    admitted = corpus_scope(plan, "construction", None, _admission(("t-c",)))
    assert admitted.ids == ("t-c",) and admitted.name == "world-learning-construction"
    with pytest.raises(ValueError, match="not reserved for world_learning"):
        corpus_scope(plan, "retrieval", None, _admission(("t-b",)))
    rule = ScopeRule(rule=TagRule(name="keep", require_any=("q",)))
    rows = trace_admissions(
        (admitted, None), (corpus_scope(plan, "retrieval", None, None), rule), _admission(("t-c",))
    )
    assert [(row.trace_id, row.corpora, row.cluster_id) for row in rows] == [
        ("t-a", ("retrieval",), None),
        ("t-c", ("construction", "retrieval"), "c00"),
    ]
    assert (
        rows[1].reason
        == "construction: c00 why; retrieval: tag rule 'keep' over the task view selected it"
    )


def test_probe_selection_is_empty_without_error_tags() -> None:
    """No error bank means no ranking and no selection."""
    assert probe_selection(_plan(0), (), None, 3) == ((), (), ())
