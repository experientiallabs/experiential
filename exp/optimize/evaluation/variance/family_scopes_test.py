"""Per-family scoping of a real variance build, with a scripted compatibility reviewer."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.tasks import load_task_set
from exp.optimize.evaluation.variance.build import plan_variance_build, run_variance_build
from exp.optimize.evaluation.variance.build_records import (
    VarianceBuildSpec,
    load_exposure_plan,
    load_tag_retrieval_binding,
)
from exp.optimize.evaluation.variance.build_test import (
    OBSERVATIONS_BANK,
    TASK_BANK,
    FakeDecisions,
    _tagger,
    variance_project,
)
from exp.optimize.evaluation.variance.family_simulators import family_simulator
from exp.simulation.mining.environment_families import (
    EnvironmentFamilySpec,
    load_environment_families,
)
from exp.simulation.mining.environment_families_test import answer, reviewer, split
from exp.simulation.mining.environment_review import CompatibilitySpec
from exp.simulation.mining.tag_clusters import ClusteringSpec
from exp.simulation.retrieval.store import load_rag_index
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact

CLUSTERING = ClusteringSpec(min_clusters=1, max_clusters=1)
"""One cluster over every world-learning trace, so the review decides the families."""


def first_half_compatible(members: frozenset[str], variant: str) -> JsonObject:
    """Split the whole cluster in halves; the first half is compatible, the second uncertain."""
    del variant
    ordered = sorted(members)
    if len(ordered) > len(FIRST) + 1:
        half = len(ordered) // 2
        return answer(
            "split", members, subgroups=split(tuple(ordered[:half]), tuple(ordered[half:]))
        )
    return answer("compatible" if ordered == FIRST else "uncertain", members)


FIRST: list[str] = []


def spec(**families: object) -> VarianceBuildSpec:
    """A build registering one cluster and environment families."""
    return VarianceBuildSpec(
        task_bank=TASK_BANK,
        observations_bank=OBSERVATIONS_BANK,
        clustering=CLUSTERING,
        environment_families=EnvironmentFamilySpec.model_validate(
            {"review": CompatibilitySpec(min_reviews=1), **families}
        ),
    )


def test_each_family_gets_its_own_index_world_and_rule_inputs_and_uncertainty_stays_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A split cluster yields one scoped family and one uncertain group, replayed exactly."""
    project = variance_project(tmp_path, monkeypatch)
    plain = run_variance_build(
        project,
        VarianceBuildSpec(
            task_bank=TASK_BANK, observations_bank=OBSERVATIONS_BANK, clustering=CLUSTERING
        ),
        code_revision="test",
        tagger=_tagger(project, FakeDecisions()),
    )
    learning = sorted(plain.retrieval_scope.ids)
    assert len(learning) >= 4
    FIRST[:] = learning[: len(learning) // 2]
    registered = spec()
    planned = plan_variance_build(project, registered, code_revision="test")
    families = registered.environment_families
    assert families is not None and planned.review_calls == families.calls(len(learning))
    with pytest.raises(ValueError, match="admitted reviewer"):
        run_variance_build(
            project, registered, code_revision="test", tagger=_tagger(project, FakeDecisions())
        )

    script, bound = reviewer(first_half_compatible)
    report = run_variance_build(
        project,
        registered,
        code_revision="test",
        tagger=_tagger(project, FakeDecisions()),
        reviewer=bound,
    )

    scoping = report.environment_families
    assert scoping is not None and script.requests
    (scope,) = scoping.scopes
    assert scope.status == "compatible" and scope.path == "c00/0"
    assert list(scope.members) == FIRST and list(scope.rules_traces) == FIRST
    assert set(scope.retrieval_scope.ids) == set(FIRST)
    index = load_rag_index(project.artifacts, scope.index.artifact_id)
    assert {row.trace_id for row in index.transitions} == set(FIRST)
    assert index.index.corpus_scope == scope.retrieval_scope
    world = load_grounded_world_model_artifact(project.artifacts, scope.world_model)
    assert world.serving_rag == scope.index and scope.world_model != report.world_model
    (uncertain,) = scoping.uncertain
    record = load_environment_families(project.artifacts, scoping.families)
    assert record.uncertain[0].group_id == uncertain
    assert record.uncertain[0].members == tuple(learning[len(learning) // 2 :])

    plan = load_exposure_plan(project, report.exposure)
    build = project.load_project().build
    assert build is not None
    probes = {
        case.task_id
        for case in load_task_set(project.artifacts, build.task_set.artifact_id).tasks
        if plan.roles[case.lineage_group_id] == "mining_probes"
    }
    assert {row.task_id for row in scoping.placements} == probes
    placed = {row.task_id for row in scoping.placements if row.family_id == scope.family_id}
    assert placed <= set(scope.tasks)
    assert all(row.family_id is None for row in scoping.placements if row.task_id not in placed)
    assert placed and scope.tag_retrieval is not None
    assert set(load_tag_retrieval_binding(project, scope.tag_retrieval).tasks) == placed
    shared = family_simulator(record.families[0], scope, scoping.families)
    assert shared.family_id == scope.family_id and shared.index == scope.index

    _, refusing = reviewer(first_half_compatible, refuse=True)
    replay = run_variance_build(
        project,
        registered,
        code_revision="test",
        tagger=_tagger(project, FakeDecisions()),
        reviewer=refusing,
    )
    assert replay == report
    saved_replay = run_variance_build(
        project, registered, code_revision="test", tagger=_tagger(project, FakeDecisions())
    )
    assert saved_replay == report

    admitted = run_variance_build(
        project,
        spec(admit_uncertain=(uncertain,)),
        code_revision="test",
        tagger=_tagger(project, FakeDecisions()),
        reviewer=refusing,
    ).environment_families
    assert admitted is not None and not admitted.uncertain
    assert [s.status for s in admitted.scopes] == ["compatible", "admitted_uncertain"]
    assert admitted.scopes[1].family_id == uncertain
