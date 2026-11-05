"""Per-family scoping of a real variance build, with a scripted compatibility reviewer."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.tasks import TaskCase, load_task_set
from exp.common.traces import load_trace_dataset
from exp.optimize.evaluation.variance.build import plan_variance_build, run_variance_build
from exp.optimize.evaluation.variance.build_records import (
    ConstructionSettings,
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
from exp.optimize.evaluation.variance.exposure import ExposurePlan, plan_exposure
from exp.optimize.evaluation.variance.family_scopes import _place
from exp.optimize.evaluation.variance.family_simulators import family_simulator
from exp.optimize.evaluation.variance.family_simulators_test import ref
from exp.simulation.mining.bindings import load_task_set_lineage_bindings
from exp.simulation.mining.environment_families import (
    EnvironmentFamilySpec,
    load_environment_families,
    review_environment_families,
)
from exp.simulation.mining.environment_families_test import (
    answer,
    clustering,
    file_trace,
    order_trace,
    reviewer,
    split,
)
from exp.simulation.mining.environment_review import CompatibilityReviewer, CompatibilitySpec
from exp.simulation.mining.tag_clusters import ClusteringSpec
from exp.simulation.mining.tagging import TagMatrix, TagRow
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
    assert families is not None and planned.review_calls == families.calls(
        len(learning), roots=CLUSTERING.max_clusters
    )
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
    assert {t.task_id for t in shared.tasks} == set(scope.tasks)
    assert all(shared.task(task_id).role == "mining_probes" for task_id in placed)

    _, refusing = reviewer(first_half_compatible, refuse=True)
    replay = run_variance_build(
        project,
        registered,
        code_revision="test",
        tagger=_tagger(project, FakeDecisions()),
        reviewer=refusing,
    )
    assert replay == report
    with pytest.raises(ValueError, match="admitted reviewer"):
        run_variance_build(
            project, registered, code_revision="test", tagger=_tagger(project, FakeDecisions())
        )
    other_script, other = reviewer(lambda members, variant: answer("uncertain", members))
    other = CompatibilityReviewer(
        draw=other.draw, model=other.model.model_copy(update={"model_id": "other-reviewer"})
    )
    changed = run_variance_build(
        project,
        registered,
        code_revision="test",
        tagger=_tagger(project, FakeDecisions()),
        reviewer=other,
    )
    assert other_script.requests and changed.identity != report.identity
    assert changed.environment_families is not None
    assert not changed.environment_families.scopes

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


def construct(body: JsonObject, operation: str) -> JsonObject:
    """Return fixed construction knowledge without contacting a model."""
    del body, operation
    return {
        "interface": {
            "tools": [{"name": "reply", "description": "Observed", "parameters": {}}],
            "state_schema": {"type": "object"},
        },
        "claims": [
            {
                "statement": "Unobserved behavior is unknown",
                "status": "unknown",
                "origin": "model_prior",
                "scope": "unseen calls",
            }
        ],
        "text_rules": "Retain uncertainty.",
    }


def test_a_family_world_is_constructed_from_its_own_members_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With construction, the family's knowledge never cites the excluded uncertain group."""
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
    FIRST[:] = learning[: len(learning) // 2]
    registered = spec().model_copy(update={"construction": ConstructionSettings()})
    _, bound = reviewer(first_half_compatible)

    report = run_variance_build(
        project,
        registered,
        code_revision="test",
        tagger=_tagger(project, FakeDecisions()),
        reviewer=bound,
        constructor=construct,
    )

    assert report.constructed is not None
    pooled = load_grounded_world_model_artifact(project.artifacts, report.constructed)
    assert pooled.knowledge is not None and set(pooled.knowledge.source_trace_ids) == set(learning)
    scoping = report.environment_families
    assert scoping is not None
    (scope,) = scoping.scopes
    world = load_grounded_world_model_artifact(project.artifacts, scope.world_model)
    assert world.knowledge is not None
    assert set(world.knowledge.source_trace_ids) == set(FIRST) == set(scope.members)
    assert world.construction_scope is not None
    assert set(world.construction_scope.ids) == set(FIRST)


def test_a_probe_is_never_placed_in_a_family_whose_contracts_lack_its_tools() -> None:
    """After a split, a filesystem probe stays out of the only admitted (order) child."""
    traces = {"o1": order_trace("o1"), "f1": file_trace("f1")}
    fitted = clustering({"c00": ("o1", "f1")})

    def decide(members: frozenset[str], variant: str) -> JsonObject:
        """Split by environment; the filesystem child stays uncertain."""
        del variant
        if len(members) > 1:
            return answer("split", members, subgroups=split(("o1",), ("f1",)))
        return answer("compatible" if "o1" in members else "uncertain", members)

    _, bound = reviewer(decide)
    record = review_environment_families(fitted, bound, traces=traces, eligible=traces, held_out=())
    probe = file_trace("probe-files")
    task = TaskCase(
        task_id="file-probe",
        lineage_group_id="probe-lineage",
        partition="fit",
        instruction=probe.task,
        tools=probe.tools,
        source_trace_ids=("probe-files",),
        workload_weight=1,
        origin="mined",
    )
    plan = ExposurePlan(
        bindings=ref("bindings"),
        seed=0,
        probe_fraction=0.2,
        roles={"probe-lineage": "mining_probes"},
        traces={"probe-files": "probe-lineage"},
    )
    matrix = TagMatrix(
        view="task",
        bank_id="fixture",
        bank_version=1,
        question_ids=("q0",),
        rows=(TagRow(trace_id="probe-files", unit_hash="fixture", values=(1.0,)),),
    )

    placements, placed = _place(plan, (task,), fitted, (matrix,), record.families)

    assert placements[0].family_id is None and not placed
    assert placements[0].cluster_id == "c00" and "covering" in placements[0].reason


def test_native_quote_counts_registered_candidate_roots_and_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real build planner quotes bounded roots, rather than hypothetical deeper trees."""
    project = variance_project(tmp_path, monkeypatch)
    review = CompatibilitySpec(max_depth=0, max_turns=2, repairs=1)
    baseline = plan_variance_build(project, spec(review=review), code_revision="test")
    count = baseline.roles["world_learning"].traces
    assert count > 1
    assert baseline.review_calls == 11
    bounded = spec(
        review=review.model_copy(
            update={
                "maximum_candidate_members": 1,
                "maximum_candidate_packet_bytes": 48_000,
            }
        )
    )
    planned = plan_variance_build(project, bounded, code_revision="test")
    assert planned.review_calls == count * 11
    assert planned.identity != baseline.identity


def test_native_quote_bounds_uneven_candidates_by_their_available_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full native quote counts one singleton root and at most two roots in the larger shard."""
    project = variance_project(tmp_path, monkeypatch)
    review = CompatibilitySpec(max_depth=0, max_turns=2, repairs=1)
    baseline = plan_variance_build(project, spec(review=review), code_revision="test")
    count = baseline.roles["world_learning"].traces
    assert count > 3
    bounded_review = review.model_copy(
        update={
            "maximum_candidate_members": count - 1,
            "maximum_candidate_packet_bytes": 48_000,
        }
    )
    registered = spec(review=bounded_review).model_copy(
        update={"clustering": ClusteringSpec(min_clusters=1, max_clusters=2)}
    )
    planned = plan_variance_build(project, registered, code_revision="test")
    assert planned.review_calls == 33
    assert planned.review_calls < min(count, 2 * 2) * review.calls_per_group()

    build = project.load_project().build
    assert build is not None
    rows = load_task_set_lineage_bindings(project.artifacts, build.task_set.artifact_id).bindings
    exposure = plan_exposure(
        build.task_set,
        rows,
        seed=registered.seed,
        probe_fraction=registered.probe_fraction,
    )
    traces = {
        trace.trace_id: trace
        for trace in load_trace_dataset(project.artifacts, build.trace_dataset.artifact_id).traces
        if exposure.role_of(trace.trace_id) == "world_learning"
    }
    assert len(traces) == count
    ordered = sorted(traces)
    script, bound = reviewer(lambda members, _variant: answer("compatible", members))
    record = review_environment_families(
        clustering({"c00": tuple(ordered[::2]), "c01": tuple(ordered[1::2])}),
        bound,
        traces=traces,
        eligible=ordered,
        held_out=(),
        spec=bounded_review,
    )
    assert len(record.families) == 3
    assert sorted(member for family in record.families for member in family.members) == ordered
    assert len(script.requests) <= planned.review_calls
