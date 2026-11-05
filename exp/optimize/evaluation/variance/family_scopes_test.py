"""Per-family scoping of a real variance build, with a scripted compatibility reviewer."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase, load_task_set
from exp.common.traces import load_trace_dataset
from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapPreparationSpec
from exp.optimize.evaluation.variance.build import plan_variance_build, run_variance_build
from exp.optimize.evaluation.variance.build_records import (
    ConstructionSettings,
    VarianceBuildReport,
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


@pytest.fixture
def multi_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ProjectStore, BootstrapPreparationSpec, VarianceBuildReport, VarianceBuildReport]:
    """Build two reviewed bases twice, reusing all canonical tag and review decisions."""
    from dataclasses import replace

    from exp.common.core.artifacts import sha256_json
    from exp.optimize.evaluation.variance.bootstrap_selection_test import preparation_fixture

    project, preparation, _ = preparation_fixture(tmp_path, monkeypatch)
    whole: frozenset[str] | None = None

    def two_bases(members: frozenset[str], variant: str) -> JsonObject:
        """Split one source cluster into two independent compatible child bases."""
        nonlocal whole
        del variant
        if whole is None:
            whole = members
        if members == whole:
            ordered = sorted(members)
            half = len(ordered) // 2
            return answer(
                "split", members, subgroups=split(tuple(ordered[:half]), tuple(ordered[half:]))
            )
        return answer("compatible", members)

    script, bound = reviewer(two_bases)
    bound = replace(bound, configuration_sha256=sha256_json(preparation.build.reviewer))
    decisions = FakeDecisions()
    legacy = run_variance_build(
        project,
        preparation.build.spec,
        code_revision="test",
        tagger=_tagger(project, decisions),
        reviewer=bound,
    )
    settings = preparation.build.spec.environment_families
    assert settings is not None
    enabled = preparation.build.spec.model_copy(
        update={
            "environment_families": settings.model_copy(
                update={"probe_placement": "all_exact_schemas"}
            )
        }
    )
    before = len(decisions.asked), len(script.requests)
    report = run_variance_build(
        project, enabled, code_revision="test", tagger=_tagger(project, decisions), reviewer=bound
    )
    assert (len(decisions.asked), len(script.requests)) == before
    preparation = preparation.model_copy(
        update={"build": preparation.build.model_copy(update={"spec": enabled})}
    )
    return project, preparation, legacy, report


def test_many_base_placement_keeps_one_canonical_task_in_each_qualified_base(
    multi_base: tuple[
        ProjectStore, BootstrapPreparationSpec, VarianceBuildReport, VarianceBuildReport
    ],
) -> None:
    """Opt-in expands exact-schema pairs while legacy ambiguity remains an honest refusal."""
    from exp.common.core.artifacts import sha256_json
    from exp.optimize.evaluation.variance.family_scopes import probe_placement_protocol_sha256

    project, preparation, legacy, report = multi_base
    old, current = legacy.environment_families, report.environment_families
    assert old is not None and current is not None
    assert len(current.scopes) == 2
    assert report.exposure == legacy.exposure and report.build.task_set == legacy.build.task_set
    assert old.placement is None and "placement" not in old.model_dump(mode="json")
    assert legacy.spec.environment_families is not None
    assert "probe_placement" not in legacy.spec.environment_families.model_dump(mode="json")
    assert not any(row.family_id for row in old.placements)
    assert current.placement is not None and current.placement.mode == "all_exact_schemas"
    assert current.placement.protocol_sha256 == probe_placement_protocol_sha256()
    accepted = [row for row in current.placements if row.family_id is not None]
    grouped = current.placements_by_task()
    assert grouped and all(len(families) == 2 for families in grouped.values())
    assert len(accepted) == 2 * len(grouped)
    assert len({(row.task_id, row.family_id) for row in accepted}) == len(accepted)
    assert all(
        row.tool_coverage is not None and row.tool_coverage.status == "schema_matches"
        for row in accepted
    )
    canonical = {
        task.task_id: task
        for task in load_task_set(project.artifacts, report.build.task_set.artifact_id).tasks
    }
    plan = load_exposure_plan(project, report.exposure)
    record = load_environment_families(project.artifacts, current.families)
    for family in record.families:
        scope = current.scope(family.family_id)
        plan.require(scope.members, "world_learning")
        plan.require(scope.rules_traces, "world_learning")
        for task in family.tasks:
            original = canonical[task.task_id]
            assert task.lineage_id == original.lineage_group_id
            assert task.source_trace_ids == original.source_trace_ids
            assert task.start.initial_context_sha256 == sha256_json(original.initial_context)
            assert task.start.checkpoint is None
            assert task.role == plan.roles[original.lineage_group_id]
            assert original.partition == "fit"
    first, second = current.scopes
    assert first.index != second.index and first.world_model != second.world_model
    assert set(first.members).isdisjoint(second.members)
    assert preparation.build.spec.environment_families is not None


@pytest.mark.parametrize("status", ["schema_unknown", "schema_mismatch", "missing_tools"])
def test_many_base_schema_rejections_remain_candidate_diagnostics(
    multi_base: tuple[
        ProjectStore, BootstrapPreparationSpec, VarianceBuildReport, VarianceBuildReport
    ],
    status: str,
) -> None:
    """Unknown or unequal declarations never attach the rejected candidate to the task."""
    from exp.optimize.evaluation.variance.build_records import load_tag_matrix
    from exp.simulation.mining.tag_clusters import load_tag_clustering
    from exp.simulation.rules.contracts import parse_tool_contracts

    project, _, _, report = multi_base
    scoping = report.environment_families
    assert scoping is not None and report.clustering is not None
    record = load_environment_families(project.artifacts, scoping.families)
    native = file_trace("schema-evidence")
    contracts = parse_tool_contracts((native,))
    task_id = next(iter(scoping.placements_by_task()))
    task = next(
        t
        for t in load_task_set(project.artifacts, report.build.task_set.artifact_id).tasks
        if t.task_id == task_id
    )
    task = task.model_copy(update={"tools": native.tools})
    changed = (
        {}
        if status == "missing_tools"
        else {
            name: contract.model_copy(
                update={
                    "declared_schema": None if status == "schema_unknown" else {"type": "array"}
                }
            )
            for name, contract in contracts.items()
        }
    )
    good = record.families[0].model_copy(update={"tool_contracts": contracts})
    bad = record.families[1].model_copy(update={"tool_contracts": changed})
    rows, placed = _place(
        load_exposure_plan(project, report.exposure),
        (task,),
        load_tag_clustering(project.artifacts, report.clustering),
        [load_tag_matrix(project, view.matrix) for view in report.tagging],
        (good, bad),
        policy="all_exact_schemas",
    )
    assert len(rows) == 2 and set(placed) == {good.family_id}
    rejected = next(row for row in rows if row.family_id is None)
    assert rejected.candidate_family_id == bad.family_id
    assert rejected.tool_coverage is not None and rejected.tool_coverage.status == status
    assert rejected.tool_coverage.issues
    assert all(t.start.checkpoint is None for tasks in placed.values() for t in tasks)


def test_many_base_selection_counts_starts_and_refuses_partial_base_groups(
    multi_base: tuple[
        ProjectStore, BootstrapPreparationSpec, VarianceBuildReport, VarianceBuildReport
    ],
) -> None:
    """A canonical task expands to both bases, with distinct native identities and exact quotes."""
    from exp.optimize.evaluation.variance.bootstrap import task_identity
    from exp.optimize.evaluation.variance.bootstrap_plan import plan_bootstrap_preparation
    from exp.optimize.evaluation.variance.bootstrap_selection import (
        select_bootstrap_families,
        select_probe_task_bases,
    )

    project, preparation, _, report = multi_base
    assert report.environment_families is not None
    task_id = next(iter(report.environment_families.placements_by_task()))
    exact = preparation.model_copy(update={"task_ids": (task_id,)})
    rows = select_bootstrap_families(
        project, report, exact.bootstrap, exact.limits, task_ids=exact.task_ids
    )
    assert len(rows) == 2
    assert all(len(row.spec.families[0].tasks) == 1 for row in rows)
    assert {row.spec.families[0].tasks[0].task.task_id for row in rows} == {task_id}
    identities = {
        task_identity(row.spec, row.spec.families[0], row.spec.families[0].tasks[0], "test")
        for row in rows
    }
    assert len(identities) == 2
    assert all(row.simulator.runtime is None for row in rows)
    with pytest.raises(ValueError, match="whole task/base groups exceed"):
        select_bootstrap_families(
            project,
            report,
            exact.bootstrap,
            exact.limits.model_copy(update={"maximum_tasks": 1}),
            task_ids=exact.task_ids,
        )
    automatic = select_probe_task_bases(
        report, exact.limits.model_copy(update={"maximum_tasks": 3})
    )
    assert len(automatic.task_ids) == 1 and len(automatic.task_bases()) == 2
    assert automatic.omissions and any(
        "capacity" in reason for reason in automatic.omissions.values()
    )
    plan = plan_bootstrap_preparation(project, exact, code_revision="test")
    assert plan.task_bound == 2 and plan.remaining_tasks == (task_id,)
    assert plan.task_base_progress is not None
    assert plan.task_base_progress.canonical_task_bound == 1
    assert plan.task_base_progress.task_base_start_bound == 2
    bad_registration = report.environment_families.placement
    assert bad_registration is not None
    malformed = report.model_copy(
        update={
            "environment_families": report.environment_families.model_copy(
                update={
                    "placement": bad_registration.model_copy(update={"protocol_sha256": "0" * 64})
                },
            )
        }
    )
    with pytest.raises(ValueError, match="native policy or protocol"):
        select_probe_task_bases(malformed, exact.limits, task_ids=exact.task_ids)
    assert len(plan.task_base_progress.remaining) == 2
    assert {key.family_id for key in plan.task_base_progress.remaining} == {
        row.scope.family_id for row in rows
    }
    unsaved = exact.model_copy(
        update={
            "build": exact.build.model_copy(
                update={"spec": exact.build.spec.model_copy(update={"probe_limit": 987})}
            )
        }
    )
    quoted = plan_bootstrap_preparation(project, unsaved, code_revision="test")
    assert not quoted.build_saved and quoted.task_bound == 2
    assert quoted.task_base_progress is not None
    assert quoted.task_base_progress.canonical_task_bound == 1
    assert quoted.task_base_progress.task_base_start_bound == 2


def test_many_base_progress_does_not_hide_a_pending_base_behind_one_finished_task(
    multi_base: tuple[
        ProjectStore, BootstrapPreparationSpec, VarianceBuildReport, VarianceBuildReport
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed completed refusals are counted per base, never as passed task-start QA."""
    from exp.optimize.evaluation.variance import bootstrap_plan as owner
    from exp.optimize.evaluation.variance.bootstrap import task_identity
    from exp.optimize.evaluation.variance.bootstrap_contracts import BootstrapRefusal
    from exp.optimize.evaluation.variance.bootstrap_selection import select_bootstrap_families

    project, preparation, _, report = multi_base
    assert report.environment_families is not None
    task_id = next(iter(report.environment_families.placements_by_task()))
    spec = preparation.model_copy(update={"task_ids": (task_id,)})
    selected = select_bootstrap_families(
        project, report, spec.bootstrap, spec.limits, task_ids=spec.task_ids
    )
    outcomes: dict[str, BootstrapRefusal] = {}
    for row in selected:
        family = row.spec.families[0]
        identity = task_identity(row.spec, family, family.tasks[0], "test")
        outcomes[identity] = BootstrapRefusal(
            family_id=row.scope.family_id,
            task_id=task_id,
            identity=identity,
            kind="author_unusable",
            diagnosis="Typed refusal fixture, never a validated start.",
            rounds=(),
        )
    first = next(iter(outcomes))

    def completed_first(_project: ProjectStore, identity: str) -> BootstrapRefusal | None:
        """Expose one completed diagnostic outcome at the native outcome-reader boundary."""
        return outcomes.get(identity) if identity == first else None

    monkeypatch.setattr(owner, "load_outcome", completed_first)
    partial = owner.plan_bootstrap_preparation(project, spec, code_revision="test")
    assert partial.completed_tasks == () and partial.remaining_tasks == (task_id,)
    assert not partial.complete and partial.task_bound == 1
    assert partial.task_base_progress is not None
    assert (
        len(partial.task_base_progress.completed) == len(partial.task_base_progress.remaining) == 1
    )
    assert (
        partial.task_base_progress.completed[0].family_id
        != partial.task_base_progress.remaining[0].family_id
    )

    def completed_both(_project: ProjectStore, identity: str) -> BootstrapRefusal | None:
        """Expose both complete refusals, still without creating any runnable start."""
        return outcomes.get(identity)

    monkeypatch.setattr(owner, "load_outcome", completed_both)
    complete = owner.plan_bootstrap_preparation(project, spec, code_revision="test")
    assert complete.completed_tasks == (task_id,) and complete.remaining_tasks == ()
    assert complete.complete and complete.task_bound == 0
    assert (
        complete.task_base_progress is not None and len(complete.task_base_progress.completed) == 2
    )
    assert not complete.task_base_progress.remaining


def test_native_build_plan_and_scoping_use_the_registered_finite_review_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The native build quotes and dispatches only the registered roots and saves the queue."""
    from exp.simulation.mining.environment_families import EnvironmentReviewWindow

    project = variance_project(tmp_path, monkeypatch)
    registered = spec(
        review=CompatibilitySpec(
            variants=("contract_first",),
            min_reviews=1,
            max_turns=1,
            repairs=0,
            max_depth=0,
            maximum_candidate_members=1,
            maximum_candidate_packet_bytes=48_000,
            maximum_parallel_groups=2,
        ),
        review_window=EnvironmentReviewWindow(start=1, count=2),
    )
    script, bound = reviewer(lambda members, _variant: answer("compatible", members))
    plan = plan_variance_build(
        project,
        registered,
        code_revision="test",
        reviewer_model=bound.model,
        reviewer_configuration_sha256=bound.configuration_sha256,
    )
    assert plan.review_calls == 4
    report = run_variance_build(
        project,
        registered,
        code_revision="test",
        tagger=_tagger(project, FakeDecisions()),
        reviewer=bound,
    )
    assert report.identity == plan.identity
    assert len(script.requests) == plan.review_calls
    scoping = report.environment_families
    assert scoping is not None
    record = load_environment_families(project.artifacts, scoping.families)
    assert registered.environment_families is not None
    assert record.review_window == registered.environment_families.review_window
    assert len(record.families) == len(scoping.scopes) == 2
    assert record.queued
    assert not record.uncertain
    actual = [member for row in (*record.families, *record.queued) for member in row.members]
    assert sorted(actual) == sorted(report.retrieval_scope.ids)
    assert len(actual) == len(set(actual))
