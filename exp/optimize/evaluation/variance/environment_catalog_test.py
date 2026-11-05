"""Named views and independent base plans over actual native offline build artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import load_task_set
from exp.optimize.evaluation.variance.build import run_variance_build
from exp.optimize.evaluation.variance.build_records import (
    REPORT_TYPE,
    VarianceBuildReport,
    VarianceBuildSpec,
    save_artifact,
)
from exp.optimize.evaluation.variance.build_test import (
    OBSERVATIONS_BANK,
    TASK_BANK,
    FakeDecisions,
    _tagger,
    variance_project,
)
from exp.optimize.evaluation.variance.environment_catalog import (
    EnvironmentBaseSpec,
    EnvironmentCatalogSpec,
    TagTaskSetSpec,
    load_environment_catalog,
    plan_environment_catalog,
    save_environment_catalog,
)
from exp.optimize.evaluation.variance.family_simulators import (
    family_simulator,
    save_family_simulator,
)
from exp.simulation.mining.environment_families import (
    EnvironmentFamilySpec,
    load_environment_families,
)
from exp.simulation.mining.environment_families_test import answer, reviewer, split
from exp.simulation.mining.environment_review import CompatibilitySpec
from exp.simulation.mining.questions import resolve_question_bank
from exp.simulation.mining.tag_clusters import ClusteringSpec
from exp.simulation.mining.tag_selection import TagRule

Built = tuple[
    ProjectStore, ArtifactInput, VarianceBuildReport, EnvironmentCatalogSpec, FakeDecisions
]
"""Actual offline native build, catalog spec and its scripted analysis boundary."""


@pytest.fixture
def built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, ArtifactInput, VarianceBuildReport, EnvironmentCatalogSpec, FakeDecisions]:
    """Build two native bases and overlapping tag views using only scripted local clients."""
    project = variance_project(tmp_path, monkeypatch)
    whole: frozenset[str] | None = None

    def two_bases(members: frozenset[str], variant: str) -> JsonObject:
        """Split the initial cluster and admit both independent children."""
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

    _, bound = reviewer(two_bases)
    decisions = FakeDecisions()
    report = run_variance_build(
        project,
        VarianceBuildSpec(
            task_bank=TASK_BANK,
            observations_bank=OBSERVATIONS_BANK,
            clustering=ClusteringSpec(min_clusters=1, max_clusters=1),
            environment_families=EnvironmentFamilySpec(review=CompatibilitySpec(min_reviews=1)),
        ),
        code_revision="test",
        tagger=_tagger(project, decisions),
        reviewer=bound,
    )
    assert report.environment_families is not None
    scopes = report.environment_families.scopes
    assert len(scopes) == 2
    families = load_environment_families(project.artifacts, report.environment_families.families)
    bases = tuple(
        EnvironmentBaseSpec(
            name=f"base-{index}",
            family_id=scope.family_id,
            simulator=save_family_simulator(
                project.artifacts,
                family_simulator(
                    next(row for row in families.families if row.family_id == scope.family_id),
                    scope,
                    report.environment_families.families,
                ),
                code_revision="test",
            ),
        )
        for index, scope in enumerate(scopes)
    )
    spec = EnvironmentCatalogSpec(
        task_sets=(
            TagTaskSetSpec(rule=TagRule(name="broad", require_all=("shape.q0",), threshold=0.05)),
            TagTaskSetSpec(rule=TagRule(name="focused", require_all=("shape.q1",))),
        ),
        bases=bases,
        banks={"task": resolve_question_bank(TASK_BANK)},
    )
    ref = artifact_input(project.artifacts.read(report.identity + "-result").manifest)
    return project, ref, report, spec, decisions


def test_overlapping_sets_share_unique_pairs_and_independent_native_bases(built: Built) -> None:
    """Views reuse one physical task set and two independently scoped native learning inputs."""
    project, report_ref, report, spec, decisions = built
    before = tuple(decisions.asked)
    catalog = plan_environment_catalog(project, report_ref, spec)
    broad, focused = catalog.task_sets
    assert broad.task_ids and focused.task_ids
    assert set(focused.task_ids) <= set(broad.task_ids)
    assert catalog.task_set == report.build.task_set
    assert catalog.bases[0].scope.index != catalog.bases[1].scope.index
    assert catalog.bases[0].scope.world_model != catalog.bases[1].scope.world_model
    assert set(catalog.bases[0].scope.members).isdisjoint(catalog.bases[1].scope.members)
    assert len(catalog.instances) == 2 * len(broad.task_ids)
    assert len({(row.task_id, row.base_name) for row in catalog.instances}) == len(
        catalog.instances
    )
    for task_id in focused.task_ids:
        rows = [row for row in catalog.instances if row.task_id == task_id]
        assert len(rows) == 2 and all(row.task_sets == ("broad", "focused") for row in rows)
        assert sum(row.native_family_member for row in rows) <= 1
    assert all(row.task_start_qa == "pending" and row.start is None for row in catalog.instances)
    assert any(not row.native_family_member for row in catalog.instances)
    assert not any(base.runtime_bound for base in catalog.bases)
    original = {
        task.task_id: task
        for task in load_task_set(project.artifacts, catalog.task_set.artifact_id).tasks
    }
    for row in catalog.instances:
        assert row.partition == original[row.task_id].partition == "fit"
        assert row.lineage_id == original[row.task_id].lineage_group_id
        assert row.role == "mining_probes"
    assert all(
        catalog.omitted_tasks[t.task_id] == "global final evaluation"
        for t in original.values()
        if t.partition == "held_out"
    )
    saved = save_environment_catalog(project, report_ref, spec, code_revision="test")
    assert load_environment_catalog(project, saved) == catalog
    assert save_environment_catalog(project, report_ref, spec, code_revision="test") == saved
    assert tuple(decisions.asked) == before


def test_exact_bank_text_and_source_pins_are_required(built: Built) -> None:
    """A same-ID edited bank or substituted artifact pin cannot reuse saved tags."""
    project, report_ref, _, spec, _ = built
    bank = spec.banks["task"]
    edited = bank.model_copy(
        update={
            "questions": (
                bank.questions[0].model_copy(update={"text": "Changed?"}),
                *bank.questions[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="question-bank digest"):
        plan_environment_catalog(
            project, report_ref, spec.model_copy(update={"banks": {"task": edited}})
        )
    with pytest.raises(ValueError, match="pinned"):
        plan_environment_catalog(project, report_ref.model_copy(update={"sha256": "0" * 64}), spec)


def test_views_cannot_relabel_heldout_or_duplicate_physical_inputs(built: Built) -> None:
    """Final evaluation has no selectable role, and aliases cannot duplicate plan work."""
    project, report_ref, _, spec, _ = built
    with pytest.raises(ValidationError):
        TagTaskSetSpec.model_validate(
            {"rule": spec.task_sets[0].rule.model_dump(), "role": "final_evaluation"}
        )
    duplicate = spec.bases[0].model_copy(update={"name": "another-name"})
    with pytest.raises(ValueError, match="same physical"):
        plan_environment_catalog(
            project, report_ref, spec.model_copy(update={"bases": (spec.bases[0], duplicate)})
        )
    with pytest.raises(ValueError, match="task-set names"):
        plan_environment_catalog(
            project,
            report_ref,
            spec.model_copy(update={"task_sets": (spec.task_sets[0], spec.task_sets[0])}),
        )


def test_a_scope_cannot_read_heldout_or_another_global_plan(built: Built) -> None:
    """Held-out evidence and mismatched plan scope remain invalid across every view."""
    project, _, report, spec, _ = built
    assert report.environment_families is not None
    scopes = report.environment_families.scopes
    heldout = next(
        task
        for task in load_task_set(project.artifacts, report.build.task_set.artifact_id).tasks
        if task.partition == "held_out"
    )
    for changed, match in (
        (
            scopes[0].model_copy(update={"members": (heldout.source_trace_ids[0],)}),
            "world_learning",
        ),
        (
            scopes[0].model_copy(
                update={
                    "retrieval_scope": scopes[0].retrieval_scope.model_copy(
                        update={"source_plan": "0" * 64}
                    )
                }
            ),
            "another global exposure",
        ),
    ):
        bad = report.model_copy(
            update={
                "environment_families": report.environment_families.model_copy(
                    update={"scopes": (changed, scopes[1])}
                )
            }
        )
        ref = save_artifact(
            project.artifacts, REPORT_TYPE, {"report.json": bad}, (report.build.task_set,), "test"
        )
        with pytest.raises(ValueError, match=match):
            plan_environment_catalog(project, ref, spec)


def test_only_actual_frozen_native_start_receives_passed_qa(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual native bootstrap validation attaches passed QA without making a new request."""
    from exp.optimize.evaluation.variance.bootstrap_plan_test import successful_preparation
    from exp.optimize.evaluation.variance.family_simulators import load_family_simulator

    project, _, prepared, script = successful_preparation(tmp_path, monkeypatch)
    assert prepared.bindings and len(prepared.simulators) == 1
    shared = load_family_simulator(project.artifacts, prepared.simulators[0])
    spec = EnvironmentCatalogSpec(
        task_sets=(
            TagTaskSetSpec(
                rule=TagRule(name="all-tagged-probes", require_all=("shape.q0",), threshold=0.05)
            ),
        ),
        bases=(
            EnvironmentBaseSpec(
                name="native-base",
                family_id=shared.family_id,
                simulator=prepared.simulators[0],
                starts=prepared.bindings,
            ),
        ),
        banks={"task": resolve_question_bank(TASK_BANK)},
    )
    before = len(script.requests)
    catalog = plan_environment_catalog(project, prepared.build_report, spec)
    passed = {row.task_id for row in catalog.instances if row.task_start_qa == "passed"}
    assert passed == {binding.task_id for binding in prepared.bindings}
    assert all(
        row.start is not None and row.start.provenance == "authored"
        for row in catalog.instances
        if row.task_id in passed
    )
    assert all(
        row.state_evidence == "native_start_validated"
        for row in catalog.instances
        if row.task_id in passed
    )
    assert len(script.requests) == before
    bare = spec.bases[0].model_copy(update={"simulator": None})
    with pytest.raises(ValueError, match="pinned runtime-bound simulator"):
        plan_environment_catalog(
            project, prepared.build_report, spec.model_copy(update={"bases": (bare,)})
        )


def test_missing_analysis_remains_diagnostic_without_tagging_again(built: Built) -> None:
    """An observation-only learning matrix cannot silently supply probe task tags."""
    project, report_ref, _, spec, decisions = built
    observation_spec = TagTaskSetSpec(
        view="observations",
        rule=TagRule(
            name="unobserved-probes",
            require_all=("obs.q0",),
            threshold=0.05,
        ),
    )
    expanded = spec.model_copy(
        update={
            "task_sets": (*spec.task_sets, observation_spec),
            "banks": {**spec.banks, "observations": resolve_question_bank(OBSERVATIONS_BANK)},
        }
    )
    before = tuple(decisions.asked)
    catalog = plan_environment_catalog(project, report_ref, expanded)
    assert catalog.task_sets[-1].task_ids == ()
    assert set(catalog.task_sets[-1].missing_tag_tasks) == set(catalog.task_sets[0].task_ids)
    assert all("unobserved-probes" not in row.task_sets for row in catalog.instances)
    assert tuple(decisions.asked) == before


def test_observed_tool_declarations_are_only_schema_diagnostics() -> None:
    """Actual trace-parsed tool contracts distinguish absent, unknown and unequal schemas."""
    from exp.common.tasks import TaskCase
    from exp.simulation.mining.environment_families_test import file_trace
    from exp.simulation.rules.contracts import parse_tool_contracts, tool_schema_coverage

    trace = file_trace("source")
    task = TaskCase(
        task_id="task-file",
        lineage_group_id="lineage-file",
        partition="fit",
        instruction=trace.task,
        tools=trace.tools,
        workload_weight=1.0,
        source_trace_ids=(trace.trace_id,),
    )
    contracts = parse_tool_contracts((trace,))
    assert tool_schema_coverage(task.tools, contracts).status == "schema_matches"
    assert tool_schema_coverage(task.tools, {}).status == "missing_tools"
    unknown = {
        name: contract.model_copy(update={"declared_schema": None})
        for name, contract in contracts.items()
    }
    assert tool_schema_coverage(task.tools, unknown).status == "schema_unknown"
    changed = {
        **contracts,
        "read_file": contracts["read_file"].model_copy(
            update={"declared_schema": {"type": "array"}}
        ),
    }
    assert tool_schema_coverage(task.tools, changed).status == "schema_mismatch"
