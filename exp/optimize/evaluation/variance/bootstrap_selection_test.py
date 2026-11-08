"""Selected build tasks retain exact family worlds and frozen knowledge during preparation."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.cli.build.app_test import _catalog, _RuntimeCatalog
from exp.common.core.artifacts import JsonObject, sha256_bytes, sha256_json
from exp.common.models import (
    ConnectionConfig,
    ModelCapabilities,
    ModelSnapshot,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable import grounded
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.stage import RulesStageSpec
from exp.optimize.evaluation.variance.bootstrap_contracts import (
    BootstrapSettings,
    FamilyInput,
    JobTemplate,
    RuntimeTemplate,
)
from exp.optimize.evaluation.variance.bootstrap_plan import (
    BootstrapPreparationSpec,
    plan_bootstrap_preparation,
)
from exp.optimize.evaluation.variance.bootstrap_selection import (
    BootstrapLimits,
    select_bootstrap_families,
)
from exp.optimize.evaluation.variance.build import run_variance_build
from exp.optimize.evaluation.variance.build_records import VarianceBuildReport
from exp.optimize.evaluation.variance.build_test import FakeDecisions, _tagger, variance_project
from exp.optimize.evaluation.variance.family_scopes_test import spec as family_spec
from exp.optimize.evaluation.variance.family_simulators import save_family_simulator
from exp.optimize.evaluation.variance.pipeline_spec import BuildSettings
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.mining.environment_families_test import answer, reviewer
from exp.simulation.mining.environment_review import CompatibilityReviewer
from exp.simulation.mining.tag_clusters import ClusterAdmissionSpec, ClusteringSpec


def compatible(members: frozenset[str], variant: str) -> JsonObject:
    """Declare one fixture environment compatible with source-cited semantics."""
    del variant
    first = sorted(members)[0]
    return answer(
        "compatible",
        members,
        state_domains=[
            {
                "name": "tickets",
                "description": "authored ticket state",
                "evidence": [f"trace:{first}"],
            }
        ],
    )


def preparation_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, BootstrapPreparationSpec, CompatibilityReviewer]:
    """Build real evidence and freeze provider-free preparation settings with tiny budgets."""

    def exact_snapshot(
        runtime: _RuntimeCatalog, alias: str
    ) -> tuple[ModelSnapshot, ModelCapabilities]:
        """Keep the fake embedding wire but use the real catalog snapshot contract."""
        return RuntimeModelCatalog(runtime._catalog).snapshot(alias)

    def exact_catalog(root: Path) -> None:
        """Declare fixture author and predictor capabilities before evidence freezes them."""
        _catalog(root)
        path = root / "models.toml"
        catalog = load_model_catalog(path)
        world = catalog.models["world"].model_copy(
            update={
                "capabilities": ModelCapabilities(
                    supports_tools=True,
                    supports_structured_output=True,
                    context_window_tokens=200000,
                    maximum_output_tokens=16000,
                    input_cost_per_million_tokens_usd=0.0,
                    output_cost_per_million_tokens_usd=0.0,
                )
            }
        )
        write_model_catalog(
            path,
            catalog.model_copy(
                update={
                    "models": {
                        **catalog.models,
                        "world": world,
                        "judge": catalog.models["judge"].model_copy(
                            update={"capabilities": world.capabilities}
                        ),
                    },
                    "connections": {
                        "fixture": ConnectionConfig(
                            provider="openai-compatible",
                            base_url="https://fixture.invalid/v1",
                            api_key_env="FIXTURE_API_KEY",
                        )
                    },
                }
            ),
        )

    monkeypatch.setattr("exp.optimize.evaluation.variance.build_test._catalog", exact_catalog)
    monkeypatch.setenv("FIXTURE_API_KEY", "fixture-only")
    monkeypatch.setattr(_RuntimeCatalog, "snapshot", exact_snapshot)
    project = variance_project(tmp_path, monkeypatch)
    catalog = load_model_catalog(project.model_catalog_path)
    model, _ = RuntimeModelCatalog(catalog, environment={}).snapshot("world")
    digest = sha256_json(catalog.model_dump(mode="json"))
    author = AuthorSettings(
        model_alias="world",
        model=model,
        catalog_sha256=digest,
        maximum_physical_attempts=1,
        deadline=datetime(2099, 1, 1, tzinfo=UTC),
    )
    _, bound = reviewer(compatible)
    build = project.load_project().build
    assert build is not None
    runtime_file = Path(grounded.__file__)
    settings = BootstrapSettings(
        author=author,
        rules=RulesStageSpec(author=author, profile="single_draw"),
        research_rounds=1,
        research_items=2,
        runtime=RuntimeTemplate(
            runtime_file=runtime_file,
            runtime_sha256=sha256_bytes(runtime_file.read_bytes()),
            world_model=build.world_model,
            catalog_sha256=digest,
            maximum_physical_attempts=2,
            deadline=author.deadline,
        ),
        job=JobTemplate(actor_alias="world", maximum_physical_attempts=2, deadline=author.deadline),
    )
    build_spec = family_spec()
    assert build_spec.environment_families is not None
    spec = BootstrapPreparationSpec(
        build=BuildSettings(
            spec=build_spec,
            reviewer=author.model_copy(
                update={
                    "model": bound.model,
                    "output_tokens": build_spec.environment_families.review.maximum_output_tokens,
                }
            ),
        ),
        bootstrap=settings,
        limits=BootstrapLimits(
            maximum_tasks=2, maximum_families=2, maximum_family_traces=20, maximum_family_tools=10
        ),
    )
    return project, spec, replace(bound, configuration_sha256=sha256_json(spec.build.reviewer))


def saved_build(
    project: ProjectStore, spec: BootstrapPreparationSpec, bound: CompatibilityReviewer
) -> VarianceBuildReport:
    """Run the real family review and scope owners with scripted decisions and reviewer."""
    return run_variance_build(
        project,
        spec.build.spec,
        code_revision="test",
        reviewer=bound,
        tagger=_tagger(project, FakeDecisions()),
    )


def test_selected_tasks_preserve_scope_without_initial_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual placements select only probes, and scoped runtimes replace the global template."""
    project, spec, bound = preparation_fixture(tmp_path, monkeypatch)
    report = saved_build(project, spec, bound)
    rows = select_bootstrap_families(project, report, spec.bootstrap, spec.limits)
    assert rows and sum(len(row.spec.families[0].tasks) for row in rows) == 2
    for row in rows:
        assert (
            row.spec.runtime.world_model
            == row.scope.world_model
            != spec.bootstrap.runtime.world_model
        )
        assert row.spec.runtime.tag_retrieval == row.scope.tag_retrieval
        assert row.spec.families[0].family.runtime is None
        assert row.spec.families[0].family.learning_traces == row.scope.rules_traces
        assert all(task.task.task_id in row.scope.tasks for task in row.spec.families[0].tasks)
    with pytest.raises(ValueError, match="compatible family placement"):
        select_bootstrap_families(
            project, report, spec.bootstrap, spec.limits, task_ids=("not-selected",)
        )


def test_frozen_family_scope_drift_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Knowledge from another world never reaches bootstrap roles."""
    project, spec, bound = preparation_fixture(tmp_path, monkeypatch)
    report = saved_build(project, spec, bound)
    row = select_bootstrap_families(project, report, spec.bootstrap, spec.limits)[0]
    wrong = row.simulator.model_copy(update={"world_model": spec.bootstrap.runtime.world_model})
    ref = save_family_simulator(project.artifacts, wrong, code_revision="test")
    with pytest.raises(ValueError, match="saved scope"):
        select_bootstrap_families(project, report, spec.bootstrap, spec.limits, simulators=(ref,))


def test_a_family_over_the_trace_bound_learns_from_a_bounded_subset_instead_of_refusing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Intent: a 1,000-member family still bootstraps; rules learn from at most the bound.

    The family keeps every member for its corpus and world; only rule learning is bounded, by
    the stratified subset the task types chose or else an evenly spaced one.
    """
    project, spec, bound = preparation_fixture(tmp_path, monkeypatch)
    report = saved_build(project, spec, bound)
    limits = spec.limits.model_copy(update={"maximum_family_traces": 1})
    rows = select_bootstrap_families(project, report, spec.bootstrap, limits)
    assert rows
    for row in rows:
        family = row.spec.families[0].family
        full = report.environment_families.scope(row.scope.family_id).rules_traces
        assert len(family.learning_traces) == 1 and set(family.learning_traces) <= set(full)
        assert family.member_traces == row.scope.members
    target = rows[0].scope.family_id
    members = report.environment_families.scope(target).rules_traces
    chosen = select_bootstrap_families(
        project, report, spec.bootstrap, limits, learning_traces={target: (members[-1],)}
    )
    assert chosen[0].spec.families[0].family.learning_traces == (members[-1],)
    with pytest.raises(ValueError, match="within maximum_family_traces"):
        select_bootstrap_families(
            project, report, spec.bootstrap, limits, learning_traces={target: members[:2]}
        )


def test_narrow_retrieval_preserves_full_family_learning_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrieval admission narrows the index while rules and author context keep the full family."""
    project, spec, bound = preparation_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(
        update={
            "build": spec.build.model_copy(
                update={
                    "spec": spec.build.spec.model_copy(
                        update={"cluster_admission": ClusterAdmissionSpec(total=1)}
                    )
                }
            )
        }
    )
    report = saved_build(project, spec, bound)
    row = select_bootstrap_families(project, report, spec.bootstrap, spec.limits)[0]
    family = row.spec.families[0].family
    assert family.corpus is not None
    assert len(family.corpus.ids) == 1 < len(family.member_traces)
    assert set(family.corpus.ids) < set(family.member_traces)
    assert family.learning_traces == row.scope.rules_traces == family.member_traces
    with pytest.raises(ValueError, match="rules traces outside"):
        FamilyInput.model_validate({**family.model_dump(), "rules_traces": ("foreign-trace",)})
    with pytest.raises(ValueError, match="retrieval traces outside"):
        FamilyInput.model_validate(
            {
                **family.model_dump(),
                "corpus": family.corpus.model_copy(
                    update={"ids": ("foreign-trace",), "ids_sha256": sha256_json(["foreign-trace"])}
                ),
            }
        )


def test_automatic_selection_skips_families_excluded_from_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unserved family's placement cannot block valid probes or bypass explicit selection."""
    project, spec, bound = preparation_fixture(tmp_path, monkeypatch)
    build = spec.build.spec.model_copy(
        update={
            "clustering": ClusteringSpec(min_clusters=2, max_clusters=2),
            "cluster_admission": ClusterAdmissionSpec(total=1),
            "probe_fraction": 0.5,
        }
    )
    spec = spec.model_copy(update={"build": spec.build.model_copy(update={"spec": build})})
    report = saved_build(project, spec, bound)
    scoping = report.environment_families
    assert scoping is not None and scoping.left_out and scoping.scopes
    rows = select_bootstrap_families(project, report, spec.bootstrap, spec.limits)
    available = {scope.family_id for scope in scoping.scopes}
    assert rows and all(row.simulator.family_id in available for row in rows)
    plan = plan_bootstrap_preparation(project, spec, code_revision="test")
    assert plan.remaining_tasks and plan.task_bound == len(plan.remaining_tasks)
    excluded = next(row.task_id for row in scoping.placements if row.family_id in scoping.left_out)
    with pytest.raises(ValueError, match="compatible family placement"):
        select_bootstrap_families(
            project, report, spec.bootstrap, spec.limits, task_ids=(excluded,)
        )
