"""Native family scope and private query alias stay independent of global project retrieval."""

from pathlib import Path

import pytest

from exp.common.models import load_model_catalog
from exp.optimize.evaluation.audit.panel_test import (
    USERS,
    SimulatedProvider,
    build_project,
    episode,
)
from exp.optimize.evaluation.executable.catalog_binding import (
    load_runtime_catalog,
    register_runtime_catalog,
)
from exp.optimize.evaluation.executable.model_costs import catalog_model_cost
from exp.optimize.evaluation.fidelity.retrieval_binding import rag_index_embedder_alias
from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapPreparationReport
from exp.optimize.evaluation.variance.bootstrap_stage import bootstrapped_registration
from exp.optimize.evaluation.variance.build_records import VarianceBuildReport
from exp.optimize.evaluation.variance.council_qualification_test import _fixture
from exp.optimize.evaluation.variance.family_simulators import load_family_simulator
from exp.optimize.evaluation.variance.pipeline_family_rag import prepared_family_build
from exp.optimize.evaluation.variance.pipeline_spec import BuildOutputs
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.retrieval.store import load_rag_index


def test_exact_prepared_family_view_keeps_real_native_refs_and_refuses_changed_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No generated artifact or index pointer is relabeled to manufacture a scoped result."""
    project, prepared, _, contract, wire = _fixture(tmp_path, monkeypatch)
    report = BootstrapPreparationReport.model_validate_json(
        project.artifacts.read_bytes(prepared.artifact_id, "report.json")
    )
    build = VarianceBuildReport.model_validate_json(
        project.artifacts.read_bytes(report.build_report.artifact_id, "report.json")
    )
    inputs = BuildOutputs(
        report=report.build_report,
        exposure=build.exposure,
        world_model=build.world_model,
        tag_retrieval=build.tag_retrieval,
        index=build.index,
        retrieval_scope=build.retrieval_scope,
        trace_dataset=build.build.trace_dataset,
        rules_traces=build.rules_stage_traces,
        selected_probes=build.selected_probes,
    )
    family = load_family_simulator(project.artifacts, contract.family_simulator)
    start = bootstrapped_registration(project, contract.bootstrap)
    before = (
        project.artifacts.list_ids(),
        project.model_catalog_path.read_bytes(),
        len(wire.requests),
    )
    view = prepared_family_build(
        project,
        inputs,
        family_ref=contract.family_simulator,
        binding=contract.bootstrap,
        job=start.job,
    )
    assert view.index == family.index and view.world_model == family.world_model
    assert view.retrieval_scope == family.corpus
    assert view.rules_traces == family.rules_traces
    assert view.report == inputs.report == report.build_report
    assert inputs.index == build.index
    with pytest.raises(ValueError, match="exact prepared family"):
        prepared_family_build(
            project,
            inputs,
            family_ref=contract.family_simulator,
            binding=contract.bootstrap,
            job=start.job.model_copy(update={"configuration": {}}),
        )
    assert (
        project.artifacts.list_ids(),
        project.model_catalog_path.read_bytes(),
        len(wire.requests),
    ) == before


def test_explicit_private_query_alias_and_native_price_ignore_old_project_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real saved index snapshot selects its configured native alias, never a name guess."""
    _, project, _ = build_project(
        tmp_path, monkeypatch, SimulatedProvider([episode(i) for i in range(USERS)])
    )
    build = project.load_project().build
    assert build is not None
    index = load_rag_index(project.artifacts, build.fit_rag.artifact_id).index
    before = project.model_catalog_path.read_bytes()
    catalog = load_model_catalog(project.model_catalog_path)
    original = catalog.models["embed"]
    target = catalog.model_copy(
        update={
            "models": {
                **catalog.models,
                "embed": original.model_copy(update={"model": "old-global-embedder"}),
                "family-query": original,
            }
        }
    )
    binding = register_runtime_catalog(
        project, target, embedder_alias="family-query", code_revision="offline"
    )
    native = load_runtime_catalog(project, binding)
    with pytest.raises(ValueError, match="was embedded"):
        rag_index_embedder_alias(project, native, index)
    alias = rag_index_embedder_alias(
        project, native, index, configured_alias=binding.embedder_alias
    )
    assert alias == "family-query"
    assert RuntimeModelCatalog(native, environment={}).snapshot(alias)[0] == index.embedder
    component = catalog_model_cost(native, alias, 3, 0)
    assert component.name == alias and component.attempts == 3
    assert project.model_catalog_path.read_bytes() == before
