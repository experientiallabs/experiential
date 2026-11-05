"""Native historical source proofs stay read-only and never authorize current execution."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance.bootstrap_history import (
    read_bootstrap_preparation_provenance,
)
from exp.optimize.evaluation.variance.bootstrap_pilot_test import pilot_fixture, register_pilot
from exp.optimize.evaluation.variance.bootstrap_protocols import (
    current_bootstrap_source,
    propose_bootstrap_protocol_revision,
    register_bootstrap_protocol_revision,
)
from exp.optimize.evaluation.variance.bootstrap_resume import (
    load_bootstrap_resume,
    register_bootstrap_resume,
)


def test_historical_reader_pins_real_source_record_without_granting_current_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Intrinsic original source remains readable after the consuming source has changed."""
    project, spec, root, pilot = pilot_fixture(tmp_path, monkeypatch)
    previous = register_pilot(project, spec, root, pilot)
    parent = load_bootstrap_resume(project, previous, spec=spec, code_revision="pilot-executor")
    proposal = propose_bootstrap_protocol_revision(
        project, previous_binding=previous, code_revision="saved-source-executor"
    )
    control = register_bootstrap_protocol_revision(
        project, proposal, code_revision="saved-source-executor"
    )
    reference = register_bootstrap_resume(
        project,
        original_inputs=parent.original_inputs,
        code_revision="saved-source-executor",
        permitted_changes=(*parent.permitted_changes, "preparation_protocol_revision"),
        previous_binding=previous,
        review_revision=parent.review_revision,
        pilot_revision=parent.pilot_revision,
        execution_deadline=parent.execution_deadline,
        execution_window=parent.execution_window,
        protocol_revision=control,
    )
    ceiling = operation_spend_ceiling(project, parent.original_operation)
    before = (ceiling.summary(), project.artifacts.list_ids())
    source = current_bootstrap_source()
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.bootstrap_protocols.current_bootstrap_source",
        lambda: source.model_copy(update={"protocols": {**source.protocols, "consumer": "new"}}),
    )
    with pytest.raises(ValueError, match="differs from reviewed source"):
        load_bootstrap_resume(project, reference, spec=spec, code_revision="saved-source-executor")
    proof = read_bootstrap_preparation_provenance(project, reference, spec=spec)
    assert proof.binding.original_operation == parent.original_operation
    assert proof.producer_revision == "saved-source-executor"
    assert proof.source_ref == control
    assert proof.source_type == "bootstrap-protocol-revision"
    assert proof.source_sha256 == sha256_json(proposal.source)
    assert proof.execution_spec.task_ids == (pilot.task_id,)
    assert (ceiling.summary(), project.artifacts.list_ids()) == before
    with pytest.raises(ValueError, match="reference differs"):
        read_bootstrap_preparation_provenance(
            project, reference.model_copy(update={"sha256": "0" * 64}), spec=spec
        )


def test_fresh_native_binding_records_real_source_and_current_drift_refuses_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New native writers record actual package bytes, with no producer-label hashing."""
    project, spec, reference, _ = pilot_fixture(tmp_path, monkeypatch)
    from exp.optimize.evaluation.variance.bootstrap_source import current_bootstrap_execution_source

    proof = read_bootstrap_preparation_provenance(project, reference, spec=spec)
    witness = current_bootstrap_execution_source("corrected")
    assert proof.source_type == "bootstrap-resume-binding"
    assert proof.source_ref == reference and proof.source_sha256 == witness.source_sha256
    assert proof.binding.execution_source == witness
    before = project.artifacts.list_ids()
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.bootstrap_resume.current_bootstrap_execution_source",
        lambda producer: witness.model_copy(update={"source_sha256": "0" * 64}),
    )
    with pytest.raises(ValueError, match="registered witness"):
        load_bootstrap_resume(project, reference, spec=spec, code_revision="corrected")
    assert read_bootstrap_preparation_provenance(project, reference, spec=spec) == proof
    assert project.artifacts.list_ids() == before


def test_current_protocol_source_precedes_inherited_embedding_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source-only successor uses its own source proof while retaining older corpus evidence."""
    from exp.common.project import ProjectStore
    from exp.optimize.evaluation.variance import bootstrap_history as owner
    from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapPreparationSpec
    from exp.optimize.evaluation.variance.bootstrap_resume import BootstrapResumeBinding

    project, spec, root, pilot = pilot_fixture(tmp_path, monkeypatch)
    previous = register_pilot(project, spec, root, pilot)
    parent = load_bootstrap_resume(project, previous, spec=spec, code_revision="pilot-executor")
    revision = propose_bootstrap_protocol_revision(
        project, previous_binding=previous, code_revision="current-source"
    )
    control = register_bootstrap_protocol_revision(
        project, revision, code_revision="current-source"
    )
    reference = register_bootstrap_resume(
        project,
        original_inputs=parent.original_inputs,
        code_revision="current-source",
        permitted_changes=(*parent.permitted_changes, "preparation_protocol_revision"),
        previous_binding=previous,
        review_revision=parent.review_revision,
        pilot_revision=parent.pilot_revision,
        execution_deadline=parent.execution_deadline,
        execution_window=parent.execution_window,
        protocol_revision=control,
    )
    actual = read_bootstrap_preparation_provenance(project, reference, spec=spec)
    inherited = actual.binding.model_copy(update={"embedding_revision": previous})

    def verified_history(
        project: ProjectStore, reference: ArtifactInput, *, spec: BootstrapPreparationSpec
    ) -> BootstrapResumeBinding:
        """Supply an already verified inherited corpus at the historical-reader boundary."""
        return inherited

    def execution(
        project: ProjectStore, binding: BootstrapResumeBinding, spec: BootstrapPreparationSpec
    ) -> BootstrapPreparationSpec:
        """Keep the verified original execution settings while isolating source selection."""
        return actual.execution_spec

    monkeypatch.setattr(owner, "read_historical_bootstrap_resume", verified_history)
    monkeypatch.setattr(owner, "bootstrap_execution_spec", execution)
    proof = owner.read_bootstrap_preparation_provenance(project, reference, spec=spec)
    assert proof.source_ref == control and proof.source_type == "bootstrap-protocol-revision"
    assert proof.source_sha256 == sha256_json(revision.source)
    assert proof.binding.embedding_revision == previous


def test_historical_floor_does_not_authorize_changed_current_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Original obligations remain priceable after source drift, while execution still refuses."""
    from exp.optimize.evaluation.variance import bootstrap_plan as planner
    from exp.optimize.evaluation.variance.bootstrap_history import read_bootstrap_preparation_plan

    project, spec, root, pilot = pilot_fixture(tmp_path, monkeypatch)
    reference = register_pilot(project, spec, root, pilot)
    changed_runtime = tmp_path / "current_runtime.py"
    changed_runtime.write_text('"""A different current runtime fixture."""\n')
    monkeypatch.setattr(planner.grounded, "__file__", str(changed_runtime))
    with pytest.raises(ValueError, match="maintained grounded.py"):
        planner.plan_bootstrap_preparation(
            project, spec, code_revision="pilot-executor", resume_binding=reference
        )
    floor = read_bootstrap_preparation_plan(project, reference, spec=spec)
    assert floor.build_saved and floor.remaining_tasks == (pilot.task_id,)
    assert floor.code_revision == "pilot-executor"

    monkeypatch.setattr(planner, "sha256_bytes", lambda content: "0" * 64)
    with pytest.raises(ValueError, match="original saved pin"):
        read_bootstrap_preparation_plan(project, reference, spec=spec)
