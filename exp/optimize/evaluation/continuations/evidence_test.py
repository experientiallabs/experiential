"""Qualification handoff binds generation to challenged tasks without executing during export."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.models import load_model_catalog
from exp.common.project import artifact_input
from exp.common.project.causality import cites
from exp.optimize.evaluation.continuations.admission import (
    ContinuationAdmission,
    admit_continuations,
)
from exp.optimize.evaluation.continuations.evidence import (
    qualified_training_job,
    verify_training_evidence,
)
from exp.optimize.evaluation.continuations.job import (
    continuation_result_reference,
    execute_continuation_job,
)
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.equivalence.evidence import equivalence_metrics
from exp.optimize.evaluation.equivalence.policy_test import fixture_policy, registered_policy
from exp.optimize.evaluation.equivalence.review import ReviewModel
from exp.optimize.evaluation.equivalence.rubric_test import frozen_rubric
from exp.optimize.evaluation.equivalence.runner import (
    EquivalenceCase,
    EquivalenceSpec,
    run_equivalence,
)
from exp.optimize.evaluation.equivalence.runner_test import response
from exp.optimize.evaluation.executable.qualification import (
    load_executable_qualification,
    qualify_executable_task,
)
from exp.optimize.evaluation.executable.qualification_test import qualified_fixture
from exp.optimize.model.sft.builder import (
    build_sft_dataset,
    load_verified_sft_dataset,
    write_sft_dataset,
)
from exp.optimize.model.sft.continuation_source import resolve_continuation_source
from exp.optimize.model.sft.contracts import SFTBuildSpec
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog


def test_qualified_job_binds_exact_execution_and_refuses_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checked task can prepare generation, but changed effects cannot inherit qualification."""
    project, _, spec = qualified_fixture(tmp_path, monkeypatch)
    reference = qualify_executable_task(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    job = qualified_training_job(project, qualification=reference)
    assert job.study.training_evidence is not None
    assert job.study.training_evidence.qualification == reference
    evidence = verify_training_evidence(
        project.artifacts, job.study, execution_revision=job.implementation("test")
    )
    assert evidence is not None and evidence.qualification == reference
    assert "native-fidelity" in evidence.qualification_scope
    changed = job.model_copy(update={"configuration": {"new": "world"}})
    with pytest.raises(ValueError, match="qualified execution"):
        verify_training_evidence(
            project.artifacts, changed.study, execution_revision=changed.implementation("test")
        )
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport",
        lambda: pytest.fail("invalid evidence must refuse before provider construction"),
    )
    with pytest.raises(ValueError, match="qualified execution"):
        execute_continuation_job(project, changed, code_revision="test")


def test_refused_qualification_cannot_prepare_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved refusal is useful evidence, never an authority for positive training targets."""
    project, _, spec = qualified_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(
        update={"cases": tuple(c for c in spec.cases if "side-effect" not in c.challenge.id)}
    )
    reference = qualify_executable_task(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    with pytest.raises(ValueError, match="unqualified"):
        qualified_training_job(project, qualification=reference)


def test_export_does_not_silently_load_native_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credential-free readers must reject execution-requiring evidence before loading code."""

    project, _, spec = qualified_fixture(tmp_path, monkeypatch)
    reference = qualify_executable_task(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    report = load_executable_qualification(project, reference)
    changed = report.model_copy(
        update={"spec": spec.model_copy(update={"replay_native_records": True})}
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.executable.qualification._read_qualification", lambda *_: changed
    )
    with pytest.raises(ValueError, match="native replay"):
        load_executable_qualification(project, reference, allow_native_replay=False)


def test_qualified_run_and_export_retain_evidence_without_fidelity_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real XLSX execution, saved evidence and target hydration retain their limited scope."""

    project, _, spec = qualified_fixture(tmp_path, monkeypatch)
    qualification = qualify_executable_task(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    job = qualified_training_job(project, qualification=qualification)
    # Obtain a fresh admitted transport with the same scripted responses, in another project.
    _, _, transport = job_fixture(tmp_path / "transport", monkeypatch)
    policy = registered_policy(project.artifacts)
    execute_continuation_job(project, job, code_revision="test")
    result = continuation_result_reference(project, job, code_revision="test")
    assert len(transport.requests) == 3
    reference = admit_continuations(
        project.artifacts,
        result,
        policy=policy,
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    admission = ContinuationAdmission.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "admission.json")
    )
    assert admission.evidence is not None and admission.evidence.qualification == qualification
    assert sum(row.admitted for row in admission.branches) == 1
    job.runtime_file.unlink()
    prepared, _, _ = resolve_continuation_source(project, reference)
    assert len(prepared) == 1 and len(transport.requests) == 3
    dataset = build_sft_dataset(
        store=project,
        production_sources=(),
        teacher_sources=(),
        continuation_sources=(reference,),
        spec=SFTBuildSpec(held_out_fraction=0),
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    assert dataset.dataset.status == "accepted"
    assert all(
        row.example.source.model_dump().get("qualification") == qualification.model_dump()
        for row in dataset.rows
    )
    write_sft_dataset(project, dataset)
    assert load_verified_sft_dataset(project, dataset.dataset.dataset_id) == dataset
    assert len(transport.requests) == 3
    assert qualification in project.artifacts.read(reference.artifact_id).manifest.inputs


@pytest.mark.parametrize("changed_configuration", [False, True])
def test_saved_comparison_only_attaches_to_its_exact_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_configuration: bool
) -> None:
    """All failed reviewer evidence remains attached; matching schema alone cannot transfer it."""

    project, _, controls = qualified_fixture(tmp_path, monkeypatch)
    qualification = qualify_executable_task(
        project, controls, created_at=datetime.now(UTC), code_revision="test"
    )
    job = qualified_training_job(project, qualification=qualification)
    compared = job.model_copy(
        update={
            "study": job.study.model_copy(
                update={
                    "training_specification": None,
                    "training_evidence": None,
                    "repeats": 1,
                }
            )
        }
    )
    if changed_configuration:
        compared = compared.model_copy(
            update={"configuration": {**compared.configuration, "unused": True}}
        )
    judge = ReviewModel(
        alias="judge",
        model=RuntimeModelCatalog(
            load_model_catalog(project.model_catalog_path), environment={}
        ).snapshot("judge")[0],
        output_tokens=1024,
    )
    other_actor = compared.model_copy(
        update={
            "actor_alias": "judge",
            "study": compared.study.model_copy(update={"actor_model": judge.model}),
        }
    )
    # The comparison is registered under its own policy, which the study then reaches by citation.
    comparison_policy = registered_policy(project.artifacts, fixture_policy(margin=0.4))
    spec = EquivalenceSpec(
        rubric=frozen_rubric(project, judge),
        cases=(
            EquivalenceCase(
                reference=compared,
                candidate=compared,
                initial_state_paths=("/input_digest",),
                final_state_paths=("/rows", "/input_digest"),
            ),
            EquivalenceCase(
                reference=other_actor,
                candidate=other_actor,
                initial_state_paths=("/input_digest",),
                final_state_paths=("/rows", "/input_digest"),
            ),
        ),
        judge=judge,
        analysts=(judge,),
        maximum_review_attempts=6,
        deadline=compared.deadline,
        policy=comparison_policy,
    )
    call = workbook_case().first
    actor = ScriptedJsonTransport(
        tuple(
            reply
            for index in range(6)
            for reply in (
                response(
                    "world-v1" if index < 3 else "judge-v1",
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call.call_id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(
                                        {
                                            **call.arguments,
                                            "deletions": [
                                                {
                                                    "sheet": "Orders",
                                                    "start": 2,
                                                    "count": 2 if index == 3 else 1,
                                                }
                                            ],
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                    tools=True,
                ),
                response(
                    "world-v1" if index < 3 else "judge-v1",
                    {"role": "assistant", "content": "done"},
                    tools=False,
                ),
            )
        )
    )
    reviews = ScriptedJsonTransport(
        tuple(
            response(
                "judge-v1",
                {"role": "assistant", "content": "invalid uncalibrated review"},
                tools=False,
            )
            for _ in range(6)
        )
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actor
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.equivalence.review.HttpxJsonTransport", lambda: reviews
    )
    report = run_equivalence(project, spec, code_revision="test")
    reference = artifact_input(project.artifacts.read(spec.identity("test") + "-result").manifest)
    if changed_configuration:
        with pytest.raises(ValueError, match="comparison candidate differs"):
            qualified_training_job(
                project, qualification=qualification, comparison=reference, comparison_case=0
            )
    else:
        prepared = qualified_training_job(
            project, qualification=qualification, comparison=reference, comparison_case=0
        )
        evidence = verify_training_evidence(
            project.artifacts, prepared.study, execution_revision=prepared.implementation("test")
        )
        assert evidence is not None
        assert evidence.comparison_metrics == equivalence_metrics(
            tuple(row for row in report.measurements if row.case == 0)
        )
        assert evidence.comparison_metrics != report.metrics
        assert any(
            row.case == 1 and row.material_agreement == "different" for row in report.measurements
        )
        assert evidence.comparison_scope == report.scope
        assert report.analysis_failures
        assert all(
            row.judgment is not None
            and row.judgment.status == "failed"
            and row.judgment.score_100 is None
            for row in report.measurements
        )
        with pytest.raises(ValueError, match="outside"):
            qualified_training_job(
                project, qualification=qualification, comparison=reference, comparison_case=2
            )
        # A study carrying that comparison cites its policy, yet only the pinned one admits it.
        job_fixture(tmp_path / "transport", monkeypatch)
        execute_continuation_job(project, prepared, code_revision="test")
        result = continuation_result_reference(project, prepared, code_revision="test")
        registration = result.artifact_id.removesuffix("-result")
        assert cites(project.artifacts, registration, comparison_policy)
        assert prepared.study.admission_policy != comparison_policy
        with pytest.raises(ValueError, match="only the policy a study pins"):
            admit_continuations(
                project.artifacts,
                result,
                policy=comparison_policy,
                created_at=datetime.now(UTC),
                code_revision="test",
            )
    assert len(actor.requests) == 12 and len(reviews.requests) == 6
