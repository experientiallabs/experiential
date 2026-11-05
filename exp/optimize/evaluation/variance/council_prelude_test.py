"""Native prelude qualification and pricing before any full actor-critic specification exists."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, sha256_json
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.executable.costs import evaluation_costs, job_costs
from exp.optimize.evaluation.variance.behavior_differences import BehaviorDifferenceSpec
from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapPreparationReport
from exp.optimize.evaluation.variance.build_records import VarianceBuildReport
from exp.optimize.evaluation.variance.council_prelude import (
    ConditionalCouncilPreludeRegistration,
    CouncilPreludeContract,
    freeze_council_prelude,
    load_qualified_prelude,
    qualify_council_prelude,
    register_council_prelude,
)
from exp.optimize.evaluation.variance.council_prelude_hook import CouncilPreludeCheckpoint
from exp.optimize.evaluation.variance.council_prelude_inventory import prelude_inventory
from exp.optimize.evaluation.variance.council_preparation import CouncilPreparationContract
from exp.optimize.evaluation.variance.council_qualification import (
    CONSENT_TYPE,
    CouncilConditionalConsent,
    CouncilStudyContract,
)
from exp.optimize.evaluation.variance.council_qualification_test import (
    NativePreparationScript,
    _fixture,
)
from exp.optimize.evaluation.variance.pipeline_inputs import rubric_draws
from exp.optimize.evaluation.variance.pipeline_spec import (
    AuthoringSettings,
    DecisionVarianceSpec,
    RecordsSettings,
    RulesSettings,
    TraceMLSettings,
    cost_lines,
)
from exp.optimize.evaluation.variance.pipeline_test import write_bank
from exp.simulation.mining.traceml import traceml_bank


def _contract(study: CouncilStudyContract) -> CouncilPreparationContract:
    """Project the native source and task evidence without creating any placeholder study."""
    return CouncilPreparationContract(
        preparation_spec=study.preparation_spec,
        bootstrap=study.bootstrap,
        exposure=study.exposure,
        family_simulator=study.family_simulator,
        preparation_producer=study.preparation_producer,
        preparation_source_sha256=study.preparation_source_sha256,
        preparation_provenance=study.preparation_provenance,
        preparation_provenance_type=study.preparation_provenance_type,
    )


def _prelude_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    ProjectStore, ArtifactInput, ArtifactInput, CouncilPreludeContract, NativePreparationScript
]:
    """Use native preparation, then freeze real pipeline settings with an unauthored rubric."""
    project, prepared, resume, full, wire = _fixture(tmp_path, monkeypatch)
    report = BootstrapPreparationReport.model_validate_json(
        project.artifacts.read_bytes(prepared.artifact_id, "report.json")
    )
    build = VarianceBuildReport.model_validate_json(
        project.artifacts.read_bytes(report.build_report.artifact_id, "report.json")
    )
    role = full.mining.judge
    bank_parent = tmp_path / "prelude-banks"
    bank_parent.mkdir()
    bank_root = write_bank(bank_parent)
    bank = traceml_bank(bank_root)
    questions = bank.question_ids()
    behavior = BehaviorDifferenceSpec(
        bank_id=bank.bank_id,
        author=role,
        reviewer=role,
        positive_tags=(questions[0],),
        negative_tags=questions[1:],
    )
    rules = full.preparation_spec.bootstrap.rules
    assert rules is not None
    pipeline = DecisionVarianceSpec(
        build=full.preparation_spec.build.model_copy(
            update={"spec": build.spec, "saved_report": report.build_report}
        ),
        rules=RulesSettings(spec=rules),
        traceml=TraceMLSettings(move_judge=role, bank_root=bank_root),
        mining=full.mining.model_copy(
            update={
                "job": None,
                "family_simulator": full.family_simulator,
                "rubric": None,
                "group_physical_attempts": 320,
                "behaviors": behavior,
            }
        ),
        authoring=AuthoringSettings(
            decision_author=role,
            scenario_author=role,
            world_author=role,
            rubric_author=role,
        ),
        records=RecordsSettings(intent="Read-only native council prelude fixture."),
    )
    contract = freeze_council_prelude(
        _contract(full),
        pipeline,
        actor_model_ids=tuple(f"open-fixture-{index}" for index in range(5)),
        code_revision="council-executor",
    )
    assert isinstance(wire, NativePreparationScript)
    return project, prepared, resume, contract, wire


def test_native_prelude_qualifies_without_a_rubric_or_full_study_and_prices_real_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native jobs, physical bounds and all other models' mandatory work precede the rubric."""
    project, prepared, resume, contract, wire = _prelude_fixture(tmp_path, monkeypatch)
    assert contract.pipeline.mining.rubric is None
    assert contract.pipeline.mining.job is None
    calls = len(wire.requests)
    qualified = qualify_council_prelude(
        project,
        prepared,
        resume,
        contract,
        code_revision="council-executor",
    )
    assert len(qualified.jobs) == 5
    assert len({job.study.actor_model.model_id for job in qualified.jobs}) == 5
    job = qualified.jobs[0]
    checkpoint = CouncilPreludeCheckpoint(
        phase="traceml_group",
        actor_alias=job.actor_alias,
        job=job,
    )
    inventory = prelude_inventory(project, qualified, checkpoint)
    assert inventory.costs == cost_lines(job_costs(project, job))
    assert inventory.jobs == (job,) and inventory.attempts > 0
    assert inventory.remaining.groups == 4 * 16
    assert inventory.remaining.group_judgments == 5 * 16
    assert inventory.remaining.selectors == 5 * 8
    assert inventory.remaining.critics == 5 * 8
    assert inventory.remaining.suffixes == 5 * 8
    assert inventory.remaining.suffix_judgments == 5 * 8
    assert inventory.remaining.attempts > 0
    assert len(wire.requests) == calls


def test_native_rubric_prelude_is_priced_by_its_actual_role_and_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No synthetic rubric or actor-critic registration is used to obtain author authority."""
    project, prepared, resume, contract, wire = _prelude_fixture(tmp_path, monkeypatch)
    qualified = qualify_council_prelude(
        project,
        prepared,
        resume,
        contract,
        code_revision="council-executor",
    )
    base = contract.pipeline.mining.model_copy(update={"job": None})
    assert base.rubric is None
    from exp.optimize.evaluation.variance.bootstrap import registered_job

    job = registered_job(project, contract.preparation.bootstrap)
    checkpoint = CouncilPreludeCheckpoint(phase="mining_rubric", job=job)
    calls = len(wire.requests)
    inventory = prelude_inventory(project, qualified, checkpoint)
    expected = tuple(
        row.scaled(rubric_draws(contract.pipeline.authoring.rubric_profile))
        for row in evaluation_costs(project, contract.pipeline.authoring.rubric_author)
    )
    assert inventory.costs == cost_lines(expected) and inventory.attempts > 0
    assert not inventory.jobs and inventory.remaining.rubric == 0
    assert inventory.remaining.groups == 5 * 16
    assert len(wire.requests) == calls


@pytest.mark.parametrize("change", ["G", "H", "group_cap", "source", "family"])
def test_native_prelude_refuses_reduced_controls_or_impossible_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    """Source and requested-study constraints fail before provider calls or ledger registration."""
    project, prepared, resume, contract, wire = _prelude_fixture(tmp_path, monkeypatch)
    mining = contract.pipeline.mining
    if change == "G":
        mining = mining.model_copy(update={"attempts": 15})
    elif change == "group_cap":
        mining = mining.model_copy(update={"group_physical_attempts": 15})
    elif change == "H":
        from exp.optimize.evaluation.variance.bootstrap import registered_job

        job = registered_job(project, contract.preparation.bootstrap)
        mining = mining.model_copy(
            update={"job": job.model_copy(update={"maximum_model_calls": 19})}
        )
    elif change == "family":
        mining = mining.model_copy(update={"family_simulator": None})
    else:
        contract = contract.model_copy(update={"sources": {}})
    contract = contract.model_copy(
        update={"pipeline": contract.pipeline.model_copy(update={"mining": mining})}
    )
    calls = len(wire.requests)
    with pytest.raises(ValueError):
        qualify_council_prelude(
            project,
            prepared,
            resume,
            contract,
            code_revision="council-executor",
        )
    assert len(wire.requests) == calls


def _register_fixture(
    project: ProjectStore,
    prepared: ArtifactInput,
    resume: ArtifactInput,
    contract: CouncilPreludeContract,
) -> ConditionalCouncilPreludeRegistration:
    """Persist exact test consent for genuine preparation on its original temporary ledger."""
    from exp.optimize.evaluation.variance.bootstrap_history import (
        read_bootstrap_preparation_provenance,
    )

    qualified = qualify_council_prelude(
        project, prepared, resume, contract, code_revision="council-executor"
    )
    binding = read_bootstrap_preparation_provenance(
        project, resume, spec=contract.preparation.preparation_spec
    ).binding
    consent = CouncilConditionalConsent(
        operation=binding.original_operation,
        ledger_sha256=binding.ledger_sha256,
        contract_sha256=sha256_json(contract),
        preparation=prepared,
        resume_binding=resume,
        open_actors=qualified.actors,
        legacy_attempt_limit=binding.unpriced_attempts,
        maximum_known_usd=binding.ceiling_usd,
        deadline=datetime.now(UTC) + timedelta(hours=1),
        producer_revision="council-executor",
    )
    reference = artifact_input(
        project.artifacts.write_json(
            artifact_id="explicit-native-prelude-consent",
            artifact_type=CONSENT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision="council-executor",
                inputs=(prepared, resume),
            ),
            files={"consent.json": consent},
        )
    )
    return register_council_prelude(
        project, prepared, resume, contract, reference, code_revision="council-executor"
    )


def test_native_prelude_registration_is_readable_and_preserves_original_grants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving exact consent adds a parent policy without rewriting native preparation grants."""
    from exp.optimize.evaluation.spending import operation_spend_ceiling
    from exp.optimize.evaluation.variance.bootstrap_history import (
        read_bootstrap_preparation_provenance,
    )

    project, prepared, resume, contract, wire = _prelude_fixture(tmp_path, monkeypatch)
    binding = read_bootstrap_preparation_provenance(
        project, resume, spec=contract.preparation.preparation_spec
    ).binding
    ceiling = operation_spend_ceiling(project, binding.original_operation)
    before = ceiling.summary()
    calls = len(wire.requests)
    registration = _register_fixture(project, prepared, resume, contract)
    loaded = load_qualified_prelude(project, registration.qualification)
    assert loaded.preparation == prepared and loaded.resume_binding == resume
    assert loaded.producer_revision == "council-executor"
    assert ceiling.summary() == before
    assert len(wire.requests) == calls
