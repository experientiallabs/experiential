"""Exact native per-job world authority with scripted preparation and no real services."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, sha256_json
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import load_continuation_runtime
from exp.optimize.evaluation.executable.runtime_authority import runtime_authority_scope
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance.bootstrap_resume import load_bootstrap_resume
from exp.optimize.evaluation.variance.council_checkpoint import CouncilCheckpoint
from exp.optimize.evaluation.variance.council_inventory import CouncilPhaseInventory
from exp.optimize.evaluation.variance.council_parallel import council_draws
from exp.optimize.evaluation.variance.council_qualification import (
    CONSENT_TYPE,
    ConditionalCouncilRegistration,
    CouncilConditionalConsent,
    CouncilStudyContract,
    qualify_council,
    register_conditional_council,
)
from exp.optimize.evaluation.variance.council_qualification_test import _fixture
from exp.optimize.evaluation.variance.council_world import (
    register_council_runtime,
    save_council_inventory,
)
from exp.runtime.models.scoped_spending import (
    SpendEnvelopePlan,
    open_spend_envelope,
    spend_envelope_scope,
)


def register_fixture(
    project: ProjectStore,
    prepared: ArtifactInput,
    resume: ArtifactInput,
    spec: CouncilStudyContract,
) -> ConditionalCouncilRegistration:
    """Register actual native offline preparation, its QA and explicit synthetic test consent."""
    qualified = qualify_council(project, prepared, resume, spec, code_revision="corrected")
    binding = load_bootstrap_resume(
        project, resume, spec=spec.preparation_spec, code_revision="corrected"
    )
    consent = CouncilConditionalConsent(
        operation=binding.original_operation,
        ledger_sha256=binding.ledger_sha256,
        contract_sha256=sha256_json(spec),
        preparation=prepared,
        resume_binding=resume,
        open_actors=qualified.actors,
        legacy_attempt_limit=binding.unpriced_attempts,
        maximum_known_usd=binding.ceiling_usd,
        deadline=datetime.now(UTC) + timedelta(hours=1),
        producer_revision="corrected",
    )
    reference = artifact_input(
        project.artifacts.write_json(
            artifact_id="explicit-native-world-test-consent",
            artifact_type=CONSENT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision="corrected",
                inputs=(prepared, resume),
            ),
            files={"consent.json": consent},
        )
    )
    return register_conditional_council(
        project, prepared, resume, spec, reference, code_revision="corrected"
    )


def test_actual_loader_requires_exact_registered_job_envelope_and_preserves_resume_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real native actor wave restores the frozen world with no trial-receipt fabrication."""
    project, prepared, resume, contract, wire = _fixture(tmp_path, monkeypatch)
    registration = register_fixture(project, prepared, resume, contract)
    binding = load_bootstrap_resume(
        project, resume, spec=contract.preparation_spec, code_revision="corrected"
    )
    ceiling = operation_spend_ceiling(project, binding.original_operation)
    spec = contract.studies[0]
    checkpoint = CouncilCheckpoint(
        run_identity=spec.identity("corrected"), phase="actors", jobs=(spec.initial,)
    )
    reference = save_council_inventory(
        project, registration.qualification, spec, checkpoint, code_revision="corrected"
    )
    inventory = CouncilPhaseInventory.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "inventory.json")
    )
    # This test exercises one native runtime boundary, not a full-study future-floor planner.
    envelope_plan = SpendEnvelopePlan(
        policy=registration.policy,
        scope="actual-actor-wave",
        inventory=reference,
        maximum_attempts=inventory.attempts,
        maximum_known_usd=inventory.known_usd,
        remaining_minimum_attempts=0,
        requirements_sha256=sha256_json(contract),
    )
    envelope = open_spend_envelope(ceiling, envelope_plan)
    calls = len(wire.requests)
    before = ceiling.summary()
    with spend_envelope_scope(ceiling, envelope):
        grant = register_council_runtime(
            project,
            registration.qualification,
            reference,
            spec.initial,
            code_revision="corrected",
        )
        assert grant == register_council_runtime(
            project,
            registration.qualification,
            reference,
            spec.initial,
            code_revision="corrected",
        )
        grant.require_configuration(
            project, spec.initial.configuration, runtime_sha256=spec.initial.runtime_sha256
        )
        with pytest.raises(ValueError, match="exact job or native wave"):
            grant.require_configuration(project, {})
        with pytest.raises(ValueError, match="exact job or native wave"):
            grant.require_configuration(
                project, spec.initial.configuration, runtime_sha256="0" * 64
            )
        with pytest.raises(ValueError, match="exact admitted native wave"):
            register_council_runtime(
                project,
                registration.qualification,
                reference,
                contract.studies[1].initial,
                code_revision="corrected",
            )

        def restore() -> str:
            """Use the actual loader in copied bounded worker contexts without model calls."""
            with load_continuation_runtime(project, spec.initial) as runtime:
                return runtime.probe_sha256

        with runtime_authority_scope(lambda project, job: grant):
            with load_continuation_runtime(project, spec.initial) as runtime:
                assert runtime.probe_sha256 == spec.initial.study.probe_sha256
            other = contract.studies[1].initial
            assert other.configuration == spec.initial.configuration
            assert other.runtime_sha256 == spec.initial.runtime_sha256
            assert other != spec.initial
            with pytest.raises(ValueError, match="another exact incoming job"):
                load_continuation_runtime(project, other)

        with runtime_authority_scope(
            lambda project, job: register_council_runtime(
                project, registration.qualification, reference, job, code_revision="corrected"
            )
        ):
            assert council_draws((restore, restore), 2) == (
                spec.initial.study.probe_sha256,
                spec.initial.study.probe_sha256,
            )
    assert ceiling.summary() == before and len(wire.requests) == calls
    with pytest.raises(ValueError, match="active native spend envelope"):
        grant.require_configuration(project, spec.initial.configuration)
    second = open_spend_envelope(
        ceiling, envelope_plan.model_copy(update={"scope": "must-not-refill-same-job"})
    )
    with spend_envelope_scope(ceiling, second):
        with pytest.raises(ValueError, match="cannot refill"):
            register_council_runtime(
                project,
                registration.qualification,
                reference,
                spec.initial,
                code_revision="corrected",
            )
        with pytest.raises(ValueError, match="exact job or native wave"):
            grant.require_configuration(project, spec.initial.configuration)
    assert not project.artifacts.list_ids(artifact_type="executable-trial")


def test_suffix_authority_rechecks_seeded_group_and_actual_selector_citations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A suffix cannot borrow a different boundary, even when its job shape matches."""
    from exp.optimize.evaluation.continuations.job_test import job_fixture
    from exp.optimize.evaluation.variance.actor_critic import run_actor_critic
    from exp.optimize.evaluation.variance.actor_critic_test import (
        _done,
        _edit,
        _exposure,
        _point,
        _spec,
        _tool_reply,
        _wire,
    )
    from exp.optimize.evaluation.variance.council_checkpoint import point_job
    from exp.optimize.evaluation.variance.council_inventory import phase_inventory
    from exp.optimize.evaluation.variance.council_world import _selected_jobs

    project, job, _ = job_fixture(tmp_path, monkeypatch)
    spec = _spec(project, job)
    point = {**_point(0), "evidence_refs": ["point:0", "judgment:0"]}
    roles = [
        _tool_reply(
            "choose_points",
            {"points": [point], "audit": [], "limitations": ["Scripted offline fixture."]},
        ),
        _tool_reply(
            "estimate_points",
            {
                "estimates": [
                    {
                        "step": 0,
                        "probability": 0.5,
                        "rationale": "Attempt-specific initial LLM feedback is supplied.",
                        "evidence_refs": ["judgment:0"],
                    }
                ],
                "limitations": ["Scripted offline fixture."],
            },
        ),
    ]
    actor, judge = _wire(
        monkeypatch,
        [reply for row in (2, 3, 4, 2, 3) for reply in (_edit(row), _done())],
        roles,
    )
    report = run_actor_critic(
        project,
        spec,
        exposure=_exposure(project, job, "mining_probes"),
        code_revision="suffix-authority",
    )
    assert report.initial_judged is not None and len(report.points) == 1
    checkpoint = CouncilCheckpoint(
        run_identity=spec.identity("suffix-authority"),
        phase="continuations",
        group=report.group,
        initial_judged=report.initial_judged,
        selected=(0,),
        wave=(0,),
        jobs=(point_job(spec, report.points[0].checkpoint),),
    )
    inventory = phase_inventory(project, spec, checkpoint, code_revision="suffix-authority")
    calls = (len(actor.requests), len(judge.judged))
    _selected_jobs(project, inventory, spec, code_revision="suffix-authority")
    with pytest.raises(ValueError, match="selected native checkpoint"):
        _selected_jobs(
            project,
            inventory.model_copy(
                update={"checkpoint": checkpoint.model_copy(update={"wave": (1,)})}
            ),
            spec,
            code_revision="suffix-authority",
        )
    with pytest.raises(ValueError, match="actual initial LLM feedback"):
        _selected_jobs(
            project,
            inventory.model_copy(
                update={"checkpoint": checkpoint.model_copy(update={"initial_judged": None})}
            ),
            spec,
            code_revision="suffix-authority",
        )
    assert (len(actor.requests), len(judge.judged)) == calls
