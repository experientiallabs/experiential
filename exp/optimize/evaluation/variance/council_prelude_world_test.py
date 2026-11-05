"""Native initial-world authority under an exact prelude and original paid envelope."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_json
from exp.optimize.evaluation.continuations.job import load_continuation_runtime
from exp.optimize.evaluation.executable.runtime_authority import runtime_authority_scope
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance.bootstrap_history import read_bootstrap_preparation_provenance
from exp.optimize.evaluation.variance.council_parallel import council_draws
from exp.optimize.evaluation.variance.council_prelude import qualify_council_prelude
from exp.optimize.evaluation.variance.council_prelude_hook import CouncilPreludeCheckpoint
from exp.optimize.evaluation.variance.council_prelude_inventory import prelude_inventory
from exp.optimize.evaluation.variance.council_prelude_test import (
    _prelude_fixture,
    _register_fixture,
)
from exp.optimize.evaluation.variance.council_prelude_world import (
    register_prelude_runtime,
    save_prelude_inventory,
)
from exp.runtime.models.scoped_spending import (
    SpendEnvelopePlan,
    open_spend_envelope,
    spend_envelope_scope,
)


def test_native_initial_job_restores_before_full_council_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prelude grants only its exact initial job and cannot refill its world or invent a trial."""
    project, prepared, resume, contract, wire = _prelude_fixture(tmp_path, monkeypatch)
    registration = _register_fixture(project, prepared, resume, contract)
    qualified = qualify_council_prelude(
        project, prepared, resume, contract, code_revision="council-executor"
    )
    binding = read_bootstrap_preparation_provenance(
        project, resume, spec=contract.preparation.preparation_spec
    ).binding
    ceiling = operation_spend_ceiling(project, binding.original_operation)
    job = qualified.jobs[0]
    checkpoint = CouncilPreludeCheckpoint(
        phase="traceml_group", actor_alias=job.actor_alias, job=job
    )
    inventory = prelude_inventory(project, qualified, checkpoint)
    reference = save_prelude_inventory(
        project, registration.qualification, checkpoint, code_revision="council-executor"
    )
    envelope_plan = SpendEnvelopePlan(
        policy=registration.policy,
        scope="native-initial-job",
        inventory=reference,
        maximum_attempts=inventory.attempts,
        maximum_known_usd=inventory.known_usd,
        remaining_minimum_attempts=inventory.remaining.attempts,
        requirements_sha256=sha256_json(contract),
    )
    missing_floor = open_spend_envelope(
        ceiling,
        envelope_plan.model_copy(
            update={"scope": "missing-future-work", "remaining_minimum_attempts": 0}
        ),
    )
    with spend_envelope_scope(ceiling, missing_floor):
        with pytest.raises(ValueError, match="exact admitted native prelude"):
            register_prelude_runtime(
                project,
                registration.qualification,
                reference,
                job,
                code_revision="council-executor",
            )
    envelope = open_spend_envelope(ceiling, envelope_plan)
    calls, before = len(wire.requests), ceiling.summary()
    with spend_envelope_scope(ceiling, envelope):
        grant = register_prelude_runtime(
            project, registration.qualification, reference, job, code_revision="council-executor"
        )
        assert grant == register_prelude_runtime(
            project, registration.qualification, reference, job, code_revision="council-executor"
        )
        with pytest.raises(ValueError, match="exact job or native wave"):
            grant.require_configuration(project, {})
        with pytest.raises(ValueError, match="exact job or native wave"):
            grant.require_configuration(project, job.configuration, runtime_sha256="0" * 64)
        changed = grant.model_copy(
            update={
                "limits": grant.limits.model_copy(
                    update={"maximum_physical_attempts": grant.limits.maximum_physical_attempts + 1}
                )
            }
        )
        with pytest.raises(ValueError, match="exact job or native wave"):
            changed.require_configuration(project, job.configuration)
        with pytest.raises(ValueError, match="exact admitted native prelude"):
            register_prelude_runtime(
                project,
                registration.qualification,
                reference,
                qualified.jobs[1],
                code_revision="council-executor",
            )

        def restore() -> str:
            """Restore using the genuine frozen runtime and copied bounded native contexts."""
            with load_continuation_runtime(project, job) as runtime:
                return runtime.probe_sha256

        with runtime_authority_scope(lambda project, incoming: grant):
            assert council_draws((restore, restore), 2) == (
                job.study.probe_sha256,
                job.study.probe_sha256,
            )
            with pytest.raises(ValueError, match="another exact incoming job"):
                load_continuation_runtime(project, qualified.jobs[1])
    assert ceiling.summary() == before and len(wire.requests) == calls
    with pytest.raises(ValueError, match="active native spend envelope"):
        grant.require_configuration(project, job.configuration)
    second = open_spend_envelope(
        ceiling, envelope_plan.model_copy(update={"scope": "no-world-refill"})
    )
    with spend_envelope_scope(ceiling, second):
        with pytest.raises(ValueError, match="cannot refill"):
            register_prelude_runtime(
                project,
                registration.qualification,
                reference,
                job,
                code_revision="council-executor",
            )
    assert not project.artifacts.list_ids(artifact_type="executable-trial")
