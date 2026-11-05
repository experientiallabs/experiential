"""Actual native owner/cost-plan checks with scripted offline transports only."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.continuations.job import execute_continuation_job
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.variance.actor_critic_test import _spec
from exp.optimize.evaluation.variance.council_checkpoint import CouncilCheckpoint
from exp.optimize.evaluation.variance.council_inventory import phase_inventory
from exp.optimize.evaluation.variance.judged import judge_study
from exp.optimize.evaluation.variance.judged_test import route_judge


def test_pending_native_group_and_judgments_are_priced_then_exact_owners_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither a saved actor group nor a raw reply silently removes its pending LLM grader."""
    project, job, actor = job_fixture(tmp_path, monkeypatch)
    spec = _spec(project, job).model_copy(update={"initial": job})
    checkpoint = CouncilCheckpoint(
        run_identity=spec.identity("inventory"),
        phase="actors",
        jobs=(job,),
    )
    pending = phase_inventory(project, spec, checkpoint, code_revision="inventory")
    assert pending.authority == "inventory_only"
    assert pending.attempts > 0 and not actor.requests and not pending.replayed

    execute_continuation_job(project, job, code_revision="inventory")
    group_saved = phase_inventory(project, spec, checkpoint, code_revision="inventory")
    assert 0 < group_saved.attempts < pending.attempts
    assert len(group_saved.replayed) == 1
    source = group_saved.replayed[0]

    router = route_judge(monkeypatch, [])
    judged_ref, _ = judge_study(
        project,
        source,
        spec.episode_judge,
        configuration=job.configuration,
        code_revision="inventory",
    )
    assert len(router.judged) == job.study.repeats
    fully_saved = phase_inventory(project, spec, checkpoint, code_revision="inventory")
    assert fully_saved.attempts == 0
    assert set(fully_saved.replayed) == {source, judged_ref}
    assert len(router.judged) == job.study.repeats
    assert sum(row.attempts for row in fully_saved.costs) == 0


def test_phase_inventory_rejects_native_identity_count_and_job_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inventory cannot obtain a lower quote by changing requested draws or native jobs."""
    project, job, actor = job_fixture(tmp_path, monkeypatch)
    spec = _spec(project, job)
    owner = spec.identity("inventory")
    with pytest.raises(ValueError, match="exact native council"):
        phase_inventory(
            project,
            spec,
            CouncilCheckpoint(run_identity="other", phase="selectors"),
            code_revision="inventory",
        )
    with pytest.raises(ValueError, match="drop or duplicate"):
        phase_inventory(
            project,
            spec,
            CouncilCheckpoint(run_identity=owner, phase="selectors", draws=()),
            code_revision="inventory",
        )
    missing = ArtifactInput(artifact_id="missing", sha256="0" * 64)
    changed = job.model_copy(update={"study": job.study.model_copy(update={"repeats": 1})})
    with pytest.raises(ValueError, match="reduce repeats"):
        phase_inventory(
            project,
            spec,
            CouncilCheckpoint(
                run_identity=owner,
                phase="continuations",
                wave=(0,),
                jobs=(changed,),
            ),
            code_revision="inventory",
        )
    with pytest.raises(ValueError, match="committed native group"):
        phase_inventory(
            project,
            spec,
            CouncilCheckpoint(run_identity=owner, phase="initial_judging"),
            code_revision="inventory",
        )
    assert missing.artifact_id == "missing" and not actor.requests


def test_scoped_parent_keeps_existing_bootstrap_operation_identity(tmp_path: Path) -> None:
    """The additive parent points at the existing native ledger, not a new identity namespace."""
    from datetime import UTC, datetime, timedelta

    from exp.optimize.evaluation.factory.author_retries import operation_identity
    from exp.runtime.models.scoped_spending import ParentSpendPolicy, register_parent_policy
    from exp.runtime.models.spend_ceiling import OperationSpendCeiling

    ceiling = OperationSpendCeiling(tmp_path / "original.sqlite", tariffs={})
    ceiling.authorize_policy(ceiling_usd=0, unpriced_attempts=2)
    proof = ArtifactInput(artifact_id="fixture-proof", sha256="0" * 64)
    policy = ParentSpendPolicy(
        operation="original-operation",
        legacy_attempt_limit=2,
        parent_attempt_limit=5,
        maximum_known_usd=0,
        deadline=datetime.now(UTC) + timedelta(hours=1),
        aliases=("actor",),
        authorization=proof,
        preparation=proof,
        execution=proof,
        producer_revision="fixture",
    )
    reference = register_parent_policy(ceiling, policy)
    assert reference.ledger_sha256 == operation_identity(ceiling)
