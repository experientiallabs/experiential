"""Fresh baseline native authority, lineage boundaries and committed completion."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.project import ProjectStore
from exp.common.tasks import load_task_set
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance import baseline_execution as owner
from exp.optimize.evaluation.variance.baseline_execution import (
    BASELINE_RESULT_TYPE,
    BaselineExecutionApproval,
    BaselineExecutionProposal,
    execute_baseline_execution,
    plan_baseline_execution,
    propose_baseline_execution,
    register_baseline_execution,
)
from exp.optimize.evaluation.variance.bootstrap_history import read_bootstrap_preparation_plan
from exp.optimize.evaluation.variance.bootstrap_pilot_test import pilot_fixture, register_pilot
from exp.optimize.evaluation.variance.bootstrap_resume_test import PartialReviewWire
from exp.optimize.evaluation.variance.build_test import TASK_BANK, DecisionWire


def _proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, BaselineExecutionProposal]:
    """Create genuine historical native evidence and a separate fresh native scope."""
    project, spec, parent, pilot = pilot_fixture(tmp_path, monkeypatch)
    previous = register_pilot(project, spec, parent, pilot)
    assert spec.build.reviewer is not None and spec.build.spec.environment_families is not None
    settings = spec.build.model_copy(
        update={
            "reviewer": spec.build.reviewer.model_copy(
                update={"model": spec.bootstrap.author.model}
            ),
            "spec": spec.build.spec.model_copy(
                update={
                    "environment_families": spec.build.spec.environment_families.model_copy(
                        update={"probe_placement": "all_exact_schemas"}
                    )
                }
            ),
        }
    )
    return project, propose_baseline_execution(
        project, original_resume=previous, build=settings, code_revision="fresh-executor"
    )


def _register(project: ProjectStore, proposal: BaselineExecutionProposal) -> ArtifactInput:
    """Register exact fresh consent without creating any additional grant."""
    return register_baseline_execution(
        project,
        proposal,
        approval=BaselineExecutionApproval(proposal_sha256=sha256_json(proposal)),
        code_revision="fresh-executor",
    )


def test_actual_native_report_receipt_replays_and_retains_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual native clients write physical receipts and replay their committed report for free."""
    project, proposal = _proposal(tmp_path, monkeypatch)
    ceiling = operation_spend_ceiling(project, proposal.original_operation)
    held = ceiling.reserve(alias="unpriced", url="https://fixture.invalid/v1", payload={})
    ceiling.mark_dispatched(held, "historical-unknown")
    ceiling.retain_unknown(held)
    before = ceiling.summary()
    reference = _register(project, proposal)
    assert ceiling.summary() == before
    plan = plan_baseline_execution(project, reference, code_revision="fresh-executor")
    assert plan.accounted_unpriced_attempts == 1
    assert plan.preparation_floor.build_saved and plan.preparation_floor.remaining_tasks
    review = PartialReviewWire()
    review.initial = False
    wire = DecisionWire()
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: review
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.build_clients.HttpxJsonTransport", lambda: wire
    )
    receipt = execute_baseline_execution(project, reference, code_revision="fresh-executor")
    assert review.requests
    manifest = project.artifacts.read(receipt.artifact_id).manifest
    assert manifest.artifact_type == BASELINE_RESULT_TYPE
    result = ArtifactInput.model_validate_json(
        project.artifacts.read_bytes(receipt.artifact_id, "report.json")
    )
    assert result.artifact_id == proposal.plan.identity + "-result"
    assert result in manifest.inputs and reference in manifest.inputs
    assert ceiling.summary().authorizations == before.authorizations
    assert ceiling.summary().unpriced_attempts >= 1
    charged, count = ceiling.summary(), len(review.requests)
    assert execute_baseline_execution(project, reference, code_revision="fresh-executor") == receipt
    assert len(review.requests) == count and ceiling.summary() == charged
    assert plan_baseline_execution(project, reference, code_revision="fresh-executor").complete


@pytest.mark.parametrize("field", ["catalog_sha256", "original_source_sha256", "ledger_sha256"])
def test_substituted_pins_refuse_before_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Caller approval cannot manufacture native catalog, source or physical-root facts."""
    project, proposal = _proposal(tmp_path, monkeypatch)
    before = project.artifacts.list_ids()
    with pytest.raises(ValueError, match="changed"):
        _register(project, proposal.model_copy(update={field: "0" * 64}))
    assert project.artifacts.list_ids() == before


def test_full_quote_preserves_original_pending_reserve_and_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fitting fresh work alone does not admit it when original mandatory work cannot fit."""
    project, proposal = _proposal(tmp_path, monkeypatch)
    parent, floor, native, costs = owner._verified(project, proposal, "fresh-executor")
    assert floor.costs and floor.unpriced_attempts > 0
    pending = sum(row.attempts for row in costs if row.usd is None)
    with pytest.raises(ValueError, match="original preparation reserve"):
        owner._admit(
            parent.model_copy(update={"preparation_unpriced_attempts": pending}),
            floor,
            native,
            costs,
        )
    with pytest.raises(ValueError, match="deadline expired"):
        owner._admit(
            parent.model_copy(update={"deadline": datetime(2000, 1, 1, tzinfo=UTC)}),
            floor,
            native,
            costs,
        )


def test_roles_and_canonical_universe_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Held-out, old learning and already used mining lineages retain their proper roles."""
    project, proposal = _proposal(tmp_path, monkeypatch)
    exposure = proposal.exposure
    for role in ("final_evaluation", "world_learning"):
        lineage = next(key for key, value in exposure.roles.items() if value == role)
        moved = exposure.model_copy(update={"roles": {**exposure.roles, lineage: "mining_probes"}})
        with pytest.raises(ValueError, match="historical learning or held-out"):
            owner._preserve_roles(project, exposure, moved, proposal.used_task_ids)
    tasks = load_task_set(project.artifacts, exposure.bindings.artifact_id)
    task = next(row for row in tasks.tasks if row.task_id == proposal.used_task_ids[0])
    moved = exposure.model_copy(
        update={"roles": {**exposure.roles, task.lineage_group_id: "world_learning"}}
    )
    with pytest.raises(ValueError, match="mining_probes"):
        owner._preserve_roles(project, exposure, moved, proposal.used_task_ids)
    with pytest.raises(ValueError, match="canonical trace/task"):
        owner._preserve_roles(
            project, exposure, exposure.model_copy(update={"traces": {}}), proposal.used_task_ids
        )


def test_bank_and_operational_config_substitutions_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bank bytes and full operational settings stay pinned even beyond semantic build identity."""
    project, proposal = _proposal(tmp_path, monkeypatch)
    changed = proposal.model_copy(
        update={
            "build": proposal.build.model_copy(
                update={"spec": proposal.build.spec.model_copy(update={"concurrency": 1})}
            )
        }
    )
    assert sha256_json(changed) != sha256_json(proposal)
    with pytest.raises(ValueError, match="exact proposal"):
        register_baseline_execution(
            project,
            changed,
            approval=BaselineExecutionApproval(proposal_sha256=sha256_json(proposal)),
            code_revision="fresh-executor",
        )
    path = tmp_path / "banks" / (TASK_BANK + ".toml")
    path.write_text(path.read_text().replace("Does proposition 0", "Does changed proposition 0"))
    with pytest.raises(ValueError, match="changed"):
        _register(project, proposal)


def test_historical_floor_prices_native_original_inventory_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reserve comes from genuine native saved selection and outcome identities."""
    project, proposal = _proposal(tmp_path, monkeypatch)
    spec, proof = owner._history(project, proposal.original_resume)
    before = project.artifacts.list_ids()
    floor = read_bootstrap_preparation_plan(project, proposal.original_resume, spec=spec)
    assert floor.identity == proof.binding.original_operation
    assert floor.code_revision == proof.producer_revision
    assert floor.build_identity == proof.binding.build_identity
    assert floor.remaining_tasks == proposal.used_task_ids
    assert project.artifacts.list_ids() == before


def test_unknown_native_author_intent_never_creates_completion_or_redraw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted native request remains a liability and cannot buy an implicit replacement."""
    from collections.abc import Mapping

    from exp.common.core.artifacts import JsonObject
    from exp.runtime.models.providers.transport import JsonHttpResponse, ProviderTransportError

    project, proposal = _proposal(tmp_path, monkeypatch)
    reference = _register(project, proposal)
    calls: list[bool] = []

    class UnknownWire(PartialReviewWire):
        """Lose one actual native response after its durable physical dispatch intent."""

        def post(
            self,
            url: str,
            *,
            headers: Mapping[str, str],
            payload: JsonObject | None,
            timeout_seconds: float,
        ) -> JsonHttpResponse:
            """Retain an unknown physical outcome instead of fabricating a provider response."""
            calls.append(True)
            raise ProviderTransportError("fixture response lost")

    wire = UnknownWire()
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: wire
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.build_clients.HttpxJsonTransport", DecisionWire
    )
    from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError

    with pytest.raises((ProviderTransportError, DispatchCapExhaustedError)):
        execute_baseline_execution(project, reference, code_revision="fresh-executor")
    assert calls
    count = len(calls)
    ceiling = operation_spend_ceiling(project, proposal.original_operation)
    charged = ceiling.summary()
    with pytest.raises(ValueError, match="unknown outcome"):
        execute_baseline_execution(project, reference, code_revision="fresh-executor")
    assert len(calls) == count and ceiling.summary() == charged
    assert not project.artifacts.exists(proposal.plan.identity + "-result")
    assert not any(
        project.artifacts.read(identity).manifest.artifact_type == BASELINE_RESULT_TYPE
        for identity in project.artifacts.list_ids()
    )
