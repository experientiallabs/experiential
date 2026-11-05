"""Native returned-owner settlement and retained unknown liability for real initial groups."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.continuations.job import (
    continuation_result_reference,
    execute_continuation_job,
)
from exp.optimize.evaluation.variance.bootstrap import resolve_bootstrap
from exp.optimize.evaluation.variance.council_prelude_execution import (
    NativePreludeExecution,
    PreludeDisposition,
)
from exp.optimize.evaluation.variance.council_prelude_hook import (
    CouncilPreludeCheckpoint,
    prelude_phase,
)
from exp.optimize.evaluation.variance.council_prelude_test import (
    _prelude_fixture,
    _register_fixture,
)
from exp.optimize.evaluation.variance.pipeline_spec import (
    StageContext,
    StageDependencies,
    StagePlan,
)
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.scoped_spending import ScopeBudgetRefused, parent_scope_summary
from exp.runtime.models.spend_ceiling import current_spend_ceiling


class ActorResponses(ScriptedJsonTransport):
    """Use ordinary native scripted actor replies under each requested exact model identity."""

    def post(
        self, url: str, *, headers: Mapping[str, str], payload: JsonObject, timeout_seconds: float
    ) -> JsonHttpResponse:
        """Keep native request recording while binding the response to its requested model."""
        response = super().post(
            url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
        )
        return JsonHttpResponse(response.status_code, {**response.body, "model": payload["model"]})


def test_native_returned_groups_settle_but_unknown_physical_claim_keeps_wave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real native G16 groups preserve all counts, original grants and unknown liabilities."""
    project, prepared, resume, contract, _ = _prelude_fixture(tmp_path, monkeypatch)
    registration = _register_fixture(project, prepared, resume, contract)
    owner = NativePreludeExecution(project, registration, code_revision="council-executor")
    context = StageContext(
        project=project,
        spec=resolve_bootstrap(project, contract.pipeline),
        identity="native-prelude-execution-test",
        code_revision="council-executor",
        upstream={},
        dependencies=StageDependencies(council_prelude_admission=owner),
        plan=StagePlan(stage="traceml", identity="native-prelude-execution-test", saved=False),
    )
    reply = JsonHttpResponse(
        200,
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Done."},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    actors = ActorResponses((reply,) * 32)
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actors
    )
    before = owner.ceiling.summary()
    first, second = owner.qualified.jobs[:2]
    checkpoint = CouncilPreludeCheckpoint(
        phase="traceml_group", actor_alias=first.actor_alias, job=first
    )
    with prelude_phase(context, checkpoint) as completed:
        group = execute_continuation_job(project, first, code_revision="council-executor")
        reference = continuation_result_reference(project, first, code_revision="council-executor")
        assert len(group.branches) == 16
        completed((reference,))
    with prelude_phase(context, checkpoint) as completed:
        execute_continuation_job(project, first, code_revision="council-executor")
        completed((reference,))
    after = owner.ceiling.summary()
    assert after.authorizations == before.authorizations
    assert len(actors.requests) == 16
    completions = project.artifacts.list_ids(artifact_type="native-spend-envelope-completion")
    assert len(completions) == 1
    disposition = PreludeDisposition.model_validate_json(
        project.artifacts.read_bytes(completions[0], "disposition.json")
    )
    assert len(disposition.failures) == 16
    with owner.ceiling._transaction() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM scoped_envelope_settlements").fetchone()[0]
            == 1
        )
    summary = parent_scope_summary(owner.ceiling)
    assert summary is not None and summary.legacy_attempt_limit == before.unpriced_attempt_ceiling
    checkpoint = CouncilPreludeCheckpoint(
        phase="traceml_group", actor_alias=second.actor_alias, job=second
    )
    with pytest.raises(ScopeBudgetRefused, match="unknown or unsettled"):
        with prelude_phase(context, checkpoint) as completed:
            execute_continuation_job(project, second, code_revision="council-executor")
            with pytest.raises(ValueError, match="exact job or attempts"):
                completed((reference,))
            scoped = current_spend_ceiling()
            assert scoped is not None
            reservation = scoped.reserve(
                alias=second.actor_alias, url="https://fixture.invalid", payload={}
            )
            scoped.mark_dispatched(reservation, "fixture-missing-response")
            scoped.retain_unknown(reservation)
            reference = continuation_result_reference(
                project, second, code_revision="council-executor"
            )
            completed((reference,))
    assert len(actors.requests) == 32
    assert owner.ceiling.summary().authorizations == before.authorizations
    retained = parent_scope_summary(owner.ceiling)
    assert retained is not None and retained.encumbered_attempts > summary.encumbered_attempts
    with owner.ceiling._transaction() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM scoped_envelope_settlements").fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM reservations WHERE state='unknown'"
            ).fetchone()[0]
            == 1
        )
