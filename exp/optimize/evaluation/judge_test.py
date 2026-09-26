"""Unusable judge output produces priced terminal evidence instead of paid replay loops."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from exp.common.models import AssistantAction, ModelClient, ModelRequest, ModelResponse
from exp.optimize.evaluation.contracts import EvaluationBudget
from exp.optimize.evaluation.prepare_test import _prepare
from exp.optimize.evaluation.runtime import run_prepared_model_evaluation
from exp.optimize.router.automatic.service_test import _REVISION, _TIME, _RuntimeCatalog
from exp.runtime.models import CatalogRoleName, ResolvedModel, RuntimeModelCatalog
from exp.runtime.models.providers.async_transport import ProviderDeadlineExceeded
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.transport import ProviderTransportError


class _FailingClient:
    """Retain real provider-shaped economics while returning invalid structured output."""

    def __init__(self, delegate: ModelClient, failure: str) -> None:
        """Retain the instrumented deterministic provider client."""
        self._delegate = delegate
        self._failure = failure

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return a paid response that cannot be parsed as a judgment."""
        if self._failure == "parameters":
            raise ProviderParameterError(
                message="temperature is incompatible with configured reasoning",
                param="temperature",
                code="invalid_parameter",
            )
        response = self._delegate.complete(request)
        if self._failure == "transport":
            raise ProviderTransportError("connection reset by provider")
        if self._failure == "deadline":
            raise ProviderDeadlineExceeded("provider deadline exceeded")
        if self._failure == "identity":
            return response.model_copy(
                update={"model": response.model.model_copy(update={"model_id": "wrong"})}
            )
        return response.model_copy(update={"output": AssistantAction(content="not JSON")})


class _FailingCatalog(_RuntimeCatalog):
    """Change only judge provider output, leaving the complete runtime path intact."""

    failure: str = "malformed"

    def resolve(self, alias: str, *, role: CatalogRoleName | None = None) -> ResolvedModel:
        """Wrap the selected judge provider with the malformed-output regression."""
        resolved = super().resolve(alias, role=role)
        return (
            replace(resolved, client=_FailingClient(resolved.client, self.failure))
            if alias == "judge"
            else resolved
        )


@pytest.mark.parametrize("failure", ["malformed", "transport", "deadline", "identity"])
def test_judge_failure_has_durable_cost_and_never_redispatches(
    tmp_path: Path, failure: str
) -> None:
    """Each admitted unusable response is excluded once and retained in report-generation spend."""
    project, catalog, state, prepared = _prepare(tmp_path)
    failing = _FailingCatalog(catalog, state)
    failing.failure = failure
    runtime = cast(RuntimeModelCatalog, failing)
    budget = EvaluationBudget(
        maximum_cost_usd=prepared.cost.maximum_cost_usd,
        maximum_judgments=prepared.cost.judgment_count,
    )
    result = run_prepared_model_evaluation(
        project,
        prepared,
        runtime,
        budget=budget,
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert result.report.compared_cells == 0
    assert result.report.excluded_cells == prepared.cost.scenario_count
    assert result.judge_cost_usd > 0
    calls = len(state.completion_calls)
    replay = run_prepared_model_evaluation(
        project,
        prepared,
        runtime,
        budget=budget,
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert replay == result
    assert len(state.completion_calls) == calls


def test_judge_parameter_error_stops_without_excluding_cells(tmp_path: Path) -> None:
    """A shared local request bug is actionable, rather than a failure for every candidate."""
    project, catalog, state, prepared = _prepare(tmp_path)
    failing = _FailingCatalog(catalog, state)
    failing.failure = "parameters"
    with pytest.raises(ValueError, match="judge request settings are invalid.*temperature"):
        run_prepared_model_evaluation(
            project,
            prepared,
            cast(RuntimeModelCatalog, failing),
            budget=EvaluationBudget(maximum_cost_usd=100, maximum_judgments=100),
            provider_spend_consented=True,
            created_at=_TIME,
            code_revision=_REVISION,
        )
    assert not any(alias == "judge" for alias, _ in state.completion_calls)
    assert not any(
        project.artifacts.read(key).manifest.artifact_type == "judgment-exclusion"
        for key in project.artifacts.list_ids()
    )
