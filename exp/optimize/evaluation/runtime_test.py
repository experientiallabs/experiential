"""Catalog-backed prepared evaluation through real simulator, LM judge and persisted reports."""

from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest

from exp.optimize.evaluation.contracts import EvaluationBudget
from exp.optimize.evaluation.prepare_test import _prepare
from exp.optimize.evaluation.runtime import run_prepared_model_evaluation
from exp.optimize.router.automatic.service_test import _REVISION, _TIME, _RuntimeCatalog
from exp.runtime.models import RuntimeModelCatalog


def test_prepared_evaluation_runs_real_lm_judge_and_replays_without_model_calls(
    tmp_path: Path,
) -> None:
    """Only provider transport is deterministic; all evaluation execution is production code."""
    project, catalog, state, prepared = _prepare(tmp_path)
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))
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
    assert result.report.compared_cells == prepared.cost.scenario_count
    assert all(row.quality == 1 for row in result.report.models)
    assert {alias for alias, _ in state.completion_calls} == {
        "candidate-a",
        "candidate-b",
        "world",
        "judge",
    }
    before = (len(state.completion_calls), len(state.embedding_calls))
    replay = run_prepared_model_evaluation(
        project,
        prepared,
        runtime,
        budget=budget,
        provider_spend_consented=True,
        created_at=_TIME + timedelta(hours=1),
        code_revision=_REVISION,
    )
    assert replay == result
    assert before == (len(state.completion_calls), len(state.embedding_calls))


@pytest.mark.parametrize("consent, funded", [(False, True), (True, False)])
def test_consent_and_full_credit_ceiling_precede_credentials_and_writes(
    tmp_path: Path,
    consent: bool,
    funded: bool,
) -> None:
    """Underfunded or unapproved execution never creates a runtime client or artifact."""
    project, catalog, state, prepared = _prepare(tmp_path)
    before = (project.artifacts.list_ids(), state.credential_resolutions)
    budget = EvaluationBudget(
        maximum_cost_usd=prepared.cost.maximum_cost_usd * (1 if funded else 0.5),
        maximum_judgments=prepared.cost.judgment_count,
    )
    with pytest.raises(ValueError, match="consent|reserved quote"):
        run_prepared_model_evaluation(
            project,
            prepared,
            cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
            budget=budget,
            provider_spend_consented=consent,
            created_at=_TIME,
            code_revision=_REVISION,
        )
    assert before == (project.artifacts.list_ids(), state.credential_resolutions)


@pytest.mark.parametrize("drift", ["quote", "agent", "redaction", "worker"])
def test_runtime_refuses_changed_accepted_inputs_before_provider_dispatch(
    tmp_path: Path,
    drift: str,
) -> None:
    """A stored quote is not authority to execute changed selection or agent settings."""
    project, catalog, state, prepared = _prepare(tmp_path)
    if drift == "quote":
        prepared = prepared.model_copy(
            update={
                "cost": prepared.cost.model_copy(
                    update={"maximum_cost_usd": 0.0},
                )
            }
        )
    elif drift == "agent":
        prepared = prepared.model_copy(update={"agent_factory_sha256": "a" * 64})
    elif drift == "redaction":
        prepared = prepared.model_copy(update={"redacted_field_names": ("private-value",)})
    else:
        selected = catalog.models["candidate-b"]
        catalog = catalog.model_copy(
            update={
                "models": {
                    **catalog.models,
                    "candidate-b": selected.model_copy(update={"model": "other-model"}),
                }
            }
        )
    before = (project.artifacts.list_ids(), len(state.completion_calls), len(state.embedding_calls))
    with pytest.raises(ValueError, match="changed"):
        run_prepared_model_evaluation(
            project,
            prepared,
            cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
            budget=EvaluationBudget(maximum_cost_usd=1_000, maximum_judgments=100),
            provider_spend_consented=True,
            created_at=_TIME,
            code_revision=_REVISION,
        )
    assert before == (
        project.artifacts.list_ids(),
        len(state.completion_calls),
        len(state.embedding_calls),
    )


def test_prepared_runtime_is_public() -> None:
    """Hosting uses the public engine API instead of copying its runtime construction."""
    import exp

    assert exp.run_prepared_model_evaluation is run_prepared_model_evaluation
