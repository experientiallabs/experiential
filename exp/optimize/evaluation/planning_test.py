"""Read-only quote binding and cost component regressions."""

from pathlib import Path

import pytest

from exp.common.judging import verify_persisted_calibration
from exp.optimize.evaluation.planning import estimate_model_evaluation
from exp.optimize.evaluation.service_test import _prepared
from exp.optimize.router.composition_test import _completion_reservation
from exp.simulation.engines.text.resume import MAXIMUM_CELL_ATTEMPTS


def test_evaluation_quote_is_read_only_and_prices_every_stage(tmp_path: Path) -> None:
    """Quotes include both workers, the environment, retrieval and the persisted judge."""
    project, setup = _prepared(tmp_path, multiple=True)
    calibration, _ = verify_persisted_calibration(
        project, setup.simulation_protocol.judge_calibration_id
    )
    request = _completion_reservation("judge-model").model_copy(
        update={"model": calibration.judge_model}
    )
    before = project.artifacts.list_ids()
    quote = estimate_model_evaluation(project, setup, judge_request=request)
    assert quote.worker_count == 2
    assert quote.judgment_count == quote.scenario_count * 2
    assert quote.maximum_cost_usd >= quote.estimated_cost_usd > 0
    assert quote.workers.estimated_cost_usd > 0
    assert quote.simulation.estimated_cost_usd > 0
    assert quote.judge.estimated_cost_usd > 0
    assert quote.workers.maximum_cost_usd == pytest.approx(
        quote.scenario_count
        * 2
        * setup.maximum_steps
        * MAXIMUM_CELL_ATTEMPTS
        * _completion_reservation("candidate-a").absolute_maximum_call_cost_usd()
    )
    assert quote.judge.maximum_cost_usd == pytest.approx(
        quote.judgment_count * request.absolute_maximum_call_cost_usd()
    )
    assert project.artifacts.list_ids() == before
    changed = estimate_model_evaluation(
        project,
        setup.model_copy(update={"maximum_steps": setup.maximum_steps + 1}),
        judge_request=request,
    )
    assert changed.quote_sha256 != quote.quote_sha256
    assert changed.maximum_cost_usd > quote.maximum_cost_usd


def test_quote_refuses_unbound_or_wrong_judge_pricing(tmp_path: Path) -> None:
    """Missing pricing and mismatched judge identities fail closed, never return free work."""
    project, setup = _prepared(tmp_path, multiple=True)
    with pytest.raises(ValueError, match="persisted judge model"):
        estimate_model_evaluation(project, setup, judge_request=_completion_reservation("wrong"))
    with pytest.raises(ValueError, match="frozen completion"):
        estimate_model_evaluation(
            project,
            setup.model_copy(update={"simulation_completion_input": None}),
            judge_request=_completion_reservation("wrong"),
        )


def test_quote_refuses_calibration_status_drift(tmp_path: Path) -> None:
    """An authored default cannot be represented as a human-calibrated judge in a quote."""
    project, setup = _prepared(tmp_path, multiple=True)
    with pytest.raises(ValueError, match="judge status or rubric"):
        estimate_model_evaluation(
            project,
            setup.model_copy(update={"judgment_status": "human_calibrated"}),
            judge_request=_completion_reservation("judge-model"),
        )
