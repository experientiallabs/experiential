"""Read-only quote binding and cost component regressions."""

from pathlib import Path

import pytest

from exp.common.judging import verify_persisted_calibration
from exp.common.models import ModelSnapshot
from exp.common.tasks import ToolSchema
from exp.common.traces import Trace
from exp.optimize.evaluation.planning import estimate_model_evaluation
from exp.optimize.evaluation.prepare_test import _prepare
from exp.optimize.evaluation.service_test import _prepared
from exp.optimize.router.automatic import service_test as build_fixtures
from exp.optimize.router.composition_test import _completion_reservation
from exp.simulation.engines.text.grounding import maximum_query_reservation
from exp.simulation.engines.text.resume import MAXIMUM_CELL_ATTEMPTS
from exp.simulation.specs import load_simulation_completion_contract


@pytest.mark.parametrize("tools", [False, True])
def test_quote_bounds_parallel_tool_retrieval_by_worker_output_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tools: bool
) -> None:
    """A tool-bearing turn can need multiple retrievals; a text turn still needs only one."""
    original_trace = build_fixtures._trace
    tool = ToolSchema(name="lookup", description="Look up an account.", input_schema={})

    def trace(index: int, model: ModelSnapshot) -> Trace:
        """Build immutable scenarios with or without declared tool schemas."""
        return original_trace(index, model).model_copy(update={"tools": (tool,) if tools else ()})

    monkeypatch.setattr(build_fixtures, "_trace", trace)
    project, _, _, prepared = _prepare(tmp_path)
    setup = prepared.setup
    query = setup.world_model_settings.query_embedding
    assert query is not None and setup.simulation_completion_input is not None
    unit = maximum_query_reservation(query).cost_usd
    assert unit is not None
    completion, _ = load_simulation_completion_contract(
        project.artifacts, setup.simulation_completion_input.artifact_id
    )
    queries_per_step = sum(
        min(
            request.request.maximum_output_tokens * setup.maximum_steps,
            setup.maximum_rollout_output_tokens,
        )
        if tools
        else setup.maximum_steps
        for request in completion.candidate_requests
    )
    expected = prepared.cost.scenario_count * queries_per_step * unit.value * MAXIMUM_CELL_ATTEMPTS
    assert prepared.cost.retrieval.maximum_cost_usd == pytest.approx(expected)
    assert 0 < prepared.cost.retrieval.estimated_cost_usd < expected


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
    assert changed.estimated_cost_usd == quote.estimated_cost_usd


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


def test_expected_cost_tracks_repeats_not_unused_rollout_limits(tmp_path: Path) -> None:
    """A large output allowance cannot masquerade as expected consumption on every turn."""
    project, _, _, prepared = _prepare(tmp_path)
    setup = prepared.setup.model_copy(update={"maximum_steps": 100})
    quote = estimate_model_evaluation(project, setup, judge_request=prepared.judge_request)
    larger = estimate_model_evaluation(
        project,
        setup.model_copy(
            update={
                "maximum_steps": 1000,
                "maximum_rollout_output_tokens": 10_000_000,
            }
        ),
        judge_request=prepared.judge_request,
    )
    assert larger.estimated_cost_usd == quote.estimated_cost_usd
    repeated = estimate_model_evaluation(
        project, setup.model_copy(update={"repeats": 3}), judge_request=prepared.judge_request
    )
    assert repeated.estimated_cost_usd == pytest.approx(3 * quote.estimated_cost_usd)
    assert quote.measured_turns == 0
    assert quote.captured_turns == 6


def test_cumulative_output_bound_keeps_unknown_retry_output_reserved(tmp_path: Path) -> None:
    """The cumulative success budget does not erase potentially billed failed provider attempts."""
    project, _, _, prepared = _prepare(tmp_path)
    setup = prepared.setup.model_copy(
        update={"maximum_steps": 100, "maximum_rollout_output_tokens": 1000}
    )
    quote = estimate_model_evaluation(project, setup, judge_request=prepared.judge_request)
    assert setup.simulation_completion_input is not None
    contract, _ = load_simulation_completion_contract(
        project.artifacts, setup.simulation_completion_input.artifact_id
    )
    retry_output = (
        sum(
            100
            * (item.request.maximum_attempts - 1)
            * 1000
            * item.request.output_usd_per_million_tokens
            / 1_000_000
            for item in contract.candidate_requests
        )
        * quote.scenario_count
        * MAXIMUM_CELL_ATTEMPTS
    )
    assert quote.workers.maximum_cost_usd >= retry_output
