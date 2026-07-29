from typing import Any, cast

from router_real_compare import _cell_and_model

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry


def _matrix(rows: list[tuple[str, str, float | None]]) -> OutcomeMatrix:
    models = sorted({model for _scenario, model, _reward in rows})
    return OutcomeMatrix(
        pool=[
            PoolEntry(
                name=model,
                kind=ProviderKind.OPENAI,
                model=model,
                input_per_mtok=1.0,
                output_per_mtok=1.0,
            )
            for model in models
        ],
        outcomes=[
            ScenarioOutcome(
                scenario_id=scenario,
                task=scenario,
                model=model,
                reward=reward,
                cost_usd=0.1,
            )
            for scenario, model, reward in rows
        ],
    )


def test_cell_comparison_reports_sparse_coverage_by_model() -> None:
    real = _matrix(
        [
            ("s1", "a", 1.0),
            ("s2", "a", 0.0),
            ("s1", "b", 0.0),
            ("s2", "b", 1.0),
        ]
    )
    simulated = _matrix(
        [
            ("s1", "a", 1.0),
            ("s2", "a", None),
            ("s1", "b", 1.0),
            ("s2", "b", 1.0),
        ]
    )

    result = _cell_and_model("example", real, {seed: simulated for seed in range(5)})

    cell = cast(dict[str, Any], result["cell"])
    seed = cast(list[dict[str, Any]], cell["by_seed"])[0]
    assert seed["real_gradeable"] == 4
    assert seed["simulated_gradeable"] == 3
    assert seed["paired_cells"] == 3
    assert seed["paired_coverage_of_real"] == 0.75
    model = cast(dict[str, Any], result["model"])
    model_seed = cast(list[dict[str, Any]], model["by_seed"])[0]
    coverage = cast(dict[str, dict[str, Any]], model_seed["coverage_by_model"])
    assert coverage["a"]["paired_cells"] == 1
    assert coverage["a"]["paired_coverage_of_real"] == 0.5
    assert coverage["b"]["paired_cells"] == 2
    assert coverage["b"]["paired_coverage_of_real"] == 1.0
