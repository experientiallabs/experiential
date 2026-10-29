"""Optional aggregate spending limits preserve finite numeric and dispatch authority."""

import pytest
from pydantic import ValidationError

from exp.optimize.evaluation.contracts import EvaluationBudget


@pytest.mark.parametrize("ceiling", [0, -1, float("inf"), float("nan")])
def test_evaluation_budget_rejects_non_authoritative_ceilings(ceiling: float) -> None:
    """An enabled aggregate limit must be positive and finite."""
    with pytest.raises(ValidationError):
        EvaluationBudget(maximum_cost_usd=ceiling, maximum_judgments=1)


def test_evaluation_budget_defaults_to_uncapped_accounting() -> None:
    """Omitted and explicit null limits agree without removing the judgment-count ceiling."""
    budget = EvaluationBudget(maximum_judgments=1)
    assert budget.maximum_cost_usd is None
    assert budget == EvaluationBudget(maximum_cost_usd=None, maximum_judgments=1)
    assert EvaluationBudget.model_validate_json(budget.model_dump_json()) == budget
    with pytest.raises(ValidationError):
        EvaluationBudget(maximum_judgments=0)
