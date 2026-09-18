"""Validation tests for finite evaluation authority."""

import pytest
from pydantic import ValidationError

from exp.optimize.evaluation.contracts import EvaluationBudget


@pytest.mark.parametrize("ceiling", [0, -1, float("inf"), float("nan")])
def test_evaluation_budget_rejects_non_authoritative_ceilings(ceiling: float) -> None:
    """Every evaluation must receive a positive finite ceiling before dispatch."""
    with pytest.raises(ValidationError):
        EvaluationBudget(maximum_cost_usd=ceiling, maximum_judgments=1)
