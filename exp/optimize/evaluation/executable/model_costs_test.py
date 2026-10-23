"""Per-role pricing primitives keep unknown tariffs unknown for fidelity runs."""

from pathlib import Path

from exp.optimize.evaluation.executable.model_costs import fidelity_costs, total_cost
from exp.optimize.evaluation.fidelity.runner_test import source_project


def test_fidelity_inventory_names_every_paid_role_at_the_run_ceiling(tmp_path: Path) -> None:
    """Predictor and judge each carry the run's whole attempt ceiling; unknown stays unknown."""
    project, spec = source_project(tmp_path)
    rows = fidelity_costs(project, spec)
    assert [row.name for row in rows] == ["model", "model"]
    assert all(row.attempts == spec.maximum_physical_attempts for row in rows)
    assert total_cost(rows) is None
    unjudged = fidelity_costs(project, spec.model_copy(update={"judged": False}))
    assert [row.name for row in unjudged] == ["model"]
