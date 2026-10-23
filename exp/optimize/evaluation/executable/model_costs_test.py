"""Per-role pricing primitives keep unknown tariffs unknown for fidelity runs."""

from pathlib import Path

import pytest

from exp.common.models import load_model_catalog, write_model_catalog
from exp.optimize.evaluation.executable.model_costs import fidelity_costs, model_cost, total_cost
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


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("window", [None, 200000])
def test_inventory_uses_same_total_generation_bound_as_physical_admission(
    tmp_path: Path, reasoning: bool, window: int | None
) -> None:
    """Even none/default reasoning cannot make a requested visible limit a billed cap."""
    project, _ = source_project(tmp_path)
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["model"]
    assert record.capabilities is not None
    record = record.model_copy(
        update={
            "capabilities": record.capabilities.model_copy(
                update={
                    "supports_reasoning": reasoning,
                    "context_window_tokens": window,
                    "input_cost_per_million_tokens_usd": 1.0,
                    "output_cost_per_million_tokens_usd": 2.0,
                }
            )
        }
    )
    write_model_catalog(
        project.model_catalog_path, catalog.model_copy(update={"models": {"model": record}})
    )
    result = model_cost(project, "model", 3, 16384, input_tokens=1000)
    if window is None:
        assert result.usd is None
    else:
        assert result.usd == pytest.approx(3 * (1000 + 2 * window) / 1000000)
