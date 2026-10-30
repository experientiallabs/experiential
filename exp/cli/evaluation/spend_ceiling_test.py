"""Commands without an operation file enforce one ceiling and save its receipt."""

import json
from pathlib import Path

from exp.cli.evaluation.spend_ceiling import ceiling_line, enforced_ceiling, inventory_terms
from exp.common.project import ProjectConfig, ProjectStore
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.runtime.models.spend_ceiling import current_spend_ceiling


def test_inventory_terms_split_dollars_from_unpriced_attempts() -> None:
    """Priced roles form the dollar ceiling; unpriced roles only an attempt ceiling."""
    rows = (CostComponent("actor", 3, 0.25), CostComponent("world", 10, None))
    assert inventory_terms(rows) == (0.25, 10)
    assert "at most 10 physical attempts" in ceiling_line(0.25, 10)
    assert ceiling_line(0.25, 0) == "Spend ceiling: $0.250000 across every priced role."


def test_enforced_ceiling_installs_one_ceiling_and_saves_its_receipt(tmp_path: Path) -> None:
    """Inside the block transports see the ceiling; leaving it saves the receipt."""
    project = ProjectStore(tmp_path, "case")
    project.initialize(ProjectConfig(project_id="case"))
    with enforced_ceiling(
        project,
        "learning-case",
        stage="learning-case",
        ceiling_usd=0.5,
        unpriced_attempts=2,
        command="exp build case",
    ) as ceiling:
        assert current_spend_ceiling() is ceiling
    assert current_spend_ceiling() is None
    receipt = json.loads(
        (project.paths.runtime_directory / "spend" / "learning-case" / "receipt.json").read_text()
    )
    assert (receipt["ceiling_usd"], receipt["unpriced_attempt_ceiling"]) == (0.5, 2)
    # A pass that asked nothing adds no authorization and keeps the earlier one.
    with enforced_ceiling(
        project,
        "learning-case",
        stage="learning-case",
        ceiling_usd=None,
        unpriced_attempts=0,
        command="exp build case",
    ) as ceiling:
        assert len(ceiling.summary().authorizations) == 1
