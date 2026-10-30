"""Unknown components cannot disappear behind a useful known-price subtotal."""

from io import StringIO

from rich.console import Console

from exp.cli.evaluation.spend import review_components
from exp.optimize.evaluation.executable.model_costs import CostComponent


def test_review_names_unknown_runtime_beside_priced_actor() -> None:
    """Show the counted actor budget but never imply that it covers an unpriced dependency."""
    output = StringIO()
    total = review_components(
        Console(file=output, width=200),
        (
            CostComponent("actor", 3, 0.1),
            CostComponent("simulator", 10, None),
        ),
    )
    assert total is None
    text = output.getvalue()
    assert "actor: 3 physical attempts" in text
    assert "simulator: 10 physical attempts, unpriced" in text
    assert "Known components: $0.100000; total remains unknown" in text
