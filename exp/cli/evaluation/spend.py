"""Component cost review feeding the shared CLI spend-consent owner."""

import math

from rich.console import Console
from rich.text import Text

from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.costs import CostedSpec, evaluation_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent, total_cost


def review_cost(console: Console, project: ProjectStore, spec: CostedSpec) -> float | None:
    """Show known components and unresolved dependencies before requesting authorization."""
    return review_components(console, evaluation_costs(project, spec))


def review_components(console: Console, components: tuple[CostComponent, ...]) -> float | None:
    """Keep unpriced dependencies visible and never present a known subtotal as the total."""
    grouped: dict[str, list[CostComponent]] = {}
    for row in components:
        grouped.setdefault(row.name, []).append(row)
    for name, rows in sorted(grouped.items()):
        attempts = sum(row.attempts for row in rows)
        amount = total_cost(tuple(rows))
        label = "unpriced" if amount is None else f"known-rate estimate ${amount:.6f}"
        console.print(Text(f"  {name}: {attempts} physical attempts, {label}"))
    total = total_cost(components)
    if total is None:
        known = math.fsum(row.usd for row in components if row.usd is not None)
        console.print(Text(f"Known components: ${known:.6f}; total remains unknown."))
    return total


def review_ceiling(
    console: Console, project: ProjectStore, spec: CostedSpec
) -> tuple[float, tuple[CostComponent, ...]]:
    """Show the spec's inventory and return its known subtotal and unpriced components.

    Args:
        console: Command-owned output console.
        project: Project whose catalog prices the inventory.
        spec: Paid action payload.

    Returns:
        The known-rate subtotal, which becomes the enforced ceiling by default, and every
        unpriced component.
    """
    components = evaluation_costs(project, spec)
    review_components(console, components)
    known = math.fsum(row.usd for row in components if row.usd is not None)
    return known, tuple(row for row in components if row.usd is None)
