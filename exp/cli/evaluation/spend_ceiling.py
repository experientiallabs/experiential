"""Enforced operation ceilings for paid commands that consent without an operation file.

Only pricing primitives are imported, so ``exp build`` learning and the held-out audit can set
their ceiling without loading the executable runtime inventories.
"""

import math
from collections.abc import Iterator
from contextlib import contextmanager

from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.spending import operation_spend_ceiling, save_spend_receipt
from exp.runtime.models.spend_ceiling import (
    OperationSpendCeiling,
    install_spend_ceiling,
    operation_spend_scope,
)


def inventory_terms(components: tuple[CostComponent, ...]) -> tuple[float, int]:
    """Return an inventory's known-rate subtotal and its unpriced physical attempts.

    Args:
        components: Complete role inventory of one operation.

    Returns:
        The dollar ceiling the priced roles need and the attempt ceiling of the unpriced ones.
    """
    known = math.fsum(row.usd for row in components if row.usd is not None)
    return known, sum(row.attempts for row in components if row.usd is None)


@contextmanager
def enforced_ceiling(
    store: ProjectStore,
    operation: str,
    *,
    stage: str,
    ceiling_usd: float | None,
    unpriced_attempts: int,
    command: str,
) -> Iterator[OperationSpendCeiling]:
    """Enforce one consented ceiling over every paid role of the enclosed execution.

    The authorization is appended to the operation's durable ledger, and the operation's spend
    receipt is saved on the way out, including after a stop.

    Args:
        store: Project owning the operation's runtime directory.
        operation: Stable identity of the operation and its resumes.
        stage: Identity of the consented inventory; a resume replaces its earlier consent.
        ceiling_usd: Displayed and authorized dollar ceiling, or None when nothing was asked:
            earlier authorizations stand, and without any every request is refused.
        unpriced_attempts: Displayed and authorized ceiling on unpriced requests.
        command: Command identity shown at consent.

    Yields:
        The installed ceiling.
    """
    ceiling = operation_spend_ceiling(store, operation)
    if ceiling_usd is not None:
        ceiling.authorize(stage, ceiling_usd=ceiling_usd, unpriced_attempts=unpriced_attempts)
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        try:
            yield ceiling
        finally:
            save_spend_receipt(store, operation, ceiling, command)


def ceiling_line(known_usd: float, unpriced_attempts: int) -> str:
    """Describe the ceiling an operation without an operation file will enforce.

    Args:
        known_usd: Known-rate subtotal of the priced roles.
        unpriced_attempts: Physical attempts of the unpriced roles.

    Returns:
        One plain line for the consent preamble.
    """
    line = f"Spend ceiling: ${known_usd:.6f} across every priced role"
    if unpriced_attempts:
        line += (
            f"; unpriced roles may use at most {unpriced_attempts:,} physical attempts, "
            "authorized only by an explicit confirmation"
        )
    return line + "."
