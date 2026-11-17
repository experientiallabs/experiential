"""Deterministic held-out selection over independent leakage units.

A split assigns whole units (a lineage, or a component of lineages that share evidence) to one
side, never parts of a unit. Units are ranked by a salted SHA-256 of their identity, so the
assignment depends only on the unit identities and the salt: adding a unit moves no other unit
across the boundary unless the held-out count changes, and changing the salt draws a fresh split.
"""

import hashlib
import math
from collections.abc import Iterable


def held_out_count(units: int, fraction: float) -> int:
    """Choose how many units to hold out, keeping at least one unit on the training side.

    Args:
        units: Independent units eligible for the split.
        fraction: Requested held-out share in [0, 1); exactly zero selects all training.

    Returns:
        The rounded held-out count, at least one when any unit can be spared and the fraction
        is positive, and never every unit.
    """
    if units < 2 or fraction == 0:
        return 0
    target = math.floor(units * fraction + 0.5)
    return min(units - 1, max(1, target))


def salted_order(identities: Iterable[str], salt: str) -> tuple[str, ...]:
    """Order unit identities by a salted digest, the ranking every split draws from.

    Args:
        identities: Unit identities; duplicates are kept in place.
        salt: Split salt; a different salt draws an independent ranking.

    Returns:
        The identities, ranked by ``sha256(f"{salt}:{identity}")``.
    """
    return tuple(
        sorted(
            identities,
            key=lambda identity: hashlib.sha256(f"{salt}:{identity}".encode()).hexdigest(),
        )
    )
