"""Held-out selection keeps whole units, a training side and a salt-determined ranking."""

import hashlib

from exp.common.evaluations.splits import held_out_count, salted_order


def test_held_out_count_rounds_and_keeps_a_training_unit() -> None:
    """Rounding never empties either side once a unit can be spared."""
    assert held_out_count(1, 0.5) == 0
    assert held_out_count(5, 0) == 0
    assert held_out_count(2, 0.01) == 1
    assert held_out_count(10, 0.25) == 3
    assert held_out_count(4, 0.99) == 3


def test_salted_order_ranks_by_the_salted_digest_only() -> None:
    """Input order is irrelevant; a different salt draws a different ranking."""
    units = [f"lineage-{index}" for index in range(12)]
    ranked = salted_order(units, "salt-a")
    assert ranked == salted_order(reversed(units), "salt-a")
    assert ranked == tuple(
        sorted(units, key=lambda unit: hashlib.sha256(f"salt-a:{unit}".encode()).hexdigest())
    )
    assert ranked != salted_order(units, "salt-b")
