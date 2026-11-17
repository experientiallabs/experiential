"""Round identities are fixed by climb and round number."""

from exp.optimize.evaluation.hillclimb.history import round_identity


def test_round_identity_depends_only_on_climb_and_number() -> None:
    """A round's record and registration are found again from its climb and number."""
    assert round_identity("hill-climb-a", 1) == round_identity("hill-climb-a", 1)
    assert round_identity("hill-climb-a", 1) != round_identity("hill-climb-a", 2)
    assert round_identity("hill-climb-a", 1) != round_identity("hill-climb-b", 1)
