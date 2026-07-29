from router_real_analyze import _oracle_choices


def test_oracle_choices_preserves_scenarios_with_no_scored_model() -> None:
    choices = _oracle_choices(
        ["scored", "missing"],
        ["baseline", "challenger"],
        {
            ("scored", "baseline"): {"reward": 0.5, "cost": 0.1},
            ("scored", "challenger"): {"reward": 1.0, "cost": 0.2},
        },
        default_model="baseline",
    )

    assert choices == {
        "scored": "challenger",
        "missing": "baseline",
    }
