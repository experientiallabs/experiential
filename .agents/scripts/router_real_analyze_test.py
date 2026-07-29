from router_real_analyze import _oracle_choices
from router_real_ids import canonical_tau2_scenario_id


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


def test_tau2_id_canonicalization_does_not_rewrite_colons_inside_task_id() -> None:
    assert canonical_tau2_scenario_id("airline:44") == "airline/44"
    assert (
        canonical_tau2_scenario_id(
            "telecom/[mms_issue]break_apn[PERSONA:Hard]"
        )
        == "telecom/[mms_issue]break_apn[PERSONA:Hard]"
    )
