import pytest
import router_real_analyze
from router_real_analyze import _bootstrap, _oracle_choices
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


def test_bootstrap_redraws_degenerate_zero_baseline_cost_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(router_real_analyze, "BOOTSTRAP_DRAWS", 200)
    routed = [
        {"scenario_id": "free", "group": "g", "reward": 0.0, "cost": 0.0},
        {"scenario_id": "paid", "group": "g", "reward": 1.0, "cost": 2.0},
    ]
    baseline = [
        {"scenario_id": "free", "group": "g", "reward": 0.0, "cost": 0.0},
        {"scenario_id": "paid", "group": "g", "reward": 1.0, "cost": 1.0},
    ]

    result = _bootstrap([(routed, baseline)], seed=7)

    assert result["draws"] == 200
    assert result["rejected_zero_baseline_cost_draws"] > 0
