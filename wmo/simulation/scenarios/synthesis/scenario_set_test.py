"""Tests for the ScenarioSet artifact (retain semantics, persistence)."""

from __future__ import annotations

from pathlib import Path

from wmo.common.core.types import EnvState
from wmo.simulation.scenarios.mining.facets import Outcome
from wmo.simulation.scenarios.synthesis import EvalScenario, ScenarioSet


def test_to_scenario_gives_the_minimal_view() -> None:
    scenario = EvalScenario(scenario_id="s1", task="do it", provenance=["t1"])
    minimal = scenario.to_scenario()
    assert minimal.task == "do it"
    assert minimal.provenance == ["t1"]


def test_retain_renormalizes_weights_and_invalidates_coverage() -> None:
    scenario_set = ScenarioSet(
        scenarios=[
            EvalScenario(scenario_id="s1", task="a", weight=0.5),
            EvalScenario(scenario_id="s2", task="b", weight=0.3),
            EvalScenario(scenario_id="s3", task="c", weight=0.2),
        ],
        corpus_traces=10,
        corpus_coverage=0.8,
        coverage_tau=0.7,
    )
    scenario_set.retain({"s1", "s3"})
    assert [s.scenario_id for s in scenario_set.scenarios] == ["s1", "s3"]
    assert sum(s.weight for s in scenario_set.scenarios) == 1.0
    assert scenario_set.scenarios[0].weight == 0.5 / 0.7
    assert scenario_set.corpus_coverage == 0.0  # stale coverage must not survive a drop
    assert scenario_set.coverage_tau == 0.0


def test_retain_nothing_is_safe() -> None:
    scenario_set = ScenarioSet(
        scenarios=[EvalScenario(scenario_id="s1", task="a", weight=1.0)], corpus_coverage=0.5
    )
    scenario_set.retain(set())
    assert scenario_set.scenarios == []
    assert scenario_set.corpus_coverage == 0.0


def test_scenario_set_save_load_roundtrip(tmp_path: Path) -> None:
    scenario_set = ScenarioSet(
        scenarios=[EvalScenario(scenario_id="s1", task="do it", weight=1.0)],
        corpus_traces=10,
        corpus_coverage=0.8,
        coverage_tau=0.7,
    )
    path = tmp_path / "scenarios.json"
    scenario_set.save(path)
    loaded = ScenarioSet.load(path)
    assert loaded == scenario_set


def test_w05_scenario_set_fixture_preserves_task_weight_and_provenance() -> None:
    """Map current `ScenarioSet` and `EvalScenario` to approved `TaskSet` and `TaskCase`.

    The fixture retains the task, checklist, source trace, outcome, corpus coverage, and workload
    weight. It freezes the current owner without adding a canonical W2 task schema.
    """
    scenario_set = ScenarioSet(
        scenarios=[
            EvalScenario(
                scenario_id="scenario-w05-refund",
                task="Refund order A-42",
                seed_state=EnvState(structured={"account_id": "acct-demo"}),
                checklist=["order A-42 status is refunded"],
                provenance=["prod-trace-w05-001"],
                cluster_name="orders/refund",
                weight=1.0,
                source_outcome=Outcome.SUCCESS,
            )
        ],
        corpus_traces=1,
        corpus_coverage=1.0,
        coverage_tau=0.8,
    )

    assert scenario_set.model_dump(mode="json") == {
        "scenarios": [
            {
                "scenario_id": "scenario-w05-refund",
                "task": "Refund order A-42",
                "seed_state": {"structured": {"account_id": "acct-demo"}, "scratchpad": ""},
                "checklist": ["order A-42 status is refunded"],
                "provenance": ["prod-trace-w05-001"],
                "cluster_name": "orders/refund",
                "weight": 1.0,
                "source_outcome": "success",
                "failure_category": None,
            }
        ],
        "clusters": [],
        "corpus_traces": 1,
        "corpus_coverage": 1.0,
        "coverage_tau": 0.8,
    }
    assert scenario_set.scenarios[0].to_scenario().model_dump(mode="json") == {
        "task": "Refund order A-42",
        "provenance": ["prod-trace-w05-001"],
    }
