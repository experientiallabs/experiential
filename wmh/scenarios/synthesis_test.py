"""Tests for scenario synthesis and the ScenarioSet artifact."""

from __future__ import annotations

from pathlib import Path

from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.scenarios.facets import Outcome, TraceFacet
from wmh.scenarios.facets_test import FakeProvider
from wmh.scenarios.synthesis import EvalScenario, ScenarioSet, ScenarioSynthesizer


def _trace() -> Trace:
    step = Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="cancel", arguments={"id": "R1"}),
        observation=Observation(content="cancelled"),
        task="cancel my reservation",
    )
    return Trace(trace_id="abcdef123456789", steps=[step])


def _facet() -> TraceFacet:
    return TraceFacet(
        trace_id="abcdef123456789",
        task_summary="Cancel a reservation",
        tool_signature="cancel",
        outcome=Outcome.SUCCESS,
    )


def test_synthesize_parses_scenario_json() -> None:
    reply = (
        '{"task": "Cancel reservation R1 for the customer.", '
        '"initial_state": "Reservation R1 exists and is active.", '
        '"checklist": ["Reservation R1 is cancelled", " ", "The customer is informed"]}'
    )
    scenario = ScenarioSynthesizer(FakeProvider(reply)).synthesize(_trace(), _facet())
    assert scenario.scenario_id == "scenario-abcdef123456"
    assert scenario.task == "Cancel reservation R1 for the customer."
    assert scenario.seed_state.scratchpad == "Reservation R1 exists and is active."
    assert scenario.checklist == ["Reservation R1 is cancelled", "The customer is informed"]
    assert scenario.provenance == ["abcdef123456789"]
    assert scenario.source_outcome is Outcome.SUCCESS


def test_synthesize_falls_back_to_facet_summary_on_garbage() -> None:
    scenario = ScenarioSynthesizer(FakeProvider("nope")).synthesize(_trace(), _facet())
    assert scenario.task == "Cancel a reservation"
    assert scenario.checklist == []
    assert scenario.seed_state.scratchpad == ""


def test_to_scenario_gives_the_minimal_view() -> None:
    scenario = EvalScenario(scenario_id="s1", task="do it", provenance=["t1"])
    minimal = scenario.to_scenario()
    assert minimal.task == "do it"
    assert minimal.provenance == ["t1"]


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
