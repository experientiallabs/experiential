"""Tests for the scenario synthesizer (LLM writer + fallbacks)."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.scenarios.mining.facets import Outcome, TraceFacet
from wmh.scenarios.mining.facets_test import FakeProvider
from wmh.scenarios.synthesis import ScenarioSynthesizer


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
    assert scenario.scenario_id == "scenario-abcdef123456789"  # full trace_id: no collision risk
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
