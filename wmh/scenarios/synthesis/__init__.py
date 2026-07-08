"""Synthesis: write self-contained, judgeable scenarios from traces or task descriptions."""

from wmh.scenarios.synthesis.from_task import scenario_from_task
from wmh.scenarios.synthesis.scenario_set import EvalScenario, ScenarioSet
from wmh.scenarios.synthesis.synthesizer import ScenarioSynthesizer

__all__ = ["EvalScenario", "ScenarioSet", "ScenarioSynthesizer", "scenario_from_task"]
