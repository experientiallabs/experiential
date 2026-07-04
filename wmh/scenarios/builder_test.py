"""Tests for the end-to-end scenario-set build pipeline (offline: fake LLM + hashing embedder)."""

from __future__ import annotations

import pytest

from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import HashingEmbedder
from wmh.scenarios.builder import ScenarioBuildConfig, build_scenario_set
from wmh.scenarios.facets import Outcome, TraceFacet


class JsonEchoProvider:
    """Answers cluster-naming, synthesis, and checklist-judge prompts with valid canned JSON."""

    def __init__(self, judge_success: bool = True) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self._judge_success = "true" if judge_success else "false"
        self.judge_calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        if "name one cluster" in system.lower():
            return Completion(text='{"name": "Cluster Name", "description": "What they share."}')
        if "grade one ai-agent episode" in system.lower():
            self.judge_calls += 1
            ok = self._judge_success
            return Completion(
                text=f'{{"passed": [{ok}], "success": {ok}, "critique": "x"}}'
            )
        return Completion(
            text=(
                '{"task": "Do the synthesized task.", "initial_state": "The world exists.", '
                '"checklist": ["it worked"]}'
            )
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _trace(trace_id: str, task: str) -> Trace:
    step = Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="go", arguments={}),
        observation=Observation(content="ok"),
        task=task,
    )
    return Trace(trace_id=trace_id, steps=[step])


# Distinct wordings (hashing trigram embeddings must not SemDeDup-collapse them) in two topics.
_SUMMARIES = [
    "cancel an upcoming flight reservation",
    "refund a cancelled airline ticket",
    "rebook a missed connection to Denver",
    "drop the extra checked bag fee",
    "switch a red-eye to a morning departure",
    "apply travel credit toward airfare",
    "update the shipping address on an order",
    "change delivery to the office location",
    "correct a typo in the street number",
    "forward a package to a new city",
    "set a preferred drop-off instruction",
    "merge duplicate customer address records",
]


def _corpus() -> tuple[list[Trace], list[TraceFacet]]:
    traces: list[Trace] = []
    facets: list[TraceFacet] = []
    for i, summary in enumerate(_SUMMARIES):
        traces.append(_trace(f"t{i}", summary))
        facets.append(
            TraceFacet(
                trace_id=f"t{i}",
                task_summary=summary,
                tool_signature="go",
                outcome=Outcome.SUCCESS,
            )
        )
    return traces, facets


def test_build_scenario_set_end_to_end_offline() -> None:
    traces, facets = _corpus()
    scenario_set = build_scenario_set(
        traces,
        facets,
        JsonEchoProvider(),
        HashingEmbedder(dim=64),
        ScenarioBuildConfig(budget=4, k=2, seed=0),
    )
    assert len(scenario_set.scenarios) == 4
    assert scenario_set.corpus_traces == 12
    assert 0.0 < scenario_set.corpus_coverage <= 1.0
    assert len(scenario_set.clusters) == 2
    assert all(s.cluster_name == "Cluster Name" for s in scenario_set.scenarios)
    assert all(s.checklist == ["it worked"] for s in scenario_set.scenarios)
    assert sum(s.weight for s in scenario_set.scenarios) == pytest.approx(1.0)


def test_build_scenario_set_rejects_misaligned_inputs() -> None:
    traces, facets = _corpus()
    with pytest.raises(ValueError, match="facets"):
        build_scenario_set(
            traces,
            facets[:-1],
            JsonEchoProvider(),
            HashingEmbedder(dim=64),
            ScenarioBuildConfig(),
        )
    with pytest.raises(ValueError, match="non-empty"):
        build_scenario_set(
            [], [], JsonEchoProvider(), HashingEmbedder(dim=64), ScenarioBuildConfig()
        )


def test_build_validates_checklists_against_recorded_outcomes() -> None:
    """Traces with a recorded reward exercise the inline back-agreement gate."""
    traces, facets = _corpus()
    for trace in traces:
        trace.metadata["reward"] = 1.0  # recorded success; judge must agree

    agreeing = JsonEchoProvider(judge_success=True)
    kept = build_scenario_set(
        traces, facets, agreeing, HashingEmbedder(dim=64), ScenarioBuildConfig(budget=4, k=2)
    )
    assert len(kept.scenarios) == 4
    assert agreeing.judge_calls == 4  # one back-agreement check per scenario

    disagreeing = JsonEchoProvider(judge_success=False)
    dropped = build_scenario_set(
        traces, facets, disagreeing, HashingEmbedder(dim=64), ScenarioBuildConfig(budget=4, k=2)
    )
    assert dropped.scenarios == []  # regenerated once, still disagreeing -> dropped
    assert disagreeing.judge_calls == 8  # two judge calls per scenario (original + regen)


def test_build_validation_skips_traces_without_recorded_outcome() -> None:
    traces, facets = _corpus()  # no reward metadata anywhere
    provider = JsonEchoProvider(judge_success=False)  # would fail everything if consulted
    scenario_set = build_scenario_set(
        traces, facets, provider, HashingEmbedder(dim=64), ScenarioBuildConfig(budget=4, k=2)
    )
    assert len(scenario_set.scenarios) == 4
    assert provider.judge_calls == 0
