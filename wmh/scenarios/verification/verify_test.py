"""Tests for the verification loop (back-agreement, solvability, report semantics)."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.scenarios.mining.facets import Outcome
from wmh.scenarios.synthesis import EvalScenario, ScenarioSet
from wmh.scenarios.verification import (
    ChecklistJudge,
    ScenarioVerdict,
    VerificationReport,
    verify_scenarios,
)


class ScriptedProvider:
    """Plays back one canned reply per call, cycling; records prompts."""

    def __init__(self, replies: list[str]) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self._replies = replies
        self.calls: list[tuple[str, str]] = []
        self.last_max_tokens = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.calls.append((system, messages[0].content))
        self.last_max_tokens = max_tokens
        reply = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        return Completion(text=reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


class EmptyRetriever:
    """Retriever stub: nothing indexed, nothing retrieved; records enrichment attempts."""

    def __init__(self) -> None:
        self.added: list[Step] = []

    def index(self, traces: list[Trace]) -> None: ...

    def topk(self, state: EnvState, action: Action, k: int) -> list[Step]:
        return []

    def add(self, step: Step) -> None:
        self.added.append(step)

    def sample(self, n: int) -> list[Step]:
        return []


class OneShotAgent:
    """Takes one fixed tool call, then signals done."""

    def act(self, task: str | None, state: EnvState, history: list[Step]) -> Action:
        if history:
            return Action(kind=ActionKind.MESSAGE, content="<DONE>")
        return Action(kind=ActionKind.TOOL_CALL, name="do_it", arguments={})


def _judge_reply(*, success: bool, passed: list[bool]) -> str:
    passed_json = ", ".join("true" if p else "false" for p in passed)
    return (
        f'{{"passed": [{passed_json}], "success": {"true" if success else "false"}, '
        f'"critique": "graded"}}'
    )


def _scenario() -> EvalScenario:
    return EvalScenario(
        scenario_id="s1",
        task="do the thing",
        checklist=["the thing is done"],
        provenance=["t1"],
        source_outcome=Outcome.SUCCESS,
    )


def _source_trace(*, reward: float) -> Trace:
    step = Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="do_it", arguments={}),
        observation=Observation(content="done"),
        task="do the thing",
    )
    return Trace(trace_id="t1", steps=[step], metadata={"reward": reward})


def _world_model(provider: ScriptedProvider) -> tuple[WorldModel, EmptyRetriever]:
    retriever = EmptyRetriever()
    return WorldModel(provider, retriever, telemetry_root="/tmp/wmh-test-telemetry"), retriever


def test_verify_scenarios_agreeing_and_solvable() -> None:
    # Call order: source-trace judge, world-model step, rollout judge.
    provider = ScriptedProvider(
        [
            _judge_reply(success=True, passed=[True]),  # back-agreement on source
            '{"output": "did the thing", "is_error": false}',  # world-model observation
            _judge_reply(success=True, passed=[True]),  # rollout grade
        ]
    )
    world_model, retriever = _world_model(provider)
    report = verify_scenarios(
        ScenarioSet(scenarios=[_scenario()]),
        [_source_trace(reward=1.0)],
        world_model,
        OneShotAgent(),
        ChecklistJudge(provider),
        max_steps=3,
    )
    # Evaluation rollouts must never enrich the retrieval index (order-dependence leak).
    assert retriever.added == []
    verdict = report.verdicts[0]
    assert verdict.back_agreement is True
    assert verdict.solvable is True
    assert verdict.ok
    assert report.back_agreement_rate == 1.0
    assert report.solvable_rate == 1.0


def test_verify_scenarios_flags_disagreement_with_recorded_failure() -> None:
    # The corpus says the source episode FAILED (reward 0) but the judge grades it a success.
    provider = ScriptedProvider(
        [
            _judge_reply(success=True, passed=[True]),
            '{"output": "did the thing", "is_error": false}',
            _judge_reply(success=False, passed=[False]),
        ]
    )
    report = verify_scenarios(
        ScenarioSet(scenarios=[_scenario()]),
        [_source_trace(reward=0.0)],
        _world_model(provider)[0],
        OneShotAgent(),
        ChecklistJudge(provider),
        max_steps=3,
    )
    verdict = report.verdicts[0]
    assert verdict.back_agreement is False
    assert verdict.solvable is False
    assert not verdict.ok


def test_verify_scenarios_without_source_trace_skips_back_agreement() -> None:
    provider = ScriptedProvider(
        [
            '{"output": "did the thing", "is_error": false}',
            _judge_reply(success=True, passed=[True]),
        ]
    )
    report = verify_scenarios(
        ScenarioSet(scenarios=[_scenario()]),
        [],  # no source corpus
        _world_model(provider)[0],
        OneShotAgent(),
        ChecklistJudge(provider),
        max_steps=3,
    )
    assert report.verdicts[0].back_agreement is None
    assert report.verdicts[0].solvable is True


def test_world_model_enrichment_resumes_after_verification() -> None:
    provider = ScriptedProvider(['{"output": "ok", "is_error": false}'])
    world_model, retriever = _world_model(provider)
    verify_scenarios(
        ScenarioSet(scenarios=[_scenario()]),
        [],
        world_model,
        OneShotAgent(),
        ChecklistJudge(provider),
        max_steps=2,
    )
    assert retriever.added == []
    # `frozen` is scoped to the verification run: serving sessions must enrich again afterwards.
    session = world_model.new_session(task="t")
    world_model.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="do_it", arguments={}))
    assert len(retriever.added) == 1


def test_verification_report_rates_handle_empty() -> None:
    report = VerificationReport(verdicts=[])
    assert report.back_agreement_rate == 0.0
    assert report.solvable_rate == 0.0
    only_unknown = VerificationReport(
        verdicts=[ScenarioVerdict(scenario_id="s", back_agreement=None, solvable=True)]
    )
    assert only_unknown.back_agreement_rate == 0.0
