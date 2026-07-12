"""Tests for the research build/eval primitives (no network: fake provider + judge)."""

from __future__ import annotations

import pytest

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.engine.replay import ReplayReport
from wmh.optimize.judge import JudgeResult
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.research.pipeline import optimize_prompt, score_prompt


class FakeProvider:
    """Records rollout temperatures; returns a fixed prediction and a fixed reflection mutation."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self.rollout_temps: list[float] = []

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        if "improve the system prompt" in system:
            return Completion(text="IMPROVED")
        self.rollout_temps.append(temperature)
        return Completion(text="predicted")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


class FakeJudge:
    def __init__(self, score: float = 0.6) -> None:
        self._score = score
        self.calls = 0

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        self.calls += 1
        # A high mean-of-5 driven by format/plausibility, but low factuality — the exact split the
        # score_dimension lever exists to separate.
        return JudgeResult(
            score=self._score,
            critique="tweak it",
            dimensions={
                "format": 0.9,
                "factuality": 0.2,
                "consistency": 0.8,
                "realism": 0.9,
                "quality": 0.7,
            },
        )


def _trace(tid: str, n: int = 2) -> Trace:
    steps = [
        Step(
            action=Action(kind=ActionKind.TOOL_CALL, name="f", arguments={"i": i}),
            observation=Observation(content=f"real-{i}"),
            state_before=EnvState(structured={"loc": "shop"}),
            task="t",
        )
        for i in range(n)
    ]
    return Trace(trace_id=tid, steps=steps)


def test_optimize_prompt_returns_winner_at_deterministic_temperature() -> None:
    provider = FakeProvider()
    result = optimize_prompt(
        [_trace("tr1"), _trace("tr2")],
        [_trace("te1")],
        "BASE",
        provider=provider,
        judge=FakeJudge(),
        embedder=None,
        budget=8,
        seed=3,
    )
    assert result.prompt  # non-empty winning prompt
    # Rollouts run deterministically (T=0): no sampling knob is exposed by the harness.
    assert provider.rollout_temps and all(t == 0.0 for t in provider.rollout_temps)


def test_score_prompt_delegates_to_replay_and_returns_mean() -> None:
    provider = FakeProvider()
    judge = FakeJudge(score=0.6)
    held_out = [_trace("te1", n=3)]
    mean = score_prompt(
        "PROMPT",
        held_out,
        provider=provider,
        judge=judge,
        embedder=None,
        train=None,
    )
    assert abs(mean - 0.6) < 1e-9
    assert judge.calls == 3
    # replay scores each held-out step once, deterministically.
    assert provider.rollout_temps == [0.0, 0.0, 0.0]


def test_score_prompt_score_dimension_isolates_factuality() -> None:
    factuality = score_prompt(
        "PROMPT",
        [_trace("te1", n=3)],
        provider=FakeProvider(),
        judge=FakeJudge(score=0.72),  # headline mean; dimension is what we ask for
        embedder=None,
        train=None,
        score_dimension="factuality",
    )
    assert abs(factuality - 0.2) < 1e-9  # returns the factuality dimension, not the 0.72 mean


def test_score_prompt_rejects_unknown_score_dimension() -> None:
    # An unknown dimension is a caller error, caught up front — not a silent 0.0 or a run.
    with pytest.raises(ValueError, match="score_dimension must be one of"):
        score_prompt(
            "P",
            [_trace("te1", n=2)],
            provider=FakeProvider(),
            judge=FakeJudge(score=0.6),
            embedder=None,
            train=None,
            score_dimension="nonexistent_dim",  # ty: ignore[invalid-argument-type]
        )


def test_optimize_prompt_forwards_hard_step_knobs(monkeypatch) -> None:  # noqa: ANN001
    import wmh.research.pipeline as pl

    seen: dict[str, object] = {}

    class _FakeOptimizer:
        def __init__(self, provider, judge, retriever=None, *, seed=0):  # noqa: ANN001, ANN204
            seen["seed"] = seed

        def optimize(  # noqa: ANN202
            self,
            train,  # noqa: ANN001
            test,  # noqa: ANN001
            base,  # noqa: ANN001
            budget,  # noqa: ANN001
            *,
            hard_step_filter,  # noqa: ANN001
            select_on_hard,  # noqa: ANN001
            recheck,  # noqa: ANN001
            minibatch_size,  # noqa: ANN001
        ):
            seen["budget"] = budget
            seen["hard_step_filter"] = hard_step_filter
            seen["select_on_hard"] = select_on_hard
            seen["recheck"] = recheck
            seen["minibatch_size"] = minibatch_size
            return "RESULT"

    monkeypatch.setattr(pl, "GEPAOptimizer", _FakeOptimizer)
    marker = lambda step: True  # noqa: E731
    recheck_traces = [_trace("rc1")]
    result = pl.optimize_prompt(
        [_trace("tr1")],
        [_trace("te1")],
        "BASE",
        provider=FakeProvider(),
        judge=FakeJudge(),
        embedder=None,
        budget=4,
        seed=7,
        hard_step_filter=marker,
        select_on_hard=True,
        recheck=recheck_traces,
        minibatch_size=8,
    )
    assert result == "RESULT"
    assert seen == {
        "seed": 7,
        "budget": 4,
        "hard_step_filter": marker,
        "select_on_hard": True,
        "recheck": recheck_traces,
        "minibatch_size": 8,
    }


def test_score_prompt_raises_on_total_judge_outage() -> None:
    # Every judgement invalid = a judge outage, not fidelity 0.0 — an ablation must not record it.
    class AlwaysInvalidJudge(FakeJudge):
        def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
            return JudgeResult(score=0.0, critique="Unparseable judge reply", valid=False)

    with pytest.raises(RuntimeError, match="judge outage"):
        score_prompt(
            "P",
            [_trace("te1", n=2)],
            provider=FakeProvider(),
            judge=AlwaysInvalidJudge(),
            embedder=None,
            train=None,
        )


def test_score_prompt_empty_holdout_is_zero() -> None:
    mean = score_prompt(
        "P",
        [],
        provider=FakeProvider(),
        judge=FakeJudge(),
        embedder=None,
        train=None,
    )
    assert mean == 0.0


class _PromptRecordingProvider(FakeProvider):
    """FakeProvider that also keeps the last user prompt for contract assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.last_user: str | None = None

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.last_user = messages[0].content
        return Completion(text='{"output": "predicted", "is_error": false, "confidence": 0.4}')


def test_score_prompt_forwards_confidence_and_hands_the_report_to_on_report() -> None:
    provider = _PromptRecordingProvider()
    reports: list[object] = []
    score = score_prompt(
        "BASE",
        [_trace("h", n=1)],
        provider=provider,
        judge=FakeJudge(0.6),
        embedder=None,
        train=None,
        confidence=True,
        on_report=reports.append,
    )
    assert score == 0.6
    assert '"confidence"' in (provider.last_user or "")  # the contract asked for it
    # The full per-step report reached the sink — calibration analysis needs the joint
    # (confidence, score) pairs the scalar return cannot carry.
    (report,) = reports
    assert isinstance(report, ReplayReport)
    assert report.results[0].confidence == 0.4


def test_score_prompt_forwards_verify_below_to_the_gate() -> None:
    provider = _PromptRecordingProvider()  # states 0.4 -> under the 0.7 gate -> verify runs
    report_box: list[object] = []
    score_prompt(
        "BASE",
        [_trace("h", n=1)],
        provider=provider,
        judge=FakeJudge(0.6),
        embedder=None,
        train=None,
        confidence=True,
        verify_below=0.7,
        on_report=report_box.append,
    )
    (report,) = report_box
    assert isinstance(report, ReplayReport)
    assert report.results[0].verified is True


class _OutageJudge:
    """Every judgement invalid — the judge-outage shape score_prompt guards against."""

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        return JudgeResult(score=0.0, critique="parse failure", valid=False)


def test_score_prompt_hands_report_to_sink_even_on_judge_outage() -> None:
    # Persistence must run BEFORE the all-invalid guard raises: the per-step report is exactly
    # what diagnoses the outage, and calibration cells must never silently skip the sink.
    provider = _PromptRecordingProvider()
    reports: list[ReplayReport] = []
    with pytest.raises(RuntimeError, match="no valid judgement"):
        score_prompt(
            "BASE",
            [_trace("h", n=1)],
            provider=provider,
            judge=_OutageJudge(),
            embedder=None,
            train=None,
            confidence=True,
            on_report=reports.append,
        )
    (report,) = reports
    assert report.n_steps == 1


class _DimensionJudge:
    """Valid judgements with a rubric dimension, for the score_dimension return path."""

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        return JudgeResult(score=0.6, critique="ok", dimensions={"factuality": 0.7})


def test_score_prompt_persists_report_for_dimension_scored_cells() -> None:
    # The "Persist FIRST" invariant's other exit: score_dimension returns a different aggregate
    # below on_report — a dimension-scored calibration cell must still hit the sink.
    provider = _PromptRecordingProvider()
    reports: list[ReplayReport] = []
    score = score_prompt(
        "BASE",
        [_trace("h", n=1)],
        provider=provider,
        judge=_DimensionJudge(),
        embedder=None,
        train=None,
        confidence=True,
        score_dimension="factuality",
        on_report=reports.append,
    )
    assert score == 0.7  # the dimension mean, not the headline
    (report,) = reports
    assert report.n_steps == 1
