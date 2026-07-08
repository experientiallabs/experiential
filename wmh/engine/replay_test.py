"""Tests for the replay/reconstruction-fidelity harness, with fakes (no network)."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.engine.replay import replay
from wmh.optimize.judge import JudgeResult
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


class FakeProvider:
    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._reply = reply
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
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


class FakeJudge:
    def __init__(self, score: float, dimensions: dict[str, float] | None = None) -> None:
        self._score = score
        self._dimensions = dimensions or {}
        self.calls = 0

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        self.calls += 1
        return JudgeResult(score=self._score, critique="ok", dimensions=dict(self._dimensions))


class PerActionJudge:
    """Scores by tool-call arg `i` (lets a trace produce a spread of per-step scores)."""

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        i = context.action.arguments.get("i", 0)
        return JudgeResult(score=1.0 if i == 0 else 0.0, critique="ok")


def _trace(tid: str, n: int = 2) -> Trace:
    return Trace(
        trace_id=tid,
        steps=[
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"i": i}),
                observation=Observation(content=f"real-{i}", is_error=False),
                state_before=EnvState(structured={"loc": "shop"}),
                task="look up",
            )
            for i in range(n)
        ],
    )


def test_replay_scores_and_aggregates() -> None:
    provider = FakeProvider('{"output": "real-0", "is_error": false}')
    report = replay("BASE", [_trace("h", n=2)], provider, FakeJudge(0.8))
    assert report.n_steps == 2
    assert report.mean_score == 0.8
    # Predicted is_error (false) matches actual (false) for both.
    assert report.error_flag_accuracy == 1.0
    assert report.results[0].actual == "real-0"


def test_replay_includes_all_prior_teacher_forced_history_by_default() -> None:
    provider = FakeProvider('{"output": "real-2", "is_error": false}')
    report = replay("BASE", [_trace("h", n=3)], provider, FakeJudge(1.0))

    assert report.n_steps == 3
    user_prompt = provider.last_user or ""
    assert "OBSERVATION (is_error=False): real-0" in user_prompt
    assert "OBSERVATION (is_error=False): real-1" in user_prompt
    assert "OBSERVATION (is_error=False): real-2" not in user_prompt


def test_replay_tracks_error_flag_mismatch() -> None:
    # Model predicts an error, but the actual observation is not an error -> flag mismatch.
    provider = FakeProvider('{"output": "boom", "is_error": true}')
    report = replay("BASE", [_trace("h", n=1)], provider, FakeJudge(0.0))
    assert report.error_flag_accuracy == 0.0
    assert report.results[0].is_error_predicted is True
    assert report.results[0].is_error_actual is False


def test_replay_rag_is_leakfree() -> None:
    # The held-out trace's own steps must never appear as demos in its prompt.
    train = [_trace("train-A", n=2)]
    holdout = [_trace("train-A", n=2)]  # same trace_id as a train trace -> must be excluded
    provider = FakeProvider('{"output": "x", "is_error": false}')
    retriever = EmbeddingRetriever(HashingEmbedder(dim=64))
    report = replay(
        "BASE", holdout, provider, FakeJudge(0.5), retriever=retriever, train=train, top_k=3
    )
    assert report.n_steps == 2
    # With train and holdout sharing the trace_id, every demo is excluded -> no leakage into prompt.
    assert "real-" not in (provider.last_user or "").split("SIMILAR PAST EXAMPLES")[-1]


def test_replay_empty_is_safe() -> None:
    report = replay("BASE", [], FakeProvider("{}"), FakeJudge(1.0))
    assert report.n_steps == 0
    assert report.mean_score == 0.0


def test_replay_reports_std_across_steps() -> None:
    # Two steps scored 1.0 and 0.0 -> mean 0.5, population std 0.5 across steps.
    report = replay("BASE", [_trace("h", n=2)], FakeProvider('{"output": "x"}'), PerActionJudge())
    assert report.n_steps == 2
    assert report.mean_score == 0.5
    assert report.score_std == 0.5


def test_replay_carries_rubric_dimensions() -> None:
    dims = {"format": 1.0, "factuality": 0.6, "consistency": 0.8, "realism": 1.0, "quality": 0.7}
    report = replay(
        "BASE", [_trace("h", n=1)], FakeProvider('{"output": "x"}'), FakeJudge(0.82, dims)
    )
    assert report.results[0].dimensions == dims


def test_replay_sampled_turns_scores_five_for_long_traces() -> None:
    judge = FakeJudge(0.5)
    report = replay(
        "BASE",
        [_trace("h", n=10)],
        FakeProvider('{"output": "x"}'),
        judge,
        sample_turns="sampled",
        seed=0,
    )
    assert report.n_steps == 5  # first, last, 3 middle
    # Deterministic under a fixed seed.
    judge2 = FakeJudge(0.5)
    report2 = replay(
        "BASE",
        [_trace("h", n=10)],
        FakeProvider('{"output": "x"}'),
        judge2,
        sample_turns="sampled",
        seed=0,
    )
    assert [r.action for r in report.results] == [r.action for r in report2.results]


def test_replay_sampled_turns_history_uses_original_trace_prefix() -> None:
    provider = FakeProvider('{"output": "x"}')
    replay("BASE", [_trace("h", n=10)], provider, FakeJudge(0.5), sample_turns="sampled", seed=0)

    user_prompt = provider.last_user or ""
    assert "OBSERVATION (is_error=False): real-8" in user_prompt
    assert "OBSERVATION (is_error=False): real-9" not in user_prompt


def test_replay_sample_turns_all_scores_every_step() -> None:
    report = replay(
        "BASE",
        [_trace("h", n=10)],
        FakeProvider('{"output": "x"}'),
        FakeJudge(0.5),
        sample_turns="all",
    )
    assert report.n_steps == 10


class InvalidOnceJudge:
    """First call is an invalid judgement (judge failure), the rest score 0.8."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        self.calls += 1
        if self.calls == 1:
            return JudgeResult(score=0.0, critique="judge broke", valid=False)
        return JudgeResult(score=0.8, critique="ok")


def test_replay_excludes_invalid_judgements_from_fidelity() -> None:
    provider = FakeProvider('{"output": "real-0", "is_error": false}')
    report = replay("BASE", [_trace("h", n=2)], provider, InvalidOnceJudge())
    assert report.n_steps == 2
    assert report.n_invalid == 1
    # The judge failure is excluded from the mean, not recorded as a 0.0 prediction score.
    assert report.mean_score == 0.8
    assert report.results[0].valid is False
    assert "invalid" in report.summary()


def test_replay_all_invalid_judgements_report_zero_with_count() -> None:
    class AlwaysInvalid:
        def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
            return JudgeResult(score=0.0, critique="judge broke", valid=False)

    provider = FakeProvider('{"output": "x"}')
    report = replay("BASE", [_trace("h", n=2)], provider, AlwaysInvalid())
    assert report.n_steps == 2
    assert report.n_invalid == 2
    assert report.mean_score == 0.0


def test_replay_concurrency_preserves_results_and_order() -> None:
    # PerActionJudge scores by step index, so the per-step result sequence is order-sensitive:
    # concurrent scoring must return the identical ordered results as serial.
    traces = [_trace("a", n=4), _trace("b", n=3)]
    fp = FakeProvider('{"output": "x"}')
    serial = replay("BASE", traces, fp, PerActionJudge(), concurrency=1)
    parallel = replay("BASE", traces, fp, PerActionJudge(), concurrency=4)
    assert serial.n_steps == parallel.n_steps == 7
    assert [r.score for r in serial.results] == [r.score for r in parallel.results]
    assert serial.mean_score == parallel.mean_score
