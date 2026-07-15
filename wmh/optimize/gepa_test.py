"""Tests for the GEPAOptimizer (drives the `gepa` engine via WorldModelGEPAAdapter).

A deterministic fake Provider + fake Judge stand in for the real models: the provider returns a
fixed env prediction and a fixed improved prompt on reflection; the judge returns a fixed score and
critique. We assert the optimizer runs a bounded loop and returns a valid frontier — no network.
"""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.optimize.gepa import (
    ENV_PROMPT_COMPONENT,
    GEPAOptimizer,
    Optimizer,
    OptimizeResult,
    WorldModelGEPAAdapter,
    _eval_steps,
    _EvalStep,
    _metric_call_budget,
    predict_observation,
)
from wmh.optimize.judge import JudgeResult
from wmh.providers.base import DEFAULT_MAX_TOKENS, Completion, Message, ProviderConfig, ProviderKind


def test_metric_call_budget_funds_exploration_not_just_seed_eval() -> None:
    # The bug: passing iterations straight through as max_metric_calls. If budget < valset, GEPA
    # spends everything on the seed valset eval and proposes nothing. The budget must always exceed
    # one valset pass (so the seed is scored AND at least one candidate can be evaluated).
    valset = 84  # tau2's held-out step count — the case that silently produced zero candidates
    assert _metric_call_budget(50, valset, minibatch=3) > valset  # was 50 < 84 -> no search
    # Scales with iterations: more iterations -> strictly more budget.
    assert _metric_call_budget(10, valset, 3) > _metric_call_budget(1, valset, 3)
    # Floor: even a single iteration funds two full valset passes (seed + one candidate).
    assert _metric_call_budget(1, valset, 3) >= 2 * valset


class FakeProvider:
    """Distinguishes reflection calls (system mentions improving the prompt) from rollouts."""

    def __init__(self, *, prediction: str = "predicted obs", mutation: str = "IMPROVED") -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self._prediction = prediction
        self._mutation = mutation
        self.reflection_calls = 0
        self.rollout_calls = 0
        self.last_rollout_user: str | None = None
        self.last_rollout_max_tokens: int | None = None

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        if "improve the system prompt" in system:
            self.reflection_calls += 1
            return Completion(text=self._mutation)
        self.rollout_calls += 1
        self.last_rollout_user = messages[0].content
        self.last_rollout_max_tokens = max_tokens
        return Completion(text=self._prediction)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


class FakeJudge:
    """Constant score + critique; counts calls so we can assert the loop is bounded."""

    def __init__(self, score: float = 0.5) -> None:
        self._score = score
        self.calls = 0

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        self.calls += 1
        return JudgeResult(score=self._score, critique="add the item total to the response")


def _trace(tid: str, n: int = 2) -> Trace:
    steps = [
        Step(
            action=Action(kind=ActionKind.TOOL_CALL, name="f", arguments={"i": i}),
            observation=Observation(content=f"real-{i}"),
            state_before=EnvState(structured={"loc": "shop"}),
            task="check out",
        )
        for i in range(n)
    ]
    return Trace(trace_id=tid, steps=steps)


def test_predict_observation_uses_provider() -> None:
    provider = FakeProvider(prediction="the cart now has 1 item")
    obs = predict_observation(
        provider,
        "PROMPT",
        task="t",
        state=EnvState(),
        action=Action(kind=ActionKind.MESSAGE, content="hi"),
        demos=[],
    )
    assert obs.content == "the cart now has 1 item"
    assert provider.last_rollout_max_tokens == DEFAULT_MAX_TOKENS


def test_predict_observation_can_include_history() -> None:
    provider = FakeProvider(prediction="next")
    predict_observation(
        provider,
        "PROMPT",
        task="t",
        state=EnvState(),
        action=Action(kind=ActionKind.MESSAGE, content="continue"),
        demos=[],
        history=[_trace("hist", n=1).steps[0]],
    )

    assert "OBSERVATION (is_error=False): real-0" in (provider.last_rollout_user or "")


def test_optimizer_satisfies_protocol() -> None:
    assert isinstance(GEPAOptimizer(FakeProvider(), FakeJudge()), Optimizer)


def test_optimize_runs_bounded_loop_and_returns_valid_frontier() -> None:
    provider = FakeProvider()
    judge = FakeJudge(score=0.5)
    opt = GEPAOptimizer(provider, judge)
    budget = 12

    result = opt.optimize([_trace("tr1"), _trace("tr2")], [_trace("te1")], "BASE", budget)

    assert isinstance(result, OptimizeResult)
    assert result.prompt  # a non-empty winning prompt
    assert len(result.frontier) >= 1
    assert all(isinstance(p, str) and p for p in result.frontier)
    # `budget` is ITERATIONS, translated to a metric-call budget that funds the seed valset eval
    # PLUS exploration (see `_metric_call_budget`). So rollouts run past `budget` itself; the point
    # of the fix is that GEPA actually explores instead of spending everything validating the seed.
    assert result.metrics.rollouts_used > budget
    # ...but it still TERMINATES near the translated metric-call budget (GEPA treats
    # max_metric_calls as a soft cap, finishing the in-flight iteration, so allow a one-iteration
    # overshoot). This upper bound is the regression guard against a runaway budget blowing up cost.
    from wmh.optimize.gepa import _metric_call_budget

    # trainset = 2 traces x 2 steps = 4 -> minibatch = min(3,4) = 3; valset = 1 trace x 2 steps.
    cap = _metric_call_budget(budget, valset_size=2, minibatch=3)
    assert result.metrics.rollouts_used <= cap + 2 + 3  # soft cap: + one valset + one minibatch
    assert 0.0 <= result.metrics.held_out_accuracy <= 1.0
    # The judge was actually consulted, and the loop terminated (didn't run forever).
    assert judge.calls > 0


def test_optimize_reports_real_metric_call_budget_via_on_budget() -> None:
    # Progress bars must be sized by the TRANSLATED metric-call budget, not the iteration count:
    # sizing by iterations made `wmh build` show 100% while GEPA was still burning valset calls.
    from wmh.optimize.gepa import _metric_call_budget

    seen: list[int] = []
    opt = GEPAOptimizer(FakeProvider(), FakeJudge(score=0.5), on_budget=seen.append)
    budget = 5

    opt.optimize([_trace("tr1"), _trace("tr2")], [_trace("te1")], "BASE", budget)

    # trainset = 2 traces x 2 steps -> minibatch 3; valset = 1 trace x 2 steps.
    assert seen == [_metric_call_budget(budget, valset_size=2, minibatch=3)]
    assert seen[0] > budget  # the whole point: the real total exceeds the iteration count


def test_optimize_can_retrieve_from_separate_rag_corpus() -> None:
    from wmh.retrieval import EmbeddingRetriever, HashingEmbedder

    class PromptRecordingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.rollout_user_messages: list[str] = []

        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Completion:
            if "improve the system prompt" not in system:
                self.rollout_user_messages.append(messages[0].content)
            return super().complete(
                system, messages, temperature=temperature, max_tokens=max_tokens
            )

    provider = PromptRecordingProvider()
    dev = _trace("dev", n=1)
    rag = _trace("rag", n=1)
    rag.steps[0].observation.content = "rag-only-observation"
    rag.steps[0].action.arguments = {"source": "rag-only"}

    result = GEPAOptimizer(
        provider, FakeJudge(), retriever=EmbeddingRetriever(HashingEmbedder(dim=64))
    ).optimize([dev], [dev], "BASE", budget=1, rag_corpus=[rag])

    assert result.prompt
    prompt_text = "\n".join(provider.rollout_user_messages)
    assert "rag-only-observation" in prompt_text
    assert "real-0" not in prompt_text


def test_optimize_with_zero_budget_returns_base_prompt() -> None:
    result = GEPAOptimizer(FakeProvider(), FakeJudge()).optimize(
        [_trace("tr1")], [_trace("te1")], "BASE", budget=0
    )
    assert result.prompt == "BASE"
    assert result.frontier == ["BASE"]


def test_optimize_with_no_traces_returns_base_prompt() -> None:
    result = GEPAOptimizer(FakeProvider(), FakeJudge()).optimize([], [], "BASE", budget=10)
    assert result.prompt == "BASE"
    assert result.frontier == ["BASE"]


def _eval_batch(trace: Trace) -> list[_EvalStep]:
    return [_EvalStep(step=s, demos=[], history=trace.steps[:i]) for i, s in enumerate(trace.steps)]


class _TempRecordingProvider(FakeProvider):
    """Records the temperature of every rollout completion."""

    def __init__(self) -> None:
        super().__init__()
        self.rollout_temps: list[float] = []

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        if "improve the system prompt" not in system:
            self.rollout_temps.append(temperature)
        return super().complete(system, messages, temperature=temperature, max_tokens=max_tokens)


def test_predict_observation_runs_deterministically() -> None:
    # Rollouts are always T=0 — the providers reject sampling params, so there is no knob.
    provider = _TempRecordingProvider()
    predict_observation(
        provider, "P", task=None, state=EnvState(), action=Action(kind=ActionKind.MESSAGE), demos=[]
    )
    assert provider.rollout_temps == [0.0]


def test_activity_logger_forwards_first_lines_and_drops_prompt_bodies() -> None:
    from wmh.optimize.gepa import _ActivityLogger

    seen: list[str] = []
    logger = _ActivityLogger(seen.append)
    logger.log("Iteration 1: Selected program 0 score: 0.5")
    logger.log("Iteration 1: Proposed new text for env_prompt: You are an env\nbody line\nmore")
    logger.log("Linear pareto front program index: 0")  # every message's first line streams
    logger.log("   \n")  # blank messages drop
    assert seen == [
        "Iteration 1: Selected program 0 score: 0.5",
        "Iteration 1: Proposed new text for env_prompt: You are an env",
        "Linear pareto front program index: 0",
    ]


def test_reflection_lm_brackets_the_call_in_activity() -> None:
    from wmh.optimize.gepa import _reflection_lm

    lines: list[str] = []
    call = _reflection_lm(FakeProvider(), lines.append)
    call("improve this prompt")
    assert lines[0] == "reflection: proposing an improved env prompt…"
    assert lines[1].startswith("reflection: proposal ready (")


def test_adapter_evaluate_streams_per_step_activity() -> None:
    lines: list[str] = []
    adapter = WorldModelGEPAAdapter(FakeProvider(), FakeJudge(score=0.5), on_activity=lines.append)
    from wmh.retrieval.leakfree import DemoRetriever

    steps = _eval_steps([_trace("t1")], DemoRetriever(None, []))
    adapter.evaluate(steps, {ENV_PROMPT_COMPONENT: "BASE"})
    # A batch-start line, then one line per step as each rollout+judge lands.
    assert lines[0] == "evaluating candidate on 2 steps…"
    assert len(lines) == 3
    assert all("fidelity 0.50" in line for line in lines[1:])


def test_adapter_evaluate_scores_and_captures_traces() -> None:
    adapter = WorldModelGEPAAdapter(FakeProvider(), FakeJudge(score=0.7))
    out = adapter.evaluate(_eval_batch(_trace("t", n=2)), {ENV_PROMPT_COMPONENT: "P"}, True)
    assert out.scores == [0.7, 0.7]
    assert out.trajectories is not None and len(out.trajectories) == 2

    reflective = adapter.make_reflective_dataset(
        {ENV_PROMPT_COMPONENT: "P"}, out, [ENV_PROMPT_COMPONENT]
    )
    records = reflective[ENV_PROMPT_COMPONENT]
    assert len(records) == 2
    assert "Feedback" in records[0] and "Generated Outputs" in records[0]


def test_adapter_evaluate_survives_rollout_failure() -> None:
    class BoomJudge(FakeJudge):
        def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
            raise RuntimeError("judge exploded")

    adapter = WorldModelGEPAAdapter(FakeProvider(), BoomJudge())
    out = adapter.evaluate(_eval_batch(_trace("t", n=1)), {ENV_PROMPT_COMPONENT: "P"}, True)
    # Per-example failure -> fallback score, never an exception.
    assert out.scores == [0.0]
    assert out.trajectories is not None and "failed" in out.trajectories[0].critique


def test_adapter_imputes_neutral_score_for_invalid_judgements() -> None:
    # A judge failure (valid=False) says nothing about the prediction: it must not become a
    # phantom 0.0 in GEPA's fitness, and its parse-error critique must not reach reflection.
    class InvalidOnSecondJudge(FakeJudge):
        def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
            self.calls += 1
            if self.calls == 2:
                return JudgeResult(score=0.0, critique="Unparseable judge reply", valid=False)
            return JudgeResult(score=0.8, critique="ok")

    adapter = WorldModelGEPAAdapter(FakeProvider(), InvalidOnSecondJudge())
    out = adapter.evaluate(_eval_batch(_trace("t", n=3)), {ENV_PROMPT_COMPONENT: "P"}, True)
    # The invalid step gets the mean of the valid scores (neutral), not 0.0.
    assert out.scores == [0.8, 0.8, 0.8]

    reflective = adapter.make_reflective_dataset(
        {ENV_PROMPT_COMPONENT: "P"}, out, [ENV_PROMPT_COMPONENT]
    )
    records = reflective[ENV_PROMPT_COMPONENT]
    assert len(records) == 2  # the judge-noise record is excluded from reflection
    assert all("Unparseable" not in str(r["Feedback"]) for r in records)


def test_adapter_all_invalid_judgements_fall_back_to_full_reflection() -> None:
    class AlwaysInvalidJudge(FakeJudge):
        def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
            return JudgeResult(score=0.0, critique="Unparseable judge reply", valid=False)

    adapter = WorldModelGEPAAdapter(FakeProvider(), AlwaysInvalidJudge())
    out = adapter.evaluate(_eval_batch(_trace("t", n=2)), {ENV_PROMPT_COMPONENT: "P"}, True)
    # No valid score to impute from: keep the raw zeros rather than inventing a signal.
    assert out.scores == [0.0, 0.0]
    reflective = adapter.make_reflective_dataset(
        {ENV_PROMPT_COMPONENT: "P"}, out, [ENV_PROMPT_COMPONENT]
    )
    # An empty reflective dataset would break GEPA's mutation step; fall back to everything —
    # but with the judge-noise critiques scrubbed so reflection never chases parse errors.
    records = reflective[ENV_PROMPT_COMPONENT]
    assert len(records) == 2
    assert all("Unparseable" not in str(r["Feedback"]) for r in records)


def test_adapter_judge_exception_is_invalid_not_a_world_model_zero() -> None:
    # A judge call that RAISES (throttle, 5xx) is judge infrastructure, not world-model signal:
    # it must flow through the same valid=False machinery as a malformed reply, and the
    # successfully generated prediction must be kept.
    class RaisingJudge(FakeJudge):
        def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("ThrottlingException: judge is down")
            return JudgeResult(score=0.8, critique="ok")

    adapter = WorldModelGEPAAdapter(FakeProvider(), RaisingJudge())
    out = adapter.evaluate(_eval_batch(_trace("t", n=2)), {ENV_PROMPT_COMPONENT: "P"}, True)
    # The judge-exception step is imputed like any invalid judgement, not scored 0.0.
    assert out.scores == [0.8, 0.8]
    assert out.trajectories is not None
    failed = [t for t in out.trajectories if not t.valid]
    assert len(failed) == 1
    assert "Judge call failed" in failed[0].critique
    assert failed[0].predicted.content  # the prediction was kept, not replaced by an error stub


def test_eval_steps_retrieves_demos_without_same_trace_leakage() -> None:
    from wmh.retrieval import EmbeddingRetriever, HashingEmbedder
    from wmh.retrieval.leakfree import DemoRetriever

    # Two train traces. Each step's nearest neighbor is its own sibling (same trace) — which must be
    # excluded — so the demo it actually gets must come from the OTHER trace.
    train = [_trace("trace-A", n=2), _trace("trace-B", n=2)]
    # Make trace-B lexically distinct so retrieval has a real choice.
    for s in train[1].steps:
        s.action.arguments = {"other": "zzz"}
        s.state_before.structured = {"loc": "warehouse"}

    demos = DemoRetriever(EmbeddingRetriever(HashingEmbedder(dim=128)), train, top_k=2)
    eval_steps = _eval_steps(train, demos)

    assert len(eval_steps) == 4
    a_ids = {id(s) for s in train[0].steps}
    for es in eval_steps:
        if id(es.step) in a_ids:  # a trace-A step
            # None of its demos may be from trace-A (no self/sibling leakage).
            assert all(id(d) not in a_ids for d in es.demos)


def test_eval_steps_zero_shot_without_retriever() -> None:
    from wmh.retrieval.leakfree import DemoRetriever

    traces = [_trace("t", n=2)]
    eval_steps = _eval_steps(traces, DemoRetriever(None, traces))  # no retriever -> zero-shot
    assert len(eval_steps) == 2
    assert all(es.demos == [] for es in eval_steps)
