"""End-to-end closed-loop tests: scripted agent + world model + judge, no network.

One provider plays all three roles (agent, world model, gold judge) by inspecting the system
prompt — the same fake-provider pattern the engine tests use.
"""

from __future__ import annotations

import threading

import pytest

from wmh.core.types import Action, ActionKind, Observation
from wmh.engine.world_model import WorldModel
from wmh.evals.closed_loop import (
    ClosedLoopReport,
    WorldModelEnvironment,
    evaluate_closed_loop,
    evaluate_with_env,
)
from wmh.evals.gold import GoldJudge, GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.harness.environment import AgentEnvironment, is_env_action
from wmh.harness.runtime import RunResult, StopReason
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


class RoleProvider:
    """Plays agent, world model, and gold judge, keyed off the system prompt."""

    def __init__(self, *, judge_passes: bool = True) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._judge_passes = judge_passes

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        if "grade whether an agent completed a task" in system:
            passed = "true" if self._judge_passes else "false"
            return Completion(
                text='{"assertions": [{"assertion": "did it", "passed": '
                + passed
                + ', "why": "x"}], "passed": '
                + passed
                + "}"
            )
        if system.startswith("You are a capable command-line agent"):
            return Completion(
                text='{"tool": "submit", "arguments": {"answer": "the answer is 42"}}'
            )
        # world model
        return Completion(text='{"output": "ok", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201 - test fake never calls it
        raise NotImplementedError


def _wm(provider: RoleProvider) -> WorldModel:
    return WorldModel(provider, EmbeddingRetriever(HashingEmbedder(dim=16)))


def test_gold_judge_no_assertions_trivially_passes() -> None:
    verdict = GoldJudge(RoleProvider()).score("task", "answer", "transcript", [])
    assert verdict == GoldVerdict.trivially_passed()


def test_closed_loop_scores_success_over_k_passes() -> None:
    provider = RoleProvider(judge_passes=True)
    tasks = [TaskSpec(task_id="q1", instruction="answer it", gold=["did it"])]
    report = evaluate_closed_loop(tasks, _wm(provider), provider, GoldJudge(provider), k=3)
    assert report.k == 3
    assert report.success_rate == 1.0
    assert report.per_task["q1"].passes == 3


def test_closed_loop_reports_failure_when_judge_rejects() -> None:
    provider = RoleProvider(judge_passes=False)
    tasks = [TaskSpec(task_id="q1", instruction="answer it", gold=["did it"])]
    report = evaluate_closed_loop(tasks, _wm(provider), provider, GoldJudge(provider), k=2)
    assert report.success_rate == 0.0
    assert report.per_task["q1"].mean_fraction == 0.0


def test_world_model_environment_steps_and_ends_session() -> None:
    provider = RoleProvider()
    wm = _wm(provider)
    env = WorldModelEnvironment(wm, task="do a thing")
    session_id = env.session_id
    obs = env.execute(Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"}))
    assert obs.content == "ok"
    env.close()
    # The session is released on close (WorldModel.end_session drops it).
    try:
        wm.get_session(session_id)
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("session should be gone after close()")
    env.close()  # idempotent


def test_rollouts_do_not_enrich_the_retrieval_buffer() -> None:
    """A rollout's PREDICTED steps must not become retrieval demos for later rollouts."""
    provider = RoleProvider()
    wm = _wm(provider)
    before = len(wm.sample_steps(1000))
    env = WorldModelEnvironment(wm, task="do a thing")
    env.execute(Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"}))
    env.execute(Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "pwd"}))
    env.close()
    assert len(wm.sample_steps(1000)) == before  # buffer unchanged: eval sessions don't enrich
    # Serve-time sessions still enrich by default.
    session = wm.new_session(task="serve")
    wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "x"}))
    assert len(wm.sample_steps(1000)) == before + 1


def test_is_env_action_gates_tool_calls() -> None:
    assert is_env_action(Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={}))
    assert not is_env_action(Action(kind=ActionKind.TOOL_CALL, name="submit", arguments={}))
    assert not is_env_action(Action(kind=ActionKind.MESSAGE, content="hi"))


def test_evaluate_rejects_k_below_one() -> None:
    provider = RoleProvider()
    tasks = [TaskSpec(task_id="q", instruction="x", gold=[])]
    try:
        evaluate_closed_loop(tasks, _wm(provider), provider, GoldJudge(provider), k=0)
    except ValueError as exc:
        assert "k must be >= 1" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for k=0")


class _StaticEnv:
    """Env fake: answers every action with a constant observation; tracks close()."""

    def __init__(self) -> None:
        self.closed = False

    def execute(self, action: Action) -> Observation:
        return Observation(content="ok")

    def close(self) -> None:
        self.closed = True


class _ScriptedRuntime:
    """Runtime fake: deterministic answer keyed off the task_id (pure in its inputs)."""

    def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
        environment.execute(
            Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "x"})
        )
        answer = "pass" if task_id.endswith("-pass") else "fail"
        return RunResult(task_id=task_id, stop_reason=StopReason.SUBMITTED, answer=answer)


class _AnswerJudge(GoldJudge):
    """Judge fake: the verdict is read straight off the runtime's answer (no LLM call)."""

    def __init__(self) -> None:
        super().__init__(RoleProvider())

    def score(self, instruction: str, answer: str, transcript: str, gold: list[str]) -> GoldVerdict:
        passed = answer == "pass"
        return GoldVerdict(passed=passed, fraction=1.0 if passed else 0.0, rationale=answer)


def _parity_tasks() -> list[TaskSpec]:
    return [
        TaskSpec(task_id="a-pass", instruction="do a", gold=["g"]),
        TaskSpec(task_id="b-fail", instruction="do b", gold=["g"]),
    ]


def test_concurrent_report_matches_sequential() -> None:
    """concurrency=0 (all cells) and =2 (capped) must reproduce the sequential report exactly."""

    def run(concurrency: int) -> ClosedLoopReport:
        return evaluate_with_env(
            _parity_tasks(),
            lambda task: _StaticEnv(),
            _ScriptedRuntime(),
            _AnswerJudge(),
            label="parity",
            k=3,
            concurrency=concurrency,
        )

    sequential = run(1)
    assert sequential.per_task["a-pass"].success_rate == 1.0
    assert sequential.per_task["b-fail"].success_rate == 0.0
    for concurrency in (0, 2):
        assert run(concurrency).model_dump() == sequential.model_dump()


def test_concurrency_zero_overlaps_all_cells() -> None:
    """With concurrency=0 every (task, attempt) cell is in flight at once: a barrier sized to the
    full cell count releases only under true parallelism (any smaller pool would deadlock and trip
    the barrier timeout)."""
    tasks = [
        TaskSpec(task_id="a-pass", instruction="do a", gold=["g"]),
        TaskSpec(task_id="b-pass", instruction="do b", gold=["g"]),
    ]
    k = 2
    barrier = threading.Barrier(len(tasks) * k)

    class BarrierRuntime:
        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            barrier.wait(timeout=30)  # BrokenBarrierError (test failure) unless all cells overlap
            return RunResult(task_id=task_id, stop_reason=StopReason.SUBMITTED, answer="pass")

    report = evaluate_with_env(
        tasks, lambda task: _StaticEnv(), BarrierRuntime(), _AnswerJudge(), k=k, concurrency=0
    )
    assert report.success_rate == 1.0
    assert barrier.broken is False


def test_on_progress_fires_once_per_cell_from_the_coordinating_thread() -> None:
    events: list[tuple[str, int, bool]] = []
    coordinator = threading.get_ident()
    callback_threads: set[int] = set()

    def on_progress(task_id: str, attempt: int, verdict: GoldVerdict) -> None:
        callback_threads.add(threading.get_ident())
        events.append((task_id, attempt, verdict.passed))

    evaluate_with_env(
        _parity_tasks(),
        lambda task: _StaticEnv(),
        _ScriptedRuntime(),
        _AnswerJudge(),
        k=3,
        concurrency=0,
        on_progress=on_progress,
    )
    assert len(events) == 6  # exactly once per (task, attempt) cell
    assert set(events) == {
        (task.task_id, attempt, task.task_id.endswith("-pass"))
        for task in _parity_tasks()
        for attempt in (1, 2, 3)
    }
    assert callback_threads == {coordinator}  # serial UI stream, never from worker threads


def test_rollout_exception_propagates_and_does_not_hang() -> None:
    class ExplodingRuntime:
        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            if task_id == "boom":
                raise RuntimeError("sandbox died")
            return RunResult(task_id=task_id, stop_reason=StopReason.SUBMITTED, answer="pass")

    tasks = [
        TaskSpec(task_id="boom", instruction="x", gold=["g"]),
        TaskSpec(task_id="ok-pass", instruction="y", gold=["g"]),
    ]
    with pytest.raises(RuntimeError, match="sandbox died"):
        evaluate_with_env(
            tasks, lambda task: _StaticEnv(), ExplodingRuntime(), _AnswerJudge(), k=2, concurrency=0
        )


def test_rollout_exception_surfaces_while_other_cells_are_still_in_flight() -> None:
    """Fail-fast must not drain in-flight cells: with sandbox rollouts those run for minutes.

    One cell blocks on an event; another raises immediately. The exception must reach the
    caller while the blocker is STILL blocked (shutdown(wait=False)) — then the blocker is
    released so its thread exits cleanly.
    """
    release = threading.Event()
    blocker_started = threading.Event()

    class HalfStuckRuntime:
        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            if task_id == "stuck":
                blocker_started.set()
                release.wait(timeout=30)
                return RunResult(task_id=task_id, stop_reason=StopReason.SUBMITTED, answer="pass")
            blocker_started.wait(timeout=30)  # raise only once the slow cell is truly in flight
            raise RuntimeError("sandbox died")

    tasks = [
        TaskSpec(task_id="stuck", instruction="x", gold=["g"]),
        TaskSpec(task_id="boom", instruction="y", gold=["g"]),
    ]
    try:
        with pytest.raises(RuntimeError, match="sandbox died"):
            evaluate_with_env(
                tasks,
                lambda task: _StaticEnv(),
                HalfStuckRuntime(),
                _AnswerJudge(),
                k=1,
                concurrency=0,
            )
        assert not release.is_set()  # the raise did not wait for the stuck cell to drain
    finally:
        release.set()  # let the worker thread exit


def test_evaluate_closed_loop_passes_concurrency_through() -> None:
    provider = RoleProvider(judge_passes=True)
    tasks = [TaskSpec(task_id="q1", instruction="answer it", gold=["did it"])]
    report = evaluate_closed_loop(
        tasks, _wm(provider), provider, GoldJudge(provider), k=2, concurrency=2
    )
    assert report.success_rate == 1.0
    assert report.per_task["q1"].passes == 2


def test_gold_judge_duplicate_assertions_cannot_pad_the_count() -> None:
    """Echoing a passing assertion twice must not substitute for an omitted one."""

    class DuplicatingJudgeProvider(RoleProvider):
        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 2048,
        ) -> Completion:
            if "grade whether an agent completed a task" in system:
                return Completion(
                    text='{"assertions": [{"assertion": "a", "passed": true, "why": ""}, '
                    '{"assertion": "a", "passed": true, "why": ""}], "passed": true}'
                )
            return super().complete(
                system, messages, temperature=temperature, max_tokens=max_tokens
            )

    verdict = GoldJudge(DuplicatingJudgeProvider()).score("t", "ans", "tr", ["a", "b"])
    assert not verdict.passed  # 'b' was never judged; duplicated 'a' doesn't cover it
    assert verdict.fraction == 0.5


def test_gold_judge_scores_against_full_gold_list() -> None:
    """A truncated judge reply that omits assertions must not be able to report success."""

    class OneAssertionJudgeProvider(RoleProvider):
        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 2048,
        ) -> Completion:
            if "grade whether an agent completed a task" in system:
                return Completion(
                    text='{"assertions": [{"assertion": "a", "passed": true, "why": ""}], '
                    '"passed": true}'
                )
            return super().complete(
                system, messages, temperature=temperature, max_tokens=max_tokens
            )

    verdict = GoldJudge(OneAssertionJudgeProvider()).score("t", "ans", "tr", ["a", "b"])
    assert not verdict.passed
    assert verdict.fraction == 0.5


def test_report_aggregates_worker_usage_from_self_metering_runtimes() -> None:
    """Cells that report worker usage sum into the report; none reported -> None."""
    from wmh.harness.runtime import TokenUsage

    class MeteredRuntime:
        def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
            return RunResult(
                task_id=task_id,
                stop_reason=StopReason.SUBMITTED,
                answer="pass",
                worker_usage=TokenUsage(input_tokens=10, output_tokens=3, calls=2),
            )

    tasks = [TaskSpec(task_id="a-pass", instruction="x", gold=["g"])]
    for concurrency in (1, 0):  # both aggregation paths
        report = evaluate_with_env(
            tasks,
            lambda task: _StaticEnv(),
            MeteredRuntime(),
            _AnswerJudge(),
            k=2,
            concurrency=concurrency,
        )
        assert report.worker_usage is not None
        assert report.worker_usage.input_tokens == 20  # 2 cells x 10
        assert report.worker_usage.output_tokens == 6
        assert report.worker_usage.calls == 4
    # Provider-wrapped runtimes report nothing -> the report says None, not zero.
    silent = evaluate_with_env(
        tasks, lambda task: _StaticEnv(), _ScriptedRuntime(), _AnswerJudge(), k=1
    )
    assert silent.worker_usage is None
