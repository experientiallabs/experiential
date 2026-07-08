"""End-to-end closed-loop tests: scripted agent + world model + judge, no network.

One provider plays all three roles (agent, world model, gold judge) by inspecting the system
prompt — the same fake-provider pattern the engine tests use.
"""

from __future__ import annotations

from wmh.core.types import Action, ActionKind
from wmh.engine.world_model import WorldModel
from wmh.evals.closed_loop import WorldModelEnvironment, evaluate_closed_loop
from wmh.evals.gold import GoldJudge, GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.harness.environment import is_env_action
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
