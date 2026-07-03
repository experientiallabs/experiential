"""Tests for closed-loop eval + gold judging, with a scripted agent/world-model/judge provider."""

from __future__ import annotations

from wmh.agent.closed_loop import evaluate_closed_loop, failing_transcripts
from wmh.agent.gold import GoldJudge, GoldVerdict
from wmh.agent.spec import HarnessSpec
from wmh.agent.tasks import TaskSpec
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


class RoleProvider:
    """One provider playing three roles by inspecting the system prompt.

    - agent (role-play): submit immediately with a fixed answer,
    - world model: canned observation,
    - gold judge: pass every assertion.
    """

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

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _wm(provider: RoleProvider) -> WorldModel:
    return WorldModel(provider, EmbeddingRetriever(HashingEmbedder(dim=16)))


def test_gold_judge_no_assertions_trivially_passes() -> None:
    verdict = GoldJudge(RoleProvider()).score("task", "answer", "transcript", [])
    assert verdict == GoldVerdict.trivially_passed()
    assert verdict.passed


def test_closed_loop_scores_success_over_k_passes() -> None:
    provider = RoleProvider(judge_passes=True)
    tasks = [TaskSpec(task_id="q1", instruction="answer it", gold=["did it"])]
    report = evaluate_closed_loop(
        HarnessSpec(), tasks, _wm(provider), provider, GoldJudge(provider), k=3
    )
    assert report.k == 3
    assert report.success_rate == 1.0
    assert report.per_task["q1"].passes == 3


def test_closed_loop_reports_failure_when_judge_rejects() -> None:
    provider = RoleProvider(judge_passes=False)
    tasks = [TaskSpec(task_id="q1", instruction="answer it", gold=["did it"])]
    report = evaluate_closed_loop(
        HarnessSpec(), tasks, _wm(provider), provider, GoldJudge(provider), k=2
    )
    assert report.success_rate == 0.0
    # The failure feedback surfaces the unmet assertion for the mutation prompt.
    feedback = failing_transcripts(report, tasks)
    assert "did it" in feedback
