"""Tests for the checklist judge (verdict padding, token headroom, garbage handling)."""

from __future__ import annotations

from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.scenarios.verification import ChecklistJudge


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


def _judge_reply(*, success: bool, passed: list[bool]) -> str:
    passed_json = ", ".join("true" if p else "false" for p in passed)
    return (
        f'{{"passed": [{passed_json}], "success": {"true" if success else "false"}, '
        f'"critique": "graded"}}'
    )


def test_checklist_judge_pads_short_verdicts_with_failures() -> None:
    provider = ScriptedProvider([_judge_reply(success=True, passed=[True])])
    result = ChecklistJudge(provider).score("task", ["a", "b", "c"], [])
    assert result.passed == [True, False, False]
    assert result.success is True
    assert result.pass_rate == 1 / 3


def test_checklist_judge_gives_reasoning_judges_token_headroom() -> None:
    # Reasoning judges think before emitting JSON; a tight max_tokens truncates the verdict
    # mid-string and silently scores as failure (found live: Gemini Flash at 1024).
    provider = ScriptedProvider([_judge_reply(success=True, passed=[True])])
    ChecklistJudge(provider).score("task", ["a"], [])
    assert provider.last_max_tokens >= 4096


def test_checklist_judge_treats_garbage_as_failure() -> None:
    result = ChecklistJudge(ScriptedProvider(["??"])).score("task", ["a"], [])
    assert result.passed == [False]
    assert result.success is False
    assert "Unparseable" in result.critique
