"""Tests for the gold-assertion judge, especially scoring against the full gold list."""

from __future__ import annotations

from wmh.agent.gold import GoldJudge, GoldVerdict
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind


class _CannedProvider:
    """Returns a fixed judge reply, regardless of prompt."""

    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._reply = reply

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - unused
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN202
        raise NotImplementedError


def _score(reply: str, gold: list[str]) -> GoldVerdict:
    return GoldJudge(_CannedProvider(reply)).score("task", "answer", "transcript", gold)


def test_no_gold_trivially_passes() -> None:
    assert _score("(never called)", []).passed


def test_all_assertions_passed() -> None:
    reply = (
        '{"assertions": [{"assertion": "a", "passed": true, "why": ""}, '
        '{"assertion": "b", "passed": true, "why": ""}], "passed": true}'
    )
    verdict = _score(reply, ["a", "b"])
    assert verdict.passed
    assert verdict.fraction == 1.0


def test_omitted_assertion_cannot_report_success() -> None:
    # Two gold assertions, but the judge only returns one (marked passed) and claims overall pass.
    # Scoring against the full gold list must treat the missing one as unmet -> not passed, 0.5.
    reply = '{"assertions": [{"assertion": "a", "passed": true, "why": ""}], "passed": true}'
    verdict = _score(reply, ["a", "b"])
    assert not verdict.passed
    assert verdict.fraction == 0.5
    assert verdict.rationale == "1/2 assertions satisfied"


def test_hallucinated_extra_assertions_do_not_over_credit() -> None:
    # The judge returns 3 passed assertions for 2 gold ones; passes are capped at the gold count.
    reply = (
        '{"assertions": [{"assertion": "a", "passed": true, "why": ""}, '
        '{"assertion": "b", "passed": true, "why": ""}, '
        '{"assertion": "c", "passed": true, "why": ""}], "passed": true}'
    )
    verdict = _score(reply, ["a", "b"])
    assert verdict.passed
    assert verdict.fraction == 1.0


def test_unparseable_reply_is_failure() -> None:
    verdict = _score("not json", ["a"])
    assert not verdict.passed
    assert verdict.fraction == 0.0
