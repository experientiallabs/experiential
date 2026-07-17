"""Verdict-parsing tests for the gold judge: the fail-closed guards in `_parse`.

The judge scores against the FULL gold list by matching assertion text the judge echoed back.
The guards under test are exactly the regression-prone ones: a truncated reply that omits an
assertion, a reply that pads the count by duplicating a passing assertion, and an unparseable
reply must all be unable to report success.
"""

from __future__ import annotations

import json

from wmh.evals.gold import GOLD_JUDGE_SYSTEM, GoldJudge, _parse
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind

_GOLD = ["the file was created", "the tests pass"]


class ScriptedJudgeProvider:
    """Returns a fixed judge reply; records the prompt it was asked to grade."""

    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._reply = reply
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        self.last_system = system
        self.last_user = messages[-1].content
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201 - test fake never calls it
        raise NotImplementedError


def _reply(assertions: list[tuple[str, bool]], *, passed: bool) -> str:
    return json.dumps(
        {
            "assertions": [
                {"assertion": text, "passed": ok, "why": "w"} for text, ok in assertions
            ],
            "passed": passed,
        }
    )


def test_full_pass_scores_every_assertion() -> None:
    verdict = _parse(_reply([(g, True) for g in _GOLD], passed=True), _GOLD)
    assert verdict.passed
    assert verdict.fraction == 1.0
    assert verdict.rationale == "2/2 assertions satisfied"


def test_truncated_reply_omitting_an_assertion_cannot_pass() -> None:
    # The judge echoes only the first assertion (e.g. reply truncated) yet claims overall success.
    verdict = _parse(_reply([(_GOLD[0], True)], passed=True), _GOLD)
    assert not verdict.passed
    assert verdict.fraction == 0.5


def test_duplicated_assertion_cannot_pad_the_count() -> None:
    # Same passing assertion twice; the other gold assertion never appears.
    verdict = _parse(_reply([(_GOLD[0], True), (_GOLD[0], True)], passed=True), _GOLD)
    assert not verdict.passed
    assert verdict.fraction == 0.5


def test_contradictory_duplicate_echo_cannot_pass() -> None:
    # The judge echoes the same assertion as failed AND passed (self-correction or a
    # hallucinated duplicate row); its own failing verdict must not be discarded.
    verdict = _parse(
        _reply([(_GOLD[0], False), (_GOLD[0], True), (_GOLD[1], True)], passed=True), _GOLD
    )
    assert not verdict.passed
    assert verdict.fraction == 0.5


def test_failed_assertion_fails_even_if_judge_claims_overall_pass() -> None:
    verdict = _parse(_reply([(_GOLD[0], True), (_GOLD[1], False)], passed=True), _GOLD)
    assert not verdict.passed
    assert verdict.fraction == 0.5


def test_whitespace_around_echoed_assertion_still_matches() -> None:
    verdict = _parse(_reply([(f"  {_GOLD[0]}  ", True), (_GOLD[1], True)], passed=True), _GOLD)
    assert verdict.passed
    assert verdict.fraction == 1.0


def test_unparseable_reply_is_a_failure_with_the_raw_text_quoted() -> None:
    verdict = _parse("I think it went great!", _GOLD)
    assert not verdict.passed
    assert verdict.fraction == 0.0
    assert "unparseable" in verdict.rationale
    assert "I think it went great!" in verdict.rationale


def test_empty_assertion_list_is_treated_as_unparseable() -> None:
    verdict = _parse('{"assertions": [], "passed": true}', _GOLD)
    assert not verdict.passed
    assert "unparseable" in verdict.rationale


def test_score_builds_prompt_with_task_answer_transcript_and_gold() -> None:
    provider = ScriptedJudgeProvider(_reply([(g, True) for g in _GOLD], passed=True))
    verdict = GoldJudge(provider).score("do the thing", "done", "ran stuff", _GOLD)
    assert verdict.passed
    assert provider.last_system == GOLD_JUDGE_SYSTEM
    assert provider.last_user is not None
    for expected in ("do the thing", "done", "ran stuff", *_GOLD):
        assert expected in provider.last_user


def test_score_placeholders_for_empty_answer_and_transcript() -> None:
    provider = ScriptedJudgeProvider(_reply([(g, True) for g in _GOLD], passed=True))
    GoldJudge(provider).score("task", "", "", _GOLD)
    assert provider.last_user is not None
    assert "(none)" in provider.last_user
    assert "(empty)" in provider.last_user
