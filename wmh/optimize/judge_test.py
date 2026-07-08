"""Tests for the RubricJudge: prompt contract, weighted scoring, truncation, and validity."""

from __future__ import annotations

import re

import pytest

from wmh.core.types import Action, ActionKind, Observation, Step
from wmh.optimize.judge import (
    JUDGE_MARKER,
    JUDGE_SYSTEM,
    OBSERVATION_HEAD_CHARS,
    OBSERVATION_TAIL_CHARS,
    RUBRIC_DIMENSIONS,
    RUBRIC_WEIGHTS,
    Judge,
    RubricJudge,
    _build_judge_prompt,
)
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind


def test_judge_system_contains_marker() -> None:
    # Run-cost attribution (wmh.tracking.metered.classify_build_call) recognizes judge calls by
    # JUDGE_MARKER. If the prompt is edited to no longer contain it, judge cost is silently
    # misattributed to GEPA — pin the coupling here.
    assert JUDGE_MARKER in JUDGE_SYSTEM


def test_judge_system_explains_payload_and_edge_rules() -> None:
    # The prompt must describe the payload fields it sends (sentinels, truncation marker) and the
    # empty/outcome-flip rules the meta-eval pins; drifting these regresses judge quality.
    assert "content_length" in JUDGE_SYSTEM
    assert "<EMPTY_PREDICTION>" in JUDGE_SYSTEM
    assert "characters omitted" in JUDGE_SYSTEM
    assert "both contents are empty" in JUDGE_SYSTEM
    assert "is_error" in JUDGE_SYSTEM


def test_rubric_weights_cover_dimensions_and_sum_to_one() -> None:
    assert set(RUBRIC_WEIGHTS) == set(RUBRIC_DIMENSIONS)
    assert sum(RUBRIC_WEIGHTS.values()) == pytest.approx(1.0)
    # Factuality carries the headline: it is the definition of functional equivalence.
    assert RUBRIC_WEIGHTS["factuality"] == max(RUBRIC_WEIGHTS.values())


class FakeProvider:
    """Returns canned completion texts in order; records prompts and call count."""

    def __init__(self, *replies: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self._replies = list(replies)
        self.calls = 0
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.last_system = system
        self.last_user = messages[0].content
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return Completion(text=reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _ctx() -> Step:
    return Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="add_to_cart", arguments={"sku": "A1"}),
        observation=Observation(content="cart has 1 item"),
    )


def _rubric_reply(
    *,
    format: float = 1.0,  # noqa: A002 - mirrors the rubric field name
    factuality: float = 1.0,
    consistency: float = 1.0,
    realism: float = 1.0,
    quality: float = 1.0,
    critique: str = "ok",
) -> str:
    return (
        f'{{"format": {format}, "factuality": {factuality}, "consistency": {consistency}, '
        f'"realism": {realism}, "quality": {quality}, "critique": "{critique}"}}'
    )


def test_rubric_judge_satisfies_protocol() -> None:
    assert isinstance(RubricJudge(FakeProvider("{}")), Judge)


def test_score_sees_both_observations_and_uses_judge_system() -> None:
    provider = FakeProvider(_rubric_reply())
    RubricJudge(provider).score(Observation(content="pred"), Observation(content="actual"), _ctx())
    assert provider.last_system == JUDGE_SYSTEM
    assert provider.last_user is not None
    assert "pred" in provider.last_user and "actual" in provider.last_user


def test_judge_prompt_makes_empty_prediction_explicit() -> None:
    prompt = _build_judge_prompt(
        Observation(content="", is_error=False),
        Observation(content="HTTP 200\n", is_error=False),
        _ctx(),
    )
    assert '"content": ""' in prompt
    assert '"content_length": 0' in prompt
    assert '"empty_content": true' in prompt
    assert '"empty_sentinel": "<EMPTY_PREDICTION>"' in prompt
    assert "<EMPTY_ACTUAL_OBSERVATION>" not in prompt
    assert "PREDICTED OBSERVATION JSON" in prompt


def test_headline_score_is_weighted_mean_of_dimensions() -> None:
    reply = _rubric_reply(format=1.0, factuality=0.0, consistency=0.5, realism=1.0, quality=0.5)
    result = RubricJudge(FakeProvider(reply)).score(
        Observation(content="p"), Observation(content="a"), _ctx()
    )
    expected = (
        RUBRIC_WEIGHTS["format"] * 1.0
        + RUBRIC_WEIGHTS["factuality"] * 0.0
        + RUBRIC_WEIGHTS["consistency"] * 0.5
        + RUBRIC_WEIGHTS["realism"] * 1.0
        + RUBRIC_WEIGHTS["quality"] * 0.5
    )
    assert result.score == pytest.approx(expected)
    assert set(result.dimensions) == set(RUBRIC_DIMENSIONS)
    assert result.valid is True


def test_equal_dimensions_score_the_same_as_the_old_unweighted_mean() -> None:
    # Comparability guard: any reply with all dimensions equal is unaffected by the reweighting,
    # so uniformly-judged steps keep their pre-overhaul scores.
    reply = _rubric_reply(format=0.5, factuality=0.5, consistency=0.5, realism=0.5, quality=0.5)
    result = RubricJudge(FakeProvider(reply)).score(
        Observation(content="p"), Observation(content="a"), _ctx()
    )
    assert result.score == pytest.approx(0.5)


def test_parse_handles_fenced_json_and_prose() -> None:
    fenced = f"Sure:\n```json\n{_rubric_reply(critique='fenced')}\n```\nDone."
    result = RubricJudge(FakeProvider(fenced)).score(
        Observation(content="p"), Observation(content="a"), _ctx()
    )
    assert result.score == 1.0
    assert result.critique == "fenced"
    prose = f"My verdict is {_rubric_reply()} overall."
    assert (
        RubricJudge(FakeProvider(prose))
        .score(Observation(content="p"), Observation(content="a"), _ctx())
        .score
        == 1.0
    )


# --- validity: judge failures must be flagged, not scored as world-model failures ---------------


def test_missing_dimension_triggers_retry_then_uses_good_reply() -> None:
    provider = FakeProvider('{"format": 1.0, "factuality": 0.8}', _rubric_reply())
    result = RubricJudge(provider).score(Observation(content="p"), Observation(content="a"), _ctx())
    assert provider.calls == 2  # first reply invalid -> one retry
    assert result.valid is True
    assert result.score == 1.0
    # The retry must tell the judge what was wrong — at temperature 0 an identical re-ask just
    # reproduces the same malformed reply (observed on Bedrock in the meta-eval).
    assert provider.last_user is not None
    assert "invalid" in provider.last_user
    assert "missing dimension 'consistency'" in provider.last_user


def test_persistent_missing_dimensions_flag_invalid_not_zero_fidelity() -> None:
    provider = FakeProvider('{"format": 1.0, "factuality": 0.8}')
    result = RubricJudge(provider).score(Observation(content="p"), Observation(content="a"), _ctx())
    assert provider.calls == 2
    assert result.valid is False
    assert "missing" in result.critique


def test_unparseable_reply_retries_then_flags_invalid() -> None:
    provider = FakeProvider("the model rambled with no json at all")
    result = RubricJudge(provider).score(Observation(content="p"), Observation(content="a"), _ctx())
    assert provider.calls == 2
    assert result.valid is False
    assert result.score == 0.0
    assert "Unparseable" in result.critique


def test_scale_confused_dimension_flags_invalid() -> None:
    # A judge answering on a 0-100 scale must not be clamped into a perfect 1.0.
    provider = FakeProvider(_rubric_reply(factuality=85))
    result = RubricJudge(provider).score(Observation(content="p"), Observation(content="a"), _ctx())
    assert result.valid is False
    assert "out of range" in result.critique


def test_minor_float_slop_is_clamped_not_invalidated() -> None:
    result = RubricJudge(FakeProvider(_rubric_reply(factuality=1.1))).score(
        Observation(content="p"), Observation(content="a"), _ctx()
    )
    assert result.valid is True
    assert result.dimensions["factuality"] == 1.0


# --- truncation: huge observations stay gradeable without hiding the tail -----------------------


def test_long_content_is_truncated_with_head_tail_and_full_length() -> None:
    content = "x" * 8000 + "MIDDLE" + "y" * 8000 + "TAIL-END"
    prompt = _build_judge_prompt(Observation(content="short"), Observation(content=content), _ctx())
    assert "characters omitted" in prompt
    assert prompt.count("x" * 100) > 0  # head survives
    assert "TAIL-END" in prompt  # tail survives
    assert "MIDDLE" not in prompt  # middle is dropped
    assert f'"content_length": {len(content)}' in prompt  # full length still reported


def test_short_content_is_never_truncated() -> None:
    content = "z" * (OBSERVATION_HEAD_CHARS + OBSERVATION_TAIL_CHARS)
    prompt = _build_judge_prompt(Observation(content="short"), Observation(content=content), _ctx())
    assert "characters omitted" not in prompt
    assert content in prompt
    assert "content_sha256" not in prompt  # hash only accompanies truncated content


def test_truncated_payloads_carry_a_hash_so_middle_divergence_is_visible() -> None:
    # Two equal-length observations diverging ONLY in the omitted middle produce identical
    # visible text and identical content_length — the hash is the only remaining tell, and the
    # prompt instructs the judge to use it.
    head, tail = "x" * 7000, "y" * 7000
    actual = head + "REAL-MIDDLE-" + "a" * 4000 + tail
    predicted = head + "FAKE-MIDDLE-" + "b" * 4000 + tail
    assert len(actual) == len(predicted)
    prompt = _build_judge_prompt(
        Observation(content=predicted), Observation(content=actual), _ctx()
    )
    hashes = re.findall(r'"content_sha256": "([0-9a-f]{64})"', prompt)
    assert len(hashes) == 2
    assert hashes[0] != hashes[1]  # differing hashes expose the hidden divergence
    assert "content_sha256" in JUDGE_SYSTEM  # the prompt explains the field
