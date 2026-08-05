"""Tests for sample-rollout rendering: single-pass decode, fragments, truncation."""

from __future__ import annotations

import pytest

from wmo.optimize.model.samples import (
    FRAGMENT_BREAK,
    MAX_EPISODE_CHARS,
    SAMPLE_SEPARATOR,
    SampleRollout,
    render_episode_text,
    sample_rollouts,
    samples_markdown,
    truncate_middle,
)
from wmo.optimize.model.tokens import TrialRecord
from wmo.providers.tinker import TokenSpan

_SPECIAL_BASE = 1000
"""Fake special-token ids start here; the decoder renders them as `<|sp|>`."""


class _Decoder:
    """A recording specials decoder: ids under 1000 are chars, others `<|sp|>`."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def decode_with_specials(self, token_ids: list[int]) -> str:
        self.calls.append(list(token_ids))
        return "".join("<|sp|>" if t >= _SPECIAL_BASE else chr(t) for t in token_ids)


def _record(
    spans: list[TokenSpan],
    *,
    trial_name: str = "task-a__s1",
    reward: float = 1.0,
    passed: bool = True,
    stop_reason: str | None = "submitted",
) -> TrialRecord:
    return TrialRecord(
        task_id="task-a",
        attempt=1,
        trial_name=trial_name,
        reward=reward,
        passed=passed,
        spans=spans,
        stop_reason=stop_reason,
        artifact_dir=f"/tmp/{trial_name}",
    )


def _span(call_index: int, prompt: list[int], sampled: list[int]) -> TokenSpan:
    return TokenSpan(
        call_index=call_index,
        prompt_token_ids=prompt,
        sampled_token_ids=sampled,
        sampled_logprobs=[-0.1] * len(sampled),
    )


# -- render_episode_text -----------------------------------------------------------------------


def test_prefix_clean_episode_decodes_in_one_pass() -> None:
    """The final prompt plus its sampled tokens IS the whole conversation."""
    decoder = _Decoder()
    first = _span(0, [_SPECIAL_BASE, 72, 105], [111, 107])  # <|sp|>Hi -> ok
    # The second prompt extends prompt + sampled verbatim, then adds "!".
    second = _span(1, [_SPECIAL_BASE, 72, 105, 111, 107, 33], [100])  # ... ! -> d

    text = render_episode_text(_record([second, first]), decoder)  # any span order

    assert decoder.calls == [[_SPECIAL_BASE, 72, 105, 111, 107, 33, 100]]
    assert "<|sp|>Hiok!d" in text
    assert FRAGMENT_BREAK not in text
    # The header carries the trial identity and outcome.
    assert text.startswith("### trial task-a__s1\n")
    assert "reward: 1" in text
    assert "passed: True" in text
    assert "stop reason: submitted" in text
    assert "spans: 2" in text
    assert "fragments: 1" in text
    assert "episode tokens: 7" in text


def test_fragmented_episode_marks_every_break() -> None:
    decoder = _Decoder()
    first = _span(0, [72, 105], [111])  # Hi -> o
    edited = _span(1, [88, 89], [122])  # XY -> z (NOT a prefix extension)

    text = render_episode_text(_record([first, edited]), decoder)

    assert decoder.calls == [[72, 105, 111], [88, 89, 122]]
    assert text.count(FRAGMENT_BREAK) == 1
    before, after = text.split(FRAGMENT_BREAK)
    assert "Hio" in before
    assert "XYz" in after
    assert "fragments: 2" in text


def test_spanless_trial_renders_a_note_without_decoding() -> None:
    decoder = _Decoder()
    text = render_episode_text(_record([], reward=0.0, passed=False, stop_reason=None), decoder)
    assert decoder.calls == []
    assert "no token spans were recorded" in text
    assert "spans: 0" in text
    assert "stop reason: unknown" in text


def test_render_episode_text_truncates_the_middle_of_long_bodies() -> None:
    class _LongDecoder:
        def decode_with_specials(self, token_ids: list[int]) -> str:
            del token_ids
            return "H" * 30_000 + "M" * 60_000 + "T" * 30_000

    text = render_episode_text(_record([_span(0, [72], [105])]), _LongDecoder())
    assert "chars elided" in text
    assert "M" * 100 not in text  # the middle is gone
    assert "H" * 100 in text  # head kept
    assert "T" * 100 in text  # tail kept
    assert len(text) < MAX_EPISODE_CHARS + 500


# -- truncate_middle ---------------------------------------------------------------------------


def test_truncate_middle_keeps_head_and_tail_exactly() -> None:
    text = "H" * 30_000 + "M" * 50_000 + "T" * 30_000
    out = truncate_middle(text)
    head = MAX_EPISODE_CHARS // 2
    tail = MAX_EPISODE_CHARS - head
    assert out.startswith(text[:head])
    assert out.endswith(text[-tail:])
    assert f"[... {len(text) - MAX_EPISODE_CHARS} chars elided" in out


def test_truncate_middle_leaves_short_text_untouched() -> None:
    assert truncate_middle("short") == "short"
    assert truncate_middle("x" * 10, limit=10) == "x" * 10


def test_truncate_middle_rejects_a_nonpositive_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        truncate_middle("text", limit=0)


# -- sample_rollouts and the document ----------------------------------------------------------


def test_sample_rollouts_takes_the_first_span_bearing_records() -> None:
    decoder = _Decoder()
    span = _span(0, [72], [105])
    records = [
        _record([], trial_name="t0"),  # span-less: skipped
        _record([span], trial_name="t1", reward=0.0, passed=False),
        _record([span], trial_name="t2"),
        _record([span], trial_name="t3"),  # beyond the limit
    ]
    samples = sample_rollouts(records, decoder, 2)
    assert [sample.trial_name for sample in samples] == ["t1", "t2"]
    assert samples[0].reward == 0.0
    assert samples[0].text.startswith("### trial t1\n")
    assert sample_rollouts(records, decoder, 0) == []


def test_samples_markdown_joins_with_the_separator() -> None:
    doc = samples_markdown(
        [
            SampleRollout(trial_name="a", reward=1.0, text="### trial a\nA\n"),
            SampleRollout(trial_name="b", reward=0.0, text="### trial b\nB\n"),
        ]
    )
    assert doc.startswith("### trial a\n")
    assert doc.count(SAMPLE_SEPARATOR) == 1
    assert doc.endswith("### trial b\nB\n")
