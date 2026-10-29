"""Release-rule tests for incremental redaction of a streamed completion."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.guardrails.regex import (
    BuiltinPattern,
    RegexAdapterDocument,
    RegexClassifier,
)
from exp.runtime.gateway.guardrails.streaming import (
    StreamableClassifier,
    StreamingRedactor,
    release_segment,
)


class _NeverSettles:
    """A redactor that always keeps the whole tail buffered."""

    def release_boundary(self, text: str) -> int:
        """Return zero settled characters."""
        return 0

    def redact(self, text: str) -> tuple[bool, str]:
        """Return the text unchanged."""
        return False, text


class _Overreaching:
    """A redactor that names a boundary past the end of the tail."""

    def release_boundary(self, text: str) -> int:
        """Return a boundary beyond the buffered text."""
        return len(text) + 25

    def redact(self, text: str) -> tuple[bool, str]:
        """Return the text unchanged."""
        return False, text


def _detector(window: int = 64) -> RegexClassifier:
    """Return one deterministic detector over the built-in families."""
    return RegexClassifier(
        RegexAdapterDocument(
            adapter_id="detector",
            builtin_patterns=(
                BuiltinPattern.EMAIL,
                BuiltinPattern.CREDIT_CARD,
                BuiltinPattern.API_KEY,
            ),
            stream_window_characters=window,
        )
    )


def _stream(redactor: StreamingRedactor, chunks: list[str]) -> str:
    """Drive one completion through the release rule chunk by chunk.

    Args:
        redactor: Deterministic redactor under test.
        chunks: Provider deltas in arrival order.

    Returns:
        Everything the caller would have seen, in order.
    """
    released: list[str] = []
    pending = ""
    for chunk in chunks:
        pending += chunk
        segment = release_segment(redactor=redactor, pending=pending, final=False)
        released.append(segment.release)
        pending = segment.pending
    segment = release_segment(redactor=redactor, pending=pending, final=True)
    released.append(segment.release)
    return "".join(released)


def _splits(text: str) -> list[list[str]]:
    """Return every two-way split of ``text`` at each character offset."""
    return [[text[:offset], text[offset:]] for offset in range(len(text) + 1)]


@pytest.mark.parametrize(
    "completion",
    [
        "contact me at ada@example.com please",
        "card 4111 1111 1111 1111 on file",
        "key sk-abcdefghijklmnopqrstuvwx tail",
    ],
)
def test_match_split_at_every_offset_is_still_redacted(completion: str) -> None:
    """A match straddling any delta boundary is redacted exactly once."""
    detector = _detector()
    _, whole = detector.redact(completion)
    assert "[REDACTED]" in whole
    for chunks in _splits(completion):
        assert _stream(detector, chunks) == whole


def test_single_character_deltas_still_redact() -> None:
    """A completion arriving one character at a time is redacted whole."""
    detector = _detector()
    completion = "mail ada@example.com and card 4111 1111 1111 1111"
    _, whole = detector.redact(completion)
    assert _stream(detector, list(completion)) == whole


def test_release_precedes_the_end_of_a_long_completion() -> None:
    """Text older than the trailing window is released before the stream ends."""
    detector = _detector()
    pending = "safe prose " * 40
    segment = release_segment(redactor=detector, pending=pending, final=False)
    assert segment.release
    assert len(segment.pending) <= 64
    assert segment.release + segment.pending == pending


def test_final_segment_releases_everything_buffered() -> None:
    """No text survives the final segment, whatever the adapter's boundary."""
    segment = release_segment(redactor=_NeverSettles(), pending="tail", final=True)
    assert segment.release == "tail"
    assert segment.pending == ""


def test_boundary_beyond_the_tail_is_clamped() -> None:
    """An adapter cannot release text the caller has not presented."""
    segment = release_segment(redactor=_Overreaching(), pending="abc", final=False)
    assert segment.release == "abc"
    assert segment.pending == ""


def test_flag_reports_the_segment_that_matched() -> None:
    """The flag follows the released segment, not the whole completion."""
    detector = _detector()
    pending = "ada@example.com " + "x" * 200 + " done "
    segment = release_segment(redactor=detector, pending=pending, final=False)
    assert segment.flagged
    assert "[REDACTED]" in segment.release


def test_only_the_trailing_candidate_run_is_held() -> None:
    """Prose releases at once: a family cannot reach back across a space."""
    detector = _detector(window=512)
    segment = release_segment(
        redactor=detector,
        pending="the quick brown fox jumps over",
        final=False,
    )
    assert segment.release == "the quick brown fox jumps "
    assert segment.pending == "over"


def test_an_unbroken_digit_run_holds_the_whole_tail() -> None:
    """A run past the window settles nothing, so the tail stays buffered."""
    detector = RegexClassifier(
        RegexAdapterDocument(
            adapter_id="detector",
            builtin_patterns=(BuiltinPattern.CREDIT_CARD,),
            stream_window_characters=64,
        )
    )
    segment = release_segment(redactor=detector, pending="1" * 400, final=False)
    assert segment.release == ""
    assert segment.pending == "1" * 400


def test_a_match_longer_than_the_window_is_never_released_in_pieces() -> None:
    """The built-in expressions are unbounded, so a long run buffers whole.

    An address whose local part outruns the window must come out redacted,
    not as a released prefix completed by a later delta.
    """
    detector = _detector(window=64)
    completion = "start " + "a" * 600 + "@example.com done"
    _, whole = detector.redact(completion)
    assert "a" * 600 not in whole
    assert _stream(detector, [completion[:300], completion[300:]]) == whole


def test_an_authored_pattern_is_not_streamable() -> None:
    """An expression of unknown span keeps its completions buffered.

    Nothing bounds how far one authored match can reach, so no trailing
    window can prove a released prefix is settled.
    """
    detector = RegexClassifier(
        RegexAdapterDocument(
            adapter_id="detector",
            patterns=(r"SECRET-[0-9]+",),
            builtin_patterns=(BuiltinPattern.EMAIL,),
            stream_window_characters=65_536,
        )
    )
    assert detector.stream_redactor() is None


def test_detector_advertises_the_streaming_capability() -> None:
    """The deterministic detector satisfies the capability protocol."""
    detector = _detector()
    assert isinstance(detector, StreamableClassifier)
    assert detector.stream_redactor() is detector
