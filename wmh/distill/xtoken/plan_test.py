"""Tests for joining a student datum to a teacher render.

The byte-boundary fallback exists because the DP refuses long, poorly-anchored
reasoning islands and dropping them cost 65,169 of 102,346 student tokens in one
live step. These tests pin that a refusal keeps full coverage, that the fallback
agrees with the DP where both apply, and that byte-misaligned pairs are still
rejected however they were produced.

They also pin the run-to-message mapping, which a per-turn datum depends on: the
render covers the whole conversation prefix a turn was sampled under, so the
datum's single run has to be aligned against ITS assistant message rather than
the render's first one.
"""

from __future__ import annotations

from typing import cast

from wmh.distill.data import TrainDatum
from wmh.distill.xtoken.aligner import align_tokens
from wmh.distill.xtoken.byte_offsets_test import FakeByteLevelTokenizer
from wmh.distill.xtoken.plan import (
    SurfaceTokenizer,
    _pair_is_byte_aligned,
    boundary_partition,
    build_chunk_plan,
    sampled_runs,
)
from wmh.distill.xtoken.teacher_render import ContentIsland, TeacherRender


def _ends(pieces: list[bytes]) -> list[int]:
    """Cumulative byte-end offsets for a token sequence."""
    out: list[int] = []
    total = 0
    for piece in pieces:
        total += len(piece)
        out.append(total)
    return out


def test_sampled_runs_finds_each_contiguous_loss_span() -> None:
    datum = TrainDatum(
        trial_name="t",
        fragment_index=0,
        model_input_tokens=list(range(10)),
        loss_mask=[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        sampled_logprobs=[0.0] * 10,
    )
    assert sampled_runs(datum) == [(1, 3), (5, 8)]


def test_sampled_runs_handles_a_run_ending_at_the_last_token() -> None:
    datum = TrainDatum(
        trial_name="t",
        fragment_index=0,
        model_input_tokens=[1, 2, 3],
        loss_mask=[0.0, 1.0, 1.0],
        sampled_logprobs=[0.0, -1.0, -1.0],
    )
    assert sampled_runs(datum) == [(1, 3)]


def test_boundary_partition_cuts_only_at_shared_boundaries() -> None:
    # Same 6 bytes both sides: student 'ab|cd|ef', teacher 'abc|d|ef'.
    # Shared boundaries are at bytes 4 and 6, so the first two student tokens
    # pair with the first two teacher tokens, then 'ef' pairs one to one.
    student = _ends([b"ab", b"cd", b"ef"])
    teacher = _ends([b"abc", b"d", b"ef"])
    pairs = boundary_partition(student, teacher, 0, 0, (0, 3), (0, 3))
    assert [(p.student_start, p.student_end, p.teacher_start, p.teacher_end) for p in pairs] == [
        (0, 2, 0, 2),
        (2, 3, 2, 3),
    ]
    # Only the one-to-one cell claims exactness.
    assert [p.exact for p in pairs] == [False, True]


def test_boundary_partition_covers_every_token() -> None:
    student = _ends([b"aa", b"bb", b"cc", b"dd"])
    teacher = _ends([b"a", b"abb", b"ccd", b"d"])
    pairs = boundary_partition(student, teacher, 0, 0, (0, 4), (0, 4))
    # Coverage is total on both sides: no token is left out.
    assert pairs[0].student_start == 0
    assert pairs[-1].student_end == 4
    assert pairs[0].teacher_start == 0
    assert pairs[-1].teacher_end == 4
    spans = [(p.student_start, p.student_end) for p in pairs]
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))


def test_boundary_partition_honours_island_origins() -> None:
    # The island starts 3 bytes into the student span and 5 into the render.
    student = _ends([b"XXX", b"ab", b"cd"])
    teacher = _ends([b"YYYYY", b"abc", b"d"])
    pairs = boundary_partition(student, teacher, 3, 5, (1, 3), (1, 3))
    assert [(p.student_start, p.student_end, p.teacher_start, p.teacher_end) for p in pairs] == [
        (1, 3, 1, 3)
    ]


def test_fallback_agrees_with_the_dp_where_both_apply() -> None:
    # A small byte-identical case the DP handles comfortably: the exact partition
    # and the DP must produce the same cells, which is what makes the fallback a
    # safe substitute rather than a different objective.
    student_pieces = [b"ls", b" -", b"la", b" |", b" wc"]
    teacher_pieces = [b"l", b"s -", b"la ", b"|", b" wc"]
    dp = align_tokens(
        [piece.decode() for piece in student_pieces],
        [piece.decode() for piece in teacher_pieces],
    )
    assert dp is not None
    exact = boundary_partition(
        _ends(student_pieces),
        _ends(teacher_pieces),
        0,
        0,
        (0, len(student_pieces)),
        (0, len(teacher_pieces)),
    )
    as_tuples = [(p.student_start, p.student_end, p.teacher_start, p.teacher_end) for p in exact]
    dp_tuples = [(p.student_start, p.student_end, p.teacher_start, p.teacher_end) for p in dp]
    assert as_tuples == dp_tuples


def test_byte_alignment_guard_rejects_a_shifted_pair() -> None:
    # The reviewer's repro: student '..' + '.....' against a 5-way teacher split
    # of the same 7 bytes. Pairing student token 1 with teacher tokens 0-2 covers
    # bytes 2..7 against 0..5, which must be refused.
    student = _ends([b"..", b"....."])
    teacher = _ends([b"...", b".", b".", b".", b"."])
    assert _pair_is_byte_aligned(student, teacher, 0, 0, (1, 2), (0, 3)) is False
    assert _pair_is_byte_aligned(student, teacher, 0, 0, (0, 2), (0, 5)) is True


def test_byte_alignment_guard_accepts_every_fallback_pair() -> None:
    # Whatever the fallback emits must survive the guard by construction, since
    # both are derived from the same shared byte boundaries.
    student_pieces = [b"aa", b"bb", b"cc", b"dd", b"ee"]
    teacher_pieces = [b"a", b"abb", b"c", b"cdd", b"ee"]
    student = _ends(student_pieces)
    teacher = _ends(teacher_pieces)
    pairs = boundary_partition(student, teacher, 0, 0, (0, 5), (0, 5))
    assert pairs
    for pair in pairs:
        assert _pair_is_byte_aligned(
            student,
            teacher,
            0,
            0,
            (pair.student_start, pair.student_end),
            (pair.teacher_start, pair.teacher_end),
        )


def test_boundary_partition_on_a_single_shared_boundary_is_one_cell() -> None:
    # Worst case: the only shared boundary is the island end, so the whole island
    # becomes one coarse chunk. Still full coverage, just blunt signal.
    student = _ends([b"abc", b"de"])
    teacher = _ends([b"ab", b"cde"])
    pairs = boundary_partition(student, teacher, 0, 0, (0, 2), (0, 2))
    assert [(p.student_start, p.student_end, p.teacher_start, p.teacher_end) for p in pairs] == [
        (0, 2, 0, 2)
    ]
    assert pairs[0].exact is False


# --- the run-to-message mapping a per-turn datum needs -----------------------------------------

_FRAME = 10
_PIECES = {1: b"action", 2: b" 0", 3: b" 1", 4: b"P"}
_SPECIALS = {_FRAME: "<|s|>"}


def _two_turn_render() -> TeacherRender:
    """A render of TWO assistant turns, with an island over each one's text.

    Decodes to `<|s|>action 0<|s|>action 1`, which is the shape a per-turn datum
    is scored against: the conversation prefix holds earlier assistant messages
    that this datum trains none of.
    """
    return TeacherRender(
        token_ids=[_FRAME, 1, 2, _FRAME, 1, 3],
        islands=[
            ContentIsland(
                kind="text",
                message_index=1,
                text="action 0",
                teacher_start=1,
                teacher_end=3,
                byte_start=5,
                byte_end=13,
            ),
            ContentIsland(
                kind="text",
                message_index=3,
                text="action 1",
                teacher_start=4,
                teacher_end=6,
                byte_start=18,
                byte_end=26,
            ),
        ],
    )


def _second_turn_datum() -> TrainDatum:
    """One turn's datum: a prompt token, then the ids that decode to `action 1`."""
    return TrainDatum(
        trial_name="t",
        fragment_index=1,
        span_indices=[1],
        model_input_tokens=[4, 1, 3],
        loss_mask=[0.0, 1.0, 1.0],
        sampled_logprobs=[0.0, -0.5, -0.25],
    )


def test_a_per_turn_datum_aligns_against_the_assistant_message_it_holds() -> None:
    """With the mapping, turn 1's run is scored against turn 1's island."""
    # `SurfaceTokenizer` is nominal, not a Protocol, so the fake is cast exactly
    # as the loop casts the real (untyped) cookbook and HuggingFace tokenizers.
    tokenizer = cast("SurfaceTokenizer", FakeByteLevelTokenizer(_PIECES, _SPECIALS))

    plan = build_chunk_plan(
        _second_turn_datum(),
        _two_turn_render(),
        tokenizer,
        tokenizer,
        assistant_indices=[3],
    )

    assert [
        (chunk.student_start, chunk.student_end, chunk.teacher_start, chunk.teacher_end)
        for chunk in plan.chunks
    ] == [(1, 2, 4, 5), (2, 3, 5, 6)]
    assert plan.scored_student_tokens == 2
    assert plan.fragment_index == 1


def test_without_the_mapping_a_per_turn_datum_is_aligned_against_the_wrong_turn() -> None:
    """Why the mapping is passed: the island-order default takes the FIRST message.

    Here that leaves the datum unscored (turn 0's text is not in turn 1's bytes),
    which is the benign outcome; the dangerous one is two turns whose text
    coincides. Either way the default is only correct for a whole-episode datum,
    so the caller states the mapping.
    """
    tokenizer = cast("SurfaceTokenizer", FakeByteLevelTokenizer(_PIECES, _SPECIALS))

    plan = build_chunk_plan(_second_turn_datum(), _two_turn_render(), tokenizer, tokenizer)

    assert plan.chunks == []
