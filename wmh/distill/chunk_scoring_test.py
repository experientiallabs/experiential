"""Tests for `_chunk_scored_datums`, the cross-tokenizer scoring orchestration.

Scope: which datums reach the teacher, what conversation each one is scored
against, and why the rest are dropped. The alignment itself (byte offsets,
island location, partition intersection, advantage math) is covered by
`wmh/distill/xtoken/*_test.py`, so the render and plan calls are stubbed here.
Two things must be right at this layer. Every drop is counted under its own
reason (a run that trains on 3 of 64 trajectories and reports only "coverage was
low" cannot be debugged), and a PER-TURN datum is paired with the conversation as
it stood for that turn: the ephemeral-reasoning history makes every turn its own
datum, so pairing one against the whole episode would score the wrong assistant
message with no error anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest
from llm_waterfall.types import ChatMessage

import wmh.distill.loop as loop_module
from wmh.distill.data import TrainDatum
from wmh.distill.loop import _chunk_scored_datums, _ChunkScoringStats
from wmh.distill.rendering import ChatRendering
from wmh.distill.tokens import ConversationReplay, TrialRecord
from wmh.distill.xtoken.byte_offsets_test import FakeByteLevelTokenizer
from wmh.distill.xtoken.chunks import ChunkPlan
from wmh.distill.xtoken.prompt_logprobs import PromptLogprobClient
from wmh.distill.xtoken.teacher_render_test import FakeTemplateTokenizer
from wmh.providers.tinker import TokenSpan

TURNS = 3


def _datum(
    trial: str,
    *,
    fragment: int = 0,
    loss_tokens: int = 4,
    span_indices: Sequence[int] | None = None,
) -> TrainDatum:
    """A datum with `loss_tokens` trainable positions after a 2-token prompt."""
    total = 2 + loss_tokens
    return TrainDatum(
        trial_name=trial,
        fragment_index=fragment,
        span_indices=[fragment] if span_indices is None else list(span_indices),
        model_input_tokens=list(range(10, 10 + total)),
        loss_mask=[0.0, 0.0] + [1.0] * loss_tokens,
        sampled_logprobs=[0.0, 0.0] + [-0.5] * loss_tokens,
    )


def _record(trial: str, *, canonical: bool = True, turns: int = 1) -> TrialRecord:
    return TrialRecord(
        task_id=trial,
        attempt=1,
        trial_name=trial,
        reward=0.0,
        passed=False,
        spans=[
            TokenSpan(
                call_index=turn,
                prompt_token_ids=[10, 11],
                sampled_token_ids=[12, 13, 14, 15],
                sampled_logprobs=[-0.5] * 4,
                # `delta_start` travels with `delta_messages` or neither is set
                # (TokenSpan enforces it): a delta boundary without its messages,
                # or messages without a boundary, cannot be replayed.
                delta_start=2 * turn if canonical else None,
                delta_messages=[] if canonical else None,
            )
            for turn in range(turns)
        ],
        artifact_dir=f"/trials/{trial}",
    )


def _replay(turns: int = 1) -> ConversationReplay:
    """A real replay of a `turns`-turn episode, reasoning included.

    Real rather than stubbed because `turn_conversation` is exercised for real
    here: slicing the replay per turn IS the behavior under test.
    """
    messages: list[ChatMessage] = []
    assistant_index_by_span: dict[int, int] = {}
    for turn in range(turns):
        messages.append(ChatMessage(role="user", content=f"observation {turn}"))
        assistant_index_by_span[turn] = len(messages)
        messages.append(
            ChatMessage(role="assistant", content=f"<think>reasoning {turn}</think>action {turn}")
        )
    return ConversationReplay(
        messages=messages, tools=None, assistant_index_by_span=assistant_index_by_span
    )


class _Plan:
    """Stands in for a ChunkPlan; only these two attributes are read here."""

    def __init__(self, *, chunks: int, scored: int) -> None:
        self.chunks = list(range(chunks))
        self.scored_student_tokens = scored


class _Render:
    def __init__(self, *, islands: int, tokens: int = 6) -> None:
        self.islands = list(range(islands))
        self.token_ids = list(range(100, 100 + tokens))


class _Teacher:
    """Records what it was asked to score; optionally fails on demand."""

    def __init__(self, *, fail_on: set[int] | None = None) -> None:
        self.calls: list[list[int]] = []
        self._fail_on = fail_on or set()

    def score(self, token_ids: Sequence[int]) -> list[float | None]:
        index = len(self.calls)
        self.calls.append(list(token_ids))
        if index in self._fail_on:
            raise RuntimeError("teacher endpoint returned garbage")
        return [None] + [-0.25] * (len(token_ids) - 1)


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy-path render/plan: one island, one chunk covering every loss token."""
    # Mirrors the real contract: None if ANY span lost its canonical messages,
    # since one re-render fallback disqualifies the whole trial.
    monkeypatch.setattr(
        loop_module,
        "reconstruct_conversation",
        lambda spans, _r: (
            _replay(len(spans))
            if spans and all(span.delta_messages is not None for span in spans)
            else None
        ),
    )
    monkeypatch.setattr(loop_module, "render_for_teacher", lambda *_a, **_k: _Render(islands=1))
    monkeypatch.setattr(
        loop_module, "build_chunk_plan", lambda *_a, **_k: _Plan(chunks=2, scored=4)
    )


def _score(
    datums: Sequence[TrainDatum], records: Sequence[TrialRecord], teacher: _Teacher
) -> tuple[list[TrainDatum], list[ChunkPlan], list[list[float | None]], _ChunkScoringStats]:
    """Run the scorer with the stubbed collaborators the `stubs` fixture installs.

    The renderer and both tokenizers are never touched here (the fixture replaces
    every function that would use them), so they are placeholders typed as the
    collaborators they stand in for; only `_Teacher.score` is really called.
    """
    return _chunk_scored_datums(
        datums,
        records,
        cast("ChatRendering", object()),
        object(),
        object(),
        cast("PromptLogprobClient", teacher),
    )


def test_scores_every_datum_when_each_trial_is_a_single_fragment(stubs: None) -> None:
    datums = [_datum("a"), _datum("b")]
    records = [_record("a"), _record("b")]
    teacher = _Teacher()

    kept, plans, rows, stats = _score(datums, records, teacher)

    assert [d.trial_name for d in kept] == ["a", "b"]
    assert len(plans) == len(rows) == 2
    assert stats.scored == 2
    # Coverage is over TRAINABLE tokens: 4 scored of 4 loss tokens, per datum.
    assert stats.scored_student_tokens == 8
    assert stats.loss_tokens == 8
    assert stats.coverage == 1.0
    assert len(teacher.calls) == 2


def test_every_per_turn_datum_of_one_trial_is_scored(stubs: None) -> None:
    """The shape the ephemeral-reasoning history produces: one datum per turn.

    Each one carries its own turn's spans, so each one is scored, and the trial's
    replay is reconstructed ONCE and reused rather than per datum.
    """
    datums = [_datum("a", fragment=turn) for turn in range(TURNS)]
    records = [_record("a", turns=TURNS)]
    teacher = _Teacher()

    kept, plans, rows, stats = _score(datums, records, teacher)

    assert [d.fragment_index for d in kept] == [0, 1, 2]
    assert len(plans) == len(rows) == TURNS
    assert stats.scored == TURNS
    assert stats.no_span_map == 0
    assert len(teacher.calls) == TURNS


def test_each_turn_is_scored_against_its_own_conversation_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-bearing pairing: turn N sees turns < N, reasoning-free.

    Getting this wrong is silent. The teacher would score a real conversation
    that is simply not the one the student sampled under, and every advantage in
    the datum would be wrong without a single warning.
    """
    monkeypatch.setattr(
        loop_module, "reconstruct_conversation", lambda spans, _r: _replay(len(spans))
    )
    seen: list[list[ChatMessage]] = []
    monkeypatch.setattr(
        loop_module,
        "render_for_teacher",
        lambda _tok, messages, _tools=None: (seen.append(list(messages)), _Render(islands=1))[1],
    )
    runs: list[Sequence[int]] = []
    monkeypatch.setattr(
        loop_module,
        "build_chunk_plan",
        lambda *_a, assistant_indices=None, **_k: (
            runs.append(list(assistant_indices or [])),
            _Plan(chunks=1, scored=4),
        )[1],
    )

    datums = [_datum("a", fragment=turn) for turn in range(TURNS)]
    _kept, _plans, _rows, stats = _score(datums, [_record("a", turns=TURNS)], _Teacher())

    assert stats.scored == TURNS
    # Turn N's view ends at its own assistant message: 2N + 2 messages.
    assert [len(messages) for messages in seen] == [2, 4, 6]
    for turn, messages in enumerate(seen):
        assistant = [message for message in messages if message.role == "assistant"]
        # The turn's OWN reasoning is kept (it is what the teacher scores) and
        # every earlier turn's is gone (the student's prompt did not carry it).
        assert assistant[-1].content == f"<think>reasoning {turn}</think>action {turn}"
        for earlier, message in enumerate(assistant[:-1]):
            assert message.content == f"action {earlier}"
    # Each datum's single run is aligned to that turn's assistant message, not
    # to the first assistant message of the render.
    assert runs == [[1], [3], [5]]


def test_a_datum_without_span_provenance_is_skipped_not_guessed(stubs: None) -> None:
    """No span map means the datum's turn is unknown; guessing would corrupt it."""
    datums = [_datum("a", span_indices=[]), _datum("b")]
    records = [_record("a"), _record("b")]
    teacher = _Teacher()

    kept, _plans, _rows, stats = _score(datums, records, teacher)

    assert [d.trial_name for d in kept] == ["b"]
    assert stats.no_span_map == 1
    assert stats.scored == 1
    # The dropped datum still counts toward the denominator: its tokens were
    # collected and paid for, and coverage must not flatter itself by ignoring
    # trajectories it failed to use.
    assert stats.loss_tokens == 8
    assert stats.coverage == pytest.approx(4 / 8)
    assert len(teacher.calls) == 1


def test_a_datum_naming_a_span_the_replay_does_not_have_is_skipped(stubs: None) -> None:
    """A datum and a replay that disagree about the episode cannot be paired."""
    _kept, _plans, _rows, stats = _score(
        [_datum("a", span_indices=[7])], [_record("a")], _Teacher()
    )

    assert stats.no_span_map == 1
    assert stats.scored == 0


def test_a_lost_canonical_history_is_counted_as_no_replay(stubs: None) -> None:
    datums = [_datum("a"), _datum("b")]
    records = [_record("a", canonical=False), _record("b")]
    teacher = _Teacher()

    kept, _plans, _rows, stats = _score(datums, records, teacher)

    assert [d.trial_name for d in kept] == ["b"]
    assert stats.no_replay == 1


def test_a_datum_with_no_matching_record_is_counted_not_crashed(stubs: None) -> None:
    """A datum whose trial record is missing must not raise a KeyError mid-step."""
    kept, _plans, _rows, stats = _score([_datum("orphan")], [], _Teacher())

    assert kept == []
    assert stats.no_replay == 1


def test_empty_islands_and_empty_chunks_are_counted_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two failures have different causes and must not be conflated.

    No islands means the teacher's render did not contain the student's text at
    all (a template or escaping problem). No chunks means the text was found but
    the two tokenizations shared no usable boundary inside it (an alignment
    problem). Collapsing them would point at the wrong fix.
    """
    monkeypatch.setattr(loop_module, "reconstruct_conversation", lambda *_a: _replay(1))
    monkeypatch.setattr(
        loop_module,
        "render_for_teacher",
        lambda _tok, messages, _tools=None: _Render(islands=0 if not messages else 1),
    )
    monkeypatch.setattr(
        loop_module, "build_chunk_plan", lambda *_a, **_k: _Plan(chunks=0, scored=0)
    )

    _kept, _plans, _rows, stats = _score([_datum("a")], [_record("a")], _Teacher())
    assert (stats.no_islands, stats.no_chunks, stats.scored) == (0, 1, 0)

    monkeypatch.setattr(loop_module, "render_for_teacher", lambda *_a, **_k: _Render(islands=0))
    _kept, _plans, _rows, stats = _score([_datum("a")], [_record("a")], _Teacher())
    assert (stats.no_islands, stats.no_chunks, stats.scored) == (1, 0, 0)


def test_one_failed_scoring_call_does_not_lose_the_other_trajectories(stubs: None) -> None:
    """A single bad endpoint response must cost one trajectory, not the step."""
    datums = [_datum("a"), _datum("b"), _datum("c")]
    records = [_record("a"), _record("b"), _record("c")]
    teacher = _Teacher(fail_on={1})

    kept, plans, rows, stats = _score(datums, records, teacher)

    assert [d.trial_name for d in kept] == ["a", "c"]
    assert stats.scoring_failed == 1
    assert stats.scored == 2
    # Kept datums, plans and rows must stay index-aligned after a mid-batch drop:
    # a row paired with the wrong datum is exactly the silent corruption this
    # whole path exists to avoid.
    assert len(kept) == len(plans) == len(rows) == 2


def test_coverage_is_zero_rather_than_undefined_for_an_empty_batch() -> None:
    _kept, _plans, _rows, stats = _score([], [], _Teacher())

    assert stats.scored == 0
    assert stats.coverage == 0.0


# --- end to end through the real render and plan -------------------------------------------------

_TURN_1_REASONING = "reasoning for turn one"


def test_a_per_turn_datum_is_scored_end_to_end_through_the_real_render_and_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole rework, with only the replay and the endpoint stubbed.

    Real `turn_conversation`, real `render_for_teacher`, real `build_chunk_plan`,
    against the deterministic tokenizer fakes the xtoken tests use. What it
    proves: turn 1's datum is scored against a render of the conversation as it
    stood for turn 1 (turn 0's reasoning gone, turn 1's kept), and the chunks land
    on turn 1's OWN tokens, reasoning included. The reasoning being teacher-scored
    is the half of the change that is easy to lose: it is absent from history and
    must still carry gradient.
    """
    replay = ConversationReplay(
        messages=[
            ChatMessage(role="user", content="observation 0"),
            ChatMessage(
                role="assistant", content="<think>reasoning for turn zero</think>action zero"
            ),
            ChatMessage(role="user", content="observation 1"),
            ChatMessage(role="assistant", content=f"<think>{_TURN_1_REASONING}</think>action one"),
        ],
        assistant_index_by_span={0: 1, 1: 3},
    )
    monkeypatch.setattr(loop_module, "reconstruct_conversation", lambda *_a: replay)
    # One byte per teacher token, so every island edge falls on a token boundary
    # (the byte-alignment guard is exercised by the xtoken tests, not this one).
    teacher_tokenizer = FakeTemplateTokenizer(chunk_bytes=1)
    student_tokenizer = FakeByteLevelTokenizer(
        {
            1: b"<think>",
            2: _TURN_1_REASONING.encode("utf-8"),
            3: b"</think>",
            4: b"action",
            5: b" one",
            6: b"P",
        }
    )
    datum = TrainDatum(
        trial_name="a",
        fragment_index=1,
        span_indices=[1],
        model_input_tokens=[6, 1, 2, 3, 4, 5],
        loss_mask=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        sampled_logprobs=[0.0, -0.5, -0.5, -0.5, -0.5, -0.5],
    )
    teacher = _Teacher()

    kept, plans, rows, stats = _chunk_scored_datums(
        [datum],
        [_record("a", turns=2)],
        cast("ChatRendering", object()),
        student_tokenizer,
        teacher_tokenizer,
        cast("PromptLogprobClient", teacher),
    )

    assert stats.scored == 1
    assert kept == [datum]
    scored_text = teacher_tokenizer.decode(teacher.calls[0])
    # Turn 0's reasoning is not in the conversation the teacher conditions on;
    # turn 1's is, because that is what is being scored.
    assert "reasoning for turn zero" not in scored_text
    assert _TURN_1_REASONING in scored_text
    assert "action zero" in scored_text, "the earlier turn's action IS still history"
    assert len(rows[0]) == len(teacher.calls[0])
    covered = sorted(
        {p for chunk in plans[0].chunks for p in range(chunk.student_start, chunk.student_end)}
    )
    # Position 2 is the reasoning token, 4 and 5 the action; 1 and 3 are the
    # think framing, which has no byte-identical counterpart under the teacher's
    # template and stays unscored.
    assert covered == [2, 4, 5]
    assert all(datum.loss_mask[position] == 1.0 for position in covered)
