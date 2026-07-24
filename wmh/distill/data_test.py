"""Tests for the prefix-merge datum builder, advantage math, and tinker conversion."""

from __future__ import annotations

import logging
import sys

import pytest

from wmh.distill.config import (
    DistillConfig,
    HarborConfig,
    RolloutConfig,
    StudentConfig,
    TeacherConfig,
    TrainConfig,
)
from wmh.distill.data import (
    CONTEXT_OVERFLOW_STOP_REASON,
    TrainDatum,
    attach_advantages,
    build_datums,
    to_tinker_datums,
)
from wmh.distill.fake_tinker import FakeDatum, FakeServiceClient, FakeTrainingClient
from wmh.distill.tokens import TrialRecord
from wmh.providers.tinker import TokenSpan


def _cfg(
    *,
    max_datum_tokens: int = 65536,
    advantage_clip: float = 4.0,
    center_advantages: bool = True,
    context_budget_tokens: int = 65536,
) -> DistillConfig:
    return DistillConfig(
        student=StudentConfig(base_model="Qwen/Qwen3-8B"),
        teacher=TeacherConfig(model="Qwen/Qwen3-32B"),
        harbor=HarborConfig(job_template="job.yaml"),
        rollout=RolloutConfig(context_budget_tokens=context_budget_tokens),
        train=TrainConfig(
            max_datum_tokens=max_datum_tokens,
            advantage_clip=advantage_clip,
            center_advantages=center_advantages,
        ),
    )


def _span(call_index: int, prompt: list[int], sampled: list[int]) -> TokenSpan:
    return TokenSpan(
        call_index=call_index,
        prompt_token_ids=prompt,
        sampled_token_ids=sampled,
        sampled_logprobs=[-0.5 - 0.125 * j for j in range(len(sampled))],
    )


def _record(
    spans: list[TokenSpan],
    *,
    trial_name: str = "task-a__x1",
    stop_reason: str | None = "submitted",
) -> TrialRecord:
    return TrialRecord(
        task_id="task-a",
        attempt=1,
        trial_name=trial_name,
        reward=1.0,
        passed=True,
        spans=spans,
        stop_reason=stop_reason,
        artifact_dir=f"/tmp/jobs/{trial_name}",
    )


def _append_only_spans() -> list[TokenSpan]:
    """A 3-turn episode where every prompt extends prior prompt+sampled exactly."""
    prompt1 = [10, 11, 12]
    sampled1 = [100, 101]
    prompt2 = prompt1 + sampled1 + [13]
    sampled2 = [102]
    prompt3 = prompt2 + sampled2 + [14, 15]
    sampled3 = [103, 104, 105]
    return [
        _span(0, prompt1, sampled1),
        _span(1, prompt2, sampled2),
        _span(2, prompt3, sampled3),
    ]


def test_append_only_episode_merges_into_one_datum_with_exact_masks() -> None:
    spans = _append_only_spans()
    datums, stats = build_datums([_record(spans)], _cfg())

    assert len(datums) == 1
    datum = datums[0]
    assert datum.trial_name == "task-a__x1"
    assert datum.fragment_index == 0
    assert datum.model_input_tokens == [10, 11, 12, 100, 101, 13, 102, 14, 15, 103, 104, 105]
    assert datum.loss_mask == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert datum.sampled_logprobs == [
        0.0,
        0.0,
        0.0,
        *spans[0].sampled_logprobs,
        0.0,
        *spans[1].sampled_logprobs,
        0.0,
        0.0,
        *spans[2].sampled_logprobs,
    ]
    assert datum.advantages == []
    assert datum.sampled_token_ids() == [100, 101, 102, 103, 104, 105]
    assert stats.datums == 1
    assert stats.fragments == 0
    assert stats.fragmentation_rate == 0.0
    assert stats.overflow_drops == 0
    assert stats.overlong_drops == 0
    assert stats.loss_tokens == 6
    assert stats.context_tokens == 6


def test_spans_are_sorted_by_call_index_before_merging() -> None:
    spans = _append_only_spans()
    shuffled = [spans[2], spans[0], spans[1]]
    datums, stats = build_datums([_record(shuffled)], _cfg())
    assert stats.datums == 1
    assert datums[0].model_input_tokens[-3:] == [103, 104, 105]


def test_edited_history_fragments_into_separate_datums() -> None:
    prompt1 = [10, 11]
    sampled1 = [100, 101]
    edited_prompt = [99, 98, 97]  # not an extension of prompt1 + sampled1
    sampled2 = [102, 103]
    spans = [_span(0, prompt1, sampled1), _span(1, edited_prompt, sampled2)]

    datums, stats = build_datums([_record(spans)], _cfg())

    assert len(datums) == 2
    assert [datum.fragment_index for datum in datums] == [0, 1]
    assert datums[0].model_input_tokens == [10, 11, 100, 101]
    assert datums[0].loss_mask == [0.0, 0.0, 1.0, 1.0]
    assert datums[1].model_input_tokens == [99, 98, 97, 102, 103]
    assert datums[1].loss_mask == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert stats.fragments == 1
    assert stats.fragmentation_rate == 0.5


def test_interleaved_prefix_and_break_spans() -> None:
    """Merge, then break, then merge again: two datums, first with two loss runs."""
    prompt1 = [10, 11]
    sampled1 = [100]
    prompt2 = prompt1 + sampled1 + [12]  # prefix extension: merges
    sampled2 = [101, 102]
    prompt3 = [50, 51]  # break: new fragment
    sampled3 = [103]
    prompt4 = prompt3 + sampled3 + [52]  # extends the new fragment: merges
    sampled4 = [104]
    spans = [
        _span(0, prompt1, sampled1),
        _span(1, prompt2, sampled2),
        _span(2, prompt3, sampled3),
        _span(3, prompt4, sampled4),
    ]

    datums, stats = build_datums([_record(spans)], _cfg())

    assert len(datums) == 2
    assert datums[0].model_input_tokens == [10, 11, 100, 12, 101, 102]
    assert datums[0].loss_mask == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    assert datums[1].model_input_tokens == [50, 51, 103, 52, 104]
    assert datums[1].loss_mask == [0.0, 0.0, 1.0, 0.0, 1.0]
    assert stats.datums == 2
    assert stats.fragments == 1
    assert stats.loss_tokens == 5
    assert stats.context_tokens == 6


def test_context_overflow_trials_are_dropped_and_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    overflowed = _record(
        _append_only_spans(),
        trial_name="task-a__of",
        stop_reason=CONTEXT_OVERFLOW_STOP_REASON,
    )
    kept = _record(_append_only_spans(), trial_name="task-a__ok")
    with caplog.at_level(logging.WARNING, logger="wmh.distill.data"):
        datums, stats = build_datums([overflowed, kept], _cfg())
    assert stats.overflow_drops == 1
    assert stats.datums == 1
    assert datums[0].trial_name == "task-a__ok"
    assert "task-a__of" in caplog.text


def test_other_stop_reasons_are_not_overflow_drops() -> None:
    for stop_reason in ("submitted", "max_turns", None):
        _, stats = build_datums([_record(_append_only_spans(), stop_reason=stop_reason)], _cfg())
        assert stats.overflow_drops == 0
        assert stats.datums == 1


def test_measured_context_overflow_drops_the_episode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The budget is enforced from the recorded spans, not just a trace marker:
    an episode whose largest call exceeds rollout.context_budget_tokens is
    dropped whole even when its stop reason is ordinary."""
    big_prompt = list(range(1500))
    overflowed = _record(
        [_span(0, big_prompt, [100, 101])],
        trial_name="task-a__big",
        stop_reason="max_turns",
    )
    kept = _record(_append_only_spans(), trial_name="task-a__ok")
    with caplog.at_level(logging.WARNING, logger="wmh.distill.data"):
        datums, stats = build_datums([overflowed, kept], _cfg(context_budget_tokens=1024))
    assert stats.overflow_drops == 1
    assert stats.overlong_drops == 0
    assert stats.datums == 1
    assert datums[0].trial_name == "task-a__ok"
    assert "task-a__big" in caplog.text
    assert "context_budget_tokens" in caplog.text

    # Exactly at the budget is kept (1022 prompt + 2 sampled = 1024).
    at_cap = _record([_span(0, list(range(1022)), [100, 101])], trial_name="task-a__cap")
    _, cap_stats = build_datums([at_cap], _cfg(context_budget_tokens=1024))
    assert cap_stats.overflow_drops == 0
    assert cap_stats.datums == 1


def test_overlong_episode_is_dropped_whole_never_truncated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    spans = _append_only_spans()  # merges to 12 tokens
    with caplog.at_level(logging.WARNING, logger="wmh.distill.data"):
        datums, stats = build_datums([_record(spans)], _cfg(max_datum_tokens=11))
    assert datums == []
    assert stats.overlong_drops == 1
    assert stats.loss_tokens == 0
    assert "never" in caplog.text and "truncated" in caplog.text

    # Exactly at the cap is kept.
    _, kept_stats = build_datums([_record(spans)], _cfg(max_datum_tokens=12))
    assert kept_stats.datums == 1
    assert kept_stats.overlong_drops == 0


def test_one_overlong_fragment_drops_the_whole_fragmented_episode() -> None:
    short = _span(0, [10, 11], [100])  # 3 tokens
    long_break = _span(1, list(range(20, 30)), [101, 102])  # 12 tokens, non-prefix
    _, stats = build_datums([_record([short, long_break])], _cfg(max_datum_tokens=8))
    assert stats.datums == 0
    assert stats.overlong_drops == 1
    assert stats.fragments == 0


def test_spanless_trials_contribute_nothing_and_are_not_drops() -> None:
    datums, stats = build_datums([_record([])], _cfg())
    assert datums == []
    assert stats.datums == 0
    assert stats.overflow_drops == 0
    assert stats.overlong_drops == 0


def test_train_datum_rejects_misaligned_lists() -> None:
    with pytest.raises(ValueError, match="must both match"):
        TrainDatum(
            trial_name="t",
            fragment_index=0,
            model_input_tokens=[1, 2],
            loss_mask=[0.0],
            sampled_logprobs=[0.0, -1.0],
        )
    with pytest.raises(ValueError, match="advantages length"):
        TrainDatum(
            trial_name="t",
            fragment_index=0,
            model_input_tokens=[1, 2],
            loss_mask=[0.0, 1.0],
            sampled_logprobs=[0.0, -1.0],
            advantages=[0.0],
        )


def _one_datum(sampled_logprobs: list[float] | None = None) -> TrainDatum:
    logprobs = sampled_logprobs if sampled_logprobs is not None else [-1.0, -2.0, -3.0]
    return TrainDatum(
        trial_name="task-a__x1",
        fragment_index=0,
        model_input_tokens=[10, 11, 100, 101, 102],
        loss_mask=[0.0, 0.0, 1.0, 1.0, 1.0],
        sampled_logprobs=[0.0, 0.0, *logprobs],
    )


def test_attach_advantages_clips_both_directions() -> None:
    datum = _one_datum(sampled_logprobs=[-1.0, -2.0, -3.0])
    # teacher - sampled per loss position: +9.5 (clips to +2), -0.5, -8.0 (clips to -2)
    teacher: list[float | None] = [None, -0.1, 8.5, -2.5, -11.0]
    attached, stats = attach_advantages(
        [datum], [teacher], _cfg(advantage_clip=2.0, center_advantages=False)
    )
    assert stats.datums == 1
    assert stats.mismatch_drops == 0
    assert stats.clipped_tokens == 2
    assert attached[0].advantages == [0.0, 0.0, 2.0, -0.5, -2.0]
    # The input datum is untouched; attachment returns new datums.
    assert datum.advantages == []


def test_attach_advantages_batch_mean_centering() -> None:
    datum = _one_datum(sampled_logprobs=[-1.0, -1.0, -1.0])
    teacher: list[float | None] = [None, -0.1, -2.0, -1.0, 0.5]  # raw: -1.0, 0.0, +1.5
    attached, _ = attach_advantages([datum], [teacher], _cfg(center_advantages=True))
    advantages = attached[0].advantages
    loss_values = [advantages[2], advantages[3], advantages[4]]
    assert sum(loss_values) == pytest.approx(0.0)
    # raw: -1.0, 0.0, +1.5; batch mean 1/6 subtracted from every loss token.
    assert loss_values == pytest.approx([-7 / 6, -1 / 6, 4 / 3])
    # Context positions stay exactly 0 even after centering.
    assert advantages[0] == 0.0 and advantages[1] == 0.0


def test_attach_advantages_length_mismatch_drops_loudly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    good = _one_datum()
    bad = _one_datum()
    full: list[float | None] = [None, -0.1, -1.0, -2.0, -3.0]
    short: list[float | None] = [None, -0.1, -1.0]
    with caplog.at_level(logging.WARNING, logger="wmh.distill.data"):
        attached, stats = attach_advantages(
            [bad, good], [short, full], _cfg(center_advantages=False)
        )
    assert stats.mismatch_drops == 1
    assert stats.datums == 1
    assert len(attached) == 1
    assert "task-a__x1" in caplog.text
    assert "3 logprob(s) for 5 token(s)" in caplog.text


def test_attach_advantages_none_at_loss_position_drops_loudly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    datum = _one_datum()
    teacher: list[float | None] = [None, -0.1, -1.0, None, -3.0]
    with caplog.at_level(logging.WARNING, logger="wmh.distill.data"):
        attached, stats = attach_advantages([datum], [teacher], _cfg())
    assert attached == []
    assert stats.mismatch_drops == 1
    assert "loss position 3" in caplog.text


def test_attach_advantages_rejects_wrong_batch_shape() -> None:
    with pytest.raises(ValueError, match="one per-position list per datum"):
        attach_advantages([_one_datum()], [], _cfg())


def test_to_tinker_datums_requires_attached_advantages() -> None:
    pytest.importorskip("tinker")
    with pytest.raises(ValueError, match="attach_advantages"):
        to_tinker_datums([_one_datum()])


def test_to_tinker_datums_import_error_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tinker", None)
    with pytest.raises(ImportError, match="uv sync --extra distill"):
        to_tinker_datums([])


def _sampled_episode() -> tuple[FakeTrainingClient, list[TokenSpan]]:
    """Build a real append-only episode by sampling through the linked fake sampler.

    Returns the training client whose ledger holds the issued spans (so its
    forward_backward TITO check has the ground truth) plus the recorded spans.
    """
    training = FakeServiceClient().create_lora_training_client("base-model", rank=8)
    sampler = training.save_weights_and_get_sampling_client("s")
    tokenizer = training.get_tokenizer()
    prompt1 = tokenizer.encode("user: run ls\nassistant:")
    seq1 = sampler.sample(prompt1, max_tokens=4, temperature=0.7)
    prompt2 = prompt1 + seq1.tokens + tokenizer.encode("\ntool: ok\nassistant:")
    seq2 = sampler.sample(prompt2, max_tokens=3, temperature=0.7)
    spans = [
        TokenSpan(
            call_index=0,
            prompt_token_ids=prompt1,
            sampled_token_ids=seq1.tokens,
            sampled_logprobs=seq1.logprobs,
        ),
        TokenSpan(
            call_index=1,
            prompt_token_ids=prompt2,
            sampled_token_ids=seq2.tokens,
            sampled_logprobs=seq2.logprobs,
        ),
    ]
    return training, spans


def test_to_tinker_datums_tito_round_trip_through_the_fake_training_client() -> None:
    """The converted shift layout must carry the sampler's exact issued tokens."""
    pytest.importorskip("tinker")
    training, spans = _sampled_episode()

    datums, stats = build_datums([_record(spans)], _cfg())
    assert stats.datums == 1
    tokens = datums[0].model_input_tokens
    teacher: list[float | None] = [None, *(-0.25 for _ in tokens[1:])]
    attached, _ = attach_advantages([datums[0]], [teacher], _cfg(center_advantages=False))

    tinker_datums = to_tinker_datums(attached)
    assert len(tinker_datums) == 1
    td = tinker_datums[0]

    # Shifted layout: input drops the last token, targets drop the first.
    assert td.model_input.to_ints() == tokens[:-1]
    # Exactly the keys the live importance_sampling loss accepts: a "mask" key is
    # rejected server-side (observed 2026-07-23), so masking rides the advantages
    # (0.0 at context positions). This keyset assertion is the offline guard for
    # that wire contract.
    assert set(td.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
    target_tokens = [int(t) for t in td.loss_fn_inputs["target_tokens"].data]
    logprobs = [float(v) for v in td.loss_fn_inputs["logprobs"].data]
    advantages = [float(v) for v in td.loss_fn_inputs["advantages"].data]
    mask = attached[0].loss_mask[1:]
    assert target_tokens == tokens[1:]
    # Masking-through-advantages invariant: every context (mask 0.0) position
    # carries advantage exactly 0.0, so dropping the mask key is loss-equivalent.
    assert all(adv == 0.0 for adv, m in zip(advantages, mask, strict=True) if m == 0.0)
    # float32 wire precision: values round-trip to ~7 significant digits.
    assert logprobs == pytest.approx(attached[0].sampled_logprobs[1:], rel=1e-6)
    assert advantages == pytest.approx(attached[0].advantages[1:], rel=1e-6, abs=1e-6)
    assert td.loss_fn_inputs["target_tokens"].dtype == "int64"
    assert td.loss_fn_inputs["advantages"].dtype == "float32"

    # Round trip into the fake training client: every nonzero-mask target run must
    # be a span the linked sampler actually issued (the TITO invariant).
    fake = FakeDatum(
        model_input_tokens=td.model_input.to_ints(),
        target_tokens=target_tokens,
        weights=mask,
        advantages=advantages,
        logprobs=logprobs,
    )
    training.forward_backward([fake], "importance_sampling")
    assert len(training.forward_backward_calls) == 1

    # Negative control: a single corrupted sampled token must trip the assertion.
    first_loss = mask.index(1.0)
    corrupted_targets = list(target_tokens)
    corrupted_targets[first_loss] += 1
    corrupted = FakeDatum(
        model_input_tokens=td.model_input.to_ints(),
        target_tokens=corrupted_targets,
        weights=mask,
        advantages=advantages,
        logprobs=logprobs,
    )
    with pytest.raises(AssertionError, match="TITO violation"):
        training.forward_backward([corrupted], "importance_sampling")
