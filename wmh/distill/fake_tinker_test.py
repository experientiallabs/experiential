"""Tests for the deterministic fake Tinker clients: determinism, TITO, echoes."""

import pytest

from wmh.distill.fake_tinker import (
    FakeDatum,
    FakeSamplingClient,
    FakeServiceClient,
    FakeTokenizer,
    FakeTrainingClient,
)


def _make_training() -> FakeTrainingClient:
    return FakeServiceClient().create_lora_training_client("base-model", rank=16)


def test_tokenizer_round_trip() -> None:
    tok = FakeTokenizer()
    text = "ls -la /tmp && echo done"
    ids = tok.encode(text)
    assert all(isinstance(t, int) for t in ids)
    assert tok.decode(ids) == text
    assert tok.encode(text) == ids


def test_sample_is_deterministic_across_clients() -> None:
    a = FakeSamplingClient(seed="tinker://fake/sampler/s/0")
    b = FakeSamplingClient(seed="tinker://fake/sampler/s/0")
    prompt = [1, 2, 3]
    seq_a = a.sample(prompt, max_tokens=16, temperature=0.7)
    seq_b = b.sample(prompt, max_tokens=16, temperature=0.7)
    assert seq_a.tokens == seq_b.tokens
    assert seq_a.logprobs == seq_b.logprobs
    assert seq_a.stop_reason == "length"
    assert len(seq_a.tokens) == 16
    assert len(seq_a.logprobs) == 16
    assert all(lp < 0 for lp in seq_a.logprobs)


def test_sample_repeated_call_identical() -> None:
    client = FakeSamplingClient(seed="s")
    first = client.sample([5, 6], max_tokens=8, temperature=0.0)
    second = client.sample([5, 6], max_tokens=8, temperature=0.0)
    assert first == second


def test_sample_varies_with_seed_prompt_and_index() -> None:
    prompt = [1, 2, 3]
    base = FakeSamplingClient(seed="s").sample(prompt, max_tokens=12, temperature=0.7)
    other_seed = FakeSamplingClient(seed="s2").sample(prompt, max_tokens=12, temperature=0.7)
    other_prompt = FakeSamplingClient(seed="s").sample([9, 9], max_tokens=12, temperature=0.7)
    other_index = FakeSamplingClient(seed="s").sample(
        prompt, max_tokens=12, temperature=0.7, sample_index=1
    )
    assert base.tokens != other_seed.tokens
    assert base.tokens != other_prompt.tokens
    assert base.tokens != other_index.tokens


def test_sample_prefix_property_and_stop() -> None:
    client = FakeSamplingClient(seed="s")
    long = client.sample([1], max_tokens=20, temperature=0.7)
    short = client.sample([1], max_tokens=5, temperature=0.7)
    assert long.tokens[:5] == short.tokens
    stop_token = long.tokens[3]
    first_hit = long.tokens.index(stop_token)
    stopped = client.sample([1], max_tokens=20, temperature=0.7, stop=[stop_token])
    assert stopped.stop_reason == "stop"
    assert stopped.tokens == long.tokens[: first_hit + 1]
    assert stopped.tokens[-1] == stop_token


def test_sample_string_stop_matches_decoded_suffix() -> None:
    # String stops mirror the real SamplingParams contract: generation ends
    # when the decoded output so far ends with the stop string.
    reference = FakeSamplingClient(seed="s").sample([1, 2], max_tokens=12, temperature=0.7)
    decoded = FakeTokenizer().decode(reference.tokens)
    stop = decoded[3:5]
    end = decoded.index(stop) + len(stop)
    stopped = FakeSamplingClient(seed="s").sample(
        [1, 2], max_tokens=12, temperature=0.7, stop=[stop]
    )
    assert stopped.stop_reason == "stop"
    assert stopped.tokens == reference.tokens[:end]


def test_sample_records_issued_spans() -> None:
    client = FakeSamplingClient(seed="s")
    seq = client.sample([7, 8], max_tokens=4, temperature=0.7)
    assert len(client.issued) == 1
    record = client.issued[0]
    assert record.prompt_ids == (7, 8)
    assert record.sampled_ids == tuple(seq.tokens)
    assert record.logprobs == tuple(seq.logprobs)


def test_compute_logprobs_echoes_issued_spans() -> None:
    client = FakeSamplingClient(seed="s")
    prompt = [10, 11, 12]
    seq = client.sample(prompt, max_tokens=6, temperature=0.7)
    full = prompt + seq.tokens
    logprobs = client.compute_logprobs(full)
    assert len(logprobs) == len(full)
    assert logprobs[0] is None
    assert logprobs[len(prompt) :] == seq.logprobs


def test_compute_logprobs_multi_turn_episode() -> None:
    client = FakeSamplingClient(seed="s")
    prompt1 = [1, 2, 3]
    seq1 = client.sample(prompt1, max_tokens=4, temperature=0.7)
    tool_tokens = [50, 51]
    prompt2 = prompt1 + seq1.tokens + tool_tokens
    seq2 = client.sample(prompt2, max_tokens=3, temperature=0.7)
    episode = prompt2 + seq2.tokens
    logprobs = client.compute_logprobs(episode)
    start1 = len(prompt1)
    assert logprobs[start1 : start1 + 4] == seq1.logprobs
    start2 = len(prompt2)
    assert logprobs[start2 : start2 + 3] == seq2.logprobs
    for position in range(start1 + 4, start2):
        assert logprobs[position] is not None


def test_compute_logprobs_unissued_positions_deterministic() -> None:
    a = FakeSamplingClient(seed="s")
    b = FakeSamplingClient(seed="s")
    ids = [1, 2, 3, 4]
    assert a.compute_logprobs(ids) == b.compute_logprobs(ids)
    assert a.compute_logprobs(ids)[0] is None
    assert all(lp is not None and lp < 0 for lp in a.compute_logprobs(ids)[1:])
    other_seed = FakeSamplingClient(seed="s2").compute_logprobs(ids)
    assert other_seed != a.compute_logprobs(ids)


def test_compute_logprobs_requires_prompt_context() -> None:
    # The issued span embedded WITHOUT its prompt directly before it is not
    # recognized: those positions get hash-derived values, not the echo.
    client = FakeSamplingClient(seed="s")
    seq = client.sample([1, 2, 3], max_tokens=4, temperature=0.7)
    detached = [99, 98] + seq.tokens
    logprobs = client.compute_logprobs(detached)
    assert logprobs[2:] != seq.logprobs


def test_tito_passes_on_faithful_datums() -> None:
    training = _make_training()
    sampler = training.save_weights_and_get_sampling_client("s0")
    prompt = [1, 2, 3]
    seq = sampler.sample(prompt, max_tokens=5, temperature=0.7)
    datum = FakeDatum(
        model_input_tokens=prompt + seq.tokens,
        target_tokens=prompt[1:] + seq.tokens,
        weights=[0.0] * (len(prompt) - 1) + [1.0] * len(seq.tokens),
        advantages=[0.0] * (len(prompt) - 1) + [0.5] * len(seq.tokens),
        logprobs=[0.0] * (len(prompt) - 1) + list(seq.logprobs),
    )
    training.forward_backward([datum], "importance_sampling")
    assert len(training.forward_backward_calls) == 1
    recorded_datums, recorded_loss = training.forward_backward_calls[0]
    assert recorded_loss == "importance_sampling"
    assert recorded_datums[0] is datum


def test_tito_fires_on_corrupted_datum() -> None:
    training = _make_training()
    sampler = training.save_weights_and_get_sampling_client("s0")
    seq = sampler.sample([1, 2], max_tokens=5, temperature=0.7)
    corrupted = list(seq.tokens)
    corrupted[2] = corrupted[2] + 1
    datum = FakeDatum(
        model_input_tokens=[1, 2] + corrupted,
        target_tokens=corrupted,
        weights=[1.0] * len(corrupted),
    )
    with pytest.raises(AssertionError, match=r"TITO violation in datum 0.*position 2"):
        training.forward_backward([datum], "importance_sampling")
    assert training.forward_backward_calls == []


def test_tito_fires_when_nothing_issued() -> None:
    training = _make_training()
    datum = FakeDatum(model_input_tokens=[1, 2], target_tokens=[3, 4])
    with pytest.raises(AssertionError, match="no eligible sampler issued any span"):
        training.forward_backward([datum], "importance_sampling")


def test_tito_ignores_zero_weight_positions() -> None:
    training = _make_training()
    sampler = training.save_weights_and_get_sampling_client("s0")
    seq = sampler.sample([9], max_tokens=3, temperature=0.7)
    # Prompt/tool tokens carry weight 0 and never need to have been issued.
    datum = FakeDatum(
        model_input_tokens=[9, 8, 7] + seq.tokens,
        target_tokens=[8, 7] + seq.tokens,
        weights=[0.0, 0.0] + [1.0] * len(seq.tokens),
    )
    training.forward_backward([datum], "importance_sampling")


def test_datum_rejects_misaligned_weights() -> None:
    with pytest.raises(ValueError, match="weights length"):
        FakeDatum(model_input_tokens=[1], target_tokens=[1, 2], weights=[1.0])


def test_optim_step_counts_and_records_lr() -> None:
    training = _make_training()
    training.optim_step(1e-4)
    training.optim_step(5e-5)
    assert training.step_count == 2
    assert training.optim_step_lrs == [1e-4, 5e-5]


def test_forward_backward_reports_a_deterministic_batch_loss() -> None:
    """The SDK-shaped output carries a stable "total_loss:sum" per batch."""
    training = _make_training()
    sampler = training.save_weights_and_get_sampling_client("s0")
    seq = sampler.sample([1, 2, 3], max_tokens=4, temperature=0.7)
    datum = FakeDatum(
        model_input_tokens=[1, 2, 3] + seq.tokens[:-1],
        target_tokens=[2, 3] + seq.tokens,
        weights=[0.0, 0.0] + [1.0] * len(seq.tokens),
    )
    first = training.forward_backward([datum], "importance_sampling")
    second = training.forward_backward([datum], "importance_sampling")
    assert first.loss_fn_output_type == "FakeLossReturn"
    assert len(first.loss_fn_outputs) == 1  # one entry per datum, like the SDK
    loss = first.metrics["total_loss:sum"]
    assert loss > 0.0
    assert second.metrics == first.metrics  # pure function of (loss_fn, batch)
    # A different loss_fn or batch reports a different (still deterministic) loss.
    other_loss = training.forward_backward([datum], "cross_entropy")
    assert other_loss.metrics["total_loss:sum"] != loss


def test_optim_step_reports_a_deterministic_grad_norm() -> None:
    """The SDK-shaped response carries a stable "grad_norm:mean" per step."""
    a = _make_training()
    b = _make_training()
    first_a = a.optim_step(1e-4)
    first_b = b.optim_step(1e-4)
    assert first_a.metrics is not None and first_b.metrics is not None
    assert first_a.metrics["grad_norm:mean"] > 0.0
    assert first_a.metrics == first_b.metrics  # pure function of (step count, lr)
    second_a = a.optim_step(1e-4)
    assert second_a.metrics != first_a.metrics  # the step count advanced


def test_save_and_load_state_across_training_clients() -> None:
    """States live on the service, so a later client restores them.

    This is the resume shape: the session that saved the checkpoint is gone
    and a freshly created (uninitialized) training client loads it, which is
    the only ordering the live service accepts.
    """
    service = FakeServiceClient()
    first = service.create_lora_training_client("base-model", rank=16)
    first.optim_step(1e-4)
    path = first.save_state()
    assert path == "tinker://fake/state/0"
    first.optim_step(1e-4)
    assert first.step_count == 2

    second = service.create_lora_training_client("base-model", rank=16)
    second.load_state(path)
    assert second.step_count == 1
    assert second.save_state() == "tinker://fake/state/1"  # the service-wide counter

    third = service.create_lora_training_client("base-model", rank=16)
    with pytest.raises(ValueError, match="never returned by save_state"):
        third.load_state("tinker://fake/state/999")
    assert third.calls == []  # a rejected restore left the client uninitialized


def test_load_state_after_a_weights_call_is_refused() -> None:
    """The live service's rule, mirrored: LoadWeights needs an uninitialized model.

    A resumed distill run that let anything touch the training client before
    restoring the checkpoint died on this exact error, so the fake fails the
    same way instead of silently accepting the restore.
    """
    service = FakeServiceClient()
    training = service.create_lora_training_client("base-model", rank=16)
    path = training.save_state()
    training.save_weights_for_sampler("s0")

    with pytest.raises(RuntimeError, match="LoadWeights can only be called on uninitialized"):
        training.load_state(path)

    # A second restore is refused for the same reason (the live failure was
    # precisely a retried load_state on one client).
    fresh = service.create_lora_training_client("base-model", rank=16)
    fresh.load_state(path)
    with pytest.raises(RuntimeError, match=r"after \['load_state'\]"):
        fresh.load_state(path)


def test_the_call_log_records_order_and_excludes_get_tokenizer() -> None:
    """`calls` is the ordered log tests assert against.

    get_tokenizer is logged but is NOT model-initializing (the SDK answers it
    from a metadata-only GetInfo), so it may precede a restore.
    """
    service = FakeServiceClient()
    training = service.create_lora_training_client("base-model", rank=16)
    path = training.save_state()

    restored = service.create_lora_training_client("base-model", rank=16)
    restored.get_tokenizer()
    restored.load_state(path)
    sampler = restored.save_weights_and_get_sampling_client("s0")
    seq = sampler.sample([1, 2], max_tokens=3, temperature=0.7)
    restored.forward_backward(
        [
            FakeDatum(
                model_input_tokens=[1, 2] + seq.tokens,
                target_tokens=list(seq.tokens),
                weights=[1.0] * len(seq.tokens),
            )
        ],
        "importance_sampling",
    )
    restored.optim_step(1e-4)

    assert restored.calls == [
        "get_tokenizer",
        "load_state",
        "save_weights_for_sampler",
        "forward_backward",
        "optim_step",
    ]


def test_sampler_refresh_distinct_but_linked() -> None:
    training = _make_training()
    first = training.save_weights_and_get_sampling_client("ck")
    second = training.save_weights_and_get_sampling_client("ck")
    assert first.seed != second.seed
    prompt = [1, 2]
    seq_first = first.sample(prompt, max_tokens=4, temperature=0.7)
    seq_second = second.sample(prompt, max_tokens=4, temperature=0.7)
    assert seq_first.tokens != seq_second.tokens
    # Spans issued by BOTH samplers satisfy TITO on the shared training client.
    for seq in (seq_first, seq_second):
        datum = FakeDatum(
            model_input_tokens=prompt + seq.tokens,
            target_tokens=list(seq.tokens),
            weights=[1.0] * len(seq.tokens),
        )
        training.forward_backward([datum], "importance_sampling")
    assert len(training.forward_backward_calls) == 2


def test_create_sampling_client_from_saved_path_is_linked() -> None:
    service = FakeServiceClient()
    training = service.create_lora_training_client("base", rank=8)
    path = training.save_weights_for_sampler("ck")
    assert path.startswith("tinker://fake/sampler/ck/")
    sampler = service.create_sampling_client(path)
    seq = sampler.sample([4, 5], max_tokens=3, temperature=0.7)
    datum = FakeDatum(
        model_input_tokens=[4, 5] + seq.tokens,
        target_tokens=list(seq.tokens),
        weights=[1.0] * len(seq.tokens),
    )
    training.forward_backward([datum], "importance_sampling")
    # A linked sampler sees peers' issued spans through the shared ledger.
    peer = service.create_sampling_client(path)
    echoed = peer.compute_logprobs([4, 5] + seq.tokens)
    assert echoed[2:] == seq.logprobs


def test_create_sampling_client_unknown_path_is_unlinked_but_tracked() -> None:
    """An unknown-path sampler (the teacher) is unlinked yet a TITO issuer.

    This replaces the pre-warmup expectation that teacher spans fail TITO:
    the warmup phase deliberately trains on TEACHER-sampled tokens, and the
    real service serves teacher and student through the same account, so a
    span a service-created client genuinely issued is legitimate training
    data. Unlinked still means its spans never echo through the student's
    shared ledger.
    """
    service = FakeServiceClient()
    teacher = service.create_sampling_client("teacher-base-model")
    seq = teacher.sample([1], max_tokens=2, temperature=0.0)
    assert len(seq.tokens) == 2
    # Unlinked: the student's samplers cannot echo the teacher's issued logprobs.
    training = service.create_lora_training_client("base")
    student = training.save_weights_and_get_sampling_client("s0")
    echoed = student.compute_logprobs([1] + seq.tokens)
    assert echoed[1:] != seq.logprobs
    # Tracked issuer: teacher-sampled spans satisfy the TITO check (warmup SFT).
    datum = FakeDatum(
        model_input_tokens=[1] + seq.tokens,
        target_tokens=list(seq.tokens),
        weights=[1.0] * len(seq.tokens),
    )
    training.forward_backward([datum], "cross_entropy")
    assert training.forward_backward_calls[-1][1] == "cross_entropy"


def test_corrupted_teacher_span_still_fails_tito() -> None:
    """The issuer extension does not weaken the assertion: fabricated ids die."""
    service = FakeServiceClient()
    teacher = service.create_sampling_client("teacher-base-model")
    seq = teacher.sample([1, 2], max_tokens=4, temperature=0.0)
    corrupted = list(seq.tokens)
    corrupted[1] = corrupted[1] + 1
    training = service.create_lora_training_client("base")
    datum = FakeDatum(
        model_input_tokens=[1, 2] + corrupted,
        target_tokens=corrupted,
        weights=[1.0] * len(corrupted),
    )
    with pytest.raises(AssertionError, match=r"TITO violation in datum 0.*position 1"):
        training.forward_backward([datum], "cross_entropy")
    assert training.forward_backward_calls == []


def test_directly_constructed_sampler_is_not_an_issuer() -> None:
    """Only service-created clients count; a bare FakeSamplingClient does not."""
    service = FakeServiceClient()
    rogue = FakeSamplingClient(seed="rogue-sampler")
    seq = rogue.sample([1], max_tokens=3, temperature=0.0)
    training = service.create_lora_training_client("base")
    datum = FakeDatum(
        model_input_tokens=[1] + seq.tokens,
        target_tokens=list(seq.tokens),
        weights=[1.0] * len(seq.tokens),
    )
    with pytest.raises(AssertionError, match="TITO violation"):
        training.forward_backward([datum], "importance_sampling")


# --- prefill top-k echo -------------------------------------------------------


def test_topk_prompt_logprobs_echo_and_determinism() -> None:
    """Top-1 is the sequence's own token with its realized logprob, ranks are
    strictly decreasing hash-derived values, and the whole result is a pure
    function of (seed, token_ids, k)."""
    client = FakeSamplingClient(seed="tinker://fake/sampler/teacher/0")
    prompt = [10, 11, 12]
    seq = client.sample(prompt, max_tokens=4, temperature=1.0)
    tokens = prompt + seq.tokens

    realized, rows = client.topk_prompt_logprobs(tokens, 4)
    realized_again, rows_again = client.topk_prompt_logprobs(tokens, 4)

    assert (realized, rows) == (realized_again, rows_again)
    # The realized logprobs ARE compute_logprobs: issued spans echo exactly.
    assert realized == client.compute_logprobs(tokens)
    assert realized[len(prompt) :] == list(seq.logprobs)
    assert rows[0] is None
    for position in range(1, len(tokens)):
        row = rows[position]
        assert row is not None
        assert len(row) == 4
        assert row[0] == (tokens[position], realized[position])
        logprobs = [lp for _, lp in row]
        assert logprobs == sorted(logprobs, reverse=True)
        assert len(set(logprobs)) == 4  # strictly decreasing, no ties
        assert len({token for token, _ in row}) == 4  # distinct candidate ids


def test_topk_prompt_logprobs_k1_is_the_pure_echo() -> None:
    client = FakeSamplingClient(seed="s")
    tokens = [1, 2, 3, 4]
    realized, rows = client.topk_prompt_logprobs(tokens, 1)
    assert rows[0] is None
    for position in range(1, len(tokens)):
        assert rows[position] == [(tokens[position], realized[position])]


def test_topk_prompt_logprobs_rejects_bad_k() -> None:
    client = FakeSamplingClient(seed="s")
    with pytest.raises(ValueError, match=">= 1"):
        client.topk_prompt_logprobs([1, 2], 0)
    with pytest.raises(ValueError, match="fake vocabulary"):
        client.topk_prompt_logprobs([1, 2], 96)


# --- topk-CE replicas: the input-side TITO variant ----------------------------


def _topk_replica_datum(
    prompt: list[int], sampled: list[int], candidate_targets: list[int]
) -> FakeDatum:
    """A shifted topk replica: candidate targets, fractional weights, flagged."""
    tokens = prompt + sampled
    # Candidate targets replace the sampled positions in target space; the
    # target index scoring the first sampled token is len(prompt) - 1.
    span_start = len(prompt) - 1
    next_tokens = tokens[1:]
    targets = next_tokens[:span_start] + candidate_targets[: len(next_tokens) - span_start]
    weights = [0.0] * span_start + [0.25] * (len(next_tokens) - span_start)
    return FakeDatum(
        model_input_tokens=tokens[:-1],
        target_tokens=targets,
        weights=weights,
        topk=True,
    )


def test_topk_datum_with_candidate_targets_passes_input_side_tito() -> None:
    """Teacher-proposed targets are fine; the INPUT is genuine sampled tokens.

    In topk-CE the targets are intentionally candidates no sampler issued
    (that is the loss), so the fake checks the model input under the
    loss-weighted positions against the ledger instead of the targets.
    """
    training = _make_training()
    sampler = training.save_weights_and_get_sampling_client("s0")
    prompt = [1, 2, 3]
    seq = sampler.sample(prompt, max_tokens=5, temperature=0.7)
    fabricated_candidates = [201, 202, 203, 204, 205]  # issued by nobody
    datum = _topk_replica_datum(prompt, list(seq.tokens), fabricated_candidates)

    training.forward_backward([datum], "cross_entropy")

    assert len(training.forward_backward_calls) == 1


def test_topk_datum_with_corrupted_input_fails_tito() -> None:
    training = _make_training()
    sampler = training.save_weights_and_get_sampling_client("s0")
    prompt = [1, 2, 3]
    seq = sampler.sample(prompt, max_tokens=5, temperature=0.7)
    corrupted = list(seq.tokens)
    corrupted[1] = corrupted[1] + 1  # the INPUT context is corrupted
    datum = _topk_replica_datum(prompt, corrupted, [201, 202, 203, 204, 205])

    with pytest.raises(AssertionError, match="TITO violation in topk datum 0"):
        training.forward_backward([datum], "cross_entropy")
    assert training.forward_backward_calls == []


def test_topk_datum_under_non_ce_loss_is_rejected() -> None:
    training = _make_training()
    sampler = training.save_weights_and_get_sampling_client("s0")
    prompt = [1, 2, 3]
    seq = sampler.sample(prompt, max_tokens=4, temperature=0.7)
    datum = _topk_replica_datum(prompt, list(seq.tokens), [201, 202, 203, 204])

    with pytest.raises(AssertionError, match="cross_entropy"):
        training.forward_backward([datum], "importance_sampling")
