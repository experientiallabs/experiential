"""Tests for teacher scoring: row alignment, error paths, fingerprints."""

import sys

import pytest

from wmh.distill.config import TeacherConfig
from wmh.distill.data import TrainDatum
from wmh.distill.fake_tinker import FakeSamplingClient, FakeTokenizer
from wmh.distill.teacher import (
    TOKENIZER_PROBE_TEXTS,
    TinkerTeacher,
    tokenizer_fingerprint_check,
)
from wmh.providers.base import ProviderKind


def _spec() -> TeacherConfig:
    return TeacherConfig(model="Qwen/Qwen3-235B-A22B-Instruct-2507")


def _datum(prompt: list[int], sampled: list[int]) -> TrainDatum:
    """One canonical (unshifted) datum with loss exactly on the sampled span."""
    return TrainDatum(
        trial_name="task-a__x1",
        fragment_index=0,
        model_input_tokens=[*prompt, *sampled],
        loss_mask=[0.0] * len(prompt) + [1.0] * len(sampled),
        sampled_logprobs=[0.0] * len(prompt) + [-0.5] * len(sampled),
    )


# --- Row alignment (off-by-one guard) ----------------------------------------


def test_score_rows_echo_issued_logprobs_exactly() -> None:
    """Pins the direct per-position alignment against the fake's echoes.

    The fake's compute_logprobs echoes the exact logprobs issued at sampling
    time at the sampled span's positions (index p = logprob of token p given
    tokens < p). If the teacher read a neighboring index (off by one in either
    direction), the returned values would be hash-derived context logprobs
    instead of the issued ones and this equality would fail.
    """
    client = FakeSamplingClient(seed="tinker://fake/sampler/teacher/0")
    prompt = [100, 101, 102, 103]
    sequence = client.sample(prompt, max_tokens=6, temperature=1.0)
    datum = _datum(prompt, sequence.tokens)
    teacher = TinkerTeacher(_spec(), sampling_client=client)

    [row] = teacher.score([datum])

    assert len(row) == len(datum.model_input_tokens)
    assert row[: len(prompt)] == [None] * len(prompt)  # context stays None
    assert row[len(prompt) :] == list(sequence.logprobs)


def test_score_preserves_datum_order_and_lengths() -> None:
    client = FakeSamplingClient(seed="s")
    prompt_a, prompt_b = [1, 2, 3], [7, 8]
    seq_a = client.sample(prompt_a, max_tokens=4, temperature=1.0)
    seq_b = client.sample(prompt_b, max_tokens=3, temperature=1.0)
    teacher = TinkerTeacher(_spec(), sampling_client=client)

    rows = teacher.score([_datum(prompt_a, seq_a.tokens), _datum(prompt_b, seq_b.tokens)])

    assert [row[len(prompt) :] for row, prompt in zip(rows, (prompt_a, prompt_b), strict=True)] == [
        list(seq_a.logprobs),
        list(seq_b.logprobs),
    ]


def test_score_empty_batch() -> None:
    teacher = TinkerTeacher(_spec(), sampling_client=FakeSamplingClient(seed="s"))
    assert teacher.score([]) == []
    assert teacher.usage() == 0


def test_loss_token_at_position_zero_stays_none() -> None:
    """A sampled token with no context cannot be scored; its None survives
    (attach_advantages drops the datum loudly) instead of raising here."""
    client = FakeSamplingClient(seed="s")
    sequence = client.sample([], max_tokens=3, temperature=1.0)
    datum = _datum([], sequence.tokens)
    teacher = TinkerTeacher(_spec(), sampling_client=client)

    [row] = teacher.score([datum])

    assert row[0] is None
    assert all(value is not None for value in row[1:])


def test_usage_counts_full_sequences() -> None:
    client = FakeSamplingClient(seed="s")
    prompt = [1, 2, 3]
    sequence = client.sample(prompt, max_tokens=5, temperature=1.0)
    datum = _datum(prompt, sequence.tokens)
    teacher = TinkerTeacher(_spec(), sampling_client=client)

    teacher.score([datum])
    teacher.score([datum])

    assert teacher.usage() == 2 * len(datum.model_input_tokens)


class _WrongLengthScorer:
    """Returns one entry too few, simulating tokenizer/SDK drift."""

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        return [None] + [-1.0] * (len(token_ids) - 2)


class _NoneAtLossScorer:
    """Returns None everywhere, so every loss position is unscoreable."""

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        return [None] * len(token_ids)


def test_wrong_length_result_is_actionable() -> None:
    teacher = TinkerTeacher(_spec(), sampling_client=_WrongLengthScorer())
    with pytest.raises(RuntimeError, match="tokenizer_fingerprint_check"):
        teacher.score([_datum([1, 2], [3, 4])])


def test_none_at_loss_position_is_actionable() -> None:
    teacher = TinkerTeacher(_spec(), sampling_client=_NoneAtLossScorer())
    with pytest.raises(RuntimeError, match="loss position"):
        teacher.score([_datum([1, 2], [3, 4])])


# --- Lazy SDK handling ------------------------------------------------------


def test_missing_extra_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tinker", None)
    teacher = TinkerTeacher(_spec())
    with pytest.raises(ImportError, match="uv sync --extra distill"):
        teacher.score([_datum([1, 2], [3])])


def test_missing_api_key_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tinker")
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    teacher = TinkerTeacher(_spec())
    with pytest.raises(RuntimeError, match="TINKER_API_KEY is not set"):
        teacher.score([_datum([1, 2], [3])])


def test_injected_client_never_touches_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tinker", None)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    client = FakeSamplingClient(seed="s")
    prompt = [1, 2]
    sequence = client.sample(prompt, max_tokens=2, temperature=1.0)
    teacher = TinkerTeacher(_spec(), sampling_client=client)
    [row] = teacher.score([_datum(prompt, sequence.tokens)])
    assert row[len(prompt) :] == list(sequence.logprobs)


# --- verify -----------------------------------------------------------------


def test_verify_ok_with_fake_client() -> None:
    teacher = TinkerTeacher(_spec(), sampling_client=FakeSamplingClient(seed="s"))
    result = teacher.verify()
    assert result.ok is True
    assert result.kind is ProviderKind.TINKER
    assert result.model == _spec().model


def test_verify_reports_checkpoint_identity() -> None:
    spec = TeacherConfig(model="base", checkpoint="tinker://run/weights/7")
    teacher = TinkerTeacher(spec, sampling_client=FakeSamplingClient(seed="s"))
    assert teacher.verify().model == "tinker://run/weights/7"


class _ExplodingScorer:
    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        raise RuntimeError("boom: model not found")


def test_verify_reports_failure_without_raising() -> None:
    teacher = TinkerTeacher(_spec(), sampling_client=_ExplodingScorer())
    result = teacher.verify()
    assert result.ok is False
    assert "boom" in result.detail


def test_verify_never_counts_usage() -> None:
    teacher = TinkerTeacher(_spec(), sampling_client=FakeSamplingClient(seed="s"))
    teacher.verify()
    assert teacher.usage() == 0


# --- tokenizer fingerprint ---------------------------------------------------


class _OffsetTokenizer:
    """Encodes like FakeTokenizer but with every id shifted, and decodes back."""

    def encode(self, text: str) -> list[int]:
        return [ord(ch) + 1 for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(t - 1) for t in token_ids)


def test_fingerprint_identical_tokenizers_pass() -> None:
    assert (
        tokenizer_fingerprint_check(
            "student-model", "teacher-model", FakeTokenizer(), FakeTokenizer()
        )
        is None
    )


def test_fingerprint_mismatch_names_models_and_fix() -> None:
    with pytest.raises(ValueError) as excinfo:
        tokenizer_fingerprint_check(
            "nvidia/Nemotron-3-Nano-30B-A3B",
            "some/Other-Teacher",
            FakeTokenizer(),
            _OffsetTokenizer(),
        )
    message = str(excinfo.value)
    assert "nvidia/Nemotron-3-Nano-30B-A3B" in message
    assert "some/Other-Teacher" in message
    assert "same-family teacher" in message
    assert "different student/teacher pair" in message


def test_fingerprint_default_corpus_covers_unicode_code_and_json() -> None:
    joined = "\n".join(TOKENIZER_PROBE_TEXTS)
    assert any(ord(ch) > 127 for ch in joined)  # mixed unicode
    assert "def " in joined  # code
    assert '{"' in joined  # JSON


def test_fingerprint_rejects_empty_probes() -> None:
    with pytest.raises(ValueError, match="probe_texts is empty"):
        tokenizer_fingerprint_check(
            "student", "teacher", FakeTokenizer(), FakeTokenizer(), probe_texts=[]
        )
