"""Tests for teacher scoring: row alignment, error paths, deadlines, fingerprints."""

import sys
import threading
from typing import TYPE_CHECKING, NoReturn, cast

import pytest

import wmh.providers.tinker as providers_tinker
from wmh.distill.config import TeacherConfig
from wmh.distill.data import TrainDatum
from wmh.distill.deadlines import TinkerDeadlineError
from wmh.distill.fake_tinker import FakeSamplingClient, FakeTokenizer
from wmh.distill.teacher import (
    TOKENIZER_PROBE_TEXTS,
    LogprobScorer,
    SdkLogprobScorer,
    TeacherTopkScores,
    TinkerTeacher,
    tokenizer_fingerprint_check,
)
from wmh.providers.base import ProviderKind

if TYPE_CHECKING:
    import tinker


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


# --- deadlines: wedged sessions become retryable errors with fresh clients ---


class _WedgedScorer:
    """A scorer whose every call reports a deadline expiry (a wedged session)."""

    def __init__(self) -> None:
        self.calls = 0

    def compute_logprobs(self, token_ids: list[int]) -> NoReturn:
        del token_ids
        self.calls += 1
        raise TinkerDeadlineError("compute_logprobs", elapsed_s=0.05, deadline_s=0.05)


def test_score_deadline_drops_and_rebuilds_the_lazy_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = TinkerTeacher(_spec())
    builds: list[LogprobScorer] = []

    def build_scorer() -> LogprobScorer:
        scorer: LogprobScorer = _WedgedScorer() if not builds else FakeSamplingClient(seed="s")
        builds.append(scorer)
        return scorer

    monkeypatch.setattr(teacher, "_build_sdk_scorer", build_scorer)

    with pytest.raises(TinkerDeadlineError, match="timed out"):
        teacher.score([_datum([1, 2], [3, 4])])
    # The wedged batch still counts: every submitted compute_logprobs call runs
    # (and bills) server-side before the pool join propagates the error, so
    # dropping it from usage would leak real spend past the budget ledger.
    assert teacher.usage() == 4
    # The retry (here: calling score again) rebuilds a fresh session and works.
    [row] = teacher.score([_datum([1, 2], [3, 4])])
    assert len(row) == 4
    assert teacher.usage() == 8
    assert len(builds) == 2


def test_injected_scorer_is_never_dropped_on_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    # An injected client cannot be rebuilt; poison the SDK so any accidental
    # rebuild attempt would fail loudly instead of hitting the network.
    monkeypatch.setitem(sys.modules, "tinker", None)
    scorer = _WedgedScorer()
    teacher = TinkerTeacher(_spec(), sampling_client=scorer)
    for _ in range(2):
        with pytest.raises(TinkerDeadlineError):
            teacher.score([_datum([1, 2], [3])])
    assert scorer.calls == 2


def test_drop_wedged_scorer_evicts_the_shared_sampling_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The lazily built scorer wraps a client from the process-wide shared
    # cache; dropping it must evict the cache entry too, or every future user
    # of the teacher's model string would inherit the wedged session.
    class _Client:
        """Identity stand-in for the cached tinker.SamplingClient."""

    wedged = cast("tinker.SamplingClient", _Client())
    spec = _spec()
    monkeypatch.setattr(providers_tinker, "_shared_samplers", {spec.model: wedged})
    teacher = TinkerTeacher(spec)
    monkeypatch.setattr(teacher, "_scorer", SdkLogprobScorer(wedged))

    teacher._drop_wedged_scorer()

    assert spec.model not in providers_tinker._shared_samplers


class _NeverResolvingFuture:
    """Mimics the SDK future of a wedged session: result(timeout) honors the timeout."""

    def __init__(self) -> None:
        self._never = threading.Event()

    def result(self, timeout: float | None = None) -> NoReturn:
        self._never.wait(timeout)
        raise TimeoutError(f"fake future gave up after {timeout}s")


def test_sdk_logprob_scorer_bounds_the_future(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tinker")
    monkeypatch.setenv("WMH_TINKER_DEADLINE_COMPUTE_LOGPROBS", "0.05")

    class _WedgedClient:
        """Wedged on `sample`, which is what scoring issues.

        `SdkLogprobScorer.compute_logprobs` calls `sample` with
        `include_prompt_logprobs=True` rather than the SDK's `compute_logprobs`
        wrapper, because that wrapper makes exactly this request and then
        discards `prompt_cache_hit_tokens` with the rest of the response. The
        deadline still carries the COMPUTE_LOGPROBS label, since the workload is
        the same prefill.
        """

        def sample(
            self,
            prompt: object,
            num_samples: int,
            sampling_params: object,
            include_prompt_logprobs: bool = False,
        ) -> _NeverResolvingFuture:
            del prompt, num_samples, sampling_params, include_prompt_logprobs
            return _NeverResolvingFuture()

    scorer = SdkLogprobScorer(cast("tinker.SamplingClient", _WedgedClient()))
    with pytest.raises(TinkerDeadlineError, match="tinker compute_logprobs timed out"):
        scorer.compute_logprobs([1, 2, 3])


# --- score_topk: the prefill top-k path ---------------------------------------


def test_score_topk_rows_pin_the_prefill_slice_exactly() -> None:
    """Pins score_topk's per-position alignment against the fake's echoes.

    The fake's prefill top-k puts, at entry p, the top-k for token p given
    tokens < p with top-1 = (token p, its realized logprob); realized rows
    echo the issued logprobs exactly like compute_logprobs. If score_topk
    sliced a neighboring index, the top-1 candidates would not be the
    datum's own sampled tokens and this equality would fail.
    """
    client = FakeSamplingClient(seed="tinker://fake/sampler/teacher/0")
    prompt = [100, 101, 102, 103]
    sequence = client.sample(prompt, max_tokens=6, temperature=1.0)
    datum = _datum(prompt, sequence.tokens)
    teacher = TinkerTeacher(_spec(), sampling_client=client)

    scores = teacher.score_topk([datum], k=4)

    [topk_row] = scores.topk
    [realized_row] = scores.realized
    n = len(datum.model_input_tokens)
    assert len(topk_row) == n and len(realized_row) == n
    # Context positions stay None in both rows.
    assert topk_row[: len(prompt)] == [None] * len(prompt)
    assert realized_row[: len(prompt)] == [None] * len(prompt)
    # Loss positions: realized rows equal score()'s rows (same metric meaning),
    # and every top-1 candidate is the realized token with its issued logprob.
    assert realized_row == teacher.score([datum])[0]
    for offset, position in enumerate(range(len(prompt), n)):
        candidates = topk_row[position]
        assert candidates is not None
        assert len(candidates) == 4
        assert candidates[0] == (sequence.tokens[offset], sequence.logprobs[offset])
        logprobs = [lp for _, lp in candidates]
        assert logprobs == sorted(logprobs, reverse=True)


def test_score_topk_preserves_order_counts_usage_and_handles_empty() -> None:
    client = FakeSamplingClient(seed="s")
    prompt_a, prompt_b = [1, 2, 3], [7, 8]
    seq_a = client.sample(prompt_a, max_tokens=4, temperature=1.0)
    seq_b = client.sample(prompt_b, max_tokens=3, temperature=1.0)
    datum_a, datum_b = _datum(prompt_a, seq_a.tokens), _datum(prompt_b, seq_b.tokens)
    teacher = TinkerTeacher(_spec(), sampling_client=client)
    assert teacher.score_topk([], k=2) == TeacherTopkScores(topk=[], realized=[])
    assert teacher.usage() == 0

    scores = teacher.score_topk([datum_a, datum_b], k=2)

    tops = [
        [row[position] for position in range(len(prompt), len(row))]
        for row, prompt in zip(scores.topk, (prompt_a, prompt_b), strict=True)
    ]
    assert [
        [candidates[0][0] for candidates in per_datum if candidates is not None]
        for per_datum in tops
    ] == [list(seq_a.tokens), list(seq_b.tokens)]
    expected_usage = len(datum_a.model_input_tokens) + len(datum_b.model_input_tokens)
    assert teacher.usage() == expected_usage


def test_score_topk_loss_position_zero_stays_none() -> None:
    """A sampled token with no context cannot be scored; its None survives in
    BOTH rows (build_topk_ce_datums drops the datum loudly downstream)."""
    client = FakeSamplingClient(seed="s")
    sequence = client.sample([], max_tokens=3, temperature=1.0)
    datum = _datum([], sequence.tokens)
    teacher = TinkerTeacher(_spec(), sampling_client=client)

    scores = teacher.score_topk([datum], k=2)

    assert scores.topk[0][0] is None
    assert scores.realized[0][0] is None
    assert all(entry is not None for entry in scores.topk[0][1:])


def test_score_topk_rejects_bad_k() -> None:
    teacher = TinkerTeacher(_spec(), sampling_client=FakeSamplingClient(seed="s"))
    with pytest.raises(ValueError, match=">= 1"):
        teacher.score_topk([_datum([1, 2], [3])], k=0)


def test_score_topk_without_a_topk_scorer_is_actionable() -> None:
    """A plain compute_logprobs client cannot serve topk_ce; the error says so."""
    teacher = TinkerTeacher(_spec(), sampling_client=_NoneAtLossScorer())
    with pytest.raises(RuntimeError, match="topk_prompt_logprobs"):
        teacher.score_topk([_datum([1, 2], [3])], k=2)


class _WrongLengthTopkScorer:
    """Returns one prefill entry too few, simulating tokenizer/SDK drift."""

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        return [None] + [-1.0] * (len(token_ids) - 1)

    def topk_prompt_logprobs(
        self, token_ids: list[int], k: int
    ) -> tuple[list[float | None], list[list[tuple[int, float]] | None]]:
        realized = self.compute_logprobs(token_ids)
        rows: list[list[tuple[int, float]] | None] = [
            [(token, -1.0)] * k for token in token_ids[:-1]
        ]
        return realized, rows


def test_score_topk_wrong_length_row_is_actionable() -> None:
    teacher = TinkerTeacher(_spec(), sampling_client=_WrongLengthTopkScorer())
    with pytest.raises(RuntimeError, match="one entry per input position"):
        teacher.score_topk([_datum([1, 2], [3, 4])], k=2)


class _EmptyCandidatesScorer:
    """Realized logprobs are fine but every top-k row is empty."""

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        return [None] + [-1.0] * (len(token_ids) - 1)

    def topk_prompt_logprobs(
        self, token_ids: list[int], k: int
    ) -> tuple[list[float | None], list[list[tuple[int, float]] | None]]:
        del k
        return self.compute_logprobs(token_ids), [None] * len(token_ids)


def test_score_topk_empty_candidates_at_loss_position_is_actionable() -> None:
    teacher = TinkerTeacher(_spec(), sampling_client=_EmptyCandidatesScorer())
    with pytest.raises(RuntimeError, match="no top-k candidates for loss position"):
        teacher.score_topk([_datum([1, 2], [3, 4])], k=2)


class _WedgedTopkScorer:
    """Every prefill top-k call reports a deadline expiry (a wedged session)."""

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        return [None] * len(token_ids)

    def topk_prompt_logprobs(self, token_ids: list[int], k: int) -> NoReturn:
        del token_ids, k
        raise TinkerDeadlineError("sample", elapsed_s=0.05, deadline_s=0.05)


def test_score_topk_deadline_drops_and_rebuilds_the_lazy_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = TinkerTeacher(_spec())
    builds: list[LogprobScorer] = []

    def build_scorer() -> LogprobScorer:
        scorer: LogprobScorer = _WedgedTopkScorer() if not builds else FakeSamplingClient(seed="s")
        builds.append(scorer)
        return scorer

    monkeypatch.setattr(teacher, "_build_sdk_scorer", build_scorer)

    with pytest.raises(TinkerDeadlineError, match="timed out"):
        teacher.score_topk([_datum([1, 2], [3, 4])], k=2)
    # The wedged batch still counts, same contract as score(): every submitted
    # call runs (and bills) server-side before the pool join propagates the
    # error, so dropping it from usage would leak real spend past the ledger.
    assert teacher.usage() == 4
    scores = teacher.score_topk([_datum([1, 2], [3, 4])], k=2)
    assert len(scores.topk) == 1
    assert teacher.usage() == 8
    assert len(builds) == 2


def test_sdk_logprob_scorer_topk_uses_the_prefill_sample_call() -> None:
    """The SDK adapter sends the whole sequence as a max_tokens=1 prompt with
    both prompt-logprob switches set, and returns the response pair."""
    pytest.importorskip("tinker")

    recorded: dict[str, object] = {}

    class _Response:
        prompt_logprobs = [None, -0.5, -0.25]
        topk_prompt_logprobs = [None, [(2, -0.5), (9, -1.0)], [(3, -0.25), (9, -1.5)]]

    class _ReadyFuture:
        def result(self, timeout: float | None = None) -> _Response:
            del timeout
            return _Response()

    class _Client:
        def sample(
            self,
            prompt: object,
            num_samples: int,
            sampling_params: object,
            include_prompt_logprobs: bool = False,
            topk_prompt_logprobs: int = 0,
        ) -> _ReadyFuture:
            recorded["prompt"] = prompt
            recorded["num_samples"] = num_samples
            recorded["sampling_params"] = sampling_params
            recorded["include_prompt_logprobs"] = include_prompt_logprobs
            recorded["topk_prompt_logprobs"] = topk_prompt_logprobs
            return _ReadyFuture()

    scorer = SdkLogprobScorer(cast("tinker.SamplingClient", _Client()))
    realized, rows = scorer.topk_prompt_logprobs([1, 2, 3], 2)

    import tinker

    assert recorded["prompt"] == tinker.ModelInput.from_ints([1, 2, 3])
    assert recorded["num_samples"] == 1
    params = recorded["sampling_params"]
    assert isinstance(params, tinker.SamplingParams) and params.max_tokens == 1
    assert recorded["include_prompt_logprobs"] is True
    assert recorded["topk_prompt_logprobs"] == 2
    assert realized == [None, -0.5, -0.25]
    assert rows == [None, [(2, -0.5), (9, -1.0)], [(3, -0.25), (9, -1.5)]]


def test_sdk_logprob_scorer_topk_missing_fields_is_actionable() -> None:
    pytest.importorskip("tinker")

    class _Response:
        prompt_logprobs = None
        topk_prompt_logprobs = None

    class _ReadyFuture:
        def result(self, timeout: float | None = None) -> _Response:
            del timeout
            return _Response()

    class _Client:
        def sample(
            self,
            prompt: object,
            num_samples: int,
            sampling_params: object,
            include_prompt_logprobs: bool = False,
            topk_prompt_logprobs: int = 0,
        ) -> _ReadyFuture:
            del prompt, num_samples, sampling_params
            del include_prompt_logprobs, topk_prompt_logprobs
            return _ReadyFuture()

    scorer = SdkLogprobScorer(cast("tinker.SamplingClient", _Client()))
    with pytest.raises(RuntimeError, match="prompt_logprobs"):
        scorer.topk_prompt_logprobs([1, 2, 3], 2)


# --- service-reported prefix-cache accounting ---------------------------------


class _CachingClient:
    """A sampling client whose prefill responses report cache hits."""

    def __init__(self, hits: object) -> None:
        self._hits = hits
        self.calls = 0

    def sample(
        self,
        prompt: object,
        num_samples: int,
        sampling_params: object,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> object:
        del prompt, num_samples, sampling_params, include_prompt_logprobs, topk_prompt_logprobs
        self.calls += 1
        hits = self._hits

        class _Response:
            prompt_logprobs = [None, -0.5, -0.25]
            topk_prompt_logprobs = [None, [(2, -0.5)], [(3, -0.25)]]
            prompt_cache_hit_tokens = hits

        class _Future:
            def result(self, timeout: float | None = None) -> object:
                del timeout
                return _Response()

        return _Future()


def test_scoring_accumulates_the_services_reported_cache_hits() -> None:
    """The field the SDK's own compute_logprobs throws away.

    Tinker's prefix cache works (a repeated prompt measured 16,000 of 16,000
    tokens cached) while every `*_cached_prefill` meter read 0 by construction,
    so the run ledger overstated spend. Reading it makes the ledger a bill.
    """
    pytest.importorskip("tinker")
    client = _CachingClient(1200)
    scorer = SdkLogprobScorer(cast("tinker.SamplingClient", client))

    assert scorer.cache_hit_tokens == 0
    scorer.compute_logprobs([1, 2, 3])
    assert scorer.cache_hit_tokens == 1200
    scorer.compute_logprobs([1, 2, 3])
    assert scorer.cache_hit_tokens == 2400, "counts accumulate across calls"


def test_the_topk_path_also_reports_cache_hits() -> None:
    """Both scoring paths bill the same meters, so both must measure the same way."""
    pytest.importorskip("tinker")
    client = _CachingClient(700)
    scorer = SdkLogprobScorer(cast("tinker.SamplingClient", client))
    scorer.topk_prompt_logprobs([1, 2, 3], 2)
    assert scorer.cache_hit_tokens == 700


@pytest.mark.parametrize("hits", [None, "1200", 3.5])
def test_an_unreadable_cache_count_bills_uncached_instead_of_crashing(hits: object) -> None:
    """Cost accounting must never be able to kill a paid run.

    If a future SDK renames or retypes the field, the right outcome is a ledger
    that reverts to billing everything at the uncached rate -- an overstatement,
    which is what we already lived with -- not an exception on the hot path
    after the service has already been paid for the scoring.
    """
    pytest.importorskip("tinker")
    scorer = SdkLogprobScorer(cast("tinker.SamplingClient", _CachingClient(hits)))
    assert scorer.compute_logprobs([1, 2, 3]) == [None, -0.5, -0.25]
    assert scorer.cache_hit_tokens == 0


def test_cached_usage_is_zero_for_a_teacher_whose_scorer_cannot_report() -> None:
    """An injected fake keeps working and can only make the ledger conservative."""
    client = FakeSamplingClient(seed="tinker://fake/sampler/teacher/0")
    teacher = TinkerTeacher(_spec(), sampling_client=client)
    prompt = [100, 101, 102]
    sequence = client.sample(prompt, max_tokens=4, temperature=1.0)
    teacher.score([_datum(prompt, sequence.tokens)])
    assert teacher.usage() > 0
    assert teacher.cached_usage() == 0
