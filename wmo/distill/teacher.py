"""Teacher-side logprob scoring for on-policy distillation.

The teacher never generates during training; it scores the student's exact
sampled tokens. `TinkerTeacher` sends each `TrainDatum`'s full (unshifted)
token sequence through `tinker.SamplingClient.compute_logprobs` and keeps the
loss positions. For the `topk_ce` loss, `score_topk` instead makes one
prefill-only `sample` call per datum (`max_tokens=1`,
`include_prompt_logprobs=True`, `topk_prompt_logprobs=k`; verified live on the
pinned SDK, k <= 1000) whose response carries BOTH the per-position top-k
candidates and the realized per-position logprobs, so one request feeds the
candidate targets and the unchanged reverse-KL metric.

Indexing convention (verified against both the real SDK and the fakes):
`compute_logprobs(tokens)` returns one entry per input position, where entry p
is the logprob of token p given tokens < p, and entry 0 is None because the
first token has no context. The prefill response's `prompt_logprobs` and
`topk_prompt_logprobs` follow the same convention. The datum's
`model_input_tokens` IS that full sequence and its `loss_mask` names the
sampled positions, so the returned rows align index for index with the datum
(`teacher_test.py` pins the alignment against the fake sampler's echoed
logprobs to guard any off-by-one).

The tinker SDK is an optional extra imported lazily (`uv sync --extra
distill`), mirroring `wmo.providers.tinker`; injecting a sampling client
(tests use `wmo.distill.fake_tinker.FakeSamplingClient`) avoids the SDK
entirely. The lazily built real client comes from `wmo.providers.tinker`'s
process-wide shared cache (one `tinker.ServiceClient` and one
`SamplingClient` per model string for the whole process), so teacher
construction never adds server-side sessions of its own.

Every SDK call is deadline-bounded (`wmo.distill.deadlines`): a wedged
session raises a retryable `TinkerDeadlineError` instead of hanging, and the
teacher drops its lazily built client on expiry, evicting the shared cache
entry, so the next score() call rebuilds a fresh session (an injected client
is never dropped).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from wmo.distill.config import TeacherConfig
from wmo.distill.data import TopkCandidates, TrainDatum
from wmo.distill.deadlines import TinkerDeadlineError, wait_with_deadline
from wmo.providers.base import ProviderKind, VerifyResult
from wmo.providers.tinker import evict_shared_sampling_client, shared_sampling_client

if TYPE_CHECKING:
    import tinker

logger = logging.getLogger(__name__)

_SCORE_CONCURRENCY = 8
"""Upper bound on concurrent compute_logprobs calls in one score() batch."""

_VERIFY_PROBE_TOKEN_IDS: tuple[int, ...] = (1, 2, 3, 4)
"""A tiny fixed sequence for verify(); small ids are valid in any real vocab."""

TOKENIZER_PROBE_TEXTS: tuple[str, ...] = (
    "Hello, world! How are you today?",
    "naïve café: déjà vu, übermäßig, œuvre, 你好世界, こんにちは, 🚀✨",
    "def solve(xs: list[int]) -> int:\n    return sum(x * 2 for x in xs)\n",
    '{"tool": "bash", "arguments": {"cmd": "ls -la | wc -l", "timeout": 30}}',
    "tail -n 100 /var/log/syslog | grep -E 'error|warn' >> /tmp/out.txt",
    "Mixing scripts: Ω ≈ 3.14, привет, مرحبا, 한국어, \t tabs\nand newlines.",
)
"""Default fingerprint probe corpus: prose, unicode, code, JSON, shell."""


class TeacherTopkScores(BaseModel):
    """One `score_topk` batch: top-k candidate rows plus realized logprobs.

    Both lists align one to one with the scored datums and, per datum, index
    for index with its `model_input_tokens` (the compute_logprobs
    convention). They come from the same prefill response, so the realized
    logprobs cost no extra teacher request.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topk: list[list[TopkCandidates | None]]
    """Per datum, per position: the teacher's top-k (token_id, logprob)
    candidates when the position is a scoreable loss position (mask 1.0,
    p >= 1), None everywhere else."""

    realized: list[list[float | None]]
    """Per datum, per position: the teacher's logprob of the REALIZED token
    at scoreable loss positions, None everywhere else; identical in meaning
    to a `score()` row, so the reverse-KL metric stays comparable across
    loss modes."""


class TeacherClient(Protocol):
    """What the distillation loop needs from a teacher backend."""

    def score(self, datums: Sequence[TrainDatum]) -> list[list[float | None]]:
        """Per-position teacher logprob rows, aligned one to one with `datums`.

        Row entry p is the teacher's logprob of `model_input_tokens[p]` when p
        is a scoreable loss position (mask 1.0, p >= 1) and None everywhere
        else (context positions, and position 0, which has no context).
        """
        ...

    def score_topk(self, datums: Sequence[TrainDatum], k: int) -> TeacherTopkScores:
        """Per-position top-k candidate rows plus realized logprobs.

        Row entry p carries the teacher's top-k (token_id, logprob)
        candidates for position p when p is a scoreable loss position and
        None everywhere else, following the same convention as `score`.
        """
        ...

    def verify(self) -> VerifyResult:
        """One cheap preflight probe; reports failure, never raises."""
        ...

    def usage(self) -> int:
        """Total tokens submitted for scoring so far (the teacher_prefill meter)."""
        ...


@runtime_checkable
class CachedUsageTeacher(Protocol):
    """A teacher that can report how much of its prefill the service cached.

    Deliberately NOT part of `TeacherClient`: every existing teacher and test
    fake satisfies that seam today, and widening it would force each of them to
    grow a method whose only honest answer is 0. A caller narrows to this at
    runtime and falls back to billing everything uncached.
    """

    def cached_usage(self) -> int:
        """Of `usage()`, the tokens served from the service's prefix cache."""
        ...


class EncodingTokenizer(Protocol):
    """The tokenizer slice the fingerprint check needs: encoding only.

    HuggingFace tokenizers and the deterministic test fakes both satisfy it
    structurally (extra defaulted parameters do not matter).
    """

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids."""
        ...


class LogprobScorer(Protocol):
    """The one call the teacher makes, in token-id terms.

    `wmo.distill.fake_tinker.FakeSamplingClient` satisfies this directly;
    real `tinker.SamplingClient`s are adapted via `SdkLogprobScorer`.
    """

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        """Per-position logprobs for the sequence; entry 0 is None."""
        ...


@runtime_checkable
class CacheReportingScorer(Protocol):
    """A scorer that can report the service's own prefix-cache hit count.

    Kept a separate, runtime-checked protocol rather than folded into
    `LogprobScorer` so the deterministic fakes and any externally injected
    scorer keep satisfying the base seam unchanged; the caller treats a scorer
    without this as "cache unknown", which bills every token at the uncached
    rate exactly as before.
    """

    @property
    def cache_hit_tokens(self) -> int:
        """Prompt tokens the service reported serving from its prefix cache."""
        ...


@runtime_checkable
class TopkLogprobScorer(Protocol):
    """The prefill-only top-k call `score_topk` additionally needs.

    `FakeSamplingClient` and `SdkLogprobScorer` (and the loop's
    `SdkSamplingClient` wrapper) all provide it; `score_topk` narrows its
    `LogprobScorer` to this at runtime so plain injected scorers keep
    working for `score`.
    """

    def topk_prompt_logprobs(
        self, token_ids: list[int], k: int
    ) -> tuple[list[float | None], list[TopkCandidates | None]]:
        """(realized logprobs, top-k rows), one entry per position each."""
        ...


class SdkLogprobScorer:
    """Adapts a real `tinker.SamplingClient` to the `LogprobScorer` seam."""

    def __init__(self, client: tinker.SamplingClient) -> None:
        self._client = client
        self._cache_hit_tokens = 0

    @property
    def sdk_client(self) -> tinker.SamplingClient:
        """The wrapped SDK client (shared-cache eviction compares its identity)."""
        return self._client

    @property
    def cache_hit_tokens(self) -> int:
        """Prompt tokens the service reported serving from its prefix cache.

        Accumulated across every scoring call this scorer has made, so the
        caller can bill the cached portion at the cached rate instead of
        assuming zero.
        """
        return self._cache_hit_tokens

    def _record_cache_hits(self, response: object) -> None:
        """Add one response's reported cache hits, tolerating their absence.

        Read defensively on purpose. This is COST ACCOUNTING on the hot path of
        a paid run: if a future SDK renames or drops `prompt_cache_hit_tokens`,
        the correct outcome is a ledger that quietly reverts to billing
        everything uncached (an overstatement, the behaviour we already lived
        with), not an AttributeError that kills a run mid-batch after the
        service has already been paid for the scoring. Logged at debug once per
        call rather than warned, because a scorer that cannot report is the
        normal case for injected fakes.

        Args:
            response: The SDK sample response.
        """
        hits = getattr(response, "prompt_cache_hit_tokens", None)
        if isinstance(hits, int):
            self._cache_hit_tokens += hits
            return
        logger.debug(
            "teacher scoring response carries no usable prompt_cache_hit_tokens (%r); "
            "billing this call's prefill as uncached",
            hits,
        )

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        """One deadline-bounded prefill call on the full sequence.

        Issued as `sample` rather than the SDK's `compute_logprobs`, which is
        not a different request: the SDK's own implementation is exactly this
        call (`num_samples=1`, `max_tokens=1`, `include_prompt_logprobs=True`)
        followed by `return sample_res.prompt_logprobs`. It throws the rest of
        the response away, including `prompt_cache_hit_tokens` -- which is the
        service's own measurement of how much of this prefill it did not have to
        recompute, and the single number that decides whether our cost figures
        are a bill or a ceiling. Tinker's prefix cache demonstrably works (a
        repeated prompt measured 16,000 of 16,000 tokens cached) while every
        `*_cached_prefill` meter read 0 by construction, so the run ledger has
        been overstating spend by an unknown factor. Same request, same
        `prompt_logprobs`, one more field kept.

        The deadline keeps the `compute_logprobs` label deliberately: the
        workload is byte-for-byte the same prefill, and relabelling it `sample`
        would silently swap in a deadline tuned for generation instead.

        Raises:
            TinkerDeadlineError: If the deadline expires (the session is
                likely wedged; the caller should retry with a fresh one).
            RuntimeError: If the response carries no prompt logprobs even
                though the request asked for them (SDK drift).
        """
        import tinker

        future = self._client.sample(
            prompt=tinker.ModelInput.from_ints(token_ids),
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
        )
        response = wait_with_deadline("compute_logprobs", future)
        self._record_cache_hits(response)
        realized = response.prompt_logprobs
        if realized is None:
            raise RuntimeError(
                "the teacher's prefill sample response carries no prompt_logprobs even "
                "though include_prompt_logprobs was set; check the pinned tinker SDK "
                "version (0.23.3 populates it on request)"
            )
        return list(realized)

    def topk_prompt_logprobs(
        self, token_ids: list[int], k: int
    ) -> tuple[list[float | None], list[TopkCandidates | None]]:
        """One deadline-bounded prefill-only sample returning prompt logprobs.

        The whole sequence rides as the PROMPT of a `max_tokens=1` sample
        request with `include_prompt_logprobs=True` and
        `topk_prompt_logprobs=k` (the pinned SDK's only top-k surface,
        verified live: k <= 1000). The response's `prompt_logprobs` and
        `topk_prompt_logprobs` both follow the compute_logprobs convention
        (entry p scores token p given tokens < p; entry 0 is None), so one
        request yields the candidates AND the realized logprobs. The single
        sampled token is discarded.

        Raises:
            TinkerDeadlineError: If the deadline expires (the session is
                likely wedged; the caller should retry with a fresh one).
            RuntimeError: If the response omits either logprob field (SDK
                drift; the pinned tinker 0.23.3 populates both on request).
        """
        import tinker

        future = self._client.sample(
            prompt=tinker.ModelInput.from_ints(token_ids),
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
            topk_prompt_logprobs=k,
        )
        response = wait_with_deadline("sample", future)
        self._record_cache_hits(response)
        realized = response.prompt_logprobs
        rows = response.topk_prompt_logprobs
        if realized is None or rows is None:
            missing = "prompt_logprobs" if realized is None else "topk_prompt_logprobs"
            raise RuntimeError(
                f"the teacher's prefill sample response carries no {missing} even "
                "though the request asked for it; check the pinned tinker SDK "
                "version (0.23.3 populates both when include_prompt_logprobs and "
                "topk_prompt_logprobs are set)"
            )
        return list(realized), [None if row is None else list(row) for row in rows]


class TinkerTeacher:
    """`TeacherClient` backed by a Tinker sampling client.

    Args:
        spec: The `[teacher]` section of the run config; `model` is the base
            model name and `checkpoint` optionally pins a `tinker://` weights
            path to serve the teacher from.
        sampling_client: Optional injected scorer (tests use the fakes in
            `wmo.distill.fake_tinker`; wrap a real `tinker.SamplingClient` in
            `SdkLogprobScorer`). When None, a real client is fetched lazily
            on first use from `wmo.providers.tinker`'s process-wide shared
            cache, keyed by the teacher's model identity (checkpoint or base
            model name); an injected client bypasses the cache entirely.
    """

    def __init__(
        self, spec: TeacherConfig, *, sampling_client: LogprobScorer | None = None
    ) -> None:
        self._spec = spec
        self._scorer = sampling_client
        # Only a client the teacher built itself may be dropped and rebuilt
        # after a deadline expiry; an injected one cannot be reconstructed.
        self._owns_scorer = sampling_client is None
        self._usage_tokens = 0

    def _model_identity(self) -> str:
        return self._spec.checkpoint or self._spec.model

    def _get_scorer(self) -> LogprobScorer:
        if self._scorer is None:
            self._scorer = self._build_sdk_scorer()
        return self._scorer

    def _build_sdk_scorer(self) -> LogprobScorer:
        # Lazy: the SDK import and the API-key check happen inside the shared
        # cache path (`wmo.providers.tinker.shared_sampling_client`). One
        # process-wide ServiceClient and one SamplingClient per model string
        # are shared with the rollout providers, so building the teacher
        # never adds server-side sessions of its own.
        return SdkLogprobScorer(shared_sampling_client(self._model_identity()))

    def _drop_wedged_scorer(self) -> None:
        """Forget (and evict from the shared cache) a wedged scoring client.

        A wedged session keeps timing out while a freshly built one heals, so
        the next score() call rebuilds through `_get_scorer`. The shared
        cache entry is evicted too, not just this teacher's reference, so no
        other user of the same model string inherits the wedged session. An
        injected client is never dropped: the teacher cannot rebuild what it
        did not build.
        """
        if self._owns_scorer and self._scorer is not None:
            logger.warning(
                "dropping the tinker teacher client after a deadline expiry; "
                "the next score() call builds a fresh session"
            )
            if isinstance(self._scorer, SdkLogprobScorer):
                evict_shared_sampling_client(self._model_identity(), self._scorer.sdk_client)
            self._scorer = None

    def score(self, datums: Sequence[TrainDatum]) -> list[list[float | None]]:
        """Score each datum's full token sequence into a per-position row.

        Datums are scored with bounded concurrency (the SDK call blocks on a
        per-sequence future, so a small thread pool batches them) and results
        keep datum order. Row entry p carries the teacher logprob of the
        datum's token p exactly when p is a loss position (mask 1.0) with
        context (p >= 1); every other entry is None. A loss token at position
        0 cannot be conditioned on anything, so its None survives and
        `attach_advantages` drops that datum loudly.

        Args:
            datums: The batch to score, in the loop's unshifted layout.

        Returns:
            One per-position row per datum, aligned with its
            `model_input_tokens`.

        Raises:
            ImportError: If no client was injected and the tinker extra is
                not installed.
            RuntimeError: If the API key is missing, the teacher returns a
                result of the wrong length (tokenizer or SDK drift), or a
                scoreable loss position comes back None.
            TinkerDeadlineError: If a compute_logprobs deadline expires; the
                lazily built client is dropped first so retrying score()
                rebuilds a fresh session.
        """
        datum_list = list(datums)
        if not datum_list:
            return []
        scorer = self._get_scorer()
        workers = min(_SCORE_CONCURRENCY, len(datum_list))
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                per_datum = list(
                    pool.map(
                        lambda datum: scorer.compute_logprobs(datum.model_input_tokens), datum_list
                    )
                )
        except TinkerDeadlineError:
            # The teacher session is likely wedged; drop the lazily built
            # client so the caller's retry scores through a fresh session.
            # Every in-flight worker is itself deadline-bounded, so the
            # pool's shutdown join above stays bounded too.
            self._drop_wedged_scorer()
            raise
        finally:
            # Counted whether or not scoring succeeded: pool.map submits every
            # datum up front and the executor's shutdown join runs them all, so
            # the service bills the whole batch even when one call raises. The
            # caller's finally-charge turns this into ledgered spend.
            self._usage_tokens += sum(len(datum.model_input_tokens) for datum in datum_list)
        results: list[list[float | None]] = []
        for index, (datum, logprobs) in enumerate(zip(datum_list, per_datum, strict=True)):
            results.append(self._loss_position_row(index, datum, logprobs))
        logger.debug(
            "teacher scored %d datum(s), %d tokens total so far",
            len(datum_list),
            self._usage_tokens,
        )
        return results

    def score_topk(self, datums: Sequence[TrainDatum], k: int) -> TeacherTopkScores:
        """Top-k candidates plus realized logprobs for each datum's loss positions.

        One prefill-only sample request per datum (bounded concurrency, datum
        order preserved) returns the teacher's per-position top-k candidates
        AND the realized per-position logprobs together, so the reverse-KL
        metric costs no second pass. Rows follow `score`'s convention: entry
        p is populated exactly when p is a loss position (mask 1.0) with
        context (p >= 1); a loss token at position 0 stays None in BOTH rows
        and `build_topk_ce_datums` drops that datum loudly, mirroring
        `attach_advantages`.

        Usage accounting matches `score`: the full sequence tokens count
        toward the teacher_prefill meter. The one discarded sampled token per
        request is deliberately not counted (it is noise against the
        sequence volume and has no meter of its own).

        Args:
            datums: The batch to score, in the loop's unshifted layout.
            k: Candidates per position (`train.topk`).

        Returns:
            The batch's `TeacherTopkScores`, aligned one to one with `datums`.

        Raises:
            ValueError: If `k < 1`.
            ImportError: If no client was injected and the tinker extra is
                not installed.
            RuntimeError: If the API key is missing, the scoring client has
                no top-k surface, the teacher returns rows of the wrong
                length, or a scoreable loss position comes back without
                candidates or without a realized logprob.
            TinkerDeadlineError: If a request deadline expires; the lazily
                built client is dropped first so retrying score_topk
                rebuilds a fresh session.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        datum_list = list(datums)
        if not datum_list:
            return TeacherTopkScores(topk=[], realized=[])
        scorer = self._get_scorer()
        if not isinstance(scorer, TopkLogprobScorer):
            raise RuntimeError(
                f"the teacher's scoring client {type(scorer).__name__} has no "
                "topk_prompt_logprobs surface, so it cannot serve the topk_ce "
                "loss; inject a client with the prefill top-k call (the SDK "
                "adapters and the fakes both have it) or use "
                'train.loss = "importance_sampling" or "ppo"'
            )
        workers = min(_SCORE_CONCURRENCY, len(datum_list))
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                per_datum = list(
                    pool.map(
                        lambda datum: scorer.topk_prompt_logprobs(datum.model_input_tokens, k),
                        datum_list,
                    )
                )
        except TinkerDeadlineError:
            # Same rationale as score(): the session is likely wedged, so a
            # lazily built client is dropped for the caller's retry.
            self._drop_wedged_scorer()
            raise
        finally:
            # Counted whether or not scoring succeeded, same contract as
            # score(): every submitted call runs (and bills) server-side
            # before the pool join propagates an error.
            self._usage_tokens += sum(len(datum.model_input_tokens) for datum in datum_list)
        topk_rows: list[list[TopkCandidates | None]] = []
        realized_rows: list[list[float | None]] = []
        for index, (datum, (realized, candidates)) in enumerate(
            zip(datum_list, per_datum, strict=True)
        ):
            realized_rows.append(self._loss_position_row(index, datum, realized))
            topk_rows.append(self._loss_position_topk_row(index, datum, candidates))
        logger.debug(
            "teacher top-%d scored %d datum(s), %d tokens total so far",
            k,
            len(datum_list),
            self._usage_tokens,
        )
        return TeacherTopkScores(topk=topk_rows, realized=realized_rows)

    def _loss_position_topk_row(
        self,
        datum_index: int,
        datum: TrainDatum,
        candidates: list[TopkCandidates | None],
    ) -> list[TopkCandidates | None]:
        """Keep the prefill top-k entries at the datum's loss positions.

        Mirrors `_loss_position_row` index for index: `candidates[p]` holds
        the top-k for token p given tokens < p, so the result aligns with
        `model_input_tokens`. Position 0's None survives (no context), and
        `build_topk_ce_datums` drops that datum loudly.
        """
        tokens = datum.model_input_tokens
        if len(candidates) != len(tokens):
            raise RuntimeError(
                f"teacher returned {len(candidates)} top-k entrie(s) for the "
                f"{len(tokens)}-token sequence of datum {datum_index}; the prefill "
                "top-k must return one entry per input position. Check that the "
                "teacher model matches the student's tokenizer (run "
                "tokenizer_fingerprint_check) and that the pinned tinker SDK "
                "version is unchanged"
            )
        row: list[TopkCandidates | None] = [None] * len(tokens)
        for position, weight in enumerate(datum.loss_mask):
            if weight != 1.0 or position == 0:
                continue
            value = candidates[position]
            if not value:
                raise RuntimeError(
                    f"teacher returned no top-k candidates for loss position "
                    f"{position} of datum {datum_index}; only position 0 may be "
                    f"empty. Re-run the step; if it persists, the teacher "
                    f"{self._model_identity()!r} could not score the sequence, so "
                    "check the model/checkpoint and the tokenizer fingerprint"
                )
            row[position] = list(value)
        return row

    def _loss_position_row(
        self,
        datum_index: int,
        datum: TrainDatum,
        logprobs: list[float | None],
    ) -> list[float | None]:
        """Keep the compute_logprobs entries at the datum's loss positions.

        `logprobs[p]` is the logprob of token p given tokens < p, so the
        result aligns index for index with `model_input_tokens`.
        """
        tokens = datum.model_input_tokens
        if len(logprobs) != len(tokens):
            raise RuntimeError(
                f"teacher returned {len(logprobs)} logprobs for the {len(tokens)}-token "
                f"sequence of datum {datum_index}; compute_logprobs must return one "
                "entry per input position. Check that the teacher model matches the "
                "student's tokenizer (run tokenizer_fingerprint_check) and that the "
                "pinned tinker SDK version is unchanged"
            )
        row: list[float | None] = [None] * len(tokens)
        for position, weight in enumerate(datum.loss_mask):
            if weight != 1.0 or position == 0:
                continue
            value = logprobs[position]
            if value is None:
                raise RuntimeError(
                    f"teacher returned no logprob for loss position {position} of "
                    f"datum {datum_index}; only position 0 may be None. Re-run the "
                    f"step; if it persists, the teacher {self._model_identity()!r} "
                    "could not score the sequence, so check the model/checkpoint "
                    "and the tokenizer fingerprint"
                )
            row[position] = value
        return row

    def verify(self) -> VerifyResult:
        """One tiny compute_logprobs probe, reporting failure as ok=False.

        Mirrors `wmo.providers.base.verify_via_ping`: never raises, so
        preflight can report every misconfigured client at once.
        """
        model = self._model_identity()
        try:
            result = self._get_scorer().compute_logprobs(list(_VERIFY_PROBE_TOKEN_IDS))
            if len(result) != len(_VERIFY_PROBE_TOKEN_IDS):
                raise RuntimeError(
                    f"probe returned {len(result)} logprobs for "
                    f"{len(_VERIFY_PROBE_TOKEN_IDS)} tokens"
                )
        except Exception as exc:  # noqa: BLE001 - verify reports failure, never raises
            if isinstance(exc, TinkerDeadlineError):
                self._drop_wedged_scorer()
            return VerifyResult(ok=False, kind=ProviderKind.TINKER, model=model, detail=str(exc))
        return VerifyResult(ok=True, kind=ProviderKind.TINKER, model=model)

    def usage(self) -> int:
        """Total tokens submitted through score() so far (verify probes excluded)."""
        return self._usage_tokens

    def cached_usage(self) -> int:
        """Of `usage()`, the tokens the SERVICE reported serving from its cache.

        Zero when the active scorer cannot report it (an injected fake, or a
        future SDK that drops the field), which bills everything at the uncached
        rate — the same conservative behaviour the ledger had before this was
        readable. The count is cumulative and monotonic like `usage()`, so a
        caller takes deltas around a step.
        """
        scorer = self._scorer
        if isinstance(scorer, CacheReportingScorer):
            return scorer.cache_hit_tokens
        return 0


def tokenizer_fingerprint_check(
    student_model: str,
    teacher_model: str,
    student_tokenizer: EncodingTokenizer,
    teacher_tokenizer: EncodingTokenizer,
    probe_texts: Sequence[str] = TOKENIZER_PROBE_TEXTS,
) -> None:
    """Require the student and teacher tokenizers to be interchangeable.

    Tinker distillation scores the student's exact token ids with the teacher,
    which is only meaningful when both models tokenize identically. Every
    probe text must encode to the same token ids under both tokenizers.

    Args:
        student_model: Student base model name, for the error message.
        teacher_model: Teacher model name, for the error message.
        student_tokenizer: The student's tokenizer.
        teacher_tokenizer: The teacher's tokenizer.
        probe_texts: The probe corpus; defaults to `TOKENIZER_PROBE_TEXTS`
            (prose, mixed unicode, code, JSON, shell).

    Raises:
        ValueError: If `probe_texts` is empty, or any probe encodes
            differently; the message names both models and the fix.
    """
    if not probe_texts:
        raise ValueError(
            "probe_texts is empty, so the tokenizer fingerprint check would prove "
            "nothing; pass at least one probe text (or use the default corpus)"
        )
    for text in probe_texts:
        student_ids = student_tokenizer.encode(text)
        teacher_ids = teacher_tokenizer.encode(text)
        if student_ids == teacher_ids:
            continue
        mismatch = next(
            (
                index
                for index, (a, b) in enumerate(zip(student_ids, teacher_ids, strict=False))
                if a != b
            ),
            min(len(student_ids), len(teacher_ids)),
        )
        raise ValueError(
            f"tokenizer fingerprint mismatch between student {student_model!r} and "
            f"teacher {teacher_model!r}: probe {text!r} encodes to {len(student_ids)} "
            f"student token(s) vs {len(teacher_ids)} teacher token(s), first "
            f"difference at position {mismatch}. Tinker distillation requires "
            "identical tokenizers, so pick a same-family teacher that shares the "
            "student's tokenizer, or choose a different student/teacher pair"
        )
    logger.debug(
        "tokenizer fingerprint ok: %s and %s agree on %d probe(s)",
        student_model,
        teacher_model,
        len(probe_texts),
    )
