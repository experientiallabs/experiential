"""Teacher-side logprob scoring for on-policy distillation.

The teacher never generates during training; it scores the student's exact
sampled tokens. `TinkerTeacher` sends each `TrainDatum`'s full (unshifted)
token sequence through `tinker.SamplingClient.compute_logprobs` and keeps the
loss positions.

Indexing convention (verified against both the real SDK and the fakes):
`compute_logprobs(tokens)` returns one entry per input position, where entry p
is the logprob of token p given tokens < p, and entry 0 is None because the
first token has no context. The datum's `model_input_tokens` IS that full
sequence and its `loss_mask` names the sampled positions, so the returned rows
align index for index with the datum; `teacher_test.py` pins the alignment
against the fake sampler's echoed logprobs to guard any off-by-one.

The tinker SDK is an optional extra imported lazily (`uv sync --extra
distill`), mirroring `wmh.providers.tinker`; injecting a sampling client
(tests use `wmh.distill.fake_tinker.FakeSamplingClient`) avoids the SDK
entirely. The lazily built real client comes from `wmh.providers.tinker`'s
process-wide shared cache (one `tinker.ServiceClient` and one
`SamplingClient` per model string for the whole process), so teacher
construction never adds server-side sessions of its own.

Every SDK call is deadline-bounded (`wmh.distill.deadlines`): a wedged
session raises a retryable `TinkerDeadlineError` instead of hanging, and the
teacher drops its lazily built client on expiry, evicting the shared cache
entry, so the next score() call rebuilds a fresh session (an injected client
is never dropped).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol

from wmh.distill.config import TeacherConfig
from wmh.distill.data import TrainDatum
from wmh.distill.deadlines import TinkerDeadlineError, wait_with_deadline
from wmh.providers.base import ProviderKind, VerifyResult
from wmh.providers.tinker import evict_shared_sampling_client, shared_sampling_client

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


class TeacherClient(Protocol):
    """What the distillation loop needs from a teacher backend."""

    def score(self, datums: Sequence[TrainDatum]) -> list[list[float | None]]:
        """Per-position teacher logprob rows, aligned one to one with `datums`.

        Row entry p is the teacher's logprob of `model_input_tokens[p]` when p
        is a scoreable loss position (mask 1.0, p >= 1) and None everywhere
        else (context positions, and position 0, which has no context).
        """
        ...

    def verify(self) -> VerifyResult:
        """One cheap preflight probe; reports failure, never raises."""
        ...

    def usage(self) -> int:
        """Total tokens submitted for scoring so far (the teacher_prefill meter)."""
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

    `wmh.distill.fake_tinker.FakeSamplingClient` satisfies this directly;
    real `tinker.SamplingClient`s are adapted via `SdkLogprobScorer`.
    """

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        """Per-position logprobs for the sequence; entry 0 is None."""
        ...


class SdkLogprobScorer:
    """Adapts a real `tinker.SamplingClient` to the `LogprobScorer` seam."""

    def __init__(self, client: tinker.SamplingClient) -> None:
        self._client = client

    @property
    def sdk_client(self) -> tinker.SamplingClient:
        """The wrapped SDK client (shared-cache eviction compares its identity)."""
        return self._client

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        """One deadline-bounded compute_logprobs call on the full sequence.

        Raises:
            TinkerDeadlineError: If the deadline expires (the session is
                likely wedged; the caller should retry with a fresh one).
        """
        import tinker

        future = self._client.compute_logprobs(tinker.ModelInput.from_ints(token_ids))
        return wait_with_deadline("compute_logprobs", future)


class TinkerTeacher:
    """`TeacherClient` backed by a Tinker sampling client.

    Args:
        spec: The `[teacher]` section of the run config; `model` is the base
            model name and `checkpoint` optionally pins a `tinker://` weights
            path to serve the teacher from.
        sampling_client: Optional injected scorer (tests use the fakes in
            `wmh.distill.fake_tinker`; wrap a real `tinker.SamplingClient` in
            `SdkLogprobScorer`). When None, a real client is fetched lazily
            on first use from `wmh.providers.tinker`'s process-wide shared
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
        # cache path (`wmh.providers.tinker.shared_sampling_client`). One
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

        Mirrors `wmh.providers.base.verify_via_ping`: never raises, so
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
