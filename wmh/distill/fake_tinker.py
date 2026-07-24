"""Deterministic in-memory fakes of the Tinker SDK surface the distill loop uses.

These fakes mirror the shapes of the real SDK (tinker.ServiceClient,
tinker.TrainingClient, tinker.SamplingClient, tinker.types.Datum and
tinker.types.SampledSequence) structurally, without importing tinker at all,
so the test suite runs without the `distill` extra installed. Method names and
call shapes match the real clients closely enough that the distill loop can be
exercised end to end against them; the notable simplifications are documented
on each method (token ids are plain `list[int]` rather than ModelInput chunks,
results are returned directly rather than through futures).

Everything is deterministic: sampled tokens and logprobs are derived from
SHA-256 hashes of (seed, prompt ids, position), never from time or global
randomness, so a run replayed with the same inputs produces identical outputs.

The fakes also enforce the tokens-in-tokens-out (TITO) invariant that on-policy
distillation depends on: every sampled span trained on must be byte-identical
to a span some linked sampling client actually issued. FakeTrainingClient
raises AssertionError from forward_backward when a datum violates it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

_SAMPLED_TOKEN_BASE = 32
_SAMPLED_TOKEN_RANGE = 95
"""Sampled token ids stay in the printable ASCII range so decode() is total."""


def _digest(*parts: str) -> bytes:
    """A 32-byte SHA-256 digest of the given parts joined unambiguously."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()


def _ids_key(token_ids: list[int]) -> str:
    return ",".join(str(t) for t in token_ids)


def _derived_token(seed: str, prompt_ids: list[int], sample_index: int, position: int) -> int:
    digest = _digest("token", seed, _ids_key(prompt_ids), str(sample_index), str(position))
    value = int.from_bytes(digest[:4], "big")
    return _SAMPLED_TOKEN_BASE + (value % _SAMPLED_TOKEN_RANGE)


def _derived_logprob(*parts: str) -> float:
    """A deterministic pseudo-logprob in [-4.05, -0.05), from the given parts."""
    digest = _digest("logprob", *parts)
    value = int.from_bytes(digest[:4], "big")
    return -0.05 - (value % 4000) / 1000.0


class FakeTokenizer:
    """A tiny deterministic char-level tokenizer: token id = code point."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids (one token per character)."""
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back to text."""
        return "".join(chr(t) for t in token_ids)


@dataclass(frozen=True)
class FakeSampledSequence:
    """One sampled sequence, mirroring tinker.types.SampledSequence.

    `tokens` are the generated token ids and `logprobs[j]` is the logprob the
    sampler assigned to `tokens[j]`, aligned one to one.
    """

    tokens: list[int]
    logprobs: list[float]
    stop_reason: Literal["length", "stop"]


@dataclass(frozen=True)
class IssuedSample:
    """A record of one span a sampling client issued: the TITO ground truth."""

    prompt_ids: tuple[int, ...]
    sampled_ids: tuple[int, ...]
    logprobs: tuple[float, ...]


@dataclass
class _SpanLedger:
    """Issued-span records shared by a training client and its samplers."""

    records: list[IssuedSample] = field(default_factory=list)


class FakeDatum:
    """A training datum, mirroring tinker.types.Datum structurally.

    The real Datum carries a ModelInput plus tensor-valued loss_fn_inputs;
    here everything is plain lists. `model_input_tokens` is the full input
    sequence and the loss inputs are aligned with `target_tokens`.

    Args:
        model_input_tokens: The full input token sequence (all but the final
            target position, in the usual shifted-by-one layout; the fakes do
            not enforce that layout).
        target_tokens: Tokens the loss is computed over.
        weights: Per-target-token loss weights; positions with weight 0 are
            prompt/tool tokens outside the sampled spans. Defaults to all 1.0.
        advantages: Optional per-target-token advantages (importance_sampling).
        logprobs: Optional per-target-token behavior-policy logprobs.
    """

    def __init__(
        self,
        model_input_tokens: list[int],
        target_tokens: list[int],
        weights: list[float] | None = None,
        advantages: list[float] | None = None,
        logprobs: list[float] | None = None,
    ) -> None:
        self.model_input_tokens = list(model_input_tokens)
        self.target_tokens = list(target_tokens)
        self.weights = list(weights) if weights is not None else [1.0] * len(target_tokens)
        self.advantages = list(advantages) if advantages is not None else []
        self.logprobs = list(logprobs) if logprobs is not None else []
        if len(self.weights) != len(self.target_tokens):
            raise ValueError(
                f"weights length {len(self.weights)} does not match "
                f"target_tokens length {len(self.target_tokens)}"
            )

    def sampled_spans(self) -> list[tuple[int, ...]]:
        """Maximal contiguous runs of target tokens with nonzero weight."""
        spans: list[tuple[int, ...]] = []
        current: list[int] = []
        for token, weight in zip(self.target_tokens, self.weights, strict=True):
            if weight != 0.0:
                current.append(token)
            elif current:
                spans.append(tuple(current))
                current = []
        if current:
            spans.append(tuple(current))
        return spans


class FakeSamplingClient:
    """Deterministic stand-in for tinker.SamplingClient.

    Simplifications vs the real client: prompts are `list[int]` (the real
    client takes ModelInput), results are returned directly (no futures), and
    one call returns one sequence.

    Args:
        seed: Seed string, by convention the fake sampler weights path; all
            sampled tokens and logprobs derive from it.
        ledger: Shared issued-span ledger; samplers refreshed from the same
            training client share one ledger so TITO checks see all of them.
    """

    def __init__(self, seed: str, ledger: _SpanLedger | None = None) -> None:
        self.seed = seed
        self._ledger = ledger if ledger is not None else _SpanLedger()
        self.issued: list[IssuedSample] = []

    def sample(
        self,
        prompt_token_ids: list[int],
        max_tokens: int,
        temperature: float,
        stop: list[int] | list[str] | None = None,
        sample_index: int = 0,
    ) -> FakeSampledSequence:
        """Sample a deterministic sequence for the prompt and record it.

        Tokens and logprobs derive purely from a hash of (seed, prompt ids,
        sample_index, position): calling again with identical arguments
        returns an identical sequence. Pass distinct `sample_index` values to
        get distinct group members deterministically (the real SDK's
        num_samples analogue). `temperature` is accepted for signature parity
        and does not affect the fake's output.

        Args:
            prompt_token_ids: Prompt tokens the sample conditions on.
            max_tokens: Maximum number of tokens to generate.
            temperature: Accepted for parity; unused by the fake.
            stop: Optional stop token ids or stop strings (the real
                SamplingParams accepts both). A generated stop token is
                included in the output and ends generation with stop_reason
                "stop"; a stop string fires when the generation so far, decoded
                with the char-level FakeTokenizer convention, ends with it.
            sample_index: Deterministic nonce distinguishing group members.

        Returns:
            The sampled sequence with aligned per-token logprobs.
        """
        del temperature
        int_stops = {item for item in (stop or ()) if isinstance(item, int)}
        str_stops = [item for item in (stop or ()) if isinstance(item, str)]
        tokens: list[int] = []
        logprobs: list[float] = []
        stop_reason: Literal["length", "stop"] = "length"
        for position in range(max_tokens):
            token = _derived_token(self.seed, prompt_token_ids, sample_index, position)
            logprob = _derived_logprob(
                "issued", self.seed, _ids_key(prompt_token_ids), str(sample_index), str(position)
            )
            tokens.append(token)
            logprobs.append(logprob)
            if token in int_stops:
                stop_reason = "stop"
                break
            if str_stops:
                text = "".join(chr(t) for t in tokens)
                if any(text.endswith(item) for item in str_stops):
                    stop_reason = "stop"
                    break
        record = IssuedSample(
            prompt_ids=tuple(prompt_token_ids),
            sampled_ids=tuple(tokens),
            logprobs=tuple(logprobs),
        )
        self.issued.append(record)
        self._ledger.records.append(record)
        return FakeSampledSequence(tokens=tokens, logprobs=logprobs, stop_reason=stop_reason)

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        """Per-position logprobs for a full token sequence.

        Indexing convention (mirroring the real SDK's prompt_logprobs): index
        i holds the logprob of token i given tokens < i, and index 0 is None
        because the first token has no context (the real SDK likewise returns
        None where a logprob cannot be computed; we chose None over a
        placeholder to match it exactly).

        Positions covered by a previously issued sampled span, meaning the
        span's sampled ids appear in `token_ids` immediately after that span's
        exact prompt ids, echo the exact logprobs issued at sampling time
        (first matching record wins for overlaps). Every other position gets
        a deterministic hash-derived value from (seed, tokens <= i).

        Args:
            token_ids: The full sequence to score.

        Returns:
            One entry per input position; entry 0 is None.
        """
        n = len(token_ids)
        result: list[float | None] = [None] * n
        filled = [False] * n
        if n > 0:
            filled[0] = True  # position 0 stays None by convention
        for record in self._ledger.records:
            span_len = len(record.sampled_ids)
            prompt_len = len(record.prompt_ids)
            if span_len == 0:
                continue
            for start in range(prompt_len, n - span_len + 1):
                if tuple(token_ids[start : start + span_len]) != record.sampled_ids:
                    continue
                if tuple(token_ids[start - prompt_len : start]) != record.prompt_ids:
                    continue
                for offset in range(span_len):
                    position = start + offset
                    if not filled[position]:
                        result[position] = record.logprobs[offset]
                        filled[position] = True
        for position in range(1, n):
            if not filled[position]:
                result[position] = _derived_logprob(
                    "context", self.seed, _ids_key(token_ids[: position + 1])
                )
        return result


class FakeTrainingClient:
    """Deterministic stand-in for tinker.TrainingClient.

    Create via FakeServiceClient.create_lora_training_client. Simplifications
    vs the real client: results return directly (no APIFuture), datums are
    FakeDatum (plain lists rather than tensors), and optim_step takes a bare
    learning rate rather than AdamParams.
    """

    def __init__(self, service: FakeServiceClient, base_model: str, rank: int) -> None:
        self.base_model = base_model
        self.rank = rank
        self.step_count = 0
        self.forward_backward_calls: list[tuple[list[FakeDatum], str]] = []
        self.optim_step_lrs: list[float] = []
        self._service = service
        self._ledger = _SpanLedger()
        self._saved_states: dict[str, int] = {}
        self._state_counter = 0
        self._sampler_counter = 0

    def get_tokenizer(self) -> FakeTokenizer:
        """The deterministic char-level tokenizer for this fake model."""
        return FakeTokenizer()

    def forward_backward(self, datums: list[FakeDatum], loss_fn: str) -> None:
        """Record a training batch after asserting the TITO invariant.

        Every sampled span in every datum (maximal nonzero-weight run of
        target tokens) must exactly equal the sampled ids of some span a
        linked sampling client previously issued.

        Args:
            datums: The batch to train on.
            loss_fn: Loss function name (e.g. "importance_sampling"); recorded
                but not interpreted.

        Raises:
            AssertionError: If a sampled span was never issued by a linked
                sampler; the message names the datum, the span, and the first
                mismatching token position against the closest issued span.
        """
        issued = {record.sampled_ids for record in self._ledger.records}
        for datum_index, datum in enumerate(datums):
            for span in datum.sampled_spans():
                if span in issued:
                    continue
                raise AssertionError(self._tito_message(datum_index, span))
        self.forward_backward_calls.append((list(datums), loss_fn))

    def _tito_message(self, datum_index: int, span: tuple[int, ...]) -> str:
        """Build the TITO failure message naming the first mismatch position."""
        best: IssuedSample | None = None
        best_prefix = -1
        for record in self._ledger.records:
            prefix = 0
            for a, b in zip(span, record.sampled_ids, strict=False):
                if a != b:
                    break
                prefix += 1
            if prefix > best_prefix:
                best_prefix = prefix
                best = record
        if best is None:
            return (
                f"TITO violation in datum {datum_index}: sampled span of length "
                f"{len(span)} trained on, but no linked sampler issued any span"
            )
        mismatch = best_prefix
        if mismatch < len(span) and mismatch < len(best.sampled_ids):
            detail = (
                f"first mismatch at position {mismatch}: datum has {span[mismatch]}, "
                f"closest issued span has {best.sampled_ids[mismatch]}"
            )
        else:
            detail = (
                f"first mismatch at position {mismatch}: datum span length {len(span)} "
                f"vs closest issued span length {len(best.sampled_ids)}"
            )
        return f"TITO violation in datum {datum_index}: {detail}"

    def optim_step(self, learning_rate: float) -> None:
        """Apply one optimizer step: increments the step counter."""
        self.optim_step_lrs.append(learning_rate)
        self.step_count += 1

    def save_state(self) -> str:
        """Save training state, returning a fake tinker:// state path."""
        path = f"tinker://fake/state/{self._state_counter}"
        self._state_counter += 1
        self._saved_states[path] = self.step_count
        return path

    def load_state(self, path: str) -> None:
        """Restore a previously saved state.

        Args:
            path: A path returned by save_state on this client.

        Raises:
            ValueError: If the path was never saved by this client.
        """
        if path not in self._saved_states:
            raise ValueError(
                f"unknown state path {path!r}: it was never returned by save_state "
                "on this training client"
            )
        self.step_count = self._saved_states[path]

    def save_weights_for_sampler(self, name: str) -> str:
        """Save current weights for sampling, returning a fake sampler path.

        The path can be exchanged for a linked FakeSamplingClient via
        FakeServiceClient.create_sampling_client.
        """
        path = f"tinker://fake/sampler/{name}/{self._sampler_counter}"
        self._sampler_counter += 1
        self._service._register_sampler_path(path, self)
        return path

    def save_weights_and_get_sampling_client(self, name: str) -> FakeSamplingClient:
        """Save current weights and return a fresh linked sampling client.

        Each call yields a distinct sampler path (so a refreshed sampler
        samples different tokens) whose client shares this training client's
        span ledger (so TITO checks see every sampler's issued spans).
        """
        path = self.save_weights_for_sampler(name)
        return FakeSamplingClient(seed=path, ledger=self._ledger)


class FakeServiceClient:
    """Deterministic stand-in for tinker.ServiceClient."""

    def __init__(self) -> None:
        self._sampler_paths: dict[str, FakeTrainingClient] = {}

    def create_lora_training_client(self, base_model: str, rank: int = 32) -> FakeTrainingClient:
        """Create a fake LoRA training client for the given base model."""
        return FakeTrainingClient(service=self, base_model=base_model, rank=rank)

    def create_sampling_client(self, model_path: str) -> FakeSamplingClient:
        """Create a sampling client for a saved sampler path.

        Paths produced by a linked training client's save_weights_for_sampler
        yield clients that share that training client's span ledger; any other
        path (e.g. a base model name for a standalone teacher) yields an
        unlinked client seeded with the path.
        """
        training = self._sampler_paths.get(model_path)
        if training is not None:
            return FakeSamplingClient(seed=model_path, ledger=training._ledger)
        return FakeSamplingClient(seed=model_path)

    def _register_sampler_path(self, path: str, training: FakeTrainingClient) -> None:
        self._sampler_paths[path] = training
