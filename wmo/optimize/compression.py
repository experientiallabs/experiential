"""Segment-aware context compression behind the optimizer interface (D-COMPRESS seam).

A compressor removes low-information tokens from the model's input before the provider call.
The seam is SEGMENT-aware, not string-aware: the caller splits the conversation into
(cached-prefix, turn-local) segments and hands the compressor ONLY the segments it is allowed
to change. The provider prompt cache is prefix-matched, so recompressing an incumbent
conversation's cached prefix would forfeit ~0.9x discounts to save ~0.1x of tokens; keeping
that prefix out of the compressor's hands makes cache safety a property of the construction,
not a convention each compressor must remember.

This module ships the protocol plus the two reference implementations the seam is proven
with: `identity` (today's behavior, bit-for-bit) and `truncate` (the trivial ratio-matched
control every learned compressor must beat). Real compressors are chosen by the research
track and register here; nothing else in the harness hardcodes one.

Every compressor must be deterministic: the same segments and config always produce the same
output (append-stability of the serving path depends on it, and so does replaying a request
log). Token counts reported here are a deterministic chars/4 PROXY for accounting the
compressor's own effect; billable truth stays with the provider-reported `TokenUsage`.
"""

from __future__ import annotations

import math
import time
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from wmo.providers.base import Embedder

# The proxy tokenizer: ~4 chars per token, the industry rule of thumb. Deterministic and
# provider-agnostic; used only for the compressor's own raw-vs-compressed accounting.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Deterministic proxy token count of `text` (ceil of chars/4; 0 only for empty text)."""
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


class CompressionConfig(BaseModel):
    """Per-cluster compression choice carried on the policy artifact (D-COMPRESS shape).

    `aggressiveness` is a compressor-DEFINED dial in [0, 1], deliberately not a removal
    fraction. Two invariants bind every implementation: 0.0 is a strict bit-for-bit no-op, and
    the dial is monotone (a higher value never removes less than a lower one). Requiring an
    exact removal fraction instead would force per-input percentile selection, which is the
    cache-hostile selection rule this track rejected (it rewrites the already-emitted prefix
    every turn, see `Compressor.append_stable`); the cache-safe learned compressors are
    fixed-threshold ones, whose removal varies with the input by construction.

    The ACHIEVED removal ratio is therefore an outcome, not a setting: read it per call off
    `CompressionResult` (tokens_in_compressed / tokens_in_raw), and match ratio-matched controls
    on that achieved ratio rather than on the nominal dial.

    Aggressiveness is risk-tiered per cluster and learned within bounds by the research track.
    `compressor_version` records the version the config was fitted against (provenance);
    serving logs the version that actually ran.
    """

    compressor_id: str = Field(min_length=1)
    compressor_version: str = "1"
    aggressiveness: float = Field(default=0.0, ge=0.0, le=1.0)


class CompressionResult(BaseModel):
    """What one compress() call did: the output segments plus its own accounting.

    `segments` has the same length and order as the input (a compressor rewrites segments,
    it never merges, splits, or reorders them; the caller owns the conversation structure).
    Token counts use `estimate_tokens`; `cost_usd` and `latency_s` are the compressor's OWN
    inference cost and wall-clock, which sit inside effective cost per the track's rules.
    """

    segments: list[str]
    tokens_in_raw: int
    tokens_in_compressed: int
    latency_s: float
    cost_usd: float = 0.0


class CompressionStats(BaseModel):
    """Request-level accounting of one applied compression stage (D-SERVING-LOG / eval shape).

    `compressor_version` is the version that actually RAN (the config's copy is fit-time
    provenance). Token counts are whole-request `estimate_tokens` totals: what the input would
    have measured raw vs what was sent, so on-vs-off savings read straight off the record.
    """

    compressor_id: str
    compressor_version: str
    aggressiveness: float
    tokens_in_raw: int
    tokens_in_compressed: int
    latency_s: float
    cost_usd: float = 0.0


@runtime_checkable
class Compressor(Protocol):
    """The pluggable compressor seam.

    Three contracts bind an implementation: it is deterministic (same segments and config, same
    bytes out), it honors the `aggressiveness` dial's invariants (0.0 is a strict bit-for-bit
    no-op, and the dial is monotone; see `CompressionConfig`), and it attests `append_stable`
    truthfully.

    One `compress` call takes ALL of a request's mutable segments at once, so an implementation
    backed by a network endpoint pays one round trip per request rather than one per message,
    and can batch internally. Per-segment determinism still holds: a segment's output may not
    depend on which other segments accompanied it.

    `append_stable` is the implementation's ATTESTATION that appending a segment never rewrites
    the bytes of the segments already emitted (C1's audit calls this append-only; the selection
    rule decides it, not the scorer). It is a serving admission ticket, not a hint: C2's cache
    simulation measured churny full recompression at up to 2.65x the input cost of no
    compression at all on cached providers, because every turn forfeits the whole prompt-cache
    prefix to save a fraction of the tokens. v1 serves append-stable compressors only; a churny
    method needs turn-local-commit support that does not exist yet, so it is refused at mount
    rather than silently allowed to lose money.
    """

    id: str
    version: str
    append_stable: bool

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        """Compress the mutable segments; never sees the cached prefix by construction."""
        ...


class IdentityCompressor:
    """Compression off: returns every segment bit-for-bit. The seam's do-no-harm proof."""

    id = "identity"
    version = "1"
    append_stable = True  # it changes nothing, so it cannot churn anything

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        del config
        start = time.monotonic()
        raw = sum(estimate_tokens(segment) for segment in segments)
        return CompressionResult(
            segments=list(segments),
            tokens_in_raw=raw,
            tokens_in_compressed=raw,
            latency_s=time.monotonic() - start,
        )


class TruncateCompressor:
    """The trivial ratio-matched control: drop the trailing words of every segment.

    Keeps the leading ceil((1 - aggressiveness) * n) whitespace-delimited tokens of each
    segment, which is this compressor's own reading of the dial. Being structural, it can hit a
    removal fraction exactly; a learned compressor generally cannot (see `CompressionConfig`),
    which is why controls are matched on ACHIEVED ratio. Deterministic per segment, so an
    unchanged segment always compresses to the same bytes (append-stability). This is a CONTROL,
    not a method: a learned compressor that does not beat it at equal achieved ratio has learned
    nothing (the track's mandatory baseline).

    Append-stable by C1's round-0 semantics: head-keep truncation is head-absolute per segment,
    the kept prefix of a segment depends on nothing outside that segment, and a segment already
    emitted is never revisited. (C1's correction to the lit review applies here: the "ratio
    budgets are never append-only" rule is a fact about percentile SELECTION, not about
    head-keep truncation, whose kept prefix only ever extends.)
    """

    id = "truncate"
    version = "1"
    append_stable = True

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        start = time.monotonic()
        out: list[str] = []
        for segment in segments:
            words = segment.split()
            keep = math.ceil(len(words) * (1.0 - config.aggressiveness))
            out.append(segment if keep >= len(words) else " ".join(words[:keep]))
        return CompressionResult(
            segments=out,
            tokens_in_raw=sum(estimate_tokens(segment) for segment in segments),
            tokens_in_compressed=sum(estimate_tokens(segment) for segment in out),
            latency_s=time.monotonic() - start,
        )


_COMPRESSORS: dict[str, Compressor] = {
    IdentityCompressor.id: IdentityCompressor(),
    TruncateCompressor.id: TruncateCompressor(),
}


def register_compressor(compressor: Compressor) -> None:
    """Register a compressor implementation under its `id` (research-track entry point).

    The module docstring's contract: real compressors are chosen by the research track and
    register here. Registration is idempotent for the same object and refuses to silently
    replace a DIFFERENT implementation under an existing id, because a policy artifact
    referencing that id must always resolve to the bytes it was fitted against.

    The `append_stable` attestation is required at registration rather than at mount, so an
    implementation that forgot to state it fails in the researcher's own process instead of on
    the first served request.
    """
    if not isinstance(getattr(compressor, "append_stable", None), bool):
        raise ValueError(
            f"compressor '{compressor.id}' does not declare `append_stable`; a compressor must "
            "attest whether appending a segment rewrites already-emitted bytes (see the "
            "Compressor protocol). Measure it with the append-stability audit, then set the "
            "attribute."
        )
    existing = _COMPRESSORS.get(compressor.id)
    if existing is not None and existing is not compressor:
        raise ValueError(
            f"compressor id '{compressor.id}' is already registered by "
            f"{type(existing).__name__}; ids are stable policy references and cannot be "
            "silently rebound"
        )
    _COMPRESSORS[compressor.id] = compressor


class CompressingEmbedder:
    """Embeds the COMPRESSED form of each text, so a fit sees what serving will see.

    The fit-side half of representation consistency (C2's Q2 result). A routing bank and its
    novelty floor are geometry: fit them on raw task text and then query them with compressed
    text and the queries land farther from every bank row, the floor trips 10-13x more often,
    and the router abstains to the expensive fallback on 20-30% more requests. Wrapping the fit's
    embedder is all it takes to move the bank and the floor onto the served representation.

    Serving does NOT wrap its embedder: the compression stage has already rewritten the request
    by the time the router embeds it, so wrapping there would compress twice.

    One `compress` call per `embed` batch, not one per text: fitting a bank means compressing
    every fit scenario, which would otherwise be a round trip each against an endpoint-backed
    compressor.
    """

    def __init__(self, inner: Embedder, config: CompressionConfig) -> None:
        self._inner = inner
        self._config = config
        self._compressor = get_compressor(config.compressor_id)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed(self._compressor.compress(list(texts), self._config).segments)


def compression_signature(config: CompressionConfig | None) -> str:
    """How a compression config reads in an error message; None is spelled out as raw text."""
    if config is None:
        return "raw text (no compression)"
    return (
        f"compressor '{config.compressor_id}' version {config.compressor_version} "
        f"at aggressiveness {config.aggressiveness:g}"
    )


def same_compression(left: CompressionConfig | None, right: CompressionConfig | None) -> bool:
    """Whether two configs produce the same representation: same compressor, version, and level.

    None (raw) equals only None. Anything else is compared on the whole triple, because a
    version bump changes the emitted bytes exactly as a different id would.
    """
    if left is None or right is None:
        return left is None and right is None
    return (
        left.compressor_id == right.compressor_id
        and left.compressor_version == right.compressor_version
        and left.aggressiveness == right.aggressiveness
    )


def servable_compressor(config: CompressionConfig | None) -> Compressor | None:
    """The compressor `config` names, checked against the v1 serving rule (None: compress off).

    Raises when the id is unknown, or when the implementation does not attest append stability:
    a churny compressor recompresses the cached prefix on every turn, which C2 measured as a net
    LOSS of up to 2.65x on cached providers. Serving it would quietly cost more than compressing
    nothing, so the mount is refused instead.
    """
    if config is None:
        return None
    compressor = get_compressor(config.compressor_id)
    if compressor.version != config.compressor_version:
        # The id alone does not identify the bytes. A stamped version that this build cannot
        # produce means the artifact was fitted against a DIFFERENT implementation of the same
        # id, so its bank sits in that implementation's geometry: the same failure as serving a
        # different compressor entirely, and invisible without this check.
        raise ValueError(
            f"compressor '{config.compressor_id}' is version {compressor.version} in this "
            f"build, but the policy was fitted against version {config.compressor_version}. "
            "The bytes a compressor emits are versioned, so its routing evidence does not "
            "transfer across versions: refit under the running version "
            "(`wmo optimize route fit --compressor <id> --aggressiveness <a>`), or deploy the "
            "build whose compressor matches the artifact."
        )
    if not compressor.append_stable:
        raise ValueError(
            f"compressor '{config.compressor_id}' is not attested append-stable, so it cannot be "
            "served in v1: it rewrites the already-compressed prefix on every turn, which "
            "forfeits the provider prompt cache and measured as a net cost INCREASE (up to 2.65x) "
            "against no compression at all. Serve an append-stable compressor (identity, "
            "truncate), or wait for turn-local-commit support."
        )
    return compressor


def get_compressor(compressor_id: str) -> Compressor:
    """Resolve a compressor by id, or raise naming the known ids (fail at mount, not mid-call)."""
    compressor = _COMPRESSORS.get(compressor_id)
    if compressor is None:
        known = ", ".join(sorted(_COMPRESSORS))
        raise ValueError(
            f"unknown compressor '{compressor_id}'; known compressors: {known}. "
            "Register new compressors with wmo.optimize.compression.register_compressor "
            "before referencing them in a policy's compression config."
        )
    return compressor
