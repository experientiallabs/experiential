"""Segment-aware context compression behind the optimizer interface (D-COMPRESS seam).

A compressor removes low-information tokens from the model's input before the provider call.
The seam is SEGMENT-aware, not string-aware: the caller splits the conversation into
(cached-prefix, turn-local) segments and hands the compressor ONLY the segments it is allowed
to change. The provider prompt cache is prefix-matched, so recompressing an incumbent
conversation's cached prefix would forfeit ~0.9x discounts to save ~0.1x of tokens; keeping
that prefix out of the compressor's hands makes cache safety a property of the construction,
not a convention each compressor must remember.

WHICH segments are eligible is the config's `scope`, and it is decided in exactly one place:
`compress_conversation` (via `compressible_positions`) is the only path from a caller's messages
to a compressor, so scope selection and the unconditional task protection on top of it cannot be
re-implemented per call site. The task or question segment (the first user segment of a
conversation) is never compressed in any scope at any aggressiveness.

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
from collections.abc import Callable, Sequence
from typing import Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from wmo.providers.base import Embedder

# The proxy tokenizer: ~4 chars per token, the industry rule of thumb. Deterministic and
# provider-agnostic; used only for the compressor's own raw-vs-compressed accounting.
_CHARS_PER_TOKEN = 4

# Default `CompressionConfig.bulk_min_chars`: how large a user segment has to be before "bulk"
# scope treats it as pasted document rather than dialogue. 2000 chars is roughly 500 proxy
# tokens, well above any measured conversational turn and below every pasted-document segment
# in the corpora the scope was ruled on.
DEFAULT_BULK_MIN_CHARS = 2000

CompressionScope = Literal["conversation", "observations", "bulk", "all"]
"""Which segments of a conversation a config may rewrite (see `CompressionConfig`)."""

COMPRESSION_SCOPES: tuple[CompressionScope, ...] = (
    "conversation",
    "observations",
    "bulk",
    "all",
)
"""Every accepted scope, in escalating order. The one source `--scope`'s help renders from."""


def parse_compression_scope(value: str) -> CompressionScope:
    """Narrow a `--scope` string to the literal, naming the choices when it is not one of them.

    Raises:
        ValueError: `value` is not a scope, listing the ones that are.
    """
    if value not in COMPRESSION_SCOPES:
        raise ValueError(
            f"unknown compression scope {value!r}; choose one of {' | '.join(COMPRESSION_SCOPES)}"
        )
    return cast("CompressionScope", value)


SegmentKind = Literal["system", "user", "assistant", "observation"]
"""What one segment IS, in the seam's own vocabulary rather than any caller's role enum.

Serving maps OpenAI roles onto this (`tool` results are observations); the eval threading maps
provider roles onto it. Keeping the seam's vocabulary separate is what lets scope selection be
one shared rule instead of one per caller.
"""


def estimate_tokens(text: str) -> int:
    """Deterministic proxy token count of `text` (ceil of chars/4; 0 only for empty text)."""
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


class Segment(BaseModel):
    """One unit of conversation text as the compression seam sees it: what it is, and its bytes.

    Frozen because the scope rules read it and the choke point maps results back onto it
    positionally; a segment whose kind could be mutated between selection and rewrite would
    let a caller compress something scope selection had already excluded.
    """

    model_config = ConfigDict(frozen=True)

    kind: SegmentKind
    text: str


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

    `scope` is WHICH segments the dial is allowed to touch, and it is identity-bearing: two
    configs that differ only in scope are two different arms, not one (see `same_compression`).
    The four values, each with the measurement behind it:

    - "conversation" (the DEFAULT, and the only behavior that existed before this field):
      user-role segments. Every artifact, matrix, and policy written without a scope loads with
      its meaning unchanged.
    - "observations": tool results and other environment output only. User and assistant
      segments pass through verbatim. With a STOCK scorer this is the safe scope: compressing
      dialogue lost 6-17 accuracy points and INVERTED cost by 2.3x, while removing 40% of
      document/observation bulk held accuracy inside the noise floor.
    - "bulk": "observations" plus user segments of at least `bulk_min_chars`, i.e. pasted
      documents rather than typed turns. The other safe scope for a stock scorer.
    - "all": user turns (bar the first), assistant turns, and observations. Dialogue included, so
      it is the target scope for a DOMAIN-ADAPTED scorer at a calibrated threshold: at a fixed
      absolute threshold, a scorer that rates dialogue as high-keep preserves conversation on its
      own, so excluding dialogue structurally is redundant and costs the savings the adapted
      scorer could have made safely. Choosing it with a stock scorer reintroduces exactly the
      measured dialogue harm above, which is why it is a deliberate selection and never a default.

    Two exclusions are STAGE LAW rather than scope rules, so no scope reaches them at any
    aggressiveness (`compressible_positions`):

    1. The TASK: the first user segment of a conversation. The one genuine deletion harm this
       track ever measured was a compressed task statement.
    2. The SYSTEM PROMPT, for two measured reasons. Instructions are the single category with a
       demonstrated deletion harm (the Block question-side failure), and a system prompt is the
       endpoint owner's instruction surface (tool schemas, policies, format contracts) rather
       than content to shorten. It is also the maximally cacheable prefix segment, so after the
       ~0.1x cache-read discount compressing it buys almost nothing while risking all of that:
       the track's caching-beats-compression rule at its strongest point.

    `bulk_min_chars` is inert outside "bulk" scope, and identity treats it that way: it is
    compared only when the scope can act on it.
    """

    compressor_id: str = Field(min_length=1)
    compressor_version: str = "1"
    aggressiveness: float = Field(default=0.0, ge=0.0, le=1.0)
    scope: CompressionScope = "conversation"
    bulk_min_chars: int = Field(default=DEFAULT_BULK_MIN_CHARS, ge=1)


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
    provenance).
    """

    compressor_id: str
    compressor_version: str
    aggressiveness: float
    # ALL segments of the request, not just the ones the scope selected: what the whole input
    # would have measured raw, and what was actually sent. The provider bills the whole prompt,
    # so this is the pair that makes `1 - compressed/raw` the honest billed-input saving, which
    # is the number the admin view quotes. The compressor's own achieved keep ratio is a
    # DIFFERENT figure (compressed/raw over eligible segments only) and is not recorded here:
    # under a narrow scope it can look excellent while the request barely shrank, so reporting
    # it as the saving would overstate every scoped endpoint.
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

    Callers hand over as many segments per call as the implementation's cap allows, chunked
    deterministically (`compress_segments`), so a network-backed implementation pays as few
    round trips as it can rather than one per message. Per-segment determinism still holds and
    is what makes chunking safe: a segment's output may not depend on which other segments
    accompanied it, so where the chunk boundaries fall cannot change the bytes.

    `max_segments_per_call` is the OPTIONAL cap (absent or None = unlimited): the most segments
    one call may carry, which for a served implementation is whatever its server enforces.
    Declaring it is how a compressor gets chunked instead of 413'd.

    A `compress` call must return EXACTLY as many segments as it was given, in order. A
    compressor rewrites segments; it never merges, splits, drops, or reorders them, because the
    caller owns the conversation structure and maps the result back onto it positionally.

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


type CompressorFactory = Callable[[], Compressor]
"""Builds a compressor on first use (see `register_compressor_factory`)."""


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

# Compressors that are too expensive to build at import (see `register_compressor_factory`).
# Constructed on first resolution, then they live in `_COMPRESSORS` like any other.
_COMPRESSOR_FACTORIES: dict[str, CompressorFactory] = {}


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
    segment_batch_limit(compressor)  # a malformed cap fails here, not on the first big batch
    existing = _COMPRESSORS.get(compressor.id)
    if existing is not None and existing is not compressor:
        raise ValueError(
            f"compressor id '{compressor.id}' is already registered by "
            f"{type(existing).__name__}; ids are stable policy references and cannot be "
            "silently rebound"
        )
    _COMPRESSORS[compressor.id] = compressor


def segment_batch_limit(compressor: Compressor) -> int | None:
    """The most segments `compressor` accepts in one call, or None for unlimited.

    Optional by design: a local compressor has no reason to declare a cap, and every
    implementation written before the attribute existed keeps working as unlimited. A served
    one declares whatever its server enforces, which is how it gets chunked instead of refused.
    """
    limit = getattr(compressor, "max_segments_per_call", None)
    if limit is None:
        return None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError(
            f"compressor '{compressor.id}' declares max_segments_per_call={limit!r}; it must be "
            "a positive integer, or None for no limit"
        )
    return limit


def compress_segments(
    compressor: Compressor, segments: list[str], config: CompressionConfig
) -> CompressionResult:
    """Run `segments` through `compressor` in as few calls as its cap allows, and check the shape.

    CHUNKING. A served compressor caps how many segments one request may carry (the endpoint
    rejects oversized batches outright), and both callers can exceed any such cap easily: a bank
    fit compresses every scenario in the matrix. Chunking is safe precisely because the protocol
    requires per-segment determinism: a segment's output cannot depend on which other segments
    travelled with it, so where the boundaries fall cannot change the bytes. For the compressors
    this matters for, that property is measured rather than assumed (fixed-threshold selection
    is per token, and fp32 output was verified batch-invariant at batch 1/4/16, which is why
    serving pins fp32).

    SHAPE. The protocol says a call returns exactly as many segments as it was given, in order,
    and nothing enforced it: one short and the caller's zip ran out mid-transcript, surfacing as
    an anonymous 502; one long and the extra was silently discarded, which serves a transcript
    nobody checked. Both are now a named error against the compressor that did it.
    """
    limit = segment_batch_limit(compressor)
    size = len(segments) if limit is None else limit
    out: list[str] = []
    tokens_in_raw = 0
    tokens_in_compressed = 0
    latency_s = 0.0
    cost_usd = 0.0
    for start in range(0, len(segments), max(size, 1)):
        chunk = segments[start : start + max(size, 1)]
        result = compressor.compress(chunk, config)
        if len(result.segments) != len(chunk):
            raise ValueError(
                f"compressor '{compressor.id}' returned {len(result.segments)} segments for "
                f"{len(chunk)}; a compressor rewrites segments in place and must return exactly "
                "as many as it was given, in order (the caller maps them back positionally)"
            )
        out.extend(result.segments)
        tokens_in_raw += result.tokens_in_raw
        tokens_in_compressed += result.tokens_in_compressed
        latency_s += result.latency_s
        cost_usd += result.cost_usd
    return CompressionResult(
        segments=out,
        tokens_in_raw=tokens_in_raw,
        tokens_in_compressed=tokens_in_compressed,
        latency_s=latency_s,
        cost_usd=cost_usd,
    )


def compressible_positions(
    segments: Sequence[Segment], config: CompressionConfig
) -> tuple[int, ...]:
    """The indices of `segments` this config lets a compressor rewrite, in order.

    THE TWO PROTECTIONS are enforced here and nowhere else, which is the point of routing every
    caller through this function. Neither is reachable by a config value, a new scope, or a
    caller that forgets to check:

    - the TASK, i.e. the first user segment of a conversation. Compressing a task statement is
      the one deletion this track measured real harm from.
    - the SYSTEM PROMPT, every one of them. It is the endpoint owner's instruction surface, and
      instructions are the category with the demonstrated deletion harm; it is also the
      maximally cacheable prefix segment, so compressing it trades a ~0.1x cache-read discount
      for a fraction of its tokens.

    Everything else follows `config.scope`; see `CompressionConfig` for what the four values
    mean and what was measured behind them. "all" therefore spans user turns after the first,
    assistant turns, and observations: it stops the SCOPE deciding what dialogue is worth
    keeping and hands that to a calibrated scorer, without reaching either protected kind.
    """
    protected = next((index for index, seg in enumerate(segments) if seg.kind == "user"), None)
    selected: list[int] = []
    for index, segment in enumerate(segments):
        if index == protected or segment.kind == "system":
            continue
        match config.scope:
            case "conversation":
                eligible = segment.kind == "user"
            case "observations":
                eligible = segment.kind == "observation"
            case "bulk":
                eligible = segment.kind == "observation" or (
                    segment.kind == "user" and len(segment.text) >= config.bulk_min_chars
                )
            case "all":
                eligible = True
        if eligible:
            selected.append(index)
    return tuple(selected)


class ConversationCompression(BaseModel):
    """What one `compress_conversation` call did: the segment texts to send, plus its own bill.

    `texts` has one entry per input segment, in order, so a caller maps it back onto whatever
    message objects it built the segments from. Segments the scope did not select (and every
    segment before `mutable_from`) come back byte-identical.
    """

    texts: list[str]
    cost_usd: float = 0.0


def compress_conversation(
    compressor: Compressor,
    segments: Sequence[Segment],
    config: CompressionConfig,
    *,
    mutable_from: int = 0,
) -> ConversationCompression:
    """Compress one conversation: the single path from a caller's messages to a compressor.

    Both production callers (serving's compression stage and the eval threading in
    `wmo.env.closed_loop`) come through here rather than selecting segments themselves, so
    task protection and scope selection are one rule with one test surface instead of two
    implementations that can drift.

    `segments` is the WHOLE conversation, always, even when only its tail may be rewritten.
    Scope selection is positional (the first user segment is the task), so a caller that handed
    over only the window it wanted compressed would shift that position and hand the task
    to the compressor on the very first turn a prefix was reused.

    `mutable_from` is the index the rewritable window starts at: everything before it is
    returned verbatim. Serving passes the length of the compressed prefix it is reusing, which
    is what keeps the provider-visible prefix byte-identical across turns (and the prompt cache
    alive). The default of 0 recompresses the whole conversation, which per-segment determinism
    makes byte-identical anyway.

    ONE compressor call carries every selected segment, so an endpoint-backed compressor pays
    one round trip per conversation rather than one per message; `compress_segments` chunks that
    to whatever cap the implementation declares.
    """
    texts = [segment.text for segment in segments]
    positions = [
        index for index in compressible_positions(segments, config) if index >= mutable_from
    ]
    if not positions:
        return ConversationCompression(texts=texts)
    result = compress_segments(compressor, [texts[index] for index in positions], config)
    for index, rewritten in zip(positions, result.segments, strict=True):
        texts[index] = rewritten
    return ConversationCompression(texts=texts, cost_usd=result.cost_usd)


class CompressingEmbedder:
    """Embeds the COMPRESSED form of each text. EXPLICIT USE ONLY: no fit reaches for this.

    Kept as a tool, not a policy. It was the fit-side half of representation consistency until
    the decision-bearing query turned out to be raw (see `routed_text_embedder`, which is what
    every fit and report goes through and which no longer wraps anything). Research code that
    deliberately wants a compressed-geometry bank constructs this directly and owns the reason;
    production fits must not, because a bank in a geometry serving does not query is the C2 Q2
    failure whichever direction it points in.

    Serving does not wrap its embedder either: the compression stage has already rewritten the
    request by the time the router embeds it, so wrapping there would compress twice.

    As few compressor calls per `embed` batch as the compressor's cap allows, not one per text:
    fitting a bank means compressing every fit scenario, which would otherwise be a round trip
    each against an endpoint-backed compressor. A matrix routinely has more scenarios than a
    served compressor accepts per call, so this goes through `compress_segments` to be chunked.
    """

    def __init__(self, inner: Embedder, config: CompressionConfig) -> None:
        self._inner = inner
        self._config = config
        self._compressor = get_compressor(config.compressor_id)

    def embed(self, texts: list[str]) -> list[list[float]]:
        compressed = compress_segments(self._compressor, list(texts), self._config)
        return self._inner.embed(compressed.segments)


# The scopes under which a fit embeds COMPRESSED text. Deliberately empty: see
# `routed_text_embedder` for why the decision-bearing query is raw in every scope. Kept as the one
# named place the rule lives, so adopting a different serving contract is an edit here rather than
# a hunt through the fit sites.
_ROUTED_TEXT_SCOPES: frozenset[CompressionScope] = frozenset()


def routed_text_embedder(inner: Embedder, config: CompressionConfig | None) -> Embedder:
    """The embedder a fit (or a report replay) must use for `config`'s arm: always the raw one.

    Representation consistency is about the query that DECIDES something, and for a knn endpoint
    that query is raw whatever the scope, for three reasons that compound:

    - the bank embeds bare TASK STATEMENTS, one row per fit scenario;
    - serving routes on the last user turn (`wmo.serving.chat._routable_text`), and on turn 1 of
      every conversation that turn IS the task, which stage law never compresses;
    - conversation affinity is sticky by default, so turns 2+ reuse the incumbent and never
      consult the bank at all.

    So the decision-bearing query is the raw task, and a bank fitted through the compressor would
    be queried with raw text essentially always: the C2 Q2 failure (queries land farther from every
    bank row, the novelty floor trips 10-13x more often, routing collapses to the expensive
    fallback, cost up 11-41% while accuracy sits flat to negative) CAUSED by wrapping rather than
    prevented by it. Hence banks are fitted raw, in every scope. This adopts the co-signed
    route-on-raw direction early and matches what serving actually embeds when it decides.

    Arm identity is untouched: `fit_compression` still stamps the whole config, so both mount gates
    and the measured-arm gate behave exactly as before. The wrap decision was never part of
    identity.

    DOCUMENTED RESIDUAL: an endpoint with stickiness OFF, under "conversation" or "all" scope,
    embeds compressed text on turns 2+ against this raw bank. That is the narrow remaining
    mismatch, it needs a non-default serving configuration to reach, and the uniform fix is the
    serving-side route-on-raw change (route on the RAW last user turn, so the routed-on
    representation is raw on every turn) which belongs to the routing co-sign round.
    """
    if config is None or config.scope not in _ROUTED_TEXT_SCOPES:
        return inner
    return CompressingEmbedder(inner, config)


def compression_signature(config: CompressionConfig | None) -> str:
    """How a compression config reads in an error message; None is spelled out as raw text.

    This string is also an ARM IDENTITY: sweep plans and the `optimize model` stage fingerprints
    persist it and compare it verbatim, so two configs that emit different bytes must render
    differently. Scope therefore appears here, but only when it departs from the default, and
    `bulk_min_chars` only in the scope that reads it. The rendering stays injective either way,
    and a fingerprint recorded before scope existed keeps matching the default-scope arm that
    wrote it instead of invalidating every resume manifest on disk.
    """
    if config is None:
        return "raw text (no compression)"
    rendered = (
        f"compressor '{config.compressor_id}' version {config.compressor_version} "
        f"at aggressiveness {config.aggressiveness:g}"
    )
    if config.scope == "bulk":
        return f"{rendered}, scope bulk over {config.bulk_min_chars} chars"
    if config.scope != "conversation":
        return f"{rendered}, scope {config.scope}"
    return rendered


def same_compression(left: CompressionConfig | None, right: CompressionConfig | None) -> bool:
    """Whether two configs produce the same representation: compressor, version, level, and scope.

    None (raw) equals only None. Anything else is compared field by field, because each one
    changes the emitted bytes: a version bump changes them exactly as a different id would, and
    a scope change moves compression onto entirely different segments (a conversation-scope arm
    and an observations-scope arm at the same dial are two arms, and a routing bank fitted under
    one is not evidence for the other).

    `bulk_min_chars` is compared only in "bulk" scope, which is the only scope that reads it.
    Comparing an inert field would refuse two arms that provably emit identical bytes.
    """
    if left is None or right is None:
        return left is None and right is None
    if left.scope == "bulk" and left.bulk_min_chars != right.bulk_min_chars:
        return False
    return (
        left.compressor_id == right.compressor_id
        and left.compressor_version == right.compressor_version
        and left.aggressiveness == right.aggressiveness
        and left.scope == right.scope
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


def resolve_compression(
    compressor_id: str,
    aggressiveness: float,
    *,
    scope: CompressionScope = "conversation",
    bulk_min_chars: int = DEFAULT_BULK_MIN_CHARS,
) -> CompressionConfig:
    """The config a `--compressor` flag names, stamped with the version that will actually RUN.

    Every command that turns a compressor id into a config comes through here (`route sweep`,
    `route fit`, `optimize model`), because two things have to happen that the config's own shape
    does not do. The version is read off the resolved implementation rather than defaulted, since
    that is what the field means: the version this config was fitted against. Defaulting it to "1"
    made every fit against a version-bumped compressor stamp a lie, which the mount gate would
    then correctly refuse with a remedy no CLI has a flag to carry out. And the result is checked
    against the v1 serving rule right here, so an arm that could never be mounted fails at the
    boundary instead of after a sweep has been paid for.

    `scope` and `bulk_min_chars` default to today's behavior, so every existing caller resolves
    the arm it always did. They are here rather than only on the model so that a scoped arm also
    picks up the running version stamp and the servability check.

    Raises:
        ValueError: The id is unknown, or the implementation cannot be served, naming which.
    """
    resolved = get_compressor(compressor_id)
    config = CompressionConfig(
        compressor_id=compressor_id,
        compressor_version=resolved.version,
        aggressiveness=aggressiveness,
        scope=scope,
        bulk_min_chars=bulk_min_chars,
    )
    servable_compressor(config)
    return config


def register_compressor_factory(compressor_id: str, factory: CompressorFactory) -> None:
    """Register a compressor to be CONSTRUCTED on first use, not at import.

    The entry point for a compressor that cannot be built for free. A served one has to reach
    its server to know what it is (the endpoint client verifies the server's selection rule
    before it will attest `append_stable`, since attesting on faith is how a churny compressor
    gets served), and doing that at import time would put a network call, a credential lookup,
    and a failure mode into every `import wmo`. Importing stays side-effect-free; the cost and
    the risk move to the first policy that actually names the compressor.

    `factory` runs at most once, on SUCCESS, and its result is registered normally (attestation
    checked, id rebinding refused). Failures are never cached: a factory that raises propagates
    with the id named and stays registered, so an endpoint that was down or an env var that was
    unset at first resolution can be fixed and retried rather than poisoning the id for the life
    of the process.

    Rebinding is refused the same way `register_compressor` refuses it, and against BOTH
    registries: an id that already resolves to a live instance cannot be shadowed by a factory
    (a test that registered a fake must not be silently replaced by the real thing), and a
    second factory cannot displace a first.
    """
    if compressor_id in _COMPRESSORS:
        raise ValueError(
            f"compressor id '{compressor_id}' is already registered by "
            f"{type(_COMPRESSORS[compressor_id]).__name__}; ids are stable policy references "
            "and cannot be silently rebound"
        )
    existing = _COMPRESSOR_FACTORIES.get(compressor_id)
    if existing is not None and existing is not factory:
        raise ValueError(
            f"compressor id '{compressor_id}' already has a registered factory; ids are stable "
            "policy references and cannot be silently rebound"
        )
    _COMPRESSOR_FACTORIES[compressor_id] = factory


def registered_compressor_ids() -> tuple[str, ...]:
    """Every compressor id `--compressor` accepts, built and lazily-registered alike.

    The one source of that list. `--compressor`'s help text renders it rather than spelling the
    ids out, because a hand-written enumeration goes stale the moment a compressor is registered:
    `llmlingua2-endpoint` shipped as a lazy factory and the help kept advertising two ids for
    several releases while the flag accepted three.
    """
    return tuple(sorted({*_COMPRESSORS, *_COMPRESSOR_FACTORIES}))


def get_compressor(compressor_id: str) -> Compressor:
    """Resolve a compressor by id, or raise naming the known ids (fail at mount, not mid-call)."""
    compressor = _COMPRESSORS.get(compressor_id)
    if compressor is not None:
        return compressor
    factory = _COMPRESSOR_FACTORIES.get(compressor_id)
    if factory is not None:
        try:
            built = factory()
        except Exception as exc:
            # Name the compressor: the bare failure from a factory that reached out to a server
            # and could not verify it says nothing about which policy field caused it.
            raise ValueError(
                f"compressor '{compressor_id}' could not be constructed: {exc}"
            ) from exc
        if built.id != compressor_id:
            # Otherwise the factory's compressor lands in the registry under ITS id while this
            # lookup hands it back for a different one, so a policy naming the registered id
            # gets an implementation that does not answer to it.
            raise ValueError(
                f"the factory registered for compressor '{compressor_id}' built one with id "
                f"'{built.id}'; a factory must build the compressor it was registered under"
            )
        register_compressor(built)
        return built
    known = ", ".join(registered_compressor_ids())
    raise ValueError(
        f"unknown compressor '{compressor_id}'; known compressors: {known}. "
        "Register new compressors with wmo.optimize.compression.register_compressor "
        "(or register_compressor_factory, for one that must be constructed on first use) "
        "before referencing them in a policy's compression config."
    )
