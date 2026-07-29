"""Tests for the segment-aware compression seam (protocol, identity, truncate, registry)."""

from __future__ import annotations

from typing import cast

import pytest

from wmo.optimize.compression import (
    DEFAULT_BULK_MIN_CHARS,
    CompressingEmbedder,
    CompressionConfig,
    CompressionResult,
    CompressionScope,
    Compressor,
    IdentityCompressor,
    Segment,
    TruncateCompressor,
    compress_conversation,
    compress_segments,
    compressible_positions,
    compression_signature,
    estimate_tokens,
    get_compressor,
    register_compressor,
    register_compressor_factory,
    registered_compressor_ids,
    resolve_compression,
    routed_text_embedder,
    same_compression,
    segment_batch_limit,
    servable_compressor,
)


def test_identity_returns_segments_bit_for_bit() -> None:
    segments = ["  leading space", "tabs\tand\nnewlines ", ""]
    result = IdentityCompressor().compress(
        segments, CompressionConfig(compressor_id="identity", aggressiveness=1.0)
    )
    # Bit-for-bit even at max aggressiveness: identity is the compression-off contract.
    assert result.segments == segments
    assert result.tokens_in_compressed == result.tokens_in_raw
    assert result.cost_usd == 0.0


def test_truncate_drops_the_trailing_fraction() -> None:
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    result = TruncateCompressor().compress(["one two three four", "a b"], config)
    assert result.segments == ["one two", "a"]
    assert result.tokens_in_compressed < result.tokens_in_raw


def test_truncate_at_zero_aggressiveness_is_a_no_op() -> None:
    segments = ["exact bytes  preserved\twhen nothing is removed"]
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.0)
    result = TruncateCompressor().compress(segments, config)
    # keep >= len(words) short-circuits to the original string, whitespace intact.
    assert result.segments == segments
    assert result.tokens_in_compressed == result.tokens_in_raw


def test_truncate_is_deterministic_per_segment() -> None:
    # Append-stability: an unchanged segment compresses to the same bytes on every call,
    # regardless of what other segments accompany it.
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.3)
    compressor = TruncateCompressor()
    alone = compressor.compress(["stable segment with several words here"], config)
    with_neighbor = compressor.compress(
        ["stable segment with several words here", "another later segment"], config
    )
    assert alone.segments[0] == with_neighbor.segments[0]


def test_compressors_preserve_segment_count_and_order() -> None:
    segments = ["first has words", "second", "", "fourth trailing"]
    for compressor_id in ("identity", "truncate"):
        config = CompressionConfig(compressor_id=compressor_id, aggressiveness=0.5)
        result = get_compressor(compressor_id).compress(segments, config)
        assert len(result.segments) == len(segments), compressor_id


def test_registry_resolves_known_ids_and_satisfies_the_protocol() -> None:
    for compressor_id in ("identity", "truncate"):
        compressor = get_compressor(compressor_id)
        assert isinstance(compressor, Compressor)
        assert compressor.id == compressor_id


def test_registry_rejects_unknown_id_with_guidance() -> None:
    # The known-ids list is open: importing `wmo.optimize` registers a factory for the endpoint
    # client, and any research compressor may register another. So assert the reference
    # implementations are named, rather than pinning the exact set and breaking on every
    # legitimate registration.
    with pytest.raises(ValueError, match="unknown compressor 'llmzip'") as caught:
        get_compressor("llmzip")
    assert "identity" in str(caught.value)
    assert "truncate" in str(caught.value)


def test_registered_ids_include_the_lazily_registered_endpoint_compressor() -> None:
    """The list `--compressor`'s help renders, so a registered id cannot go undocumented.

    `llmlingua2-endpoint` is registered as a FACTORY at import of `wmo.optimize`, which is why
    the two hand-written help strings kept advertising only identity and truncate.
    """
    ids = registered_compressor_ids()
    assert {"identity", "truncate", "llmlingua2-endpoint"} <= set(ids)
    assert list(ids) == sorted(ids)


def test_aggressiveness_is_bounded() -> None:
    with pytest.raises(ValueError):
        CompressionConfig(compressor_id="truncate", aggressiveness=1.5)
    with pytest.raises(ValueError):
        CompressionConfig(compressor_id="truncate", aggressiveness=-0.1)


def test_estimate_tokens_is_ceil_of_quarter_chars() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


class _CountingEmbedder:
    """Records exactly which texts it was asked to embed."""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.seen.append(list(texts))
        return [[float(len(text))] for text in texts]


class _Churny:
    """A compressor that admits it rewrites already-emitted bytes (C1's percentile family)."""

    id = "churny-compression-test"
    version = "1"
    append_stable = False

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        del config
        raw = sum(estimate_tokens(segment) for segment in segments)
        return CompressionResult(
            segments=list(segments), tokens_in_raw=raw, tokens_in_compressed=raw, latency_s=0.0
        )


_CHURNY = _Churny()


def test_the_reference_compressors_attest_append_stability() -> None:
    # Both are servable in v1: identity changes nothing, and truncate is head-absolute per
    # segment (C1 round 0 measured churn 0.000 on all five corpora for head-keep truncation).
    assert IdentityCompressor.append_stable is True
    assert TruncateCompressor.append_stable is True
    assert servable_compressor(CompressionConfig(compressor_id="truncate")) is not None
    assert servable_compressor(None) is None


def test_a_churny_compressor_is_not_servable() -> None:
    register_compressor(_CHURNY)
    with pytest.raises(ValueError, match="not attested append-stable"):
        servable_compressor(CompressionConfig(compressor_id=_CHURNY.id))


def test_register_compressor_refuses_to_rebind_an_id() -> None:
    register_compressor(_CHURNY)
    register_compressor(_CHURNY)  # idempotent for the same object
    assert get_compressor(_CHURNY.id) is _CHURNY

    class _Impostor:
        id = _CHURNY.id
        version = "2"
        append_stable = True

        def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
            raise NotImplementedError

    with pytest.raises(ValueError, match="already registered"):
        register_compressor(cast("Compressor", _Impostor()))


def test_same_compression_compares_the_whole_triple() -> None:
    base = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    assert same_compression(base, base.model_copy())
    assert same_compression(None, None)
    assert not same_compression(base, None)
    assert not same_compression(base, base.model_copy(update={"aggressiveness": 0.25}))
    # A version bump changes the emitted bytes exactly as a different id would.
    assert not same_compression(base, base.model_copy(update={"compressor_version": "2"}))


# ---------------------------------------------------------------- scope, identity, and the task


def test_two_configs_differing_only_in_scope_are_not_the_same_arm() -> None:
    # Scope moves compression onto entirely different segments, so a bank fitted under one scope
    # is not evidence for another. Identity has to see that, or the mount and fit gates wave it
    # through and the artifact serves a representation it was never measured in.
    conversation = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    observations = conversation.model_copy(update={"scope": "observations"})
    assert not same_compression(conversation, observations)
    assert not same_compression(observations, conversation)
    assert compression_signature(conversation) != compression_signature(observations)
    # bulk_min_chars is identity-bearing only where it can act, which is "bulk" scope alone.
    assert same_compression(conversation, conversation.model_copy(update={"bulk_min_chars": 10}))
    bulk = conversation.model_copy(update={"scope": "bulk"})
    assert not same_compression(bulk, bulk.model_copy(update={"bulk_min_chars": 10}))


def test_a_default_scope_signature_is_unchanged_so_recorded_fingerprints_still_match() -> None:
    # The signature is persisted verbatim in sweep plans and `optimize model` stage fingerprints.
    # Rendering the default scope would invalidate every one of them on disk for no new
    # information, so scope shows up only when it departs from the default.
    default = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    assert compression_signature(default) == "compressor 'truncate' version 1 at aggressiveness 0.5"
    assert compression_signature(None) == "raw text (no compression)"
    bulk = default.model_copy(update={"scope": "bulk", "bulk_min_chars": 4096})
    assert compression_signature(bulk).endswith("scope bulk over 4096 chars")


def test_resolve_compression_carries_scope_and_stamps_the_running_version() -> None:
    config = resolve_compression("truncate", 0.5, scope="observations", bulk_min_chars=1234)
    assert config.scope == "observations"
    assert config.bulk_min_chars == 1234
    assert config.compressor_version == TruncateCompressor.version


_CONVERSATION = [
    Segment(kind="system", text="a system prompt"),
    Segment(kind="user", text="the task statement"),
    Segment(kind="assistant", text="an earlier reply"),
    Segment(kind="observation", text="a tool result"),
    Segment(kind="user", text="a follow-up turn"),
]


def test_the_task_and_the_system_prompt_are_protected_in_every_scope() -> None:
    # The two stage-law exclusions, at the top of the dial with the bulk threshold as low as it
    # goes. Index 1 is the task (the measured deletion harm), index 0 is the system prompt (the
    # endpoint owner's instruction surface, and the maximally cacheable prefix segment).
    scopes: tuple[CompressionScope, ...] = ("conversation", "observations", "bulk", "all")
    for scope in scopes:
        config = CompressionConfig(
            compressor_id="truncate", aggressiveness=1.0, scope=scope, bulk_min_chars=1
        )
        positions = compressible_positions(_CONVERSATION, config)
        assert 1 not in positions, scope
        assert 0 not in positions, scope


def test_a_second_system_segment_is_protected_too_not_just_the_first() -> None:
    # The system rule is by KIND, not by position: a transcript carrying a second system turn
    # (clients do send them mid-conversation) must not have that one compressed either.
    segments = [
        Segment(kind="user", text="the task statement"),
        Segment(kind="system", text="a mid-conversation instruction update"),
        Segment(kind="user", text="a follow-up turn"),
    ]
    config = CompressionConfig(compressor_id="truncate", aggressiveness=1.0, scope="all")
    assert compressible_positions(segments, config) == (2,)


def test_each_scope_selects_the_segments_it_names() -> None:
    def positions(scope: CompressionScope, *, bulk_min_chars: int = 2000) -> tuple[int, ...]:
        return compressible_positions(
            _CONVERSATION,
            CompressionConfig(compressor_id="truncate", scope=scope, bulk_min_chars=bulk_min_chars),
        )

    assert positions("conversation") == (4,)  # user turns, minus the task
    assert positions("observations") == (3,)  # tool output only
    # "bulk" is observations plus LARGE user content; the follow-up turn is short prose, so at the
    # default threshold it is left alone and only the observation is selected.
    assert positions("bulk") == (3,)
    assert positions("bulk", bulk_min_chars=1) == (3, 4)
    # "all" spans user turns after the first, assistant turns, and observations: it stops the SCOPE
    # deciding what dialogue is worth keeping and hands that to a calibrated scorer, while both
    # stage-law exclusions (index 1 the task, index 0 the system prompt) still hold.
    assert positions("all") == (2, 3, 4)


@pytest.mark.parametrize(
    ("length", "eligible"),
    [(DEFAULT_BULK_MIN_CHARS - 1, False), (DEFAULT_BULK_MIN_CHARS, True), (2001, True)],
)
def test_the_bulk_threshold_is_inclusive_at_exactly_its_default(
    length: int, eligible: bool
) -> None:
    # Pinned at the boundary because "at least bulk_min_chars" is a decision, not an accident: a
    # segment of exactly 2000 chars IS bulk. An off-by-one here silently moves which user content a
    # bulk arm compresses, and no accuracy measurement would attribute the change to this.
    segments = [
        Segment(kind="user", text="the task statement"),
        Segment(kind="user", text="x" * length),
    ]
    config = CompressionConfig(compressor_id="truncate", scope="bulk")
    assert compressible_positions(segments, config) == ((1,) if eligible else ())


def test_the_task_is_byte_identical_through_the_stage_at_full_aggressiveness() -> None:
    config = CompressionConfig(compressor_id="truncate", aggressiveness=1.0)
    result = compress_conversation(get_compressor("truncate"), _CONVERSATION, config)
    assert result.texts[1] == "the task statement"  # verbatim, at the top of the dial
    assert result.texts[0] == "a system prompt"
    assert result.texts[2] == "an earlier reply"
    assert result.texts[3] == "a tool result"
    assert result.texts[4] == ""  # a later turn, fully removed: the dial does work


def test_mutable_from_leaves_the_reused_prefix_verbatim() -> None:
    # Serving reuses an already-compressed prefix rather than recompressing it, which is what
    # keeps the provider's prompt cache alive. Passing the WHOLE conversation is still required:
    # scope selection is positional, so a caller that handed over only its window would shift the
    # task segment and expose it.
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    result = compress_conversation(
        get_compressor("truncate"), _CONVERSATION, config, mutable_from=4
    )
    assert result.texts[:4] == [segment.text for segment in _CONVERSATION[:4]]
    assert result.texts[4] == "a follow-up"


def test_a_conversation_with_nothing_eligible_never_calls_the_compressor() -> None:
    # Not just "no change": no round trip and no bill either, which is what makes a narrow scope
    # free rather than merely harmless.
    class _Refusing:
        id = "refusing-for-tests"
        version = "1"
        append_stable = True

        def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
            raise AssertionError("the choke point handed over an ineligible segment")

    config = CompressionConfig(compressor_id="refusing-for-tests", scope="observations")
    only_the_task = [Segment(kind="user", text="the task statement")]
    result = compress_conversation(cast("Compressor", _Refusing()), only_the_task, config)
    assert result.texts == ["the task statement"]
    assert result.cost_usd == 0.0


def test_no_scope_wraps_the_fit_embedder_because_the_deciding_query_is_raw() -> None:
    # The bank embeds bare task statements; serving routes on the last user turn, which on turn 1
    # IS the task (stage law never compresses it); and sticky affinity means turns 2+ never consult
    # the bank. So the deciding query is raw in EVERY scope, and a wrapped bank would be queried
    # with raw text essentially always: the C2 Q2 failure caused by wrapping, not prevented by it.
    inner = _CountingEmbedder()
    base = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    scopes: tuple[CompressionScope, ...] = ("conversation", "observations", "bulk", "all")
    for scope in scopes:
        assert routed_text_embedder(inner, base.model_copy(update={"scope": scope})) is inner, scope
    assert routed_text_embedder(inner, None) is inner  # an uncompressed arm, unchanged as ever


def test_every_scope_fit_embeds_raw_bytes() -> None:
    # The same rule read off the BYTES handed to the inner embedder, not off types, so a future
    # wrap rule cannot pass this by returning a wrapper that happens to be a no-op.
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    scopes: tuple[CompressionScope, ...] = ("conversation", "observations", "bulk", "all")
    for scope in scopes:
        inner = _CountingEmbedder()
        routed_text_embedder(inner, config.model_copy(update={"scope": scope})).embed(
            ["one two three four"]
        )
        assert inner.seen == [["one two three four"]], scope


def test_compressing_embedder_is_still_available_for_deliberate_research_use() -> None:
    # Demoted to explicit-use-only, not deleted: research code that wants a compressed-geometry
    # bank constructs it directly and owns the reason.
    inner = _CountingEmbedder()
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    CompressingEmbedder(inner, config).embed(["one two three four"])
    assert inner.seen == [["one two"]]


def test_compressing_embedder_embeds_the_compressed_text() -> None:
    # The fit-side half of representation consistency: the bank rows must be the geometry of
    # what serving will send, not of the raw task text.
    inner = _CountingEmbedder()
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    CompressingEmbedder(inner, config).embed(["one two three four", "alpha beta"])
    assert inner.seen == [["one two", "alpha"]]


def test_the_dial_invariants_hold_for_every_reference_compressor() -> None:
    # `aggressiveness` is a compressor-defined dial, not a removal fraction, so these two
    # invariants are the whole contract an implementation has to honor: 0.0 is a strict
    # bit-for-bit no-op, and higher never removes less.
    segments = ["one two three four five six seven eight", "alpha beta gamma", ""]
    for compressor_id in ("identity", "truncate"):
        compressor = get_compressor(compressor_id)
        removed = []
        for level in (0.0, 0.25, 0.5, 0.75, 1.0):
            config = CompressionConfig(compressor_id=compressor_id, aggressiveness=level)
            result = compressor.compress(segments, config)
            if level == 0.0:
                assert result.segments == segments, compressor_id
            removed.append(result.tokens_in_raw - result.tokens_in_compressed)
        assert removed == sorted(removed), f"{compressor_id} dial is not monotone: {removed}"


def test_one_call_carries_every_segment() -> None:
    # The batching contract a network-backed compressor depends on: the caller hands over all
    # of a request's mutable segments at once, so the implementation pays one round trip and
    # can batch internally.
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    result = get_compressor("truncate").compress(["one two", "three four", "five six"], config)
    assert len(result.segments) == 3


class _V2Compressor:
    """Stands in for a compressor whose implementation was version-bumped under a stable id."""

    id = "versioned-compression-test"
    version = "2"
    append_stable = True

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        del config
        out = [segment.upper() for segment in segments]
        return CompressionResult(
            segments=out,
            tokens_in_raw=sum(estimate_tokens(segment) for segment in segments),
            tokens_in_compressed=sum(estimate_tokens(segment) for segment in out),
            latency_s=0.0,
        )


_V2 = _V2Compressor()


def test_a_config_version_the_build_cannot_produce_is_refused() -> None:
    # The id alone does not identify the bytes. An artifact stamped against a version this
    # build does not run was fitted in a DIFFERENT implementation's geometry, which is the same
    # failure as serving a different compressor and is invisible without this check.
    with pytest.raises(ValueError, match="version 1 in this build.*fitted against version 99"):
        servable_compressor(CompressionConfig(compressor_id="truncate", compressor_version="99"))


def test_a_version_bumped_implementation_refuses_the_old_stamp() -> None:
    # The other direction: the build moved forward and the artifact did not.
    register_compressor(_V2)
    assert (
        servable_compressor(CompressionConfig(compressor_id=_V2.id, compressor_version="2"))
        is not None
    )
    with pytest.raises(ValueError, match="version 2 in this build.*fitted against version 1"):
        servable_compressor(CompressionConfig(compressor_id=_V2.id, compressor_version="1"))


class _CappedCompressor:
    """A compressor whose server refuses batches over its cap, like the real endpoint's 413."""

    id = "capped-for-tests"
    version = "1"
    append_stable = True
    max_segments_per_call = 3

    def __init__(self) -> None:
        self.batches: list[int] = []

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        if len(segments) > self.max_segments_per_call:
            raise RuntimeError(
                f"413: {len(segments)} segments over cap {self.max_segments_per_call}"
            )
        self.batches.append(len(segments))
        out = [segment.split(" ")[0] for segment in segments]
        return CompressionResult(
            segments=out,
            tokens_in_raw=sum(estimate_tokens(s) for s in segments),
            tokens_in_compressed=sum(estimate_tokens(s) for s in out),
            latency_s=0.5,
            cost_usd=0.001,
        )


def test_a_capped_compressor_is_chunked_instead_of_overflowed() -> None:
    # The fit path hands over every scenario in the matrix at once, which is far past any served
    # compressor's per-call cap. Chunking is what makes those two facts compatible.
    capped = _CappedCompressor()
    segments = [f"segment {index} tail" for index in range(7)]
    result = compress_segments(capped, segments, CompressionConfig(compressor_id=capped.id))
    assert capped.batches == [3, 3, 1]  # chunked to the cap, nothing dropped
    assert result.segments == ["segment" for _ in range(7)]
    assert len(result.segments) == len(segments)
    # Per-chunk accounting is summed, not taken from the last chunk.
    assert result.latency_s == pytest.approx(1.5)
    assert result.cost_usd == pytest.approx(0.003)


def test_chunking_does_not_change_the_bytes() -> None:
    # Safe only because the protocol requires per-segment determinism; this pins that the seam
    # relies on nothing else. Same segments, two different chunk sizes, identical output.
    segments = [f"one two three {index}" for index in range(10)]
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    whole = compress_segments(get_compressor("truncate"), segments, config)

    class _Capped2(TruncateCompressor):
        max_segments_per_call = 2

    chunked = compress_segments(cast("Compressor", _Capped2()), segments, config)
    assert chunked.segments == whole.segments


def test_an_uncapped_compressor_still_gets_one_call() -> None:
    calls: list[int] = []

    class _Counting(TruncateCompressor):
        def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
            calls.append(len(segments))
            return super().compress(segments, config)

    compress_segments(
        cast("Compressor", _Counting()),
        ["a b", "c d", "e f"],
        CompressionConfig(compressor_id="truncate"),
    )
    assert calls == [3]


def test_a_malformed_cap_is_refused_at_registration() -> None:
    class _BadCap:
        id = "bad-cap-for-tests"
        version = "1"
        append_stable = True
        max_segments_per_call = 0

        def compress(
            self, segments: list[str], config: CompressionConfig
        ) -> CompressionResult:  # pragma: no cover - never reached
            raise NotImplementedError

    with pytest.raises(ValueError, match="max_segments_per_call"):
        register_compressor(cast("Compressor", _BadCap()))


class _WrongLength:
    """Returns the wrong number of segments, the contract violation nothing used to catch."""

    id = "wrong-length-for-tests"
    version = "1"
    append_stable = True

    def __init__(self, delta: int) -> None:
        self.delta = delta

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        del config
        out = list(segments)
        if self.delta < 0:
            out = out[:-1]
        else:
            out = [*out, "extra"]
        return CompressionResult(
            segments=out, tokens_in_raw=1, tokens_in_compressed=1, latency_s=0.0
        )


def test_a_short_return_is_a_named_error_not_a_stopiteration() -> None:
    # It used to surface as StopIteration from the caller's zip, i.e. an anonymous 502 that
    # blamed nothing in particular.
    with pytest.raises(ValueError, match="wrong-length-for-tests.*returned 1 segments for 2"):
        compress_segments(
            cast("Compressor", _WrongLength(-1)),
            ["a", "b"],
            CompressionConfig(compressor_id="x"),
        )


def test_a_long_return_is_refused_instead_of_silently_truncated() -> None:
    # The worse direction: the extra segment used to be discarded and the request served, so a
    # compressor that split a segment corrupted the transcript with nobody the wiser.
    with pytest.raises(ValueError, match="returned 3 segments for 2"):
        compress_segments(
            cast("Compressor", _WrongLength(1)),
            ["a", "b"],
            CompressionConfig(compressor_id="x"),
        )


def test_a_factory_compressor_is_built_on_first_resolution_not_at_import() -> None:
    # The endpoint client has to reach its server to verify the selection rule before it can
    # attest append_stable, and doing that at import would put a network call in `import wmo`.
    built: list[int] = []

    class _Lazy(TruncateCompressor):
        id = "lazy-for-tests"

    def factory() -> Compressor:
        built.append(1)
        return cast("Compressor", _Lazy())

    register_compressor_factory("lazy-for-tests", factory)
    assert built == []  # registering builds nothing

    first = get_compressor("lazy-for-tests")
    assert built == [1]
    assert get_compressor("lazy-for-tests") is first  # built once, then cached
    assert built == [1]


def test_a_failing_factory_names_the_compressor_and_can_be_retried() -> None:
    attempts: list[int] = []

    class _Flaky(TruncateCompressor):
        id = "flaky-for-tests"

    def factory() -> Compressor:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("endpoint unreachable")
        return cast("Compressor", _Flaky())

    register_compressor_factory("flaky-for-tests", factory)
    with pytest.raises(ValueError, match="'flaky-for-tests' could not be constructed"):
        get_compressor("flaky-for-tests")
    # Still registered, so a transient failure is retried rather than poisoning the id.
    assert get_compressor("flaky-for-tests") is not None


def test_a_factory_cannot_shadow_a_registered_compressor() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_compressor_factory("truncate", lambda: cast("Compressor", TruncateCompressor()))


def test_a_factory_that_builds_the_wrong_compressor_is_caught() -> None:
    # Otherwise it lands in the registry under ITS id while this lookup hands it back for a
    # different one, so a policy naming the registered id gets an implementation that does not
    # answer to it.
    register_compressor_factory(
        "mislabelled-for-tests", lambda: cast("Compressor", TruncateCompressor())
    )
    with pytest.raises(ValueError, match="built one with id 'truncate'"):
        get_compressor("mislabelled-for-tests")


def test_a_factory_failure_keeps_the_original_message_and_cause() -> None:
    # A served compressor's construction error is the actionable part (which env var, which
    # host), so the wrapper must not swallow it. The TYPE is normalized to ValueError on
    # purpose: this resolves inside a pydantic validator, which converts ValueError into a
    # ValidationError and lets anything else escape raw, and the CLI catches ValueError to turn
    # it into a usage error. The original exception stays reachable as __cause__.
    class _EndpointDown(RuntimeError):
        pass

    detail = "set WMO_COMPRESSOR_URL and WMO_COMPRESSOR_API_KEY, then retry"

    def factory() -> Compressor:
        raise _EndpointDown(detail)

    register_compressor_factory("actionable-for-tests", factory)
    with pytest.raises(ValueError) as caught:
        get_compressor("actionable-for-tests")
    assert detail in str(caught.value)  # the operator keeps the fix
    assert "actionable-for-tests" in str(caught.value)  # and learns which compressor
    assert isinstance(caught.value.__cause__, _EndpointDown)


def test_a_second_factory_cannot_displace_the_first() -> None:
    def first() -> Compressor:  # pragma: no cover - never resolved
        raise NotImplementedError

    def second() -> Compressor:  # pragma: no cover - never resolved
        raise NotImplementedError

    register_compressor_factory("one-factory-for-tests", first)
    register_compressor_factory("one-factory-for-tests", first)  # idempotent for the same object
    with pytest.raises(ValueError, match="already has a registered factory"):
        register_compressor_factory("one-factory-for-tests", second)


def test_a_factory_cannot_shadow_a_live_instance() -> None:
    # A test that registered a fake under an id must not be silently replaced by the real one.
    register_compressor(_CHURNY)
    with pytest.raises(ValueError, match="already registered"):
        register_compressor_factory(_CHURNY.id, lambda: _CHURNY)


def test_unknown_compressor_lists_registered_factories_too() -> None:
    # A factory-registered compressor IS available; leaving it out of the known list would tell
    # an operator to register something that is already there.
    def factory() -> Compressor:  # pragma: no cover - never resolved
        raise NotImplementedError

    register_compressor_factory("listed-for-tests", factory)
    with pytest.raises(ValueError, match="listed-for-tests"):
        get_compressor("definitely-not-registered")


def test_the_cap_is_read_off_the_constructed_instance() -> None:
    # A served compressor learns its cap from the server, so the value only exists after
    # construction. Nothing may require it as a class-level constant inspected beforehand.
    class _LearnsItsCap(TruncateCompressor):
        id = "learned-cap-for-tests"

        def __init__(self) -> None:
            self.max_segments_per_call = 2  # as if read from /healthz

    register_compressor_factory(
        "learned-cap-for-tests", lambda: cast("Compressor", _LearnsItsCap())
    )
    resolved = get_compressor("learned-cap-for-tests")
    assert segment_batch_limit(resolved) == 2
    result = compress_segments(
        resolved, ["a b", "c d", "e f"], CompressionConfig(compressor_id="learned-cap-for-tests")
    )
    assert len(result.segments) == 3
