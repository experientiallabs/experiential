"""Tests for MeteredProvider: every call is recorded with the right phase + usage. No network."""

from __future__ import annotations

from collections.abc import Generator, Iterator

from wmo.common.observability.metered import MeteredProvider, classify_build_call
from wmo.common.observability.tracker import Phase, RunTracker
from wmo.common.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    StreamChunk,
    TokenUsage,
)


class FakeProvider:
    """Returns canned usage; mimics the build-time call sites by branching on the system prompt."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="claude-opus-4-8")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        if "grade a world model" in system:
            return Completion(text="judged", usage=TokenUsage(input_tokens=40, output_tokens=10))
        return Completion(text="rolled", usage=TokenUsage(input_tokens=100, output_tokens=20))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def test_complete_records_usage_with_base_phase() -> None:
    tracker = RunTracker(run_id="r", kind="serve")
    metered = MeteredProvider(FakeProvider(), tracker, base_phase=Phase.SERVE)

    completion = metered.complete("anything", [Message(role="user", content="hi")])

    assert completion.text == "rolled"  # forwarded unchanged
    total = tracker.totals()
    assert total.calls == 1
    assert total.input_tokens == 100
    assert total.output_tokens == 20
    assert tracker.by_phase()[Phase.SERVE].calls == 1


def test_classify_build_call_splits_judge_from_gepa() -> None:
    tracker = RunTracker(run_id="r", kind="build")
    metered = MeteredProvider(FakeProvider(), tracker, classify=classify_build_call)

    # A GEPA env-sim rollout (any non-judge system) and a judge call.
    metered.complete("You simulate an environment", [Message(role="user", content="x")])
    metered.complete("You grade a world model ...", [Message(role="user", content="y")])

    by_phase = tracker.by_phase()
    assert by_phase[Phase.GEPA].calls == 1
    assert by_phase[Phase.GEPA].input_tokens == 100
    assert by_phase[Phase.JUDGE].calls == 1
    assert by_phase[Phase.JUDGE].input_tokens == 40


def test_classify_build_call_marker() -> None:
    assert classify_build_call("You grade a world model") is Phase.JUDGE
    assert classify_build_call("You improve the system prompt") is Phase.GEPA
    assert classify_build_call("env simulator") is Phase.GEPA


def test_embed_is_recorded_under_embed_phase() -> None:
    tracker = RunTracker(run_id="r", kind="build")
    metered = MeteredProvider(FakeProvider(), tracker)

    vectors = metered.embed(["a", "b"])

    assert vectors == [[0.0, 1.0], [0.0, 1.0]]  # forwarded unchanged
    assert tracker.by_phase()[Phase.EMBED].calls == 1


def test_embed_event_is_attributed_to_embed_model_not_completion_model() -> None:
    provider = FakeProvider()
    provider.config = ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model="claude-opus-4-8",
        embed_model="amazon.titan-embed-text-v2:0",
    )
    tracker = RunTracker(run_id="r", kind="build")
    MeteredProvider(provider, tracker).embed(["a"])

    # The EMBED event must carry the embeddings model, not the completion model.
    embed_events = [e for e in tracker.events if e.phase is Phase.EMBED]
    assert len(embed_events) == 1
    assert embed_events[0].model == "amazon.titan-embed-text-v2:0"


def test_config_is_forwarded() -> None:
    provider = FakeProvider()
    metered = MeteredProvider(provider, RunTracker(run_id="r", kind="serve"))
    assert metered.config is provider.config


class FakeFailoverProvider(FakeProvider):
    """Mimics WaterfallProvider: config reports the primary, Completion reports who served."""

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        return Completion(
            text="served by fallback",
            usage=TokenUsage(input_tokens=100, output_tokens=20),
            model="claude-haiku-4-5",  # a fallback served, not the primary opus
        )


def test_cost_attributed_to_serving_model_not_primary() -> None:
    # Regression: a failed-over call must be priced at the serving model's rate, not the
    # primary's (opus 5/25 vs haiku 1/5 per Mtok — a 5x over-report).
    tracker = RunTracker(run_id="r", kind="serve")
    metered = MeteredProvider(FakeFailoverProvider(), tracker, base_phase=Phase.SERVE)

    metered.complete("anything", [Message(role="user", content="hi")])

    (event,) = tracker._events
    assert event.model == "claude-haiku-4-5"
    assert event.cost_usd == (100 * 1.0 + 20 * 5.0) / 1_000_000


class FakeStreamingProvider(FakeProvider):
    """Streams three deltas, then a terminal chunk carrying exact usage."""

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Iterator[StreamChunk]:
        yield StreamChunk(delta="aaaa")
        yield StreamChunk(delta="bbbb")
        yield StreamChunk(delta="cccc")
        yield StreamChunk(done=True, usage=TokenUsage(input_tokens=100, output_tokens=3))


def test_stream_records_the_terminal_usage_once() -> None:
    """A stream consumed to the end records exactly the provider's own counts."""
    tracker = RunTracker(run_id="r", kind="test")
    provider = MeteredProvider(FakeStreamingProvider(), tracker, base_phase=Phase.SERVE)

    chunks = list(provider.stream("sys", [Message(role="user", content="hi")]))

    assert chunks[-1].done
    assert len(tracker.events) == 1
    assert tracker.events[0].usage.input_tokens == 100
    assert tracker.events[0].usage.output_tokens == 3


def test_abandoned_stream_records_a_chars_over_four_estimate() -> None:
    """Closing the stream before its terminal chunk records the documented estimate.

    The provider reports exact counts only in the chunk the consumer never took; the old
    behavior recorded nothing, billing the whole partial generation as free.
    """
    tracker = RunTracker(run_id="r", kind="test")
    provider = MeteredProvider(FakeStreamingProvider(), tracker, base_phase=Phase.SERVE)

    stream = provider.stream("sysprompt", [Message(role="user", content="x" * 39)])
    assert next(stream).delta == "aaaa"
    assert next(stream).delta == "bbbb"
    # The metered wrapper is a generator; closing it is how an abandoning
    # consumer (or GC) ends it, which is the path under test.
    assert isinstance(stream, Generator)
    stream.close()

    assert len(tracker.events) == 1
    event = tracker.events[0]
    # input: (9 system chars + 39 message chars) // 4; output: 8 streamed chars // 4.
    assert event.usage.input_tokens == 12
    assert event.usage.output_tokens == 2


class FailingStreamProvider(FakeProvider):
    """Streams one delta then raises, like a throttled upstream."""

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Iterator[StreamChunk]:
        yield StreamChunk(delta="partial")
        msg = "ThrottlingException"
        raise RuntimeError(msg)


def test_upstream_stream_failure_records_nothing() -> None:
    """An upstream failure must not book a full-prompt phantom estimate.

    The tracker's total feeds spend caps; a throttled retry loop estimating
    the whole prompt each attempt would trip a cap on money never spent.
    Only consumer abandonment (GeneratorExit) estimates.
    """
    tracker = RunTracker(run_id="r", kind="test")
    provider = MeteredProvider(FailingStreamProvider(), tracker, base_phase=Phase.SERVE)

    stream = provider.stream("s" * 400_000, [Message(role="user", content="hi")])
    assert next(stream).delta == "partial"
    try:
        next(stream)
        raise AssertionError("expected the upstream failure to propagate")
    except RuntimeError:
        pass

    assert tracker.events == []


class UsagelessStreamProvider(FakeProvider):
    """Finishes normally but its terminal chunk carries no usage."""

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Iterator[StreamChunk]:
        yield StreamChunk(delta="hello")
        yield StreamChunk(done=True)


def test_fully_consumed_stream_without_usage_records_nothing() -> None:
    """Normal exhaustion without a usage chunk stays unrecorded, not estimated."""
    tracker = RunTracker(run_id="r", kind="test")
    provider = MeteredProvider(UsagelessStreamProvider(), tracker, base_phase=Phase.SERVE)

    chunks = list(provider.stream("sys", [Message(role="user", content="hi")]))

    assert chunks[-1].done
    assert tracker.events == []
