"""Tests for MeteredProvider: every call is recorded with the right phase + usage. No network."""

from __future__ import annotations

from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind, TokenUsage
from wmh.tracking.metered import MeteredProvider, classify_build_call
from wmh.tracking.tracker import Phase, RunTracker


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
