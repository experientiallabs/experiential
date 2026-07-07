"""Tests for the Waterfall failover loop (fake adapters — no SDKs, no network)."""

from __future__ import annotations

import threading

import pytest

from llm_waterfall.pricing import ModelPrice
from llm_waterfall.types import (
    Backend,
    EmbeddingsUnsupported,
    Message,
    RetryPolicy,
    TokenUsage,
    WaterfallExhausted,
)
from llm_waterfall.waterfall import Waterfall


class _Throttle(Exception):
    def __init__(self) -> None:
        super().__init__("boom")
        self.response = {"Error": {"Code": "ThrottlingException", "Message": "slow down"}}


class _Validation(Exception):
    def __init__(self) -> None:
        super().__init__("request timeout too large")
        self.response = {"Error": {"Code": "ValidationException", "Message": "bad"}}


class FakeAdapter:
    """Scriptable adapter: each call pops the next behavior ('ok' or an exception)."""

    def __init__(self, backend: Backend, script: list[object]) -> None:
        self.backend = backend
        self.script = list(script)
        self.calls = 0

    def _next(self) -> None:
        self.calls += 1
        step = self.script.pop(0) if self.script else "ok"
        if isinstance(step, Exception):
            raise step

    def complete(
        self, system: str, messages: list[Message], *, temperature: float | None, max_tokens: int
    ) -> tuple[str, TokenUsage]:
        self._next()
        return f"text-from-{self.backend.model}", TokenUsage(input_tokens=100, output_tokens=10)

    def embed(self, texts: list[str]) -> tuple[list[list[float]], TokenUsage]:
        self._next()
        return [[1.0] for _ in texts], TokenUsage(input_tokens=3 * len(texts))

    def embed_model_id(self) -> str | None:
        return f"embedder-of-{self.backend.model}"


def _waterfall(
    scripts: dict[str, list[object]],
    *,
    retry: RetryPolicy | None = None,
    prices: dict[str, ModelPrice] | None = None,
) -> tuple[Waterfall, dict[str, FakeAdapter]]:
    backends = [Backend("bedrock", model) for model in scripts]
    adapters = {m: FakeAdapter(Backend("bedrock", m), s) for m, s in scripts.items()}
    wf = Waterfall(
        backends,
        retry=retry if retry is not None else RetryPolicy(),
        prices=prices,
        adapter_factory=lambda b: adapters[b.model],
    )
    return wf, adapters


MSGS = [Message(role="user", content="hi")]


def test_primary_serves_when_healthy() -> None:
    wf, adapters = _waterfall({"opus": [], "sonnet": []})
    r = wf.complete(system="s", messages=MSGS)
    assert r.text == "text-from-opus"
    assert r.model_used == "opus" and r.provider_used == "bedrock"
    assert [a.outcome for a in r.attempts] == ["ok"]
    assert adapters["sonnet"].calls == 0


def test_failover_attributes_cost_to_serving_backend() -> None:
    # Regression: the prototype attributed cost/model to the PRIMARY even when a fallback served.
    wf, _ = _waterfall({"claude-opus-4-8": [_Throttle()], "claude-sonnet-4-6": []})
    r = wf.complete(system="", messages=MSGS)
    assert r.model_used == "claude-sonnet-4-6"
    assert [a.outcome for a in r.attempts] == ["capacity_error", "ok"]
    assert r.attempts[0].error_type == "_Throttle"
    # 100 in @ $3/M + 10 out @ $15/M — sonnet's rate, not opus's.
    assert r.cost_usd == pytest.approx((100 * 3.0 + 10 * 15.0) / 1_000_000)


def test_client_error_propagates_immediately() -> None:
    wf, adapters = _waterfall({"opus": [_Validation()], "sonnet": []})
    with pytest.raises(_Validation):
        wf.complete(system="", messages=MSGS)
    assert adapters["sonnet"].calls == 0  # never masked behind a fallback


def test_exhaustion_raises_with_full_attempt_trail() -> None:
    wf, _ = _waterfall({"a": [_Throttle()], "b": [_Throttle()]})
    with pytest.raises(WaterfallExhausted) as exc_info:
        wf.complete(system="", messages=MSGS)
    exc = exc_info.value
    assert len(exc.attempts) == 2
    assert all(a.outcome == "capacity_error" for a in exc.attempts)
    assert isinstance(exc.__cause__, _Throttle)


def test_rounds_wrap_around_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("llm_waterfall.waterfall._sleep", sleeps.append)
    # Both throttle in round 1; primary recovers in round 2.
    wf, adapters = _waterfall(
        {"a": [_Throttle()], "b": [_Throttle(), _Throttle()]},
        retry=RetryPolicy(rounds=3, backoff_base_s=15.0),
    )
    r = wf.complete(system="", messages=MSGS)
    assert r.model_used == "a"
    assert len(r.attempts) == 3  # a:throttle, b:throttle, a:ok
    assert len(sleeps) == 1 and 15.0 <= sleeps[0] <= 15.0 * 1.34  # base + jitter


def test_exhaustion_after_all_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("llm_waterfall.waterfall._sleep", sleeps.append)
    wf, _ = _waterfall(
        {"a": [_Throttle()] * 3},
        retry=RetryPolicy(rounds=3, backoff_base_s=15.0, backoff_max_s=20.0),
    )
    with pytest.raises(WaterfallExhausted) as exc_info:
        wf.complete(system="", messages=MSGS)
    assert len(exc_info.value.attempts) == 3
    assert len(sleeps) == 2  # before rounds 2 and 3


def test_embed_waterfalls_and_skips_unsupported() -> None:
    wf, _ = _waterfall(
        {"anthropic-like": [EmbeddingsUnsupported("no embeddings API")], "titan": []}
    )
    r = wf.embed(["x", "y"])
    assert r.vectors == [[1.0], [1.0]]
    assert [a.outcome for a in r.attempts] == ["unsupported", "ok"]
    # Attributed to the embedding model the serving ADAPTER reports, not the chat model id.
    assert r.model_used == "embedder-of-titan"
    assert r.provider_used == "bedrock"


def test_messages_accept_raw_dicts() -> None:
    wf, _ = _waterfall({"m": []})
    r = wf.complete(system="", messages=[{"role": "user", "content": "hi"}])
    assert r.text == "text-from-m"


def test_empty_chain_rejected() -> None:
    with pytest.raises(ValueError, match="at least one backend"):
        Waterfall([])


def test_price_overrides_apply() -> None:
    wf, _ = _waterfall(
        {"custom-deploy": []},
        prices={"custom-deploy": ModelPrice(input_per_mtok=10.0, output_per_mtok=100.0)},
    )
    r = wf.complete(system="", messages=MSGS)
    assert r.cost_usd == pytest.approx((100 * 10.0 + 10 * 100.0) / 1_000_000)


def test_shared_waterfall_is_thread_safe() -> None:
    wf, adapters = _waterfall({"m": []})
    results: list[str] = []
    lock = threading.Lock()

    def hit() -> None:
        r = wf.complete(system="", messages=MSGS)
        with lock:
            results.append(r.text)

    threads = [threading.Thread(target=hit) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 32
    assert adapters["m"].calls == 32


def test_verify_pings_every_backend_and_never_raises() -> None:
    wf, _ = _waterfall({"good": [], "bad": [_Validation()]})
    results = wf.verify()
    assert [v.ok for v in results] == [True, False]
    assert "timeout too large" in results[1].detail


def test_all_unsupported_embed_raises_embeddings_unsupported_without_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: an embeddings-free chain is a static config error, not a capacity condition —
    # no misleading WaterfallExhausted, and no pointless backoff rounds.
    sleeps: list[float] = []
    monkeypatch.setattr("llm_waterfall.waterfall._sleep", sleeps.append)
    wf, _ = _waterfall(
        {"a": [EmbeddingsUnsupported("n/a")], "b": [EmbeddingsUnsupported("n/a")]},
        retry=RetryPolicy(rounds=3),
    )
    with pytest.raises(EmbeddingsUnsupported, match="no backend in this chain"):
        wf.embed(["x"])
    assert sleeps == []  # deterministic outcome: no rounds were retried


def test_jitter_never_exceeds_backoff_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: jitter was applied after the cap, sleeping up to 1.34x backoff_max_s.
    sleeps: list[float] = []
    monkeypatch.setattr("llm_waterfall.waterfall._sleep", sleeps.append)
    monkeypatch.setattr("llm_waterfall.waterfall.random.uniform", lambda a, b: b)  # worst case
    wf, _ = _waterfall(
        {"a": [_Throttle()] * 5},
        retry=RetryPolicy(rounds=5, backoff_base_s=15.0, backoff_max_s=60.0),
    )
    with pytest.raises(WaterfallExhausted):
        wf.complete(system="", messages=MSGS)
    assert all(s <= 60.0 for s in sleeps), sleeps


def test_stub_backends_fail_at_construction() -> None:
    # Regression: an unimplemented rung must break Waterfall() construction loudly, never
    # abort a live call mid-chain (a call-time NotImplementedError reads as a client error).
    # azure_openai is real but misconfiguration must still fail at construction.
    with pytest.raises(ValueError, match="endpoint"):
        Waterfall([Backend("azure_openai", "gpt-5.5", deployment="d")])
    with pytest.raises(NotImplementedError, match="aws_mantle adapter is not implemented"):
        Waterfall([Backend("aws_mantle", "anthropic.claude-opus-4-8")])


def test_verify_ping_budget_fits_reasoning_models() -> None:
    # Regression: GPT-5.x spends output tokens on reasoning before any text; a 1-token ping
    # 400s on a healthy backend. The ping must give visible-output headroom.
    captured: list[int] = []

    class _Recorder(FakeAdapter):
        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float | None,
            max_tokens: int,
        ) -> tuple[str, TokenUsage]:
            captured.append(max_tokens)
            return super().complete(
                system, messages, temperature=temperature, max_tokens=max_tokens
            )

    backend = Backend("openai", "gpt-5.5")
    wf = Waterfall([backend], adapter_factory=lambda b: _Recorder(b, []))
    assert wf.verify()[0].ok
    assert captured == [256]
