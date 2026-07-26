"""Tests for the OpenAI-compatible chat endpoint (routing, streaming, affinity, request log)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from wmh.optimize.compression import CompressionConfig, CompressionResult, Compressor
from wmh.optimize.policy import (
    KNN_BANK_FILENAME,
    POLICY_FILENAME,
    ClusterRanking,
    EmbedderSpec,
    KnnBank,
    RoutingPolicy,
)
from wmh.providers.base import (
    Completion,
    Message,
    ProviderKind,
    StreamChunk,
    TokenUsage,
    VerifyResult,
)
from wmh.providers.pool import PoolEntry
from wmh.retrieval.embedders import HashingEmbedder
from wmh.serving.chat import EndpointRuntime, RequestLog, create_chat_router

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from wmh.providers.base import ProviderConfig


class _EchoProvider:
    """Fake provider: replies with its own pool name so tests see who served."""

    def __init__(self, entry: PoolEntry) -> None:
        self.config: ProviderConfig = entry.provider_config()
        self.name = entry.name

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        return Completion(
            text=f"served by {self.name}",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Iterator[StreamChunk]:
        yield StreamChunk(delta="served by ")
        yield StreamChunk(delta=self.name)
        yield StreamChunk(done=True, usage=TokenUsage(input_tokens=10, output_tokens=5))

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def verify(self) -> VerifyResult:
        raise NotImplementedError


def _pool() -> list[PoolEntry]:
    return [
        PoolEntry(
            name="fable-5",
            kind=ProviderKind.ANTHROPIC,
            model="claude-fable-5",
        ),
        PoolEntry(
            name="haiku-4-5",
            kind=ProviderKind.ANTHROPIC,
            model="claude-haiku-4-5",
        ),
    ]


def _cluster_policy() -> RoutingPolicy:
    embedder = HashingEmbedder(dim=64)
    sql, prose = embedder.embed(["SELECT count(*) FROM superheroes", "write a friendly email"])
    return RoutingPolicy(
        kind="rank",
        default_model="haiku-4-5",
        pool=_pool(),
        embedder=EmbedderSpec(dim=64),
        top_k_clusters=1,
        clusters=[
            ClusterRanking(
                cluster_id=0, label="sql", centroid=sql, ranking=["fable-5", "haiku-4-5"]
            ),
            ClusterRanking(
                cluster_id=1, label="prose", centroid=prose, ranking=["haiku-4-5", "fable-5"]
            ),
        ],
    )


def _client(tmp_path: Path, policy: RoutingPolicy | None = None) -> tuple[TestClient, Path]:
    log_path = tmp_path / "requests.jsonl"
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=policy or RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool()),
        provider_factory=_EchoProvider,
        log=RequestLog(log_path),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    return TestClient(app), log_path


def test_completion_matches_openai_shape(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "tau-bench"  # the endpoint, not the mechanism
    assert body["choices"][0]["message"]["content"] == "served by haiku-4-5"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    assert response.headers["x-wmh-routed-model"] == "haiku-4-5"


def test_streaming_emits_openai_chunks(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    payloads = [line.removeprefix("data: ") for line in lines]
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    assert text == "served by haiku-4-5"
    # OpenAI include_usage framing: finish_reason chunk, THEN a choices-less usage chunk.
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] == 15


def test_cluster_routing_and_affinity(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, policy=_cluster_policy())
    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "SELECT count(*) FROM superheroes"}],
        },
    )
    assert first.headers["x-wmh-routed-model"] == "fable-5"
    reply = first.json()["choices"][0]["message"]["content"]
    # Turn 2 would route to the prose cluster if fresh, but the conversation prefix
    # (turn-1 user + assistant reply) pins the incumbent: affinity wins.
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "user", "content": "SELECT count(*) FROM superheroes"},
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "write a friendly email about it"},
            ],
        },
    )
    assert second.headers["x-wmh-routed-model"] == "fable-5"


def test_unknown_endpoint_404s_with_available(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    body = response.json()
    # OpenAI error shape: clients read body["error"]["message"], never FastAPI's "detail".
    assert body["error"]["code"] == "model_not_found"
    assert "tau-bench" in body["error"]["message"]


def test_models_endpoint_lists_endpoints(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["tau-bench"]


def test_request_log_rows(tmp_path: Path) -> None:
    client, log_path = _client(tmp_path, policy=_cluster_policy())
    client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "SELECT 1"}],
        },
    )
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["endpoint"] == "tau-bench"
    assert row["model"] == "fable-5"
    assert row["cluster_id"] == 0
    assert row["cluster_label"] == "sql"
    assert row["routing_reason"]
    assert row["input_tokens"] == 10
    assert row["output_tokens"] == 5
    assert row["cost_usd"] == pytest.approx((10 * 10.0 + 5 * 50.0) / 1_000_000)
    assert row["latency_ms"] >= 0
    assert row["status"] == "ok"
    assert row["ts"]
    assert row["id"]
    assert row["leg"] == "serving"
    assert row["cached_tokens"] == 0  # carried for the metering contract, not yet captured
    assert row["router_cost_usd"] == 0.0  # hashing policy routes for free; passed through


def test_create_app_mounts_endpoints_from_policies(tmp_path: Path) -> None:
    from wmh.serving.server import create_app

    app = create_app(
        artifact_dirs=(str(tmp_path),),
        world_models={},
        policies={
            "tau-bench": RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool())
        },
    )
    client = TestClient(app)
    body = client.get("/v1/models").json()
    assert [m["id"] for m in body["data"]] == ["tau-bench"]


def test_create_app_without_policies_serves_empty_model_list(tmp_path: Path) -> None:
    # A client wired up before any policy is fitted gets an empty list and an OpenAI-shaped
    # "no endpoint" error, never a bare 404 on the whole /v1 surface.
    from wmh.serving.server import create_app

    app = create_app(artifact_dirs=(str(tmp_path),), world_models={})
    client = TestClient(app)
    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"] == []
    chat = client.post(
        "/v1/chat/completions",
        json={"model": "anything", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert chat.status_code == 404
    assert chat.json()["error"]["code"] == "model_not_found"


def test_provider_failure_logs_error_and_502s(tmp_path: Path) -> None:
    class _BoomProvider(_EchoProvider):
        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Completion:
            raise RuntimeError("upstream on fire")

    log_path = tmp_path / "requests.jsonl"
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool()),
        provider_factory=_BoomProvider,
        log=RequestLog(log_path),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502
    row = json.loads(log_path.read_text().splitlines()[0])
    assert row["status"] == "error"
    assert "upstream on fire" in row["error_message"]


def test_abandoned_stream_still_records_metering(tmp_path: Path) -> None:
    # A client that disconnects mid-stream closes the generator; the upstream call still
    # consumed tokens, so a request-log row must land anyway (D-METERING: no silent loss).
    class _EndlessProvider(_EchoProvider):
        def stream(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Iterator[StreamChunk]:
            for _ in range(1_000_000):  # far more than any client buffer; never finishes
                yield StreamChunk(delta="x")

    log_path = tmp_path / "requests.jsonl"
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool()),
        provider_factory=_EndlessProvider,
        log=RequestLog(log_path),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))

    # Drive the ASGI app directly: after the request body, `receive` reports
    # http.disconnect, which is what starlette listens for to cancel a StreamingResponse
    # and close its body iterator (GeneratorExit in the generator).
    body = json.dumps(
        {"model": "tau-bench", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    ).encode()
    state = {"request_sent": False}

    async def receive() -> dict[str, object]:
        if not state["request_sent"]:
            state["request_sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        return None

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 0),
        "server": ("test", 80),
    }
    asyncio.run(app(cast("Any", scope), cast("Any", receive), cast("Any", send)))

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert "disconnected" in rows[0]["error_message"]


def test_create_app_with_injected_policies_and_no_artifact_dirs(tmp_path: Path) -> None:
    # The injected-policies test pattern must not require an artifact root for the request log.
    from wmh.serving.server import create_app

    app = create_app(
        artifact_dirs=(),
        world_models={},
        policies={"ep": RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool())},
    )
    client = TestClient(app)
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["ep"]


def _knn_policy(tmp_path: Path) -> RoutingPolicy:
    """A knn policy written to disk exactly as the fitter emits it: policy.json + npz sidecar.

    Six SQL scenarios where fable-5 wins outright and six prose ones where haiku-4-5 does, with
    haiku-4-5 as the pinned fallback. Loading it back proves the endpoint serves the artifact
    pair with no serving-side knowledge of the bank format.
    """
    sql = ["SELECT count(*) FROM superheroes", "SELECT name FROM users LIMIT 10"] * 3
    prose = ["write a friendly email to the team", "draft a thank-you note"] * 3
    rewards = [[1.0, 0.0]] * len(sql) + [[0.0, 1.0]] * len(prose)
    bank = KnnBank(
        embeddings=np.asarray(HashingEmbedder(dim=64).embed(sql + prose), dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        costs=np.asarray([[0.01, 0.001]] * len(rewards), dtype=np.float32),
        models=["fable-5", "haiku-4-5"],
        scenario_ids=[f"s{index}" for index in range(len(rewards))],
    )
    bank.save(tmp_path / KNN_BANK_FILENAME)
    RoutingPolicy(
        kind="knn",
        default_model="haiku-4-5",
        guard_model="haiku-4-5",
        pool=_pool(),
        embedder=EmbedderSpec(dim=64),
        rag_num=6,
        knn_min_pairs=4,
    ).save(tmp_path / POLICY_FILENAME)
    return RoutingPolicy.load(tmp_path / POLICY_FILENAME)


def test_knn_policy_routes_a_request_end_to_end(tmp_path: Path) -> None:
    client, log_path = _client(tmp_path, policy=_knn_policy(tmp_path))
    routed = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "SELECT count(*) FROM superheroes"}],
        },
    )
    # The SQL neighborhood is unanimous, so the guard lets the request leave the fallback.
    assert routed.headers["x-wmh-routed-model"] == "fable-5"
    assert routed.json()["choices"][0]["message"]["content"] == "served by fable-5"

    fallback = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "draft a thank-you"}]},
    )
    assert fallback.headers["x-wmh-routed-model"] == "haiku-4-5"

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [row["model"] for row in rows] == ["fable-5", "haiku-4-5"]
    assert rows[0]["routing_reason"].startswith("knn: ")
    assert rows[0]["cluster_id"] is None  # a knn decision cites neighbors, not clusters


def test_runtime_builds_the_policy_embedder_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An azure embedder spec would otherwise construct a fresh SDK client per request.
    from wmh.optimize.policy import EmbedderSpec

    builds = {"n": 0}
    original = EmbedderSpec.build

    def counting_build(self: EmbedderSpec) -> object:
        builds["n"] += 1
        return original(self)

    monkeypatch.setattr(EmbedderSpec, "build", counting_build)
    client, _ = _client(tmp_path, policy=_cluster_policy())
    for _ in range(3):
        client.post(
            "/v1/chat/completions",
            json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert builds["n"] == 1


def test_static_policy_never_builds_its_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A static route must keep serving even when the embedder spec cannot initialize here.
    from wmh.optimize.policy import EmbedderSpec

    def exploding_build(self: EmbedderSpec) -> object:
        raise RuntimeError("no credentials in this environment")

    monkeypatch.setattr(EmbedderSpec, "build", exploding_build)
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy(
            kind="static",
            default_model="haiku-4-5",
            pool=_pool(),
            embedder=EmbedderSpec(dim=64),
        ),
        provider_factory=_EchoProvider,
        log=RequestLog(tmp_path / "requests.jsonl"),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200


# --- D-COMPRESS: the serving compression stage ---


class _CapturingProvider(_EchoProvider):
    """Echo provider that also records every (system, messages) it was asked to serve."""

    def __init__(self, entry: PoolEntry) -> None:
        super().__init__(entry)
        self.seen: list[tuple[str, list[Message]]] = []

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.seen.append((system, list(messages)))
        return super().complete(system, messages, temperature=temperature, max_tokens=max_tokens)

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Iterator[StreamChunk]:
        self.seen.append((system, list(messages)))
        yield from super().stream(system, messages, temperature=temperature, max_tokens=max_tokens)


def _compressed_runtime(
    tmp_path: Path, compression: CompressionConfig | None
) -> tuple[TestClient, Path, EndpointRuntime, dict[str, _CapturingProvider]]:
    providers: dict[str, _CapturingProvider] = {}

    def factory(entry: PoolEntry) -> _CapturingProvider:
        provider = _CapturingProvider(entry)
        providers[entry.name] = provider
        return provider

    log_path = tmp_path / "requests.jsonl"
    policy = RoutingPolicy(
        kind="static", default_model="haiku-4-5", pool=_pool(), compression=compression
    )
    runtime = EndpointRuntime(
        name="tau-bench", policy=policy, provider_factory=factory, log=RequestLog(log_path)
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    return TestClient(app), log_path, runtime, providers


def _last_log_row(log_path: Path) -> dict[str, object]:
    return json.loads(log_path.read_text().splitlines()[-1])


def test_identity_compression_serves_bit_for_bit(tmp_path: Path) -> None:
    # The seam's do-no-harm proof: identity compression and no compression hand the provider
    # byte-identical (system, turns), and the log accounts raw == compressed.
    body = {
        "model": "tau-bench",
        "messages": [
            {"role": "system", "content": "be  terse\twith   spacing"},
            {"role": "user", "content": "  what is 2+2?  keep my  spacing "},
        ],
    }
    _, _, runtime_off, providers_off = _compressed_runtime(tmp_path / "off", None)
    client_off = TestClient(_app_for(runtime_off))
    client_off.post("/v1/chat/completions", json=body)
    identity = CompressionConfig(compressor_id="identity", aggressiveness=1.0)
    client_on, log_path, _, providers_on = _compressed_runtime(tmp_path / "on", identity)
    response = client_on.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert providers_on["haiku-4-5"].seen == providers_off["haiku-4-5"].seen
    row = _last_log_row(log_path)
    assert row["compressor_id"] == "identity"
    assert row["tokens_in_raw"] == row["tokens_in_compressed"]
    assert cast("int", row["tokens_in_raw"]) > 0


def _app_for(runtime: EndpointRuntime) -> FastAPI:
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    return app


def test_truncate_compresses_what_the_provider_sees(tmp_path: Path) -> None:
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    client, log_path, _, providers = _compressed_runtime(tmp_path, config)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "system", "content": "system prompts are never compressed"},
                {"role": "user", "content": "one two three four five six seven eight"},
            ],
        },
    )

    assert response.status_code == 200
    system, turns = providers["haiku-4-5"].seen[0]
    assert system == "system prompts are never compressed"  # verbatim
    assert turns[0].content == "one two three four"  # trailing half dropped
    row = _last_log_row(log_path)
    assert row["compressor_id"] == "truncate"
    assert row["compressor_version"] == "1"
    assert row["aggressiveness"] == 0.5
    assert cast("int", row["tokens_in_compressed"]) < cast("int", row["tokens_in_raw"])
    # OPAQUE: compression never surfaces in the response body or headers.
    assert "compress" not in response.text
    assert not any("compress" in key.lower() for key in response.headers)


def test_incumbent_prefix_is_reused_not_recompressed(tmp_path: Path) -> None:
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    client, _, runtime, providers = _compressed_runtime(tmp_path, config)

    class _SpyCompressor:
        """Delegates to the real compressor, recording which segments it was handed."""

        def __init__(self, inner: Compressor) -> None:
            self.inner = inner
            self.id = inner.id
            self.version = inner.version
            self.calls: list[list[str]] = []

        def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
            self.calls.append(list(segments))
            return self.inner.compress(segments, config)

    spy = _SpyCompressor(cast("Compressor", runtime._compressor))
    runtime._compressor = spy

    first_user = "alpha beta gamma delta epsilon zeta"
    first = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": first_user}]},
    )
    reply = first.json()["choices"][0]["message"]["content"]
    turn_one = list(providers["haiku-4-5"].seen[0][1])

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "user", "content": first_user},
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "eta theta iota kappa"},
            ],
        },
    )

    assert second.status_code == 200
    # The compressor only ever saw the turn-local segment on turn two: the cached prefix was
    # RETRIEVED from the affinity state, not recompressed.
    assert spy.calls == [[first_user], ["eta theta iota kappa"]]
    # And the provider-visible prefix is byte-identical across turns (prompt cache survives).
    second_turns = providers["haiku-4-5"].seen[1][1]
    assert [(m.role, m.content) for m in second_turns[: len(turn_one)]] == [
        (m.role, m.content) for m in turn_one
    ]
    assert second_turns[0].content == "alpha beta gamma"  # compressed once, reused verbatim
    assert second_turns[1].content == reply  # the model's own reply is never compressed
    assert second_turns[2].content == "eta theta"


def test_lost_affinity_recompression_is_byte_identical(tmp_path: Path) -> None:
    # Affinity eviction must not break the provider-visible prefix: per-segment determinism
    # reproduces the same bytes when the whole transcript is recompressed from scratch.
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    client, _, runtime, providers = _compressed_runtime(tmp_path, config)
    first_user = "alpha beta gamma delta epsilon zeta"
    first = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": first_user}]},
    )
    reply = first.json()["choices"][0]["message"]["content"]
    turn_one = providers["haiku-4-5"].seen[0][1]

    with runtime._lock:
        runtime._affinity.clear()
        runtime._compressed.clear()

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [
                {"role": "user", "content": first_user},
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "eta theta iota kappa"},
            ],
        },
    )
    assert second.status_code == 200
    second_turns = providers["haiku-4-5"].seen[1][1]
    assert [(m.role, m.content) for m in second_turns[: len(turn_one) + 1]] == [
        *[(m.role, m.content) for m in turn_one],
        ("assistant", reply),
    ]


def test_compression_fields_populate_on_stream_path(tmp_path: Path) -> None:
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    client, log_path, _, providers = _compressed_runtime(tmp_path, config)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "stream": True,
            "messages": [{"role": "user", "content": "one two three four five six"}],
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert "[DONE]" in body
    assert "compress" not in body  # opaque on the stream too
    system, turns = providers["haiku-4-5"].seen[0]
    assert turns[0].content == "one two three"
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 1  # one record per request, stream included
    assert rows[0]["compressor_id"] == "truncate"
    assert cast("int", rows[0]["tokens_in_compressed"]) < cast("int", rows[0]["tokens_in_raw"])


def test_compression_fields_populate_on_error_path(tmp_path: Path) -> None:
    class _FailingProvider(_EchoProvider):
        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Completion:
            raise RuntimeError("upstream on fire")

    log_path = tmp_path / "requests.jsonl"
    policy = RoutingPolicy(
        kind="static",
        default_model="haiku-4-5",
        pool=_pool(),
        compression=CompressionConfig(compressor_id="truncate", aggressiveness=0.5),
    )
    runtime = EndpointRuntime(
        name="tau-bench", policy=policy, provider_factory=_FailingProvider, log=RequestLog(log_path)
    )
    response = TestClient(_app_for(runtime)).post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "one two three four five six"}],
        },
    )
    assert response.status_code == 502
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 1  # one record per request, error included
    assert rows[0]["status"] == "error"
    assert rows[0]["compressor_id"] == "truncate"
    assert cast("int", rows[0]["tokens_in_compressed"]) < cast("int", rows[0]["tokens_in_raw"])


def test_uncompressed_rows_keep_default_compression_fields(tmp_path: Path) -> None:
    client, log_path = _client(tmp_path)
    client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
    )
    row = _last_log_row(log_path)
    assert row["compressor_id"] == ""
    assert row["compressor_version"] == ""
    assert row["tokens_in_raw"] == 0
    assert row["tokens_in_compressed"] == 0
    assert row["aggressiveness"] == 0.0
