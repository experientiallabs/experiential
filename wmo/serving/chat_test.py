"""Tests for the OpenAI-compatible chat endpoint (routing, streaming, affinity, request log)."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from wmo.optimize.policy import (
    KNN_BANK_FILENAME,
    POLICY_FILENAME,
    ClusterRanking,
    EmbedderSpec,
    KnnBank,
    RoutingPolicy,
)
from wmo.providers.base import (
    Completion,
    Message,
    ProviderKind,
    StreamChunk,
    TokenUsage,
    VerifyResult,
)
from wmo.providers.openrouter_pricing import CATALOG_PATH_ENV, PriceCatalog
from wmo.providers.pool import PoolEntry, load_pool
from wmo.retrieval.embedders import HashingEmbedder
from wmo.serving.chat import (
    EndpointRuntime,
    RequestLog,
    RequestLogRecord,
    create_chat_router,
    install_openai_error_shapes,
)
from wmo.serving.endpoint_config import ENDPOINT_CONFIG_FILENAME, EndpointConfig
from wmo.serving.savings import EndpointSavings, SavingsWindow
from wmo.tracking.pricing import ModelPrice

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from wmo.providers.base import ProviderConfig


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
    assert response.headers["x-wmo-routed-model"] == "haiku-4-5"


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
    assert first.headers["x-wmo-routed-model"] == "fable-5"
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
    assert second.headers["x-wmo-routed-model"] == "fable-5"


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
    from wmo.serving.server import create_app

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
    from wmo.serving.server import create_app

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
    from wmo.serving.server import create_app

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
        # What the fitter persists for this bank: the mean of the per-model mean cell costs.
        cost_scale=0.0055,
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
    assert routed.headers["x-wmo-routed-model"] == "fable-5"
    assert routed.json()["choices"][0]["message"]["content"] == "served by fable-5"

    fallback = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "draft a thank-you"}]},
    )
    assert fallback.headers["x-wmo-routed-model"] == "haiku-4-5"

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [row["model"] for row in rows] == ["fable-5", "haiku-4-5"]
    assert rows[0]["routing_reason"].startswith("knn: ")
    assert rows[0]["cluster_id"] is None  # a knn decision cites neighbors, not clusters


def test_runtime_builds_the_policy_embedder_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An azure embedder spec would otherwise construct a fresh SDK client per request.
    from wmo.optimize.policy import EmbedderSpec

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
    from wmo.optimize.policy import EmbedderSpec

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


# --- the cost/quality dial: GET/PUT /v1/endpoints/{name}/config -----------------------------


def _dial_client(
    tmp_path: Path, policy: RoutingPolicy, *, config_path: Path | None = None
) -> tuple[TestClient, EndpointRuntime]:
    """A client with OpenAI error shapes installed, so 400s look like the real server's."""
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=policy,
        provider_factory=_EchoProvider,
        log=RequestLog(tmp_path / "requests.jsonl"),
        config_path=config_path,
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    install_openai_error_shapes(app)
    return TestClient(app), runtime


def test_config_reports_an_as_fitted_endpoint_with_the_measured_anchors(tmp_path: Path) -> None:
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    body = client.get("/v1/endpoints/tau-bench/config").json()
    assert body["endpoint"] == "tau-bench"
    assert body["dialable"] is True
    # Nobody has set the dial, and mounting did not set one for them.
    assert body["cost_quality"] is None
    assert body["named_point"] == "as-fitted"
    # The anchors are what a slider labels itself with: measured quality and cost per position,
    # sorted, and carrying nothing else. A client interpolates between them itself, so the
    # response must never hand it a delta for a position nobody measured.
    assert [anchor["s"] for anchor in body["anchors"]] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert [anchor["label"] for anchor in body["anchors"]] == [
        "Quality max",
        "Balanced (default)",
        "Cost saver",
        "Deep saver",
        "Max savings",
    ]
    balanced = next(anchor for anchor in body["anchors"] if anchor["s"] == 0.25)
    assert balanced == {
        "s": 0.25,
        "label": "Balanced (default)",
        "quality_delta_pt": 0.99,
        "cost_delta_pct": -24.7,
    }


def test_a_dial_between_anchors_is_labelled_custom(tmp_path: Path) -> None:
    # Never borrow the nearer anchor's name: its label sits next to its measured numbers.
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    body = client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 0.26}).json()
    assert body["named_point"] == "Custom"
    assert body["cost_quality"] == 0.26


def test_put_moves_the_dial_on_the_live_endpoint(tmp_path: Path) -> None:
    client, runtime = _dial_client(tmp_path, _knn_policy(tmp_path))
    put = client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 1.0})
    assert put.status_code == 200
    body = put.json()
    assert body["cost_quality"] == 1.0
    assert body["named_point"] == "Max savings"
    assert body["knobs"] == {
        "knn_z": 0.5,
        "floor_q": 0.05,
        "pick_lam": 0.03,
        "guard_mode": "asymmetric",
    }
    # The runtime is serving the new policy, not just reporting it.
    assert runtime.policy.pick_lam == 0.03
    assert runtime.policy.guard_mode == "asymmetric"
    assert client.get("/v1/endpoints/tau-bench/config").json() == body
    # And the routed request says the knob was in play.
    routed = client.post(
        "/v1/chat/completions",
        json={
            "model": "tau-bench",
            "messages": [{"role": "user", "content": "SELECT count(*) FROM superheroes"}],
        },
    )
    assert routed.status_code == 200
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert "cost knob lam=0.03" in rows[-1]["routing_reason"]


def test_put_persists_the_dial_next_to_the_policy(tmp_path: Path) -> None:
    config_path = tmp_path / ENDPOINT_CONFIG_FILENAME
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path), config_path=config_path)
    client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 0.6})
    # A restart has to come back on the same dial, so the file is the record.
    assert EndpointConfig.load(config_path).cost_quality == 0.6
    restarted = EndpointRuntime(
        name="tau-bench",
        policy=_knn_policy(tmp_path),
        log=RequestLog(tmp_path / "requests.jsonl"),
        cost_quality=EndpointConfig.load(config_path).cost_quality,
    )
    assert restarted.cost_quality == 0.6
    assert restarted.policy.guard_mode == "asymmetric"


def test_mounting_a_dial_setting_applies_it_before_the_first_request(tmp_path: Path) -> None:
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=_knn_policy(tmp_path),
        provider_factory=_EchoProvider,
        log=RequestLog(tmp_path / "requests.jsonl"),
        cost_quality=0.75,
    )
    assert runtime.cost_quality == 0.75
    assert runtime.policy.pick_lam == pytest.approx(0.02)


def test_put_rejects_a_dial_outside_the_range(tmp_path: Path) -> None:
    client, runtime = _dial_client(tmp_path, _knn_policy(tmp_path))
    response = client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 1.5})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert "cost_quality" in response.json()["error"]["message"]
    assert runtime.cost_quality is None  # a rejected change changes nothing


def test_config_on_a_policy_kind_without_a_dial(tmp_path: Path) -> None:
    static = RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool())
    client, _ = _dial_client(tmp_path, static)
    body = client.get("/v1/endpoints/tau-bench/config").json()
    assert body["dialable"] is False
    assert body["cost_quality"] is None and body["knobs"] is None
    conflict = client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 0.5})
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "dial_unavailable"
    assert "kind='static'" in conflict.json()["error"]["message"]


def test_put_409s_when_the_policy_carries_no_cost_evidence(tmp_path: Path) -> None:
    # The coverage leg needs no prices; the price leg cannot be honored without them, and
    # saying so beats serving a dial position that silently does nothing.
    priceless = _knn_policy(tmp_path).model_copy(update={"cost_scale": 0.0})
    client, _ = _dial_client(tmp_path, priceless)
    assert (
        client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 0.2}).status_code == 200
    )
    conflict = client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 0.8})
    assert conflict.status_code == 409
    assert "cost_scale" in conflict.json()["error"]["message"]


def test_config_404s_for_an_unknown_endpoint(tmp_path: Path) -> None:
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    missing = client.get("/v1/endpoints/nope/config")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"
    assert "tau-bench" in missing.json()["error"]["message"]
    assert client.put("/v1/endpoints/nope/config", json={"cost_quality": 0.5}).status_code == 404


def test_the_openai_surface_is_untouched_by_the_dial_routes(tmp_path: Path) -> None:
    # The customer-facing contract must not grow: /v1/models still lists endpoints only, and a
    # chat completion still names the endpoint.
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 0.9})
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["tau-bench"]
    completion = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert completion.status_code == 200
    assert completion.json()["model"] == "tau-bench"
    assert "cost_quality" not in completion.text


def test_put_rejects_non_finite_dial_values(tmp_path: Path) -> None:
    # A slider bug that sends NaN must be a readable 400: NaN fails every comparison the guard
    # makes, so accepting it would quietly stop the endpoint from routing at all.
    client, runtime = _dial_client(tmp_path, _knn_policy(tmp_path))
    for payload in ('{"cost_quality": NaN}', '{"cost_quality": Infinity}'):
        response = client.put(
            "/v1/endpoints/tau-bench/config",
            content=payload,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400, payload
        assert response.json()["error"]["code"] == "invalid_request"
    assert runtime.cost_quality is None


# --- the savings card: GET /v1/endpoints/{name}/savings --------------------------------------


def _served_request(client: TestClient, content: str) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": content}]},
    )
    assert response.status_code == 200


def test_savings_start_at_the_empty_state(tmp_path: Path) -> None:
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    body = client.get("/v1/endpoints/tau-bench/savings").json()
    assert body["requests_served"] == 0
    assert body["cost_saved_usd"] == 0.0
    assert body["cost_saved_pct"] == 0.0
    assert body["time_saved_s_estimate"] == 0.0
    assert body["expected_quality_delta_pt"] == 0.0
    assert body["window"] == "all_time"
    assert body["estimate_basis"]  # never empty: the card explains the zero


def test_savings_accrue_as_the_endpoint_serves(tmp_path: Path) -> None:
    client, runtime = _dial_client(tmp_path, _knn_policy(tmp_path))
    _served_request(client, "SELECT count(*) FROM superheroes")  # routes to fable-5
    _served_request(client, "draft a thank-you")  # stays on the haiku-4-5 fallback
    body = client.get("/v1/endpoints/tau-bench/savings").json()
    assert body["requests_served"] == 2
    # fable-5 is the pricier model here, so routing away from the cheap fallback COSTS money and
    # the card says so with a negative saving rather than hiding it.
    assert body["actual_cost_usd"] > body["baseline_cost_estimate_usd"]
    assert body["cost_saved_usd"] < 0.0
    assert any("haiku-4-5" in basis for basis in body["estimate_basis"])


def test_savings_window_parameter_selects_the_period(tmp_path: Path) -> None:
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    assert client.get("/v1/endpoints/tau-bench/savings?window=7d").json()["window"] == "7d"
    bad = client.get("/v1/endpoints/tau-bench/savings?window=forever")
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "invalid_request"


def test_savings_survive_a_restart_by_reading_the_log(tmp_path: Path) -> None:
    # The persisted JSONL is the source, so a customer's savings are not reset by a deploy.
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    _served_request(client, "SELECT count(*) FROM superheroes")
    restarted, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    body = restarted.get("/v1/endpoints/tau-bench/savings").json()
    assert body["requests_served"] == 1


def test_savings_ignore_other_endpoints_rows(tmp_path: Path) -> None:
    log = RequestLog(tmp_path / "requests.jsonl")
    log.append(
        RequestLogRecord(
            id="other",
            ts=datetime.now(UTC).isoformat(),
            endpoint="somebody-else",
            model="haiku-4-5",
            provider_model="claude-haiku-4-5",
            routing_reason="test",
            input_tokens=1000,
            output_tokens=1000,
            cost_usd=1.0,
        )
    )
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=_knn_policy(tmp_path),
        provider_factory=_EchoProvider,
        log=log,
    )
    assert runtime.savings().requests_served == 0


def test_savings_skip_an_unreadable_log_row(tmp_path: Path) -> None:
    # A line truncated by a hard kill must not take the whole card down.
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    _served_request(client, "SELECT count(*) FROM superheroes")
    with (tmp_path / "requests.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"id": "truncated"\n')
    body = client.get("/v1/endpoints/tau-bench/savings").json()
    assert body["requests_served"] == 1


def test_savings_are_available_for_a_static_endpoint(tmp_path: Path) -> None:
    static = RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool())
    client, _ = _dial_client(tmp_path, static)
    _served_request(client, "hi")
    body = client.get("/v1/endpoints/tau-bench/savings").json()
    assert body["requests_served"] == 1
    assert body["cost_saved_usd"] == 0.0  # nothing to save against itself
    assert body["expected_quality_delta_pt"] == 0.0


def test_savings_404_for_an_unknown_endpoint(tmp_path: Path) -> None:
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    missing = client.get("/v1/endpoints/nope/savings")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"


def test_moving_the_dial_refreshes_the_quality_expectation(tmp_path: Path) -> None:
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    _served_request(client, "draft a thank-you")
    client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 1.0})
    body = client.get("/v1/endpoints/tau-bench/savings").json()
    assert body["expected_quality_delta_pt"] == -0.54  # the Max savings anchor, not a stale 0.0


def test_savings_are_recomputed_when_the_log_grows_not_on_a_timer(tmp_path: Path) -> None:
    # Cached between requests (a polling dashboard must not re-read the JSONL every paint), and
    # invalidated by the request that changes the total, so the card is never stale.
    client, runtime = _dial_client(tmp_path, _knn_policy(tmp_path))
    reads = {"n": 0}
    original = RequestLog.replay

    def counting_replay(self: RequestLog, endpoint: str) -> list[RequestLogRecord]:
        reads["n"] += 1
        return original(self, endpoint)

    RequestLog.replay = counting_replay
    try:
        client.get("/v1/endpoints/tau-bench/savings")
        client.get("/v1/endpoints/tau-bench/savings")
        assert reads["n"] == 1  # the second read came from the cache
        _served_request(client, "draft a thank-you")
        assert client.get("/v1/endpoints/tau-bench/savings").json()["requests_served"] == 1
        assert reads["n"] == 2
    finally:
        RequestLog.replay = original
    assert runtime.savings().requests_served == 1


def test_failed_dial_persist_leaves_the_live_endpoint_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Persist-then-install: a dial whose file write fails must not serve until restart un-sets
    # it, and must not move the live endpoint at all.
    from wmo.serving.endpoint_config import EndpointConfig

    _, runtime = _dial_client(
        tmp_path, _knn_policy(tmp_path), config_path=tmp_path / "endpoint.toml"
    )
    runtime.set_cost_quality(0.25)
    before = runtime.policy

    def exploding_save(self: EndpointConfig, path: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(EndpointConfig, "save", exploding_save)
    with pytest.raises(OSError, match="disk full"):
        runtime.set_cost_quality(1.0)
    assert runtime.policy is before  # live dial unmoved
    assert runtime.cost_quality == 0.25


def test_slow_savings_computation_cannot_resurrect_the_old_dial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A savings computation that captures the old policy, races a dial move, and finishes late
    # must NOT store its stale result under a revision the new dial answers to.
    import wmo.serving.chat as chat_module

    client, runtime = _dial_client(tmp_path, _knn_policy(tmp_path))
    runtime.set_cost_quality(0.25)
    # The empty log zeroes every savings field; serve one request so the quality expectation
    # actually reflects the dial.
    client.post(
        "/v1/chat/completions",
        json={"model": "tau-bench", "messages": [{"role": "user", "content": "hi"}]},
    )
    original = chat_module.compute_savings

    moved = {"done": False}

    def racing_compute(
        rows: list[RequestLogRecord],
        policy: RoutingPolicy,
        *,
        window: SavingsWindow = "all_time",
    ) -> EndpointSavings:
        if not moved["done"]:
            moved["done"] = True
            runtime.set_cost_quality(1.0)  # the dial moves mid-computation
        return original(rows, policy, window=window)

    monkeypatch.setattr(chat_module, "compute_savings", racing_compute)
    stale = runtime.savings()  # computed against the 0.25 policy, returned to ITS caller
    fresh = runtime.savings()  # must recompute against the 1.0 policy, not read a stale cache
    assert fresh.expected_quality_delta_pt != stale.expected_quality_delta_pt
    assert fresh.expected_quality_delta_pt == pytest.approx(-0.54, abs=0.01)


def test_config_reports_the_coverage_setting_the_policy_was_fitted_with(tmp_path: Path) -> None:
    # An as-fitted endpoint's knobs must describe THAT fit, not the dial's default: a policy
    # fitted at the quality-max coverage setting reports 0.5, and one whose fit never recorded a
    # coverage setting reports null rather than a 0.0 that reads as "no floor".
    fitted_wide = _knn_policy(tmp_path).model_copy(update={"floor_q": 0.5})
    client, _ = _dial_client(tmp_path, fitted_wide)
    assert client.get("/v1/endpoints/tau-bench/config").json()["knobs"]["floor_q"] == 0.5

    unrecorded = _knn_policy(tmp_path).model_copy(update={"floor_q": None})
    older, _ = _dial_client(tmp_path, unrecorded)
    body = older.get("/v1/endpoints/tau-bench/config").json()
    assert body["knobs"]["floor_q"] is None
    assert body["knobs"]["knn_z"] == 0.5  # the rest of the knobs still report
    # Dialing it fixes the gap: the mapping records the quantile it applied.
    dialed = older.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 0.0}).json()
    assert dialed["knobs"]["floor_q"] == 0.5


def test_the_seven_day_savings_window_is_never_served_from_cache(tmp_path: Path) -> None:
    # A bounded window ages with the clock, so an idle endpoint must not keep serving the answer
    # it computed an hour ago; the all-time card is still cached between requests.
    client, _ = _dial_client(tmp_path, _knn_policy(tmp_path))
    _served_request(client, "draft a thank-you")
    reads = {"n": 0}
    original = RequestLog.replay

    def counting_replay(self: RequestLog, endpoint: str) -> list[RequestLogRecord]:
        reads["n"] += 1
        return original(self, endpoint)

    RequestLog.replay = counting_replay
    try:
        client.get("/v1/endpoints/tau-bench/savings?window=7d")
        client.get("/v1/endpoints/tau-bench/savings?window=7d")
        assert reads["n"] == 2  # recomputed every read
        client.get("/v1/endpoints/tau-bench/savings")
        client.get("/v1/endpoints/tau-bench/savings")
        assert reads["n"] == 3  # all_time cached after the first
    finally:
        RequestLog.replay = original


def test_an_openrouter_candidate_is_served_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `kind = "openrouter"` pool entry survives the whole serving path.

    Catalog-resolved price -> valid PoolEntry -> policy artifact on disk -> mounted endpoint ->
    an OpenAI-compatible completion attributed to that candidate. The provider factory is the
    usual echo fake (no network); `pool_test.py` covers the real `pool_provider` resolution.
    """
    catalog = tmp_path / "openrouter-prices.json"
    catalog.write_text(
        PriceCatalog(
            fetched_at=time.time(),
            source="test fixture",
            prices={"z-ai/glm-4.6": ModelPrice(input_per_mtok=0.4, output_per_mtok=1.75)},
        ).model_dump_json(),
        encoding="utf-8",
    )
    monkeypatch.setenv(CATALOG_PATH_ENV, str(catalog))
    pool_file = tmp_path / "pool.toml"
    pool_file.write_text(
        '[[model]]\nname = "or-glm"\nkind = "openrouter"\nmodel = "z-ai/glm-4.6"\ntier = "open"\n',
        encoding="utf-8",
    )
    policy_path = tmp_path / POLICY_FILENAME
    RoutingPolicy(kind="static", default_model="or-glm", pool=load_pool(pool_file).models).save(
        policy_path
    )

    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy.load(policy_path),
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
    assert response.headers["x-wmo-routed-model"] == "or-glm"
    assert response.json()["choices"][0]["message"]["content"] == "served by or-glm"
