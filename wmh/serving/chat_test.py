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
from wmh.serving.chat import (
    EndpointRuntime,
    RequestLog,
    create_chat_router,
    install_openai_error_shapes,
)
from wmh.serving.endpoint_config import ENDPOINT_CONFIG_FILENAME, EndpointConfig

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
    # The anchors are what a slider labels itself with: measured quality and cost per position.
    assert [anchor["cost_quality"] for anchor in body["anchors"]] == [0.0, 0.25, 0.5, 0.75, 1.0]
    balanced = next(a for a in body["anchors"] if a["cost_quality"] == 0.25)
    assert balanced["named_point"] == "balanced"
    assert balanced["quality_delta_points"] == 0.99
    assert balanced["cost_delta_percent"] == -24.7
    assert balanced["knobs"] == {
        "knn_z": 0.5,
        "floor_q": 0.05,
        "pick_lam": 0.0,
        "guard_mode": "symmetric",
    }


def test_put_moves_the_dial_on_the_live_endpoint(tmp_path: Path) -> None:
    client, runtime = _dial_client(tmp_path, _knn_policy(tmp_path))
    put = client.put("/v1/endpoints/tau-bench/config", json={"cost_quality": 1.0})
    assert put.status_code == 200
    body = put.json()
    assert body["cost_quality"] == 1.0
    assert body["named_point"] == "savings-max"
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
