"""Adversarial online router runtime and endpoint tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from wmo.common.models import (
    AssistantAction,
    Embedding,
    ModelCapabilities,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    RoutedCandidateSnapshot,
    ToolCall,
    Usage,
)
from wmo.common.routing import KnnGuard, KnnRouterPolicy
from wmo.common.routing.bank import (
    CandidateEvidenceCount,
    KnnBankManifest,
    KnnEvidenceBank,
    bank_bytes,
)
from wmo.common.routing.features import ROUTER_FEATURE_SCHEMA_SHA256
from wmo.runtime.models import ResolvedModel, RuntimeModelCatalog
from wmo.runtime.router import RouterRuntime, RouterRuntimeIntegrityError, create_router_endpoint

_TIME = datetime(2026, 8, 12, tzinfo=UTC)
_DIGEST = "a" * 64


class _Client:
    """Deterministic embedding and completion client with captured structured requests."""

    def __init__(self) -> None:
        self.embedding_values: tuple[Embedding, ...] | Exception = (Embedding(values=(1.0, 0.0)),)
        self.requests: list[ModelRequest] = []

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        del texts
        if isinstance(self.embedding_values, Exception):
            raise self.embedding_values
        return self.embedding_values

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            output=AssistantAction(
                tool_calls=(ToolCall(call_id="call-out", name="write", arguments={"x": 1}),)
            ),
            model=_snapshot("cheap"),
            economics=OperationEconomics(
                usage=Usage(input_tokens=7, output_tokens=3, cached_input_tokens=2)
            ),
            finish_reason=ModelFinishReason.COMPLETED,
        )


class _Catalog:
    """Exact static identity catalog with no environment or provider access."""

    def __init__(self, snapshots: dict[str, ModelSnapshot], client: _Client) -> None:
        self.snapshots = snapshots
        self.client = client

    def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
        return self.snapshots[alias], ModelCapabilities(supports_embeddings=alias == "embedder")

    def resolve(self, alias: str) -> ResolvedModel:
        snapshot, capabilities = self.snapshot(alias)
        return ResolvedModel(
            alias,
            snapshot,
            capabilities,
            self.client,
            self.client if capabilities.supports_embeddings else None,
        )


@pytest.mark.parametrize(
    "bad",
    [
        RuntimeError("embed failed"),
        (),
        (Embedding(values=(1.0, 0.0)), Embedding(values=(1.0, 0.0))),
        (Embedding(values=(1.0, 0.0, 0.0)),),
        (cast(Embedding, SimpleNamespace(values=(float("nan"), 0.0))),),
        (cast(Embedding, SimpleNamespace(values=(0.0, 0.0))),),
    ],
)
def test_request_embedding_failures_always_fall_back_with_evidence(bad: object) -> None:
    """Count, dimension, NaN, zero-vector, and embed exceptions never escape selection."""
    runtime, client = _runtime()
    client.embedding_values = cast(tuple[Embedding, ...] | Exception, bad)

    decision = runtime.select(_request(), episode_id="episode-a")

    assert decision.selected_alias == runtime.policy.baseline_alias
    assert decision.fallback_reason == "embedding_error"
    assert decision.neighbor_count == decision.paired_count == 0


def test_episode_stickiness_uses_caller_identity_and_tools_affect_request_hash() -> None:
    """An episode stays on its first alias while separate IDs and tool schemas remain distinct."""
    runtime, client = _runtime()
    first = runtime.select(_request(tool_name="read"), episode_id="episode-a")
    client.embedding_values = ()
    with pytest.raises(ValueError, match="different request-visible hash"):
        runtime.select(_request(tool_name="write"), episode_id="episode-a")
    sticky = runtime.select(_request(tool_name="read"), episode_id="episode-a")
    separate = runtime.select(_request(tool_name="read"), episode_id="episode-b")

    assert first.selected_alias == sticky.selected_alias == "cheap"
    assert sticky == first
    assert sticky.decision_id == first.decision_id
    assert separate.selected_alias == "baseline"
    assert separate.episode_id_sha256 != first.episode_id_sha256
    assert set(runtime._episode_decisions) == {  # noqa: SLF001 - privacy regression
        hashlib.sha256(b"episode-a").hexdigest(),
        hashlib.sha256(b"episode-b").hexdigest(),
    }
    assert "episode-a" not in first.model_dump_json()


def test_artifact_mutation_and_pricing_or_alias_drift_block_activation() -> None:
    """Artifact integrity rejects while request-time problems conservatively fall back."""
    policy, manifest, bank, snapshots, client = _fixture()
    catalog = cast(RuntimeModelCatalog, _Catalog(snapshots, client))
    with pytest.raises(RouterRuntimeIntegrityError, match="pricing snapshot"):
        RouterRuntime(policy, manifest, bank, catalog, pricing_snapshot_id="pricing-b")
    drifted = dict(snapshots)
    drifted["cheap"] = snapshots["cheap"].model_copy(update={"connection_sha256": "f" * 64})
    with pytest.raises(RouterRuntimeIntegrityError, match="connection digest"):
        RouterRuntime(
            policy,
            manifest,
            bank,
            cast(RuntimeModelCatalog, _Catalog(drifted, client)),
            pricing_snapshot_id="pricing-a",
        )
    runtime = RouterRuntime(policy, manifest, bank, catalog, pricing_snapshot_id="pricing-a")
    bank.scores.setflags(write=True)
    bank.scores[0, 0] = 0.0
    with pytest.raises(RouterRuntimeIntegrityError, match="bank content has mutated"):
        runtime.select(_request(), episode_id="episode-a")


def test_complete_consumes_one_prior_decision_without_selecting_again() -> None:
    """Python completion validates and reuses a supplied decision exactly once."""
    policy, manifest, bank, snapshots, client = _fixture()
    recorded = []
    runtime = RouterRuntime(
        policy,
        manifest,
        bank,
        cast(RuntimeModelCatalog, _Catalog(snapshots, client)),
        pricing_snapshot_id="pricing-a",
        decision_sink=recorded.append,
    )
    request = _request(tool_name="read")
    decision = runtime.select(request, episode_id="episode-a")
    routed = runtime.complete(request, episode_id="episode-a", decision=decision)

    assert routed.decision is decision
    assert len(recorded) == 1
    assert client.requests == [request]
    with pytest.raises(ValueError, match="does not match"):
        runtime.complete(_request(tool_name="write"), episode_id="episode-a", decision=decision)


def test_endpoint_requires_episode_rejects_stream_and_preserves_tool_transcript() -> None:
    """HTTP is non-streaming and passes assistant calls plus ordered tool results unchanged."""
    runtime, client = _runtime()
    app = FastAPI()
    app.include_router(create_router_endpoint({"router-a": runtime}))
    http = TestClient(app)
    payload = {
        "model": "router-a",
        "messages": [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-in",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"a"}'},
                    }
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "call-in"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "read a file",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }
    assert http.post("/v1/chat/completions", json=payload).status_code == 400
    assert (
        http.post(
            "/v1/chat/completions",
            json={**payload, "stream": True},
            headers={"X-WMO-Episode-ID": "episode-a"},
        ).status_code
        == 400
    )
    response = http.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-WMO-Episode-ID": "episode-a"},
    )

    assert response.status_code == 200
    assert response.headers["X-WMO-Episode-ID-SHA256"] == hashlib.sha256(b"episode-a").hexdigest()
    assert "episode-a" not in response.text
    assert response.json()["choices"][0]["message"]["tool_calls"][0]["id"] == "call-out"
    captured = client.requests[-1]
    assert [message.role for message in captured.messages] == ["user", "assistant", "tool"]
    assert captured.messages[1].assistant_action is not None
    assert captured.messages[1].assistant_action.tool_calls[0].call_id == "call-in"
    assert captured.messages[2].tool_call_id == "call-in"
    conflict = http.post(
        "/v1/chat/completions",
        json={**payload, "tools": []},
        headers={"X-WMO-Episode-ID": "episode-a"},
    )
    assert conflict.status_code == 409
    assert "episode-a" not in conflict.text


def _request(*, tool_name: str | None = None) -> ModelRequest:
    from wmo.common.tasks import ToolSchema

    tools = (
        (ToolSchema(name=tool_name, description="fixture", input_schema={"type": "object"}),)
        if tool_name is not None
        else ()
    )
    return ModelRequest(messages=(ModelMessage(role="user", content="route me"),), tools=tools)


def _runtime() -> tuple[RouterRuntime, _Client]:
    policy, manifest, bank, snapshots, client = _fixture()
    return (
        RouterRuntime(
            policy,
            manifest,
            bank,
            cast(RuntimeModelCatalog, _Catalog(snapshots, client)),
            pricing_snapshot_id="pricing-a",
        ),
        client,
    )


def _fixture() -> tuple[
    KnnRouterPolicy,
    KnnBankManifest,
    KnnEvidenceBank,
    dict[str, ModelSnapshot],
    _Client,
]:
    snapshots = {alias: _snapshot(alias) for alias in ("baseline", "cheap", "embedder")}
    client = _Client()
    bank = KnnEvidenceBank(
        task_ids=tuple(f"task-{index}" for index in range(8)),
        candidate_aliases=("baseline", "cheap"),
        embeddings=np.asarray(((1.0, 0.0),) * 8, dtype=np.float32),
        scores=np.asarray(((0.8, 0.9),) * 8, dtype=np.float32),
        candidate_costs=np.asarray(((0.5, 0.1),) * 8, dtype=np.float64),
        score_counts=np.ones((8, 2), dtype=np.int32),
        cost_counts=np.ones((8, 2), dtype=np.int32),
        workload_weights=np.ones(8, dtype=np.float64),
        novelty_floor=0.5,
    )
    digest = hashlib.sha256(bank_bytes(bank)).hexdigest()
    manifest = KnnBankManifest(
        schema_version=1,
        created_at=_TIME,
        code_revision="test",
        bank_artifact_id="bank-a",
        fit_evaluation_id="fit-a",
        task_set_id="tasks-a",
        task_ids=bank.task_ids,
        candidate_aliases=bank.candidate_aliases,
        embedder_alias="embedder",
        embedder=snapshots["embedder"],
        feature_extractor_id="request-visible-v2",
        feature_schema_sha256=ROUTER_FEATURE_SCHEMA_SHA256,
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256=_DIGEST,
        bank_sha256=digest,
        embedding_dimension=2,
        novelty_floor=0.5,
        evidence_counts=tuple(
            CandidateEvidenceCount(candidate_alias=alias, scored_task_count=8, costed_task_count=8)
            for alias in bank.candidate_aliases
        ),
    )
    policy = KnnRouterPolicy(
        schema_version=1,
        created_at=_TIME,
        code_revision="test",
        policy_id="policy-a",
        baseline_alias="baseline",
        candidates=tuple(
            RoutedCandidateSnapshot(alias=alias, model=snapshots[alias])
            for alias in bank.candidate_aliases
        ),
        embedder_alias="embedder",
        embedder=snapshots["embedder"],
        feature_extractor_id=manifest.feature_extractor_id,
        feature_schema_sha256=manifest.feature_schema_sha256,
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256=_DIGEST,
        bank_artifact_id="bank-a",
        bank_sha256=digest,
        guard=KnnGuard(
            maximum_neighbors=8,
            minimum_paired_observations=8,
            relative_similarity_threshold=0.95,
            uncertainty_multiplier=0.5,
            quality_tolerance=0,
        ),
        fit_evaluation_id="fit-a",
        judgment_status="provisional",
    )
    return policy, manifest, bank, snapshots, client


def _snapshot(alias: str) -> ModelSnapshot:
    return ModelSnapshot(
        provider="test",
        model_id=alias,
        revision="fixture",
        capabilities_sha256=_DIGEST,
        connection_sha256="b" * 64,
    )
