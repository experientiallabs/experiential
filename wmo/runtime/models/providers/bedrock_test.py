"""Bedrock client, region, credential-chain, and catalog resolution tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import pytest

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    ConnectionConfig,
    EmbeddingClient,
    ModelCapabilities,
    ModelCatalog,
    ModelClient,
    ModelMessage,
    ModelRecord,
    ModelRequest,
    ModelRoles,
    ModelSnapshot,
)
from wmo.runtime.models.providers.bedrock import (
    AWS_DEFAULT_REGION_ENV,
    AWS_REGION_ENV,
    NO_REGION_ERROR,
    BedrockClient,
    BedrockRegionError,
    BedrockRuntime,
    resolve_bedrock_region,
)
from wmo.runtime.models.providers.errors import ProviderResponseError
from wmo.runtime.models.providers.transport import (
    JsonHttpResponse,
    JsonHttpTransport,
    ProviderTransportError,
)
from wmo.runtime.models.registry import ModelConnectionError, RuntimeModelCatalog


class _UnusedTransport(JsonHttpTransport):
    """Fails if catalog construction unexpectedly tries to make an HTTP call."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Reject every attempted HTTP-shaped call."""
        del url, headers, payload, timeout_seconds
        raise AssertionError("Bedrock catalog construction must not use HTTP transport")


class _FakeBedrockRuntime:
    """Records Converse and InvokeModel calls without contacting AWS."""

    def __init__(
        self,
        *,
        converse_response: Mapping[str, object] | None = None,
        invoke_bodies: list[Mapping[str, object]] | None = None,
        converse_error: Exception | None = None,
    ) -> None:
        self.converse_calls: list[Mapping[str, object]] = []
        self.invoke_calls: list[Mapping[str, object]] = []
        self._converse_response = converse_response or {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 4,
                "outputTokens": 2,
                "cacheReadInputTokens": 1,
                "cacheWriteInputTokens": 1,
            },
        }
        self._invoke_bodies = list(invoke_bodies or [{"embedding": [3.0, 4.0]}])
        self._converse_error = converse_error

    def converse(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Record one Converse request and return the frozen response."""
        self.converse_calls.append(request)
        if self._converse_error is not None:
            raise self._converse_error
        return self._converse_response

    def invoke_model(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Record one InvokeModel request and return the next embedding body."""
        self.invoke_calls.append(request)
        if not self._invoke_bodies:
            raise AssertionError("test made an unexpected embedding request")
        return {"body": json.dumps(self._invoke_bodies.pop(0))}


def _snapshot(model_id: str = "us.anthropic.claude-sonnet-4-5") -> ModelSnapshot:
    """Build an immutable Bedrock identity fixture."""
    return ModelSnapshot(
        provider="bedrock",
        model_id=model_id,
        capabilities_sha256="a" * 64,
        connection_sha256="a" * 64,
    )


def _request() -> ModelRequest:
    """Build one user completion request."""
    return ModelRequest(messages=(ModelMessage(role="user", content="Hello"),))


def test_complete_sends_the_exact_model_id_and_preserves_cache_usage() -> None:
    """Converse receives the catalog model ID and reports cache-read plus cache-write subsets."""
    runtime = _FakeBedrockRuntime()
    client = BedrockClient(
        model=_snapshot("us.anthropic.claude-sonnet-4-5"),
        region="us-west-2",
        environment={},
        runtime_factory=lambda *, region_name: runtime,
    )

    response = client.complete(_request())

    assert isinstance(client, ModelClient)
    assert runtime.converse_calls[0]["modelId"] == "us.anthropic.claude-sonnet-4-5"
    assert response.output.content == "ok"
    assert response.economics.usage is not None
    assert response.economics.usage.input_tokens == 6
    assert response.economics.usage.cached_input_tokens == 1
    assert response.economics.usage.cache_write_input_tokens == 1


def test_embeddings_validate_count_dimensions_and_normalization() -> None:
    """Titan-style embeddings stay finite, matching, and unit-normalized."""
    runtime = _FakeBedrockRuntime(
        invoke_bodies=[{"embedding": [0.0, 2.0]}, {"embedding": [2.0, 0.0]}]
    )
    client = BedrockClient(
        model=_snapshot("amazon.titan-embed-text-v2:0"),
        region="us-east-1",
        environment={},
        runtime_factory=lambda *, region_name: runtime,
    )

    embeddings = client.embed(["a", "b"])

    assert isinstance(client, EmbeddingClient)
    assert embeddings[0].values == (0.0, 1.0)
    assert embeddings[1].values == (1.0, 0.0)
    assert runtime.invoke_calls[0]["modelId"] == "amazon.titan-embed-text-v2:0"
    body = json.loads(str(runtime.invoke_calls[0]["body"]))
    assert body["inputText"] == "a"
    assert body["normalize"] is True
    mismatched = _FakeBedrockRuntime(
        invoke_bodies=[{"embedding": [1.0, 0.0]}, {"embedding": [1.0, 0.0, 0.0]}]
    )
    bad = BedrockClient(
        model=_snapshot("amazon.titan-embed-text-v2:0"),
        region="us-east-1",
        environment={},
        runtime_factory=lambda *, region_name: mismatched,
    )
    with pytest.raises(ProviderResponseError, match="dimensions"):
        bad.embed(["a", "b"])


def test_region_precedence_is_catalog_then_aws_region_then_session() -> None:
    """Catalog region wins, then AWS_REGION, then the boto session chain."""
    assert resolve_bedrock_region("eu-west-1", {AWS_REGION_ENV: "us-east-1"}) == "eu-west-1"
    assert resolve_bedrock_region(None, {AWS_REGION_ENV: "us-east-1"}) == "us-east-1"
    assert (
        resolve_bedrock_region(
            None,
            {AWS_DEFAULT_REGION_ENV: "ap-south-1"},
            session_region="ap-south-1",
        )
        == "ap-south-1"
    )
    assert resolve_bedrock_region(None, {}) is None


def test_missing_region_names_every_source() -> None:
    """A missing region lists catalog, AWS_REGION, and the boto session chain."""
    client = BedrockClient(
        model=_snapshot(),
        region=None,
        environment={},
        runtime_factory=lambda *, region_name: _FakeBedrockRuntime(),
    )

    with pytest.raises(BedrockRegionError, match="catalog connection region") as captured:
        client.complete(_request())
    assert AWS_REGION_ENV in str(captured.value)
    assert AWS_DEFAULT_REGION_ENV in str(captured.value)
    assert str(captured.value) == NO_REGION_ERROR


def test_catalog_region_is_passed_to_the_runtime_factory() -> None:
    """The constructed client receives the resolved region_name and no API key."""
    seen: list[str] = []

    def factory(*, region_name: str) -> BedrockRuntime:
        seen.append(region_name)
        return _FakeBedrockRuntime()

    client = BedrockClient(
        model=_snapshot(),
        region="ca-central-1",
        environment={AWS_REGION_ENV: "us-east-1"},
        runtime_factory=factory,
    )
    client.complete(_request())

    assert seen == ["ca-central-1"]


def test_aws_region_beats_injected_factory_when_catalog_omits_region() -> None:
    """AWS_REGION is read before a request and passed as the explicit region_name."""
    seen: list[str] = []

    def factory(*, region_name: str) -> BedrockRuntime:
        seen.append(region_name)
        return _FakeBedrockRuntime()

    client = BedrockClient(
        model=_snapshot(),
        region=None,
        environment={AWS_REGION_ENV: "us-west-2"},
        runtime_factory=factory,
    )
    client.complete(_request())

    assert seen == ["us-west-2"]


def test_construction_does_not_import_boto_or_hold_the_lock_during_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Client construction stays lazy, and request work happens outside the instance lock."""
    imported: list[str] = []

    def forbidden_boto3() -> object:
        imported.append("boto3")
        raise AssertionError("tests must not import boto3")

    monkeypatch.setattr("wmo.runtime.models.providers.bedrock._import_boto3", forbidden_boto3)
    runtime = _FakeBedrockRuntime()
    client = BedrockClient(
        model=_snapshot(),
        region="us-east-1",
        environment={},
        runtime_factory=lambda *, region_name: runtime,
    )
    assert imported == []
    client.complete(_request())
    assert imported == []
    assert runtime.converse_calls


def test_concurrent_client_construction_does_not_deadlock() -> None:
    """First-request construction can be entered from many threads without holding request locks."""
    runtime = _FakeBedrockRuntime()
    created: list[str] = []

    def factory(*, region_name: str) -> BedrockRuntime:
        created.append(region_name)
        return runtime

    client = BedrockClient(
        model=_snapshot(),
        region="us-east-1",
        environment={},
        runtime_factory=factory,
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: client.complete(_request()), range(4)))
    assert [result.output.content for result in results] == ["ok", "ok", "ok", "ok"]
    assert created == ["us-east-1"]
    assert len(runtime.converse_calls) == 4


def test_retries_stay_on_the_same_region_and_model() -> None:
    """Transient Bedrock failures retry the same client without changing region or model."""

    class _FlakyRuntime(_FakeBedrockRuntime):
        """Fail once with a retryable transport error, then succeed."""

        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def converse(self, request: Mapping[str, object]) -> Mapping[str, object]:
            self.attempts += 1
            if self.attempts == 1:
                raise ProviderTransportError(
                    "Bedrock returned ThrottlingException", status_code=429
                )
            return super().converse(request)

    runtime = _FlakyRuntime()
    client = BedrockClient(
        model=_snapshot("exact-model"),
        region="us-east-1",
        environment={},
        runtime_factory=lambda *, region_name: runtime,
    )

    response = client.complete(_request())

    assert response.output.content == "ok"
    assert runtime.attempts == 2
    assert runtime.converse_calls[0]["modelId"] == "exact-model"


def test_catalog_rejects_bedrock_api_key_env_and_resolves_without_http() -> None:
    """Bedrock catalog records cannot carry api_key_env and resolve through RuntimeModelCatalog."""
    with pytest.raises(ValueError, match="api_key_env"):
        ConnectionConfig(provider="bedrock", api_key_env="AWS_ACCESS_KEY_ID")
    runtime = _FakeBedrockRuntime()
    catalog = RuntimeModelCatalog(
        ModelCatalog(
            connections={"bedrock": ConnectionConfig(provider="bedrock", region="us-east-1")},
            models={
                "claude": ModelRecord(
                    connection="bedrock",
                    model="us.anthropic.claude-sonnet-4-5",
                    capabilities=ModelCapabilities(
                        supports_tools=True,
                        supports_completions=True,
                    ),
                ),
                "embed": ModelRecord(
                    connection="bedrock",
                    model="amazon.titan-embed-text-v2:0",
                    capabilities=ModelCapabilities(supports_embeddings=True),
                ),
            },
            roles=ModelRoles(world_model="claude", judge="claude", embedder="embed"),
        ),
        environment={},
        transport_factory=_UnusedTransport,
        bedrock_runtime_factory=lambda *, region_name: runtime,
    )

    snapshot, _capabilities = catalog.snapshot("claude")
    resolved = catalog.resolve("claude")
    embedder = catalog.resolve("embed")
    response = resolved.client.complete(_request())
    vectors = embedder.embedding_client.embed(["hello"]) if embedder.embedding_client else ()

    assert snapshot.provider == "bedrock"
    assert snapshot.model_id == "us.anthropic.claude-sonnet-4-5"
    assert isinstance(resolved.client, BedrockClient)
    assert embedder.embedding_client is embedder.client
    assert response.output.content == "ok"
    assert vectors[0].values == (0.6, 0.8)
    with pytest.raises(ModelConnectionError, match="unsupported provider"):
        RuntimeModelCatalog(
            ModelCatalog(
                connections={"other": ConnectionConfig(provider="waterfall", api_key_env="X")},
                models={
                    "x": ModelRecord(
                        connection="other",
                        model="x",
                        capabilities=ModelCapabilities(),
                    )
                },
            ),
            environment={"X": "x"},
            transport_factory=_UnusedTransport,
        ).snapshot("x")


def test_snapshot_does_not_construct_a_bedrock_runtime() -> None:
    """Static identity never opens a Bedrock client or reads AWS credentials."""

    def forbidden(*, region_name: str) -> BedrockRuntime:
        del region_name
        raise AssertionError("snapshot must not construct a Bedrock runtime")

    catalog = RuntimeModelCatalog(
        ModelCatalog(
            connections={"bedrock": ConnectionConfig(provider="bedrock", region="us-east-1")},
            models={
                "claude": ModelRecord(
                    connection="bedrock",
                    model="us.anthropic.claude-sonnet-4-5",
                    capabilities=ModelCapabilities(supports_completions=True),
                )
            },
        ),
        environment={},
        transport_factory=_UnusedTransport,
        bedrock_runtime_factory=forbidden,
    )

    snapshot, capabilities = catalog.snapshot("claude")

    assert snapshot.provider == "bedrock"
    assert capabilities.supports_completions is True
