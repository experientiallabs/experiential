"""Bedrock client, region, credential-chain, catalog resolution, and Converse translation tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest

import exp.runtime.models.providers.bedrock_endpoints as bedrock_endpoints
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    EmbeddingClient,
    ModelCapabilities,
    ModelCatalog,
    ModelClient,
    ModelFinishReason,
    ModelMessage,
    ModelRecord,
    ModelRequest,
    ModelRoles,
    ModelSnapshot,
)
from exp.runtime.models.credentials import MissingModelCredentialError
from exp.runtime.models.providers.async_transport import RequestDeadline
from exp.runtime.models.providers.bedrock import (
    AWS_DEFAULT_REGION_ENV,
    AWS_REGION_ENV,
    CONNECT_TIMEOUT_SECONDS,
    NO_REGION_ERROR,
    READ_TIMEOUT_SECONDS,
    BedrockClient,
    BedrockRegionError,
    BedrockRuntime,
    BoundedBedrockClient,
    converse_response,
    create_bedrock_runtime_client,
    resolve_bedrock_region,
)
from exp.runtime.models.providers.errors import (
    ProviderRefusalError,
    ProviderRefusalSignal,
    ProviderResponseError,
)
from exp.runtime.models.providers.transport import ProviderTransportError, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog


class _FakeBedrockRuntime:
    """Records Converse and InvokeModel calls without contacting AWS."""

    def __init__(
        self,
        *,
        converse_response: Mapping[str, object] | None = None,
        converse_stream_response: Mapping[str, object] | None = None,
        invoke_bodies: list[Mapping[str, object]] | None = None,
    ) -> None:
        self.converse_calls: list[Mapping[str, object]] = []
        self.converse_stream_calls: list[Mapping[str, object]] = []
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
        self._converse_stream_response = converse_stream_response
        self._invoke_bodies = list(invoke_bodies or [{"embedding": [3.0, 4.0]}])

    def converse(self, **request: object) -> Mapping[str, object]:
        """Record one Converse request and return the frozen response."""
        self.converse_calls.append(request)
        return self._converse_response

    def converse_stream(self, **request: object) -> Mapping[str, object]:
        """Record and return a configured stream, rejecting unexpected calls."""
        self.converse_stream_calls.append(request)
        if self._converse_stream_response is None:
            raise AssertionError("test made an unexpected ConverseStream request")
        return self._converse_stream_response

    def invoke_model(self, **request: object) -> Mapping[str, object]:
        """Record one InvokeModel request and return the next embedding body."""
        self.invoke_calls.append(request)
        if not self._invoke_bodies:
            raise AssertionError("test made an unexpected embedding request")
        return {"body": json.dumps(self._invoke_bodies.pop(0))}


def _snapshot(model_id: str = "us.anthropic.claude-sonnet-4-5") -> ModelSnapshot:
    """Build an immutable Bedrock identity fixture."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
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
        environment={"BEDROCK_ACCESS_KEY_ID": "AKIAEXAMPLEKEY0001"},
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

    monkeypatch.setattr("exp.runtime.models.providers.bedrock._import_boto3", forbidden_boto3)
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

        def converse(self, **request: object) -> Mapping[str, object]:
            """Raise one retryable throttle error, then delegate to the recording fake."""
            self.attempts += 1
            if self.attempts == 1:
                raise ProviderTransportError(
                    "Bedrock returned ThrottlingException", status_code=429
                )
            return super().converse(**request)

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


def test_catalog_requires_a_complete_bedrock_access_key_pair_and_resolves_ambient() -> None:
    """Bedrock rejects half-configured credentials and preserves ambient-chain resolution."""
    with pytest.raises(ValueError, match="api_key_env"):
        ConnectionConfig(provider="bedrock", api_key_env="AWS_ACCESS_KEY_ID")
    runtime = _FakeBedrockRuntime()
    catalog = RuntimeModelCatalog(
        ModelCatalog(
            connections={"bedrock": ConnectionConfig(provider="bedrock", region="us-east-1")},
            models={
                "claude": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="bedrock",
                    model="us.anthropic.claude-sonnet-4-5",
                    capabilities=ModelCapabilities(
                        supports_tools=True,
                        supports_completions=True,
                    ),
                ),
                "embed": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="bedrock",
                    model="amazon.titan-embed-text-v2:0",
                    capabilities=ModelCapabilities(supports_embeddings=True),
                ),
            },
            roles=ModelRoles(world_model="claude", judge="claude", embedder="embed"),
        ),
        environment={},
        transport_factory=ScriptedJsonTransport,
        bedrock_runtime_factory=lambda *, region_name: runtime,
    )

    snapshot, _capabilities = catalog.snapshot("claude")
    resolved = catalog.resolve("claude")
    embedder = catalog.resolve("embed")
    response = resolved.client.complete(_request())
    vectors = embedder.embedding_client.embed(["hello"]) if embedder.embedding_client else ()

    async def complete_through_bounded_lane() -> str | None:
        """Exercise the catalog-exposed bounded async completion contract."""
        assert isinstance(resolved.client, BoundedBedrockClient)
        async_response = await resolved.client.complete_async(
            _request(),
            deadline=RequestDeadline.after(1),
        )
        return async_response.output.content

    assert snapshot.provider == "bedrock"
    assert snapshot.model_id == "us.anthropic.claude-sonnet-4-5"
    assert isinstance(resolved.client, BoundedBedrockClient)
    assert isinstance(embedder.client, BoundedBedrockClient)
    assert embedder.embedding_client is not embedder.client
    assert isinstance(embedder.embedding_client, BedrockClient)
    assert response.output.content == "ok"
    assert asyncio.run(complete_through_bounded_lane()) == "ok"
    assert vectors[0].values == (0.6, 0.8)


def test_catalog_resolves_explicit_bedrock_access_key_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A named secret access key and non-secret key ID reach the default boto seam together."""
    runtime = _FakeBedrockRuntime()
    seen: dict[str, str | None] = {}

    def create_runtime(
        *,
        region_name: str,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> BedrockRuntime:
        """Capture explicit credentials without exposing them in production diagnostics."""
        seen.update(
            region=region_name,
            access_key_id=aws_access_key_id,
            secret_access_key=aws_secret_access_key,
        )
        return runtime

    monkeypatch.setattr(
        "exp.runtime.models.providers.bedrock.create_bedrock_runtime_client", create_runtime
    )
    connection = ConnectionConfig(
        provider="bedrock",
        region="us-west-2",
        api_key_env="BEDROCK_SECRET_ACCESS_KEY",
        aws_access_key_id_env="BEDROCK_ACCESS_KEY_ID",
    )
    catalog = RuntimeModelCatalog(
        ModelCatalog(
            connections={"bedrock": connection},
            models={
                "claude": ModelRecord(
                    billing_source=BillingSource.HOST_MANAGED,
                    connection="bedrock",
                    model="us.anthropic.claude-sonnet-4-5",
                    capabilities=ModelCapabilities(supports_completions=True),
                )
            },
            roles=ModelRoles(world_model="claude", judge="claude"),
        ),
        environment={
            "BEDROCK_ACCESS_KEY_ID": "AKIAEXAMPLEKEY0001",
            "BEDROCK_SECRET_ACCESS_KEY": "resolved-secret-access-key",
        },
    )

    response = catalog.resolve("claude").client.complete(_request())

    assert response.output.content == "ok"
    assert seen == {
        "region": "us-west-2",
        "access_key_id": "AKIAEXAMPLEKEY0001",
        "secret_access_key": "resolved-secret-access-key",
    }


def test_catalog_resolves_bedrock_api_key_without_sigv4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Bedrock API key stays a per-connection bearer through both runtime paths."""
    runtime = _FakeBedrockRuntime()
    seen: dict[str, str | None] = {}

    def create_runtime(
        *,
        region_name: str,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        bearer_token: str | None = None,
    ) -> BedrockRuntime:
        seen.update(
            region=region_name,
            access_key_id=aws_access_key_id,
            secret_access_key=aws_secret_access_key,
            bearer_token=bearer_token,
        )
        return runtime

    monkeypatch.setattr(
        "exp.runtime.models.providers.bedrock.create_bedrock_runtime_client", create_runtime
    )
    connection = ConnectionConfig(
        provider="bedrock",
        region="us-west-2",
        api_key_env="BEDROCK_API_KEY",
        bedrock_auth_mode="api_key",
    )
    catalog = RuntimeModelCatalog(
        ModelCatalog(
            connections={"bedrock": connection},
            models={
                "claude": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="bedrock",
                    model="us.anthropic.claude-sonnet-4-5",
                    capabilities=ModelCapabilities(supports_completions=True),
                )
            },
            roles=ModelRoles(world_model="claude", judge="claude"),
        ),
        environment={"BEDROCK_API_KEY": "bedrock-bearer-token"},
    )

    resolved = catalog.resolve("claude")
    response = resolved.client.complete(_request())
    assert isinstance(resolved.client, BoundedBedrockClient)
    runtime_client = resolved.client._client  # noqa: SLF001
    assert isinstance(runtime_client, BedrockClient)
    headers = runtime_client.sign_gateway_dispatch(
        url=(
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/"
            "us.anthropic.claude-sonnet-4-5/converse-stream"
        ),
        body="{}",
    )

    assert response.output.content == "ok"
    assert seen == {
        "region": "us-west-2",
        "access_key_id": None,
        "secret_access_key": None,
        "bearer_token": "bedrock-bearer-token",
    }
    assert headers["authorization"] == "Bearer bedrock-bearer-token"
    with pytest.raises(ProviderTransportError, match="differs"):
        runtime_client.sign_gateway_dispatch(
            url="https://untrusted.example/collect",
            body="{}",
        )


def test_catalog_fails_closed_when_explicit_bedrock_secret_is_missing() -> None:
    """An explicit key ID never falls back to the ambient chain when its paired secret is absent."""
    catalog = RuntimeModelCatalog(
        ModelCatalog(
            connections={
                "bedrock": ConnectionConfig(
                    provider="bedrock",
                    region="us-east-1",
                    api_key_env="BEDROCK_SECRET_ACCESS_KEY",
                    aws_access_key_id_env="BEDROCK_ACCESS_KEY_ID",
                )
            },
            models={
                "claude": ModelRecord(
                    billing_source=BillingSource.HOST_MANAGED,
                    connection="bedrock",
                    model="us.anthropic.claude-sonnet-4-5",
                    capabilities=ModelCapabilities(supports_completions=True),
                )
            },
            roles=ModelRoles(world_model="claude", judge="claude"),
        ),
        environment={"BEDROCK_ACCESS_KEY_ID": "AKIAEXAMPLEKEY0001"},
    )

    with pytest.raises(MissingModelCredentialError, match="BEDROCK_SECRET_ACCESS_KEY"):
        catalog.resolve("claude")


def test_low_level_bedrock_factory_rejects_a_half_explicit_pair() -> None:
    """The lowest credential helper never falls through to ambient auth on a half pair."""
    from exp.runtime.models.providers.bedrock import _explicit_session_kwargs

    with pytest.raises(ValueError, match="both access-key fields"):
        _explicit_session_kwargs("AKIAEXAMPLEKEY0001", None)
    with pytest.raises(ValueError, match="both access-key fields"):
        _explicit_session_kwargs(None, "secret-access-key")


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
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="bedrock",
                    model="us.anthropic.claude-sonnet-4-5",
                    capabilities=ModelCapabilities(supports_completions=True),
                )
            },
        ),
        environment={},
        transport_factory=ScriptedJsonTransport,
        bedrock_runtime_factory=forbidden,
    )

    snapshot, capabilities = catalog.snapshot("claude")

    assert snapshot.provider == "bedrock"
    assert capabilities.supports_completions is True


def test_runtime_construction_uses_the_aws_session_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session() carries no static keys; the client gets an explicit region and bounded config."""

    class _Recorded:
        """Mutable construction evidence for the fake boto session."""

        session_called = False
        service = ""
        region = ""
        connect_timeout = 0.0
        read_timeout = 0.0
        retries: Mapping[str, object] = {}
        tcp_keepalive = False

    class _FakeConfig:
        """Record botocore Config keyword arguments."""

        def __init__(
            self,
            *,
            connect_timeout: float,
            read_timeout: float,
            retries: Mapping[str, object],
            tcp_keepalive: bool,
        ) -> None:
            """Store the exact Config values used to construct bedrock-runtime."""
            _Recorded.connect_timeout = connect_timeout
            _Recorded.read_timeout = read_timeout
            _Recorded.retries = retries
            _Recorded.tcp_keepalive = tcp_keepalive

    class _FakeSession:
        """Record the service client request made from a default boto session."""

        def client(self, service_name: str, *, region_name: str, config: object) -> object:
            """Store construction arguments and return a dummy client."""
            del config
            _Recorded.service = service_name
            _Recorded.region = region_name
            return object()

    class _FakeBoto3:
        """Stand in for boto3 and prove Session() is called with no credential kwargs."""

        def Session(self) -> _FakeSession:
            """Return a default session without access-key arguments."""
            _Recorded.session_called = True
            return _FakeSession()

    monkeypatch.setattr(
        "exp.runtime.models.providers.bedrock._import_boto3",
        lambda: _FakeBoto3(),
    )
    monkeypatch.setattr(
        "exp.runtime.models.providers.bedrock._import_botocore_config",
        lambda: _FakeConfig,
    )

    runtime = create_bedrock_runtime_client(region_name="eu-central-1")

    assert _Recorded.session_called is True
    assert _Recorded.service == "bedrock-runtime"
    assert _Recorded.region == "eu-central-1"
    assert _Recorded.connect_timeout == CONNECT_TIMEOUT_SECONDS
    assert _Recorded.read_timeout == READ_TIMEOUT_SECONDS
    assert _Recorded.retries == {"max_attempts": 1, "mode": "standard"}
    assert _Recorded.tcp_keepalive is True
    assert runtime is not None


def test_bearer_client_construction_never_resolves_ambient_aws_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bearer construction is unsigned until its isolated token signer is bound."""

    class _FakeConfig:
        """Capture the selected signature version."""

        def __init__(self, **kwargs: object) -> None:
            self.signature_version = kwargs.get("signature_version")

    class _FakeSession:
        """Reject any client that was not explicitly made unsigned."""

        def client(self, service_name: str, *, region_name: str, config: object) -> object:
            assert service_name == "bedrock-runtime"
            assert region_name == "us-west-2"
            assert isinstance(config, _FakeConfig)
            assert config.signature_version is unsigned
            return object()

    class _FakeBoto3:
        """Session construction itself carries no ambient credential lookup."""

        def Session(self, **credentials: object) -> _FakeSession:
            assert credentials == {"botocore_session": isolated_session}
            return _FakeSession()

    unsigned = object()
    isolated_session = object()
    bound: list[tuple[object, str]] = []
    monkeypatch.setattr("exp.runtime.models.providers.bedrock._import_boto3", _FakeBoto3)
    monkeypatch.setattr(
        "exp.runtime.models.providers.bedrock._import_botocore_config", lambda: _FakeConfig
    )
    monkeypatch.setattr(
        "exp.runtime.models.providers.bedrock._import_botocore_unsigned", lambda: unsigned
    )
    monkeypatch.setattr(
        "exp.runtime.models.providers.bedrock._import_isolated_botocore_session",
        lambda: isolated_session,
    )
    monkeypatch.setattr(
        "exp.runtime.models.providers.bedrock._bind_bedrock_bearer",
        lambda client, token: bound.append((client, token)),
    )

    runtime = create_bedrock_runtime_client(
        region_name="us-west-2",
        bearer_token="bedrock-bearer",
    )

    assert bound == [(runtime, "bedrock-bearer")]


def test_real_bearer_client_ignores_hostile_ambient_token_and_credential_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real botocore constructs a token-only client without loading ambient auth."""
    import botocore.credentials
    import botocore.tokens

    def forbidden(*_: object, **__: object) -> object:
        raise AssertionError("ambient AWS authentication was consulted")

    monkeypatch.setattr(
        botocore.credentials.CredentialResolver,
        "load_credentials",
        forbidden,
    )
    monkeypatch.setattr(botocore.tokens.SSOTokenProvider, "load_token", forbidden)

    runtime = create_bedrock_runtime_client(
        region_name="us-west-2",
        bearer_token="bedrock-bearer",
    )

    assert runtime is not None


def test_real_bearer_client_ignores_hostile_profile_and_endpoint_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Bearer mode uses the official regional endpoint and no ambient AWS profile."""
    malformed = tmp_path / "malformed-aws-config"
    malformed.write_text("[profile broken\n", encoding="utf-8")
    monkeypatch.setenv("AWS_PROFILE", "profile-that-does-not-exist")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(malformed))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(malformed))
    monkeypatch.setenv(
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
        "https://credential-exfiltration.invalid",
    )
    monkeypatch.setenv("AWS_DEFAULTS_MODE", "invalid-ambient-mode")
    monkeypatch.setenv("AWS_USE_DUALSTACK_ENDPOINT", "true")
    monkeypatch.setenv("AWS_USE_FIPS_ENDPOINT", "true")

    runtime = create_bedrock_runtime_client(
        region_name="us-west-2",
        bearer_token="bedrock-bearer",
    )

    endpoint = cast("Any", runtime)._endpoint
    assert endpoint.host == "https://bedrock-runtime.us-west-2.amazonaws.com"


def test_real_access_key_client_ignores_hostile_ambient_aws_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An explicit pair uses only its pinned keys and the public regional endpoint."""
    malformed = tmp_path / "malformed-aws-config"
    malformed.write_text("[profile broken\n", encoding="utf-8")
    monkeypatch.setenv("AWS_PROFILE", "profile-that-does-not-exist")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(malformed))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(malformed))
    monkeypatch.setenv(
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
        "https://credential-exfiltration.invalid",
    )
    monkeypatch.setenv("AWS_DEFAULTS_MODE", "invalid-ambient-mode")
    monkeypatch.setenv("AWS_USE_DUALSTACK_ENDPOINT", "true")
    monkeypatch.setenv("AWS_USE_FIPS_ENDPOINT", "true")

    runtime = create_bedrock_runtime_client(
        region_name="us-west-2",
        aws_access_key_id="AKIAEXPLICITKEY001",
        aws_secret_access_key="explicit-secret-access-key",
    )

    scoped = cast("Any", runtime)
    assert scoped._endpoint.host == "https://bedrock-runtime.us-west-2.amazonaws.com"
    credentials = scoped._request_signer._credentials.get_frozen_credentials()
    assert credentials.access_key == "AKIAEXPLICITKEY001"
    assert credentials.secret_key == "explicit-secret-access-key"
    assert credentials.token is None


@pytest.mark.parametrize("auth_mode", ("pair", "bearer"))
@pytest.mark.parametrize(
    ("variable", "value"),
    (
        ("AWS_REQUEST_CHECKSUM_CALCULATION", "invalid-mode"),
        ("AWS_REQUEST_MIN_COMPRESSION_SIZE_BYTES", "not-an-integer"),
    ),
)
def test_real_explicit_clients_ignore_hostile_environment_client_configuration(
    monkeypatch: pytest.MonkeyPatch,
    auth_mode: str,
    variable: str,
    value: str,
) -> None:
    """Explicit pair and bearer clients ignore every environment config chain."""
    monkeypatch.setenv(variable, value)
    kwargs = (
        {"bearer_token": "bedrock-bearer"}
        if auth_mode == "bearer"
        else {
            "aws_access_key_id": "AKIAEXPLICITKEY001",
            "aws_secret_access_key": "explicit-secret-access-key",
        }
    )

    runtime = create_bedrock_runtime_client(region_name="us-west-2", **kwargs)

    assert runtime is not None


@pytest.mark.parametrize("auth_mode", ("pair", "bearer"))
def test_explicit_clients_ignore_hostile_customer_endpoint_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    auth_mode: str,
) -> None:
    """Customer botocore models cannot redirect explicit credentials or signing."""
    from botocore.loaders import Loader

    customer_data = tmp_path / "models"
    customer_data.mkdir()
    built_in = Loader(
        extra_search_paths=[Loader.BUILTIN_DATA_PATH],
        include_default_search_paths=False,
    ).load_data("endpoints")
    aws_partition = next(
        partition
        for partition in cast("Any", built_in)["partitions"]
        if partition["partition"] == "aws"
    )
    aws_partition["services"]["bedrock-runtime"] = {
        "endpoints": {
            "us-west-2": {
                "hostname": "credential-exfiltration.invalid",
                "protocols": ["https"],
                "signatureVersions": ["v4"],
            }
        }
    }
    (customer_data / "endpoints.json").write_text(
        json.dumps(built_in),
        encoding="utf-8",
    )
    monkeypatch.setattr(Loader, "CUSTOMER_DATA_PATH", str(customer_data))
    bedrock_endpoints.bedrock_runtime_origin.cache_clear()
    try:
        if auth_mode == "bearer":
            runtime = create_bedrock_runtime_client(
                region_name="us-west-2",
                bearer_token="bedrock-bearer",
            )
            native = BedrockClient(
                model=_snapshot(),
                region="us-west-2",
                environment={},
                runtime_factory=None,
                bearer_token="bedrock-bearer",
            )
        else:
            runtime = create_bedrock_runtime_client(
                region_name="us-west-2",
                aws_access_key_id="AKIAEXPLICITKEY001",
                aws_secret_access_key="explicit-secret-access-key",
            )
            native = BedrockClient(
                model=_snapshot(),
                region="us-west-2",
                environment={},
                runtime_factory=None,
                aws_access_key_id="AKIAEXPLICITKEY001",
                aws_secret_access_key="explicit-secret-access-key",
            )

        assert cast("Any", runtime)._endpoint.host == (
            "https://bedrock-runtime.us-west-2.amazonaws.com"
        )
        assert native.converse_stream_url().startswith(
            "https://bedrock-runtime.us-west-2.amazonaws.com/"
        )
    finally:
        bedrock_endpoints.bedrock_runtime_origin.cache_clear()


def test_real_bearer_request_emits_only_the_pinned_authorization_token() -> None:
    """A serialized Converse request carries the exact bearer before network dispatch."""

    class _StopBeforeNetwork(Exception):
        """Abort after botocore has serialized and signed the request."""

    captured: dict[str, str] = {}
    runtime = create_bedrock_runtime_client(
        region_name="us-west-2",
        bearer_token="bedrock-bearer",
    )

    def capture(request: object, **_: object) -> None:
        headers = cast("Any", request).headers
        value = headers.get("Authorization")
        captured["authorization"] = (
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
        )
        raise _StopBeforeNetwork

    cast("Any", runtime).meta.events.register(
        "before-send.bedrock-runtime.Converse",
        capture,
    )

    with pytest.raises(_StopBeforeNetwork):
        runtime.converse(
            modelId="amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": "ping"}]}],
        )

    assert captured == {"authorization": "Bearer bedrock-bearer"}


def test_converse_response_normalizes_cache_legs_without_double_counting() -> None:
    """Converse inputTokens exclude cache legs, so read and write are added once."""
    response = converse_response(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "done"},
                        {
                            "toolUse": {
                                "toolUseId": "call-new",
                                "name": "create_ticket",
                                "input": {"priority": "urgent"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 4,
                "cacheReadInputTokens": 6,
                "cacheWriteInputTokens": 2,
            },
        },
        configured_model=_snapshot(),
        latency_seconds=0.5,
    )

    assert response.finish_reason == ModelFinishReason.COMPLETED
    assert response.output.content == "done"
    assert response.output.tool_calls[0].call_id == "call-new"
    assert response.economics.usage is not None
    assert response.economics.usage.input_tokens == 18
    assert response.economics.usage.cached_input_tokens == 6
    assert response.economics.usage.cache_write_input_tokens == 2


def test_converse_response_maps_length_and_rejects_unsupported_blocks() -> None:
    """max_tokens becomes length, and unknown content blocks fail closed."""
    length = converse_response(
        {
            "output": {"message": {"content": [{"text": "partial"}]}},
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 3, "outputTokens": 2},
        },
        configured_model=_snapshot(),
        latency_seconds=0.1,
    )
    assert length.finish_reason == ModelFinishReason.LENGTH
    with pytest.raises(ProviderResponseError, match="unsupported block"):
        converse_response(
            {
                "output": {"message": {"content": [{"image": {"format": "png"}}]}},
                "stopReason": "end_turn",
            },
            configured_model=_snapshot(),
            latency_seconds=0.1,
        )
    with pytest.raises(ProviderRefusalError) as refusal_error:
        converse_response(
            {
                "output": {"message": {"content": [{"text": "blocked"}]}},
                "stopReason": "content_filtered",
            },
            configured_model=_snapshot(),
            latency_seconds=0.1,
        )
    assert refusal_error.value.signal is ProviderRefusalSignal.GUARDRAIL
    assert "blocked" not in str(refusal_error.value)


def _signing_client(monkeypatch: pytest.MonkeyPatch, *, token: str | None = None) -> BedrockClient:
    """Build one region-bound client over deterministic environment credentials."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
    if token is None:
        monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    else:
        monkeypatch.setenv("AWS_SESSION_TOKEN", token)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    return BedrockClient(
        model=_snapshot(),
        region="us-east-1",
        environment={},
        runtime_factory=None,
    )


def test_converse_stream_url_encodes_the_model_like_botocore() -> None:
    """The REST route keeps ``/`` and ``~`` raw and percent-encodes ``:``."""
    client = BedrockClient(
        model=_snapshot("arn:aws:bedrock:us-east-1:123:inference-profile/us.anthropic.claude"),
        region="eu-central-1",
        environment={},
        runtime_factory=None,
    )
    assert client.converse_stream_url() == (
        "https://bedrock-runtime.eu-central-1.amazonaws.com/model/"
        "arn%3Aaws%3Abedrock%3Aus-east-1%3A123%3Ainference-profile/us.anthropic.claude"
        "/converse-stream"
    )


@pytest.mark.parametrize(
    ("region", "suffix"),
    (
        ("cn-north-1", "amazonaws.com.cn"),
        ("us-iso-east-1", "c2s.ic.gov"),
        ("us-isob-east-1", "sc2s.sgov.gov"),
    ),
)
def test_converse_stream_url_uses_the_region_partition_endpoint(
    region: str,
    suffix: str,
) -> None:
    """Native streaming derives China and isolated-partition DNS correctly."""
    client = BedrockClient(
        model=_snapshot(),
        region=region,
        environment={},
        runtime_factory=None,
    )

    assert client.converse_stream_url().startswith(
        f"https://bedrock-runtime.{region}.{suffix}/model/"
    )


def test_converse_stream_url_caches_bundled_endpoint_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated URL and signing checks do not reparse botocore endpoint data."""
    real_loader = bedrock_endpoints.built_in_botocore_loader
    calls = 0

    def counted_loader() -> bedrock_endpoints.BotocoreLoader:
        nonlocal calls
        calls += 1
        return real_loader()

    monkeypatch.setattr(bedrock_endpoints, "built_in_botocore_loader", counted_loader)
    bedrock_endpoints.bedrock_runtime_origin.cache_clear()
    client = BedrockClient(
        model=_snapshot(),
        region="us-west-2",
        environment={},
        runtime_factory=None,
    )
    try:
        first = client.converse_stream_url()
        second = client.converse_stream_url()
    finally:
        bedrock_endpoints.bedrock_runtime_origin.cache_clear()

    assert first == second
    assert calls == 1


def test_sign_gateway_dispatch_matches_an_independent_sigv4_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The produced Authorization header verifies against the SigV4 spec."""
    import hashlib
    import hmac

    client = _signing_client(monkeypatch)
    url = client.converse_stream_url()
    body = '{"messages":[{"role":"user","content":[{"text":"Zürich"}]}]}'
    headers = client.sign_gateway_dispatch(url=url, body=body)

    amz_date = headers["X-Amz-Date"]
    date_stamp = amz_date[:8]
    scope = f"{date_stamp}/us-east-1/bedrock/aws4_request"
    signed_headers = "accept;content-type;host;x-amz-date"
    canonical = "\n".join(
        (
            "POST",
            "/model/us.anthropic.claude-sonnet-4-5/converse-stream",
            "",
            "accept:application/vnd.amazon.eventstream",
            "content-type:application/json",
            "host:bedrock-runtime.us-east-1.amazonaws.com",
            f"x-amz-date:{amz_date}",
            "",
            signed_headers,
            hashlib.sha256(body.encode("utf-8")).hexdigest(),
        )
    )
    string_to_sign = "\n".join(
        (
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )
    )

    def _hmac(key: bytes, value: str) -> bytes:
        """Compute one HMAC-SHA256 chain link."""
        return hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()

    key = _hmac(b"AWS4wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", date_stamp)
    key = _hmac(key, "us-east-1")
    key = _hmac(key, "bedrock")
    key = _hmac(key, "aws4_request")
    signature = hmac.new(key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    assert headers["Authorization"] == (
        f"AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    assert headers["content-type"] == "application/json"
    assert "X-Amz-Security-Token" not in headers


def test_sign_gateway_dispatch_forwards_the_session_token_and_binds_the_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session tokens are signed in, and a changed body changes the signature."""
    client = _signing_client(monkeypatch, token="the-session-token")
    url = client.converse_stream_url()
    first = client.sign_gateway_dispatch(url=url, body='{"messages":[]}')
    assert first["X-Amz-Security-Token"] == "the-session-token"
    assert "x-amz-security-token" in first["Authorization"]
    second = client.sign_gateway_dispatch(url=url, body='{"messages":[{}]}')
    if first["X-Amz-Date"] == second["X-Amz-Date"]:
        assert first["Authorization"] != second["Authorization"]


def test_bedrock_dispatch_signing_rejects_every_non_admitted_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither SigV4 nor bearer credentials can be released to another origin."""
    client = _signing_client(monkeypatch, token=None)

    with pytest.raises(ProviderTransportError, match="differs"):
        client.sign_gateway_dispatch(
            url="https://untrusted.example/collect",
            body='{"messages":[]}',
        )


def test_sign_gateway_dispatch_uses_the_explicit_access_key_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Native ConverseStream signing resolves the configured pair, not the ambient chain."""
    malformed = tmp_path / "malformed-aws-config"
    malformed.write_text("[profile broken\n", encoding="utf-8")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(malformed))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(malformed))
    monkeypatch.setenv("AWS_PROFILE", "profile-that-does-not-exist")
    monkeypatch.setenv("AWS_DEFAULTS_MODE", "invalid-ambient-mode")
    client = BedrockClient(
        model=_snapshot(),
        region="us-west-2",
        environment={},
        aws_access_key_id="AKIAEXPLICITKEY001",
        aws_secret_access_key="explicit-secret-access-key",
        runtime_factory=None,
    )

    headers = client.sign_gateway_dispatch(
        url=client.converse_stream_url(),
        body='{"messages":[]}',
    )

    assert "Credential=AKIAEXPLICITKEY001/" in headers["Authorization"]


def test_bounded_client_wire_profile_marks_the_body_for_signing() -> None:
    """The bounded adapter's profile carries the signing dialect facts."""
    client = BedrockClient(
        model=_snapshot(),
        region="us-east-1",
        environment={},
        runtime_factory=None,
    )
    profile = BoundedBedrockClient(client).gateway_wire_profile()
    assert profile.dialect == "bedrock_converse_stream"
    assert profile.signs_request_body is True
    assert profile.headers == {}
    assert profile.model_id == "us.anthropic.claude-sonnet-4-5"
    assert profile.timeout_seconds == 600.0
    assert profile.url.endswith("/model/us.anthropic.claude-sonnet-4-5/converse-stream")
