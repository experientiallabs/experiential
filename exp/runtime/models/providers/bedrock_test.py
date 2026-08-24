"""Bedrock client, region, credential-chain, catalog resolution, and Converse translation tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import pytest

from exp.common.models import (
    AssistantAction,
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
    ToolCall,
    ToolChoice,
)
from exp.common.tasks import ToolSchema
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
        invoke_bodies: list[Mapping[str, object]] | None = None,
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

    def converse(self, **request: object) -> Mapping[str, object]:
        """Record one Converse request and return the frozen response."""
        self.converse_calls.append(request)
        return self._converse_response

    def converse_stream(self, **request: object) -> Mapping[str, object]:
        """Reject streaming in non-streaming fixtures that did not configure events."""
        del request
        raise AssertionError("test made an unexpected ConverseStream request")

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


def _tool_transcript_request() -> ModelRequest:
    """Build a visible transcript containing an earlier tool call and result."""
    return ModelRequest(
        messages=(
            ModelMessage(role="system", content="You are precise."),
            ModelMessage(role="user", content="Create a ticket."),
            ModelMessage(
                role="assistant",
                assistant_action=AssistantAction(
                    tool_calls=(
                        ToolCall(
                            call_id="call-old",
                            name="create_ticket",
                            arguments={"priority": "normal"},
                        ),
                    )
                ),
            ),
            ModelMessage(role="tool", content="created", tool_call_id="call-old"),
        ),
        tools=(
            ToolSchema(
                name="create_ticket",
                description="Create one support ticket.",
                input_schema={"type": "object"},
            ),
        ),
        tool_choice=ToolChoice(name="create_ticket"),
        temperature=0.1,
        maximum_output_tokens=256,
    )


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
