"""Native Amazon Bedrock adapter using Converse and explicit embedding aliases."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    Embedding,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
)
from wmo.runtime.models.providers.base import DEFAULT_RETRY_POLICY
from wmo.runtime.models.providers.bedrock_converse import converse_request, converse_response
from wmo.runtime.models.providers.errors import (
    ProviderEndpointClass,
    ProviderError,
    ProviderResponseError,
    parse_provider_envelope,
    provider_error_from_transport,
)
from wmo.runtime.models.providers.openai_compatible import normalize_embedding_vector
from wmo.runtime.models.providers.retry import RetryPolicy, run_with_retry
from wmo.runtime.models.providers.transport import ProviderTransportError

AWS_REGION_ENV = "AWS_REGION"
AWS_DEFAULT_REGION_ENV = "AWS_DEFAULT_REGION"
CONNECT_TIMEOUT_SECONDS = 15.0
READ_TIMEOUT_SECONDS = 600.0
_REGION_SOURCES = (
    "the catalog connection region",
    AWS_REGION_ENV,
    "the boto session chain including AWS_DEFAULT_REGION, the active AWS profile region, and "
    "the instance role",
)
NO_REGION_ERROR = (
    "Bedrock has no region. Region is resolved in this order, first hit wins: "
    + ", ".join(_REGION_SOURCES)
    + ". Set one of them."
)
_CLIENT_CONSTRUCTION_LOCK = threading.Lock()
_RETRYABLE_BOTO_CODES = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelTimeoutException",
    }
)


class BedrockRegionError(ValueError):
    """Bedrock cannot build a runtime client because no region was resolved."""


class BedrockRuntime(Protocol):
    """Narrow execute-only surface over one constructed ``bedrock-runtime`` client."""

    def converse(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Send one Converse request and return the decoded response object."""

    def invoke_model(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Send one InvokeModel request and return the decoded response object."""


class BedrockRuntimeFactory(Protocol):
    """Builds one region-bound Bedrock runtime client without holding request locks."""

    def __call__(self, *, region_name: str) -> BedrockRuntime:
        """Return a runtime client for one already-resolved AWS region."""


class _BotoSession(Protocol):
    """Session surface used to resolve region and construct ``bedrock-runtime``."""

    region_name: str | None

    def client(self, service_name: str, *, region_name: str, config: object) -> object:
        """Construct one AWS service client."""


class _Boto3Module(Protocol):
    """Lazy boto3 module surface used only at request time."""

    def Session(self) -> _BotoSession:
        """Return a boto session that reads the standard AWS chain."""


class _BotoRuntimeClient(Protocol):
    """Keyword-argument boto client methods used by the execute-only wrapper."""

    def converse(self, *args: object, **kwargs: object) -> object:
        """Send one Converse request."""

    def invoke_model(self, *args: object, **kwargs: object) -> object:
        """Send one InvokeModel request."""


def resolve_bedrock_region(
    configured: str | None,
    environment: Mapping[str, str],
    *,
    session_region: str | None = None,
) -> str | None:
    """Resolve a Bedrock region without contacting instance metadata.

    Args:
        configured: Explicit catalog region, when present.
        environment: Process or injected environment mapping.
        session_region: Optional region already read from a boto session.

    Returns:
        The first configured, ``AWS_REGION``, or session region, otherwise ``None``.
    """
    if configured:
        return configured
    aws_region = environment.get(AWS_REGION_ENV)
    if aws_region:
        return aws_region
    return session_region


def create_bedrock_runtime_client(*, region_name: str) -> BedrockRuntime:
    """Construct one ``bedrock-runtime`` client with bounded timeouts and no hidden retries.

    Args:
        region_name: Already-resolved AWS region passed as ``region_name``.

    Returns:
        A thin wrapper around the constructed boto client.

    Raises:
        RuntimeError: ``boto3`` or ``botocore`` is not installed.
    """
    boto3 = _import_boto3()
    config_cls = _import_botocore_config()
    session = boto3.Session()
    with _CLIENT_CONSTRUCTION_LOCK:
        client = session.client(
            "bedrock-runtime",
            region_name=region_name,
            config=config_cls(
                connect_timeout=CONNECT_TIMEOUT_SECONDS,
                read_timeout=READ_TIMEOUT_SECONDS,
                retries={"max_attempts": 1, "mode": "standard"},
                tcp_keepalive=True,
            ),
        )
    return _BotoBedrockRuntime(cast("_BotoRuntimeClient", client))


class BedrockClient:
    """Calls one Bedrock model or inference profile without failover or API-key auth."""

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        region: str | None,
        environment: Mapping[str, str],
        runtime_factory: BedrockRuntimeFactory | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        """Create a lazy Bedrock client that does not import boto or open a session.

        Args:
            model: Resolved identity whose ``model_id`` is the exact Bedrock model ID.
            region: Optional catalog region. ``AWS_REGION`` and the boto chain follow it.
            environment: Process or injected environment mapping used for region lookup.
            runtime_factory: Optional deterministic factory used by tests.
            retry_policy: Bounded same-region retry policy applied outside botocore.
            capabilities: Catalog sampling capabilities for this model, when known.
        """
        self._model = model
        self._configured_region = region
        self._environment = environment
        self._runtime_factory = runtime_factory
        self._retry_policy = retry_policy
        self._capabilities = capabilities
        self._client: BedrockRuntime | None = None
        self._lock = threading.Lock()

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one non-streaming request through Bedrock Converse.

        Args:
            request: Visible messages, tool schemas, and sampling controls to send.

        Returns:
            The typed non-streaming model response with observed request economics.
        """
        started_at = time.monotonic()
        payload = converse_request(self._model.model_id, request, self._capabilities)
        response = self._call_with_retry(
            lambda: self._runtime().converse(payload),
            endpoint_class="converse",
        )
        return converse_response(
            response,
            configured_model=self._model,
            latency_seconds=time.monotonic() - started_at,
        )

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Embed ordered text through the configured Bedrock embedding model.

        Args:
            texts: Ordered visible text values to embed.

        Returns:
            Unit-normalized embeddings in the input order, or an empty tuple for no texts.
        """
        if not texts:
            return ()
        vectors: list[Embedding] = []
        expected_dimensions: int | None = None
        for text in texts:
            body: JsonObject = {"inputText": text, "normalize": True}
            raw = self._call_with_retry(
                lambda request_body=body: self._runtime().invoke_model(
                    {
                        "modelId": self._model.model_id,
                        "body": json.dumps(request_body),
                        "contentType": "application/json",
                        "accept": "application/json",
                    }
                ),
                endpoint_class="invoke_model",
            )
            embedding = _embedding_values(_read_invoke_body(raw))
            if expected_dimensions is None:
                expected_dimensions = len(embedding)
            elif len(embedding) != expected_dimensions:
                raise ProviderResponseError(
                    "Bedrock embedding dimensions must match across the request"
                )
            vectors.append(Embedding(values=normalize_embedding_vector(embedding)))
        if len(vectors) != len(texts):
            raise ProviderResponseError(
                f"Bedrock embedding count {len(vectors)} does not match request count {len(texts)}"
            )
        return tuple(vectors)

    def _runtime(self) -> BedrockRuntime:
        """Return the constructed runtime client, building it once without holding request locks."""
        existing = self._client
        if existing is not None:
            return existing
        with self._lock:
            if self._client is None:
                factory = self._runtime_factory or create_bedrock_runtime_client
                self._client = factory(region_name=self._region_name())
            return self._client

    def _region_name(self) -> str:
        """Resolve the region used for this client without catalog-time metadata probes."""
        region = resolve_bedrock_region(self._configured_region, self._environment)
        if region:
            return region
        if self._runtime_factory is not None:
            raise BedrockRegionError(NO_REGION_ERROR)
        session_region = _boto_session_region()
        if session_region:
            return session_region
        raise BedrockRegionError(NO_REGION_ERROR)

    def _call_with_retry(
        self,
        operation: Callable[[], Mapping[str, object]],
        *,
        endpoint_class: ProviderEndpointClass,
    ) -> Mapping[str, object]:
        """Retry one Bedrock call on the same region and model without botocore multiplication.

        Args:
            operation: One Converse or InvokeModel attempt.
            endpoint_class: Documented Bedrock operation that issued the call.

        Returns:
            The successful Bedrock response mapping.
        """

        def send() -> Mapping[str, object]:
            """Run one attempt and translate provider failures into transport errors."""
            try:
                return operation()
            except ProviderTransportError:
                raise
            except ProviderResponseError:
                raise
            except BedrockRegionError:
                raise
            except Exception as exc:
                raise _as_transport_error(exc, endpoint_class=endpoint_class) from exc

        return run_with_retry(send, policy=self._retry_policy)


class _BotoBedrockRuntime:
    """Adapts a constructed boto ``bedrock-runtime`` client to the execute-only protocol."""

    def __init__(self, client: _BotoRuntimeClient) -> None:
        """Wrap one already-constructed boto client.

        Args:
            client: ``boto3`` ``bedrock-runtime`` client.
        """
        self._client = client

    def converse(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Forward one Converse request as explicit keyword arguments."""
        return cast("Mapping[str, object]", self._client.converse(**dict(request)))

    def invoke_model(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Forward one InvokeModel request as explicit keyword arguments."""
        return cast("Mapping[str, object]", self._client.invoke_model(**dict(request)))


def _boto_session_region() -> str | None:
    """Read the boto session region, including ``AWS_DEFAULT_REGION`` and profile config."""
    session = _import_boto3().Session()
    region = session.region_name
    return region if isinstance(region, str) and region else None


def _import_boto3() -> _Boto3Module:
    """Import ``boto3`` only when a Bedrock request needs a runtime client."""
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Bedrock requires boto3; install world-model-optimizer") from exc
    return cast("_Boto3Module", boto3)


def _import_botocore_config() -> type[object]:
    """Import botocore ``Config`` only when constructing a Bedrock runtime client."""
    try:
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError("Bedrock requires botocore; install world-model-optimizer") from exc
    return Config


def _as_transport_error(
    exc: Exception,
    *,
    endpoint_class: ProviderEndpointClass,
) -> ProviderError:
    """Convert a boto failure into a secret-free retry classification boundary.

    Args:
        exc: Raw boto or runtime exception from one Bedrock attempt.
        endpoint_class: Documented Bedrock operation that issued the call.

    Returns:
        A typed sanitized failure labeled with the originating Bedrock operation.
    """
    name = type(exc).__name__
    if name in {
        "ReadTimeoutError",
        "ConnectTimeoutError",
        "EndpointConnectionError",
        "TimeoutError",
    }:
        return provider_error_from_transport(
            "Bedrock request timed out",
            provider="bedrock",
            endpoint_class=endpoint_class,
        )
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        parsed = parse_provider_envelope(cast("JsonObject", response))
        metadata = response.get("ResponseMetadata")
        status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
        status_code = status if isinstance(status, int) else None
        retryable = parsed.error_code in _RETRYABLE_BOTO_CODES if parsed.error_code else None
        return ProviderError(
            parsed.message or "Bedrock request failed",
            provider="bedrock",
            endpoint_class=endpoint_class,
            status_code=status_code or (503 if retryable else None),
            error_code=parsed.error_code,
            error_type=parsed.error_type,
            rejected_parameter=parsed.rejected_parameter,
            request_id=parsed.request_id,
            retryable=retryable,
        )
    return provider_error_from_transport(
        "Bedrock request failed",
        provider="bedrock",
        endpoint_class=endpoint_class,
        retryable=True,
    )


def _read_invoke_body(payload: Mapping[str, object]) -> JsonObject:
    """Decode an InvokeModel body from a mapping, bytes, or streaming body."""
    body = payload.get("body")
    if isinstance(body, dict):
        return cast("JsonObject", body)
    raw: object = body
    read = getattr(body, "read", None)
    if callable(read):
        raw = read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError("Bedrock embedding response body is not JSON") from exc
        if isinstance(decoded, dict):
            return cast("JsonObject", decoded)
    raise ProviderResponseError("Bedrock embedding response body must be a JSON object")


def _embedding_values(payload: JsonObject) -> list[JsonValue]:
    """Read one embedding vector from a Titan or compatible InvokeModel body."""
    values = payload.get("embedding")
    if not isinstance(values, list) or not values:
        raise ProviderResponseError("Bedrock embedding response needs a non-empty embedding array")
    return cast("list[JsonValue]", values)
