"""Typed local `.exp/models.toml` catalog loading without credential values."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

import tomli_w
from pydantic import (
    AwareDatetime,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from exp.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    JsonObject,
    SecretBoundaryError,
    Sha256,
    assert_secret_free,
    sha256_json,
    validate_artifact_id,
)
from exp.common.core.files import write_text_atomic
from exp.common.models.bedrock_connection import require_bedrock_connection_shape
from exp.common.models.catalog_prices import (
    MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS as MAXIMUM_RATE_NANO_USD_PER_MILLION_TOKENS,
)
from exp.common.models.catalog_prices import (
    GatewayLongContextTier as GatewayLongContextTier,
)
from exp.common.models.catalog_prices import (
    GatewayServiceTierPrices as GatewayServiceTierPrices,
)
from exp.common.models.catalog_prices import (
    GatewayTokenPrices as GatewayTokenPrices,
)
from exp.common.models.catalog_prices import (
    NanoUsdRatePerMillionTokens as NanoUsdRatePerMillionTokens,
)
from exp.common.models.catalog_roles import ModelRoles
from exp.common.models.discovery import DiscoveredModel
from exp.common.models.dispatch_policy import GatewayRungDispatchPolicy
from exp.common.models.failover_tokens import FailoverToken
from exp.common.models.gateway_chains import GatewayModelChain
from exp.common.models.gateway_pools import GatewayPoolRecord
from exp.common.models.model import (
    BillingSource,
    ModelCapabilities,
    ModelSnapshot,
    ReasoningEffort,
)
from exp.common.models.nano_usd_upgrade import (
    upgrade_legacy_billing_source,
    upgrade_model_catalog_document,
)

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_AZURE_API_VERSION = re.compile(r"^(?:v1|\d{4}-\d{2}-\d{2}(?:-preview)?)$")
_AWS_REGION_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_VERTEX_HOST = re.compile(
    r"(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?-)?aiplatform\.googleapis\.com"
    r"|aiplatform\.(?:us|eu)\.rep\.googleapis\.com)"
)
_FIXED_ORIGIN_PROVIDERS = frozenset(
    {"anthropic", "gemini", "openai", "openrouter", "tinker", "typesafe"}
)
_EXPLICIT_CAPABILITY_PROVIDERS = frozenset({"azure", "bedrock", "openai-compatible", "vertex"})

AzureApiSurface = Literal["openai_deployments", "model_inference"]
"""Azure wire surface a connection speaks: classic deployments or Foundry model inference."""

_FOUNDRY_HOST_SUFFIXES = (".services.ai.azure.com", ".inference.ai.azure.com")
_AZURE_OPENAI_HOST_SUFFIX = ".openai.azure.com"
_MODEL_INFERENCE_ROOT_SUFFIXES = ("/models", "/openai/v1")
_MODEL_INFERENCE_IDENTITY_SUFFIX = "/models"


def _normalize_base_url(value: str) -> str:
    """Return the stable endpoint spelling used for connection identity."""
    parsed = urlsplit(value)
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("base_url must include a hostname")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("base_url must use a valid port") from exc
    scheme = parsed.scheme.lower()
    host = hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    default_port = 443 if scheme == "https" else 80
    netloc = host if port in {None, default_port} else f"{host}:{port}"
    return urlunsplit((scheme, netloc, parsed.path.rstrip("/"), "", ""))


def infer_azure_api_surface(endpoint: str) -> AzureApiSurface | None:
    """Infer the Azure wire surface one resource endpoint serves.

    Azure AI Foundry resources (``*.services.ai.azure.com``) serve the model-inference surface,
    which carries provider-specific sampling fields such as ``top_k``. Azure OpenAI resources
    (``*.openai.azure.com``) serve only the deployment surface.

    Args:
        endpoint: Azure resource endpoint from a connection.

    Returns:
        The surface the host is known to serve, or ``None`` for an unrecognized host such as a
        private endpoint or a local recording proxy.
    """
    host = urlsplit(endpoint).hostname
    if host is None:
        return None
    host = host.lower()
    if host.endswith(_AZURE_OPENAI_HOST_SUFFIX):
        return "openai_deployments"
    if any(host.endswith(suffix) for suffix in _FOUNDRY_HOST_SUFFIXES):
        return "model_inference"
    return None


def strip_model_inference_root(value: str) -> str:
    """Remove the route suffix one Azure model-inference endpoint spelling carries.

    The model-inference surface serves ``/models`` directly off the resource, so the bare resource,
    its terminal ``/models`` form, and the Azure OpenAI ``/openai/v1`` root all name one resource.

    Args:
        value: Endpoint or endpoint path, with or without a trailing slash.

    Returns:
        The value reduced to the resource itself.
    """
    trimmed = value.rstrip("/")
    for suffix in _MODEL_INFERENCE_ROOT_SUFFIXES:
        if trimmed.lower().endswith(suffix):
            return trimmed[: -len(suffix)].rstrip("/")
    return trimmed


def _normalize_connection_base_url(connection: ConnectionConfig) -> str | None:
    """Normalize one endpoint while preserving provider-surface equivalence."""
    if connection.base_url is None:
        return None
    normalized = _normalize_base_url(connection.base_url)
    # Endpoint identity is deliberately narrower than request routing: it folds only the terminal
    # ``/models`` segment, and only for a declared surface, so no stored credential digest moves
    # for a connection the operator never edited.
    if (
        connection.provider == "azure"
        and connection.azure_api_surface == "model_inference"
        and normalized.lower().endswith(_MODEL_INFERENCE_IDENTITY_SUFFIX)
    ):
        return normalized[: -len(_MODEL_INFERENCE_IDENTITY_SUFFIX)].rstrip("/")
    return normalized


class ModelCatalogError(ValueError):
    """A local model catalog was malformed or named a credential value."""


class ConnectionConfig(ContractModel):
    """Local provider connection metadata, with an optional credential environment name only."""

    provider: str = Field(min_length=1, max_length=128)
    base_url: str | None = Field(default=None, max_length=2_048)
    api_key_env: str | None = Field(default=None, max_length=256)
    api_version: str | None = Field(default=None, max_length=64)
    azure_api_surface: Literal["openai_deployments", "model_inference"] | None = None
    region: str | None = Field(default=None, max_length=64)
    inference_geo: Literal["us"] | None = None
    """Operator-enforced Anthropic inference geography, independent of caller input."""
    aws_access_key_id_env: str | None = Field(default=None, max_length=256)
    bedrock_auth_mode: Literal["access_key_pair", "api_key"] | None = None
    # Opt-in: native provider via a trusted https base_url in its own dialect (default-off).
    trusted_custom_origin: bool = False

    @field_validator("api_key_env", "aws_access_key_id_env")
    @classmethod
    def _require_environment_variable_name(cls, value: str | None) -> str | None:
        if value is not None and not _ENVIRONMENT_NAME.fullmatch(value):
            raise ValueError("credential environment fields must name environment variables")
        return value

    @field_validator("base_url")
    @classmethod
    def _reject_embedded_credentials(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not embed credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not include query parameters or fragments")
        _normalize_base_url(value)
        return value

    @model_validator(mode="after")
    def _require_secret_free_connection_metadata(self) -> ConnectionConfig:
        if self.inference_geo is not None and self.provider != "anthropic":
            raise ValueError("inference_geo is only accepted for provider='anthropic'")
        if self.provider != "azure" and self.azure_api_surface is not None:
            raise ValueError("azure_api_surface is only accepted for provider='azure'")
        if self.provider != "bedrock" and (
            self.aws_access_key_id_env is not None or self.bedrock_auth_mode is not None
        ):
            raise ValueError(
                "aws_access_key_id_env and bedrock_auth_mode are only accepted for "
                "provider='bedrock'"
            )
        if self.trusted_custom_origin:
            if self.provider not in _FIXED_ORIGIN_PROVIDERS:
                raise ValueError("trusted_custom_origin applies only to a native provider")
            if self.base_url is None:
                raise ValueError("trusted_custom_origin requires an explicit base_url")
            if urlsplit(self.base_url).scheme != "https":
                raise ValueError("trusted_custom_origin requires an https base_url")
        elif self.provider in _FIXED_ORIGIN_PROVIDERS and self.base_url is not None:
            raise ValueError(
                f"native provider {self.provider!r} uses its built-in official endpoint; "
                "set trusted_custom_origin=True or use provider='openai-compatible'"
            )
        if self.provider == "azure":
            if self.base_url is None:
                raise ValueError("azure requires an explicit resource endpoint in base_url")
            if self.api_key_env is None:
                raise ValueError("azure requires api_key_env")
            if self.api_version is None:
                raise ValueError(
                    "azure requires an explicit api_version such as 'v1' or a dated Azure "
                    "OpenAI version"
                )
            if not _AZURE_API_VERSION.fullmatch(self.api_version):
                raise ValueError(
                    "azure api_version must be 'v1' or a dated Azure OpenAI version such as "
                    "2024-10-21"
                )
            if self.azure_api_surface == "model_inference" and self.api_version == "v1":
                raise ValueError(
                    "azure model_inference requires a dated api_version for the mandatory "
                    "api-version query parameter"
                )
            if self.region is not None:
                raise ValueError("region is only accepted for provider='bedrock'")
        elif self.provider == "bedrock":
            require_bedrock_connection_shape(
                bedrock_auth_mode=self.bedrock_auth_mode,
                api_key_env=self.api_key_env,
                aws_access_key_id_env=self.aws_access_key_id_env,
                base_url=self.base_url,
                api_version=self.api_version,
            )
            if self.region is not None and not _AWS_REGION_NAME.fullmatch(self.region):
                raise ValueError("bedrock region must be an AWS region name")
        elif self.provider == "vertex":
            if self.base_url is None:
                raise ValueError(
                    "vertex requires base_url naming the project-and-location root, such as "
                    "https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT/"
                    "locations/us-central1"
                )
            # The runtime attaches a cloud-platform OAuth token to every request, so the
            # endpoint host is pinned to Vertex AI service hosts and never operator-chosen.
            vertex_parts = urlsplit(self.base_url)
            vertex_host = (vertex_parts.hostname or "").lower()
            if vertex_parts.scheme != "https" or not _VERTEX_HOST.fullmatch(vertex_host):
                raise ValueError(
                    "vertex base_url must use an HTTPS Vertex AI host such as "
                    "https://us-central1-aiplatform.googleapis.com; OAuth tokens are never "
                    "sent to other hosts"
                )
            if self.api_key_env is None:
                raise ValueError(
                    "vertex requires api_key_env naming the environment variable that holds "
                    "the service-account JSON credential"
                )
            if self.api_version is not None:
                raise ValueError("api_version is only accepted for provider='azure'")
            if self.region is not None:
                raise ValueError(
                    "region is only accepted for provider='bedrock'; the Vertex location "
                    "lives inside base_url"
                )
        else:
            if self.api_version is not None:
                raise ValueError("api_version is only accepted for provider='azure'")
            if self.region is not None:
                raise ValueError("region is only accepted for provider='bedrock'")
        try:
            assert_secret_free(
                {
                    "provider": self.provider,
                    "base_url": self.base_url,
                    "api_version": self.api_version,
                    "azure_api_surface": self.azure_api_surface,
                    "region": self.region,
                    "bedrock_auth_mode": self.bedrock_auth_mode,
                }
            )
        except SecretBoundaryError as exc:
            raise ValueError("connection metadata must not contain credential values") from exc
        return self

    def identity_sha256(self) -> Sha256:
        """Return a deterministic digest of the secret-free provider endpoint identity.

        Returns:
            A SHA-256 digest over the provider, normalized endpoint, and any Azure API version or
            Bedrock region. Credential values and credential-environment metadata are excluded.
        """
        identity: JsonObject = {
            "provider": self.provider,
            "base_url": _normalize_connection_base_url(self),
        }
        if self.api_version is not None:
            identity["api_version"] = self.api_version
        if self.provider == "azure" and self.azure_api_surface == "model_inference":
            # Keep classic Azure revisions byte-compatible with the identity
            # contract that predates this discriminator. Only the genuinely
            # different Foundry surface needs a new credential binding.
            identity["azure_api_surface"] = "model_inference"
        if self.inference_geo is not None:
            identity["inference_geo"] = self.inference_geo
        if self.region is not None:
            identity["region"] = self.region
        if self.trusted_custom_origin:  # endpoint identity; added only when set
            identity["trusted_custom_origin"] = True
        effective_bedrock_auth_mode = self.bedrock_auth_mode
        if (
            self.provider == "bedrock"
            and effective_bedrock_auth_mode is None
            and self.api_key_env is not None
            and self.aws_access_key_id_env is not None
        ):
            effective_bedrock_auth_mode = "access_key_pair"
        if effective_bedrock_auth_mode is not None:
            identity["bedrock_auth_mode"] = effective_bedrock_auth_mode
        return sha256_json(identity)

    def canonicalized(self) -> ConnectionConfig:
        """Return the canonical persisted shape for Bedrock access-key pairs."""
        if (
            self.provider == "bedrock"
            and self.bedrock_auth_mode is None
            and self.api_key_env is not None
            and self.aws_access_key_id_env is not None
        ):
            return self.model_copy(update={"bedrock_auth_mode": "access_key_pair"})
        return self

    @model_serializer(mode="wrap")
    def _serialize_without_absent_bedrock_metadata(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        """Preserve pre-Bedrock canonical bytes on every supported Pydantic version."""
        serialized: dict[str, object] = handler(self)
        if self.inference_geo is None:
            serialized.pop("inference_geo", None)
        if self.aws_access_key_id_env is None:
            serialized.pop("aws_access_key_id_env", None)
        if self.bedrock_auth_mode is None:
            serialized.pop("bedrock_auth_mode", None)
        if not self.trusted_custom_origin:
            serialized.pop("trusted_custom_origin", None)
        return serialized


class SFTModelProvenance(ContractModel):
    """Immutable W12, W13, and base-model bindings for one registered SFT sampling handle."""

    source_dataset: ArtifactInput
    optimization_config: ArtifactInput
    training_spec_sha256: Sha256
    run_id: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=128)
    model_sha256: Sha256
    result_id: str = Field(min_length=1, max_length=128)
    result_sha256: Sha256
    base_model: ModelSnapshot
    connection_config_sha256: Sha256
    sampling_handle_sha256: Sha256


class GatewayDeploymentCapabilities(ContractModel):
    """Gateway protocol capabilities declared for one provider deployment.

    These fields are intentionally separate from ``ModelCapabilities``. The latter participates
    in frozen optimizer and runtime identities, while this declaration can evolve with the
    gateway protocol without invalidating existing router artifacts.

    Attributes:
        supports_decisions: Whether this deployment serves native typed decisions instead of
            chat.
        supports_developer_messages: Whether developer-role messages are supported.
        supports_streaming: Whether streaming responses are supported.
        supports_streaming_tool_arguments: Whether tool arguments can be streamed incrementally.
        supports_responses_logprobs: Verified native Responses probability support, opt-in.
        logprobs_reasoning_efforts: Verified Chat probability efforts; empty means unknown.
        supports_strict_tools: Whether strict function-tool schemas are supported.
        supports_parallel_tool_calls: Whether parallel tool calls are supported.
        supports_custom_tools: Whether this deployment's relevant native wire can preserve
            free-form custom tools.

            False means the capability is not declared. Public Chat still refuses custom tools
            even when a Responses-native deployment declares this, because Chat accepts function
            tools only. Read the flag with the parity row's dialect and the caller's public API
            surface.
        supports_grammar_tools: Whether grammar-constrained custom tools can be preserved on
            this deployment.

            This requires ``supports_custom_tools``. False means the capability is not declared;
            it does not describe Chat, which still refuses custom and grammar tools.
        supports_tool_call_limit: Whether a caller's Responses ``max_tool_calls`` limit can be
            preserved.

            False means the capability is not declared. The public Responses surface currently
            refuses the field, so no authored catalog should set this until a route honors the
            cap.
        supports_structured_text: Whether schema-constrained text output is supported.
        supports_stop_sequences: Whether caller-specified stop sequences are supported.
        supports_image_input: Whether the wire and model accept images; undeclared image input
            is rejected.
        supports_image_url_input: Whether the provider fetches remote images; undeclared URLs
            are rejected.

            Every image-capable wire accepts inline base64; URL support varies by route.
        supports_video_input: Whether the wire and model accept video; undeclared video input is
            rejected.

            Video carriers exist on Gemini, Bedrock Converse, and compatible ``video_url``
            wires.
        supports_video_url_input: Whether the provider fetches video URLs (Gemini and
            OpenAI-compatible wires).

            Bedrock requires inline bytes or an S3 location that the gateway does not author.
        supports_audio_input: Whether the wire and model accept audio; undeclared audio input is
            rejected.

            Supported models use compatible Chat ``input_audio`` or Gemini ``inline_data``. No
            public audio surface accepts remote URLs.
        supports_pdf_input: Whether the wire and model accept PDFs; undeclared document input is
            rejected.
        supports_pdf_url_input: Whether this route's provider fetches a caller PDF URL itself.

            Only the OpenAI Responses (``file_url``) and Anthropic Messages (``url`` source)
            wires fetch a remote document; Chat Completions ``file`` parts, Gemini, and Bedrock
            accept inline bytes only.
        supports_prompt_cache_boundaries: Whether explicit caller-selected prompt-cache
            boundaries can be preserved.

            This covers Chat ``prompt_cache_options`` and ``prompt_cache_retention``. False
            means those explicit boundaries are not declared as preserved; it does not mean
            implicit prefix caching is absent.
        supports_media_handle_input: Whether this route forwards handles to media the caller
            uploaded to its provider.

            A handle (an OpenAI or Anthropic ``file_id``, a Gemini Files URI, a ``gs://`` object
            on Vertex, an ``s3://`` object on Bedrock) is scoped to the provider that minted it
            and never portable, so admission requires both this declaration and a handle
            provider equal to the route's provider. Providers whose inference wire defines no
            uploaded-media reference (Fireworks, OpenRouter) never declare it.
        maximum_stop_sequences: Largest stop-sequence count this route accepts, when the
            provider caps it.

            ``None`` leaves the count unbounded (only ``supports_stop_sequences`` gates the
            field). A concrete value lets admission reject an over-limit list locally with a
            named parameter error instead of forwarding it and surfacing the provider's opaque
            4xx (e.g. Gemini caps ``stopSequences`` at 5).
        minimum_output_tokens: Provider output-token floor (sonar/fugu via OpenRouter, grok-4.6
            on Bedrock: 16); a smaller caller ceiling is floored to it with disclosure on every
            surface (see the profile).
        supported_reasoning_efforts: Exact caller values this deployment can preserve without
            normalization.

            An empty tuple means the gateway should use its maintained provider-family contract.
            OpenRouter and other catalog-driven providers declare the exact ordered set here
            because their supported values vary by model.
        reasoning_default_effort: The depth this deployment reasons at when the caller names
            none.

            Emitted on a wire that requires an explicit effort, and read by Messages admission
            as the depth a budget-less ``thinking`` config (``adaptive``, or Claude Code's bare
            ``{type: enabled}``) asks for on an effort rung, so a lane's think-mode depth is set
            here, not in code.
        reasoning_effort_required: Whether this deployment requires an explicit reasoning effort
            on its wire.
        reports_refusals: Whether provider refusals are reported explicitly.
        reports_cached_input_tokens: Whether cached input-token usage is reported.
        reports_cache_creation_input_tokens: Whether cache-write input-token usage is reported.
        reports_reasoning_tokens: Whether reasoning-token usage is reported separately.
        reports_model_status: Whether provider model-status metadata is preserved in the
            normalized response.

            False means the capability is not declared. Gemini ``modelStatus`` is currently
            unpreserved, so no authored catalog should set this until the response contract
            carries that field.
        supports_async_tools: Whether a tool may be flagged ``async`` so the model keeps
            generating while the caller runs it, with the result returned later on the tool
            call's ORIGINAL ``call_id`` (GPT-6 Astra Responses). Declaration-driven and off
            until the decoder + turn lifecycle honor it; a route that declares it must not drop
            an async tool call. See the platform's astra_responses helpers.
        supports_mid_turn_steering: Whether the caller may inject additional input over the
            Responses WebSocket WHILE the model is working, folded into a continuation that
            preserves completed work (GPT-6 Astra). Off until the WS transport accepts inbound
            mid-turn frames.
        supports_reasoning_effort_update: Whether a ``configuration_update`` input item may
            change reasoning effort mid-conversation without invalidating the cached prompt
            prefix -- the request-level ``reasoning.effort`` stays fixed (GPT-6 Astra). Off
            until the decoder recognizes the item (it must not hit the unknown-item reject path)
            and applies the effort forward.
        time_to_first_byte_base_seconds: Deployment override for the lane's flat
            time-to-first-byte allowance.

            ``None`` uses the serving configuration's default. The effective bound on the wait
            for a provider's response headers is this base plus the input-scaled allowance
            below, so very large prompts are not misread as a dead lane. The wait for the first
            TOKEN has its own base (``time_to_first_token_base_seconds``) and shares the slope.
        time_to_first_byte_seconds_per_million_input_tokens: Deployment override for the
            input-scaled time-to-first-byte allowance.

            Seconds added per million approximate input tokens (the request body's bytes divided
            by four; an allowance heuristic, never a billing quantity). ``None`` uses the
            serving configuration's default; ``0`` disables scaling for this deployment.
        time_to_first_token_base_seconds: Deployment override for the lane's flat
            time-to-first-TOKEN allowance.

            ``None`` uses the serving configuration's default (two minutes). The effective bound
            on the wait from the dial to the first semantic event (content, reasoning, a tool
            call; keepalive comments and role-only frames do not count) is this base plus the
            input-scaled allowance above. A stall past it fails over to the next rung. Author it
            above the lane's observed first-token p99 on a thinking model, below the stall you
            want caught.
        failover_only_on: Failure tokens this rung serves as a failover for, or ``None`` for an
            unrestricted rung.

            A rung carrying a set is never dialed first and is dialed as a successor only when
            the failure being failed over from spells one of its tokens (see
            ``exp.common.models.failover_tokens``); a rule-carrying rung reached that way
            records ``fallback_reason = failover_only_on:<token>``.
    """

    supports_decisions: bool = False

    supports_developer_messages: bool = False
    supports_streaming: bool = False
    supports_streaming_tool_arguments: bool = False
    supports_responses_logprobs: bool = False
    supports_strict_tools: bool = False
    supports_parallel_tool_calls: bool = False
    supports_custom_tools: bool = False
    supports_grammar_tools: bool = False
    supports_tool_call_limit: bool = False
    supports_structured_text: bool = False
    supports_stop_sequences: bool = False
    supports_image_input: bool = False
    supports_image_url_input: bool = False
    supports_video_input: bool = False
    supports_video_url_input: bool = False
    supports_audio_input: bool = False
    supports_pdf_input: bool = False
    supports_pdf_url_input: bool = False
    supports_prompt_cache_boundaries: bool = False
    supports_media_handle_input: bool = False
    maximum_stop_sequences: int | None = Field(default=None, ge=1)
    minimum_output_tokens: int | None = Field(default=None, ge=1)
    logprobs_reasoning_efforts: tuple[ReasoningEffort, ...] = ()
    supported_reasoning_efforts: tuple[ReasoningEffort, ...] = ()
    reasoning_default_effort: ReasoningEffort | None = None
    reasoning_effort_required: bool = False
    reports_refusals: bool = False
    reports_cached_input_tokens: bool = False
    reports_cache_creation_input_tokens: bool = False
    reports_reasoning_tokens: bool = False
    reports_model_status: bool = False
    supports_async_tools: bool = False
    supports_mid_turn_steering: bool = False
    supports_reasoning_effort_update: bool = False
    time_to_first_byte_base_seconds: float | None = Field(default=None, gt=0)
    time_to_first_byte_seconds_per_million_input_tokens: float | None = Field(default=None, ge=0)
    time_to_first_token_base_seconds: float | None = Field(default=None, gt=0)
    failover_only_on: tuple[FailoverToken, ...] | None = None

    @property
    def declares_reasoning_contract(self) -> bool:
        """Whether this metadata overrides provider-family reasoning behavior."""
        return bool(
            self.supported_reasoning_efforts
            or self.reasoning_default_effort is not None
            or self.reasoning_effort_required
        )

    @model_validator(mode="after")
    def _require_custom_tools_for_grammar(self) -> GatewayDeploymentCapabilities:
        """Reject grammar-tool support that is not backed by custom-tool support.

        Returns:
            The validated declaration.

        Raises:
            ValueError: ``supports_grammar_tools`` is true while
                ``supports_custom_tools`` is false.
        """
        if self.supports_grammar_tools and not self.supports_custom_tools:
            raise ValueError("supports_grammar_tools requires supports_custom_tools=true")
        return self

    @model_validator(mode="after")
    def _require_valid_reasoning_contract(self) -> GatewayDeploymentCapabilities:
        """Reject ambiguous or non-canonical reasoning declarations."""
        order = ("none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max")
        indexes = tuple(order.index(effort) for effort in self.supported_reasoning_efforts)
        if len(set(self.supported_reasoning_efforts)) != len(self.supported_reasoning_efforts):
            raise ValueError("supported_reasoning_efforts cannot repeat values")
        if indexes != tuple(sorted(indexes)):
            raise ValueError("supported_reasoning_efforts must use canonical order")
        if (
            self.reasoning_default_effort is not None
            and self.reasoning_default_effort not in self.supported_reasoning_efforts
        ):
            raise ValueError(
                "reasoning_default_effort must be one of the supported reasoning efforts"
            )
        if self.reasoning_effort_required and not self.supported_reasoning_efforts:
            raise ValueError(
                "reasoning_effort_required needs at least one supported reasoning effort"
            )
        if self.reasoning_effort_required and self.reasoning_default_effort is None:
            raise ValueError("reasoning_effort_required needs reasoning_default_effort")
        return self


class GatewayDeploymentMetadata(ContractModel):
    """Optional gateway-only metadata authored beside one existing model record.

    Attributes:
        cache_retention_seconds: Declared positive cache lifetime, at most one hour;
            None supplies no plausible warmth claim.
    """

    exact_model_id: ArtifactId | None = None
    capabilities: GatewayDeploymentCapabilities = Field(
        default_factory=GatewayDeploymentCapabilities
    )
    prices: GatewayTokenPrices = Field(default_factory=GatewayTokenPrices)
    pricing_source: str | None = Field(default=None, min_length=1, max_length=512)
    pricing_effective_at: AwareDatetime | None = None
    dispatch: GatewayRungDispatchPolicy | None = None
    cache_retention_seconds: float | None = Field(default=None, gt=0, le=3600, allow_inf_nan=False)


class ModelRecord(ContractModel):
    """A stable local alias, exact capability snapshot, and provider-side model name.

    An omitted capability declaration means the catalog cannot prove any optional protocol
    feature or token limit. Unknown declarations stay permissive: capability preflight blocks
    only an explicit declaration that rules a requirement out, so an undeclared model remains
    usable and the provider reports any real protocol gap.

    ``served_model_id`` accepts an alternate identifier the provider echoes in responses when it
    differs from the requested ``model``, for example a vLLM endpoint that publishes an alias in
    ``/models`` but reports its canonical served name in every completion.

    ``supported_reasoning_efforts`` preserves discovery choices for setup without changing
    the identity-bearing capability snapshot. ``None`` means the listing did not declare them.

    ``discovery`` retains published tri-state flags so an explicit denial survives reload
    and remains distinguishable from an undeclared capability's default value.
    """

    connection: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=2_048)
    revision: str | None = Field(default=None, max_length=256)
    served_model_id: str | None = Field(default=None, min_length=1, max_length=2_048)
    billing_source: BillingSource
    capabilities: ModelCapabilities | None = None
    supported_reasoning_efforts: tuple[ReasoningEffort, ...] | None = None
    discovery: DiscoveredModel | None = None
    gateway: GatewayDeploymentMetadata | None = None
    sft_provenance: SFTModelProvenance | None = None

    @model_validator(mode="after")
    def _require_secret_free_model_identity(self) -> ModelRecord:
        """Reject contradictory reasoning metadata and credential-bearing identity fields."""
        if self.discovery is not None and self.discovery.model != self.model:
            raise ValueError("discovery metadata must describe the same provider model")
        if (
            self.gateway is not None
            and self.gateway.capabilities.declares_reasoning_contract
            and (self.capabilities is None or not self.capabilities.supports_reasoning)
        ):
            raise ValueError(
                "gateway reasoning metadata requires model capabilities.supports_reasoning=true"
            )
        if (
            self.sft_provenance is not None
            and self.sft_provenance.sampling_handle_sha256
            != sha256_json({"sampling_handle": self.model})
        ):
            raise ValueError("SFT provenance does not bind this model sampling handle")
        try:
            assert_secret_free(
                {
                    "connection": self.connection,
                    "model": self.model,
                    "revision": self.revision,
                    "served_model_id": self.served_model_id,
                    "billing_source": self.billing_source.value,
                    "capabilities": (
                        self.capabilities.model_dump(mode="json")
                        if self.capabilities is not None
                        else None
                    ),
                    "gateway": (
                        self.gateway.model_dump(mode="json") if self.gateway is not None else None
                    ),
                    "discovery": (
                        self.discovery.model_dump(mode="json")
                        if self.discovery is not None
                        else None
                    ),
                    "sft_provenance": (
                        self.sft_provenance.model_dump(mode="json")
                        if self.sft_provenance is not None
                        else None
                    ),
                }
            )
        except SecretBoundaryError as exc:
            raise ValueError("model identity must not contain credential values") from exc
        return self


MODEL_CATALOG_SCHEMA_VERSION = 3
"""Authored catalog revision this build writes (3 = integer nano-USD prices)."""

SANE_MAX_MODEL_CATALOG_SCHEMA_VERSION = 10_000
"""Upper bound on an authored catalog version this parser accepts as real.

Mirrors the normalized snapshot's sane-range posture: no product will ever ship
this many authored-catalog schema revisions, so a value beyond it is corruption
and fails closed rather than being read as a future contract.
"""


class ModelCatalog(ContractModel):
    """The local model aliases, connection metadata, and project role assignments.

    Attributes:
        gateway_model_chains: Authored chains keyed by canonical model ID; empty by default.
    """

    schema_version: int = Field(
        default=MODEL_CATALOG_SCHEMA_VERSION, ge=2, le=SANE_MAX_MODEL_CATALOG_SCHEMA_VERSION
    )
    """Authored catalog contract revision. Deliberately NOT a ``Literal``.

    Schema 3 prices in integer nano-USD; schema 2 (micro-USD) documents are
    upgraded at every read boundary by ``upgrade_model_catalog_document``.

    Every cross-version hydration parses the authored document first, and a
    changed ``Literal`` value on a known field raises ``literal_error``, which
    the forward-compatible read path cannot drop — so a literal here makes any
    future authored revision warm-fatal on every older pod (the same outage
    class as the 09-02 catalog incident). A newer stamp within the sane range
    parses under this build's semantics instead. That makes additive revisions
    safe by construction; a revision that REINTERPRETS existing fields must not
    reuse this channel — it needs a new field name or a fleet-first tolerance
    release. Version 1 stays rejected here: it is only readable through
    ``upgrade_legacy_billing_source`` on the TOML load path.
    """
    connections: dict[str, ConnectionConfig]
    models: dict[str, ModelRecord]
    gateway_pools: dict[str, GatewayPoolRecord] = Field(default_factory=dict)
    gateway_model_chains: dict[str, GatewayModelChain] = Field(default_factory=dict)
    roles: ModelRoles = Field(default_factory=ModelRoles)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _require_integer_schema_version(cls, value: object) -> object:
        """Reject boolean and floating-point lookalikes at the version boundary."""
        if type(value) is not int:
            raise ValueError("model catalog schema_version must be an integer")
        return value

    @field_validator("connections")
    @classmethod
    def _require_valid_connection_names(
        cls, value: dict[str, ConnectionConfig]
    ) -> dict[str, ConnectionConfig]:
        if not value:
            raise ValueError("models.toml needs at least one connection")
        for connection_name in value:
            validate_artifact_id(connection_name)
        return value

    @field_validator("models")
    @classmethod
    def _require_valid_model_aliases(cls, value: dict[str, ModelRecord]) -> dict[str, ModelRecord]:
        for alias in value:
            validate_artifact_id(alias)
        return value

    @field_validator("gateway_pools")
    @classmethod
    def _require_valid_gateway_pool_names(
        cls, value: dict[str, GatewayPoolRecord]
    ) -> dict[str, GatewayPoolRecord]:
        """Validate authored pool identifiers before cross-reference checks."""
        for pool_id in value:
            validate_artifact_id(pool_id)
        return value

    @model_validator(mode="after")
    def _require_referenced_connections_and_roles(self) -> ModelCatalog:
        for alias, record in self.models.items():
            if record.connection not in self.connections:
                raise ValueError(
                    f"model alias {alias!r} names unknown connection {record.connection!r}"
                )
            connection = self.connections[record.connection]
            if (
                connection.provider in _EXPLICIT_CAPABILITY_PROVIDERS
                and record.capabilities is None
            ):
                raise ValueError(
                    f"{connection.provider} model alias {alias!r} needs an explicit capabilities "
                    "declaration because provider names do not imply protocol support or prices"
                )
        assigned_aliases = self.roles.candidates + tuple(
            alias
            for alias in (
                self.roles.incumbent,
                self.roles.world_model,
                self.roles.judge,
                self.roles.rubric_proposer,
                self.roles.embedder,
                self.roles.teacher,
            )
            if alias is not None
        )
        unknown_aliases = sorted(set(assigned_aliases).difference(self.models))
        if unknown_aliases:
            raise ValueError(f"roles name unknown model aliases: {', '.join(unknown_aliases)}")
        if self.roles.incumbent is not None and self.roles.incumbent not in self.roles.candidates:
            raise ValueError("incumbent must also appear in roles.candidates")
        pooled_aliases: set[str] = set()
        for pool_id, pool in self.gateway_pools.items():
            for alias in pool.deployment_aliases:
                if alias in pooled_aliases:
                    raise ValueError(
                        f"gateway deployment alias {alias!r} appears in more than one pool"
                    )
                pooled_aliases.add(alias)
                record = self.models.get(alias)
                if record is None:
                    raise ValueError(
                        f"gateway pool {pool_id!r} names unknown model alias {alias!r}"
                    )
                connection = self.connections[record.connection]
                if connection.provider == "tinker" or record.sft_provenance is not None:
                    raise ValueError(
                        f"gateway pool {pool_id!r} cannot contain training handle {alias!r}"
                    )
                if record.gateway is None or record.gateway.exact_model_id != pool.exact_model_id:
                    raise ValueError(
                        f"gateway pool {pool_id!r} requires alias {alias!r} to declare exact "
                        "model identity"
                    )
        return self


def load_model_catalog(path: Path) -> ModelCatalog:
    """Load and validate `.exp/models.toml` without reading its environment variables.

    Args:
        path: Path to the local model catalog.

    Returns:
        Typed aliases, connection metadata, and role assignments.

    Raises:
        ModelCatalogError: The catalog is missing, malformed, or violates the no-secret contract.
    """
    try:
        with path.open("rb") as handle:
            raw_catalog = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ModelCatalogError(f"model catalog does not exist: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ModelCatalogError(f"model catalog is invalid TOML: {path}") from exc
    try:
        return ModelCatalog.model_validate(
            upgrade_model_catalog_document(upgrade_legacy_billing_source(raw_catalog))
        )
    except ValueError as exc:
        raise ModelCatalogError(f"model catalog is invalid: {exc}") from exc


def write_model_catalog(path: Path, catalog: ModelCatalog) -> None:
    """Atomically write validated model metadata and environment-variable names only.

    Args:
        path: Destination `.exp/models.toml` path.
        catalog: Typed catalog containing no credential values.
    """
    payload = tomli_w.dumps(catalog.model_dump(mode="json", exclude_none=True))
    write_text_atomic(path, payload)
