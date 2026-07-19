"""Provider interface and shared config/value types."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Literal, Protocol, Self, runtime_checkable

from llm_waterfall import ChatMaxTokensField, ChatRequest, ChatResponse, ReasoningEffort
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderKind(StrEnum):
    ANTHROPIC = "anthropic"  # Opus 4.8 direct
    BEDROCK = "bedrock"  # Claude 4.8 via AWS
    AZURE_OPENAI = "azure"  # GPT 5.5 via the Azure OpenAI service
    OPENAI = "openai"  # GPT 5.5 direct
    OPENAI_RESPONSES = "openai_responses"  # GPT 5.x direct via the Responses API


class EmbedderKind(StrEnum):
    """Which embedder supplies phi for retrieval.

    `HASHING` is the offline, zero-config default (no creds, no network). The other three map 1:1 to
    the same-named `ProviderKind` and use that backend's embeddings API. Anthropic is intentionally
    absent — it has no embeddings API; configure `BEDROCK`/`OPENAI`/`AZURE_OPENAI` (or `HASHING`).
    """

    HASHING = "hashing"  # offline HashingEmbedder (default)
    BEDROCK = "bedrock"  # Titan on AWS Bedrock
    OPENAI = "openai"  # OpenAI embeddings
    AZURE_OPENAI = "azure"  # Azure OpenAI embedding deployment

    def provider_kind(self) -> ProviderKind:
        """The ProviderKind backing this embedder. Raises for `HASHING` (no provider)."""
        if self is EmbedderKind.HASHING:
            raise ValueError("HASHING is the offline embedder; it has no backing provider")
        return ProviderKind(self.value)


Role = Literal["user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class TokenUsage(BaseModel):
    # Providers may report separately billed usage dimensions. Preserve them so the hard-budget
    # boundary can price a registered inclusive subset or fail closed on an unknown dimension.
    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0


class Completion(BaseModel):
    text: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    # The model that actually served, when the provider is a failover chain and a fallback took
    # the call. None (the norm) means "the configured model" — metering falls back to config.
    # min_length=1 keeps "" impossible, so `completion.model or config.model` is exact.
    model: str | None = Field(default=None, min_length=1)
    # The provider that actually served a failover call. None means the configured provider.
    provider: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
    )
    # OpenAI-compatible providers can report a backend fingerprint independently of the served
    # model. Preserve explicit absence so a frozen scored route can distinguish it from a value.
    system_fingerprint: str | None = Field(default=None, min_length=1, max_length=512)


DEFAULT_MAX_TOKENS = 8192


class VerifyResult(BaseModel):
    ok: bool
    kind: ProviderKind
    model: str
    detail: str = ""


class ProviderConfig(BaseModel):
    """Everything needed to construct one provider.

    Credentials are read from the environment by default (keys named per backend); the explicit
    backend knobs below override. The env var names are documented in `wmh.config`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ProviderKind
    # Canonical, provider-independent identity. ``model`` remains the exact
    # provider runtime id for SDK calls and old persisted configs.
    model_type: str | None = None
    model: str
    embed_model: str | None = None  # embeddings model id / Azure embedding deployment
    embed_dim: int | None = None  # requested embedding dimension (Titan v2, text-embedding-3-*)
    # Backend knobs (only some apply per kind):
    endpoint: str | None = None  # Azure OpenAI / custom base URL
    region: str | None = None  # AWS Bedrock region
    deployment: str | None = None  # Azure OpenAI deployment name
    api_version: str | None = None  # Azure OpenAI API version
    reasoning_effort: ReasoningEffort | None = None
    # Azure reasoning/tool calls use the native Responses route alongside the dated Chat API.
    # Omit this field from unrelated persisted configs so existing identities remain stable.
    responses_api_version: Literal["v1"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    # The serialized default stays stable for persisted configs. When callers do not explicitly
    # set this field, built-in models resolve it from the canonical ProviderModel catalog.
    chat_max_tokens_field: ChatMaxTokensField = "max_completion_tokens"

    @model_validator(mode="after")
    def _validate_reasoning_effort(self) -> Self:
        """Reject settings that the selected provider and model cannot honor."""
        if self.responses_api_version is not None and (
            self.kind is not ProviderKind.AZURE_OPENAI or self.reasoning_effort is None
        ):
            raise ValueError(
                "responses_api_version is supported only by Azure reasoning configuration"
            )
        if (
            self.kind is ProviderKind.AZURE_OPENAI
            and self.reasoning_effort is not None
            and self.responses_api_version != "v1"
        ):
            raise ValueError("Azure reasoning configuration requires responses_api_version='v1'")
        if self.reasoning_effort is None:
            return self
        from wmh.providers.models import resolve_provider_model

        resolved = resolve_provider_model(self.kind, self.model)
        if self.model_type is not None:
            declared = resolve_provider_model(self.kind, self.model_type)
            if declared.model_type != resolved.model_type:
                raise ValueError(
                    f"model_type {self.model_type!r} does not match runtime model "
                    f"{self.model!r} for reasoning configuration"
                )
        if self.reasoning_effort not in resolved.reasoning_efforts:
            if self.reasoning_effort == "max":
                raise ValueError("reasoning effort 'max' is supported only by Claude Opus 4.6")
            raise ValueError(
                f"{self.kind.value}/{resolved.model_type} does not support reasoning effort "
                f"{self.reasoning_effort!r}"
            )
        return self

    def resolved_chat_max_tokens_field(self) -> ChatMaxTokensField:
        """Return the output-token field accepted by this configured model."""
        # Local import avoids a module cycle: the model catalog imports ProviderKind above.
        from wmh.providers.models import resolve_chat_max_tokens_field

        model = self.model_type or self.model
        return resolve_chat_max_tokens_field(
            self.kind,
            model,
            fallback=self.chat_max_tokens_field,
        )

    def resolved_chat_forward_temperature(self) -> bool:
        """Return whether this configured model accepts chat temperature."""
        # Local import avoids a module cycle: the model catalog imports ProviderKind above.
        from wmh.providers.models import resolve_provider_model

        # Explicit OpenAI-compatible endpoints are user-owned sampling servers even when their
        # configured model label happens to match a built-in reasoning model.
        if self.kind is ProviderKind.OPENAI and self.endpoint is not None:
            return True
        model = self.model_type or self.model
        return resolve_provider_model(self.kind, model).forward_temperature


def normalize_chat_temperature(
    request: ChatRequest,
    *,
    forward_temperature: bool,
) -> ChatRequest:
    """Apply one model's sampling capability without mutating the provider-neutral request."""
    if forward_temperature or request.temperature is None:
        return request
    return request.model_copy(update={"temperature": None})


@runtime_checkable
class Embedder(Protocol):
    """The embedding half of a provider (phi in DreamGym).

    Retrieval depends only on this narrower capability, so it accepts either a full `Provider` or a
    standalone local embedder (`wmh.retrieval.embedders.HashingEmbedder`) without requiring creds.
    """

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Provider(Protocol):
    """The single interface all four backends implement."""

    config: ProviderConfig

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        """Generate a completion. Used by the world model, GEPA, the judge, and the demo agent."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for retrieval (phi in DreamGym). May delegate to a sibling embed model."""
        ...

    def verify(self) -> VerifyResult:
        """Cheap creds/model check run on startup (`wmh providers verify`)."""
        ...


@runtime_checkable
class ToolCallingProvider(Protocol):
    """Provider capability for full structured agent requests.

    This stays separate from :class:`Provider`: world-model, judge, and prompt-optimization
    callers need only text, while agent runtimes must preserve tool schemas, tool calls, tool
    results, finish reasons, and usage end to end.
    """

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Return one non-streaming structured chat completion."""
        ...


@runtime_checkable
class SingleDispatchProvider(Provider, ToolCallingProvider, Protocol):
    """A provider whose public completion methods issue at most one paid request.

    Hard-budget settlement cannot observe SDK-internal retries or fallback requests. Providers
    advertise this narrower capability only when their clients disable those hidden attempts.
    """

    paid_request_attempts: Literal[1]


# One read-only instance reused across every verify() ping (complete() never mutates messages).
_PING_MESSAGES: list[Message] = [Message(role="user", content="ping")]

# Ping output budget. Reasoning models (GPT-5.x) spend output tokens on reasoning before any
# visible text, and OpenAI 400s ("max_tokens or model output limit was reached") when the budget
# can't cover it. Non-reasoning models stop after a token or two regardless, so the headroom costs
# nothing there.
PING_MAX_TOKENS = 2048

_STRUCTURED_TOOL_PING = ChatRequest.model_validate(
    {
        "messages": [{"role": "user", "content": "Call health_check exactly once."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "health_check",
                    "description": "Verify structured tool-call availability.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ],
        "tool_choice": "required",
        "max_completion_tokens": PING_MAX_TOKENS,
        "store": False,
    }
)

# Belt-and-suspenders for the above: if a reasoning model spends even the larger ping budget on
# reasoning before emitting output, the resulting error still PROVES the model is reachable (auth
# ok, model exists). Treat these markers as reachable so `verify` passes instead of reporting fail.
_REACHABLE_ERROR_MARKERS = (
    "max_tokens",
    "max_output_tokens",
    "output limit was reached",
    "finish the message because",
)


def verify_via_ping(provider: Provider) -> VerifyResult:
    """Shared `verify()`: one cheap short completion, reporting failure as ok=False.

    Every backend's verify() is identical apart from its kind/model (both on the config), so they
    all delegate here. Never raises — `verify_all` relies on that to not crash startup.
    """
    return _verify_call(
        provider.config,
        lambda: provider.complete("", _PING_MESSAGES, max_tokens=PING_MAX_TOKENS),
        accept_output_limit_reachability=True,
    )


def verify_via_structured_tool_ping(provider: SingleDispatchProvider) -> VerifyResult:
    """Verify the operative structured tool route with exactly one bounded dispatch."""

    def call() -> object:
        if provider.paid_request_attempts != 1:
            raise ValueError("structured provider verification requires one paid request attempt")
        response = provider.complete_chat(_STRUCTURED_TOOL_PING)
        if not response.choices:
            raise ValueError("structured provider verification returned no choices")
        calls = response.choices[0].message.tool_calls or []
        if not any(call.function.name == "health_check" for call in calls):
            raise ValueError("structured provider verification returned no health_check call")
        if response.provider_receipt is None:
            raise ValueError("structured provider verification returned no provider receipt")
        return response

    return _verify_call(provider.config, call, accept_output_limit_reachability=False)


def _verify_call(
    config: ProviderConfig,
    call: Callable[[], object],
    *,
    accept_output_limit_reachability: bool,
) -> VerifyResult:
    """Run one verification dispatch with shared reasoning-limit semantics."""
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - verify reports failure, never raises
        # A max-tokens/output-limit error confirms reachability: the request reached the model
        # (auth + model id are valid) and only failed because a reasoning model consumed the 1-token
        # ping budget before producing output. Anything else (auth, missing model, network) is a
        # real failure.
        msg = str(exc).lower()
        if accept_output_limit_reachability and any(
            marker in msg for marker in _REACHABLE_ERROR_MARKERS
        ):
            return VerifyResult(ok=True, kind=config.kind, model=config.model)
        return VerifyResult(ok=False, kind=config.kind, model=config.model, detail=str(exc))
    return VerifyResult(ok=True, kind=config.kind, model=config.model)
