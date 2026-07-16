"""Provider interface and shared config/value types."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from llm_waterfall import ChatMaxTokensField, ChatRequest, ChatResponse
from pydantic import BaseModel, Field


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
    input_tokens: int = 0
    output_tokens: int = 0


class Completion(BaseModel):
    text: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    # The model that actually served, when the provider is a failover chain and a fallback took
    # the call. None (the norm) means "the configured model" — metering falls back to config.
    # min_length=1 keeps "" impossible, so `completion.model or config.model` is exact.
    model: str | None = Field(default=None, min_length=1)


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
    reasoning_effort: str | None = None  # OpenAI Responses reasoning.effort
    # The serialized default stays stable for persisted configs. When callers do not explicitly
    # set this field, built-in models resolve it from the canonical ProviderModel catalog.
    chat_max_tokens_field: ChatMaxTokensField = "max_completion_tokens"

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


# One read-only instance reused across every verify() ping (complete() never mutates messages).
_PING_MESSAGES: list[Message] = [Message(role="user", content="ping")]

# Ping output budget. Reasoning models (GPT-5.x) spend output tokens on reasoning before any
# visible text, and OpenAI 400s ("max_tokens or model output limit was reached") when the budget
# can't cover it — which reads like bad credentials. Non-reasoning models stop after a token or
# two regardless, so the headroom costs nothing there.
PING_MAX_TOKENS = 2048

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
    cfg = provider.config
    try:
        provider.complete("", _PING_MESSAGES, max_tokens=PING_MAX_TOKENS)
    except Exception as exc:  # noqa: BLE001 - verify reports failure, never raises
        # A max-tokens/output-limit error confirms reachability: the request reached the model
        # (auth + model id are valid) and only failed because a reasoning model consumed the 1-token
        # ping budget before producing output. Anything else (auth, missing model, network) is a
        # real failure.
        msg = str(exc).lower()
        if any(marker in msg for marker in _REACHABLE_ERROR_MARKERS):
            return VerifyResult(ok=True, kind=cfg.kind, model=cfg.model)
        return VerifyResult(ok=False, kind=cfg.kind, model=cfg.model, detail=str(exc))
    return VerifyResult(ok=True, kind=cfg.kind, model=cfg.model)
