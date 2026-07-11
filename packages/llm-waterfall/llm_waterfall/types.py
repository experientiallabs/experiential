"""Public value types: backends, messages, per-call results, and errors.

`Backend` is a frozen dataclass (positional-friendly, hashable, safely shared across threads);
results are pydantic models so callers get validation and `.model_dump()` for persistence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

Role = Literal["user", "assistant"]
ChatMaxTokensField = Literal["max_completion_tokens", "max_tokens"]

PROVIDERS = ("openai", "anthropic", "azure_openai", "bedrock", "aws_mantle")


class Message(BaseModel):
    """One chat turn. The system prompt is a separate `complete()` param, not a message."""

    role: Role
    content: str


@dataclass(frozen=True)
class Backend:
    """One (provider, model, credentials) rung of the waterfall.

    Credentials come from the environment (API keys) or, for Bedrock, a named AWS profile —
    `profile` maps to `boto3.Session(profile_name=...)`, so one chain can span multiple accounts.
    """

    provider: str  # one of PROVIDERS
    model: str
    profile: str | None = None  # bedrock: named AWS profile
    region: str | None = None  # bedrock
    endpoint: str | None = None  # azure base URL / custom OpenAI base_url
    deployment: str | None = None  # azure
    api_version: str | None = None  # azure
    embed_model: str | None = None  # None → provider default
    embed_dim: int | None = None
    connect_timeout_s: float = 15.0
    # Generous read timeout: reasoning models can legitimately generate for minutes, and a
    # mid-generation cutoff wastes the whole call — but a stalled connection must still raise
    # (and thus fail over) instead of hanging forever.
    read_timeout_s: float = 600.0
    chat_max_tokens_field: ChatMaxTokensField = "max_completion_tokens"

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"unknown provider {self.provider!r}; expected one of {', '.join(PROVIDERS)}"
            )


@dataclass(frozen=True)
class RetryPolicy:
    """How many times to walk the whole chain before giving up.

    `rounds=1` (default) is pure failover: one attempt per backend, no sleeping. Higher values
    wrap around — sleep with capped exponential backoff, then restart at the primary — so a long
    unattended run survives the whole chain throttling at once. The sleep is call-local; the
    waterfall stays stateless.
    """

    rounds: int = 1
    backoff_base_s: float = 15.0
    backoff_max_s: float = 120.0

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("RetryPolicy.rounds must be >= 1")

    def backoff_before_round(self, round_index: int) -> float:
        """Seconds to sleep before round `round_index` (1-based; round 1 never sleeps)."""
        if round_index <= 1:
            return 0.0
        return min(self.backoff_base_s * 2 ** (round_index - 2), self.backoff_max_s)


class TokenUsage(BaseModel):
    """Raw token counts for one call (pricing converts to USD per 1M tokens)."""

    input_tokens: int = 0
    output_tokens: int = 0


AttemptOutcome = Literal["ok", "capacity_error", "client_error", "unsupported"]


class Attempt(BaseModel):
    """One backend try within a call — the unit of the trace the waterfall returns."""

    provider: str
    model: str
    outcome: AttemptOutcome
    latency_s: float
    error: str | None = None
    error_type: str | None = None  # exception class name


class CompletionResult(BaseModel):
    """A completion plus attribution: which backend served it, what it cost, the full path."""

    text: str
    model_used: str
    provider_used: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    attempts: list[Attempt] = Field(default_factory=list)


JsonObject = dict[str, JsonValue]


class ChatFunctionCall(BaseModel):
    """Function name plus its JSON-encoded arguments in an assistant tool call."""

    name: str
    arguments: str


class ChatToolCall(BaseModel):
    """One OpenAI-compatible assistant tool call."""

    id: str
    type: Literal["function"] = "function"
    function: ChatFunctionCall


class ChatMessage(BaseModel):
    """One structured chat turn, including tool calls and tool results."""

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: JsonValue = None
    tool_calls: list[ChatToolCall] | None = None
    tool_call_id: str | None = None


class ChatFunctionDefinition(BaseModel):
    """Function schema advertised to a tool-calling model."""

    name: str
    description: str = ""
    parameters: JsonObject = Field(default_factory=dict)


class ChatTool(BaseModel):
    """One OpenAI-compatible function tool definition."""

    type: Literal["function"] = "function"
    function: ChatFunctionDefinition


class ChatRequest(BaseModel):
    """Provider-neutral structured chat request used by agent runtimes.

    Known tool-calling fields are validated explicitly. ``extra="allow"`` preserves newer
    OpenAI-compatible request fields emitted by an agent SDK without weakening the typed core.
    Providers call :meth:`provider_payload` to force a non-streaming request for the framed pi
    transport and to stamp their own routed model/deployment.
    """

    model_config = ConfigDict(extra="allow")

    messages: list[ChatMessage] = Field(default_factory=list)
    model: str | None = None
    tools: list[ChatTool] | None = None
    tool_choice: JsonValue = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    stream_options: JsonObject | None = None

    def provider_payload(
        self,
        model: str,
        *,
        max_tokens_field: ChatMaxTokensField = "max_completion_tokens",
    ) -> JsonObject:
        """Return the non-streaming wire payload for a provider-routed model."""
        payload = self.model_dump(mode="json", exclude_none=True)
        payload["model"] = model
        payload["stream"] = False
        payload.pop("stream_options", None)
        if max_tokens_field == "max_tokens":
            alternate = payload.pop("max_completion_tokens", None)
            if alternate is not None and "max_tokens" not in payload:
                payload["max_tokens"] = alternate
        else:
            alternate = payload.pop("max_tokens", None)
            if alternate is not None and "max_completion_tokens" not in payload:
                payload["max_completion_tokens"] = alternate
        return payload


class ChatUsage(BaseModel):
    """OpenAI-compatible structured completion usage."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatChoice(BaseModel):
    """One structured completion choice."""

    model_config = ConfigDict(extra="allow")

    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    """Structured completion returned to the agent runtime."""

    model_config = ConfigDict(extra="allow")

    choices: list[ChatChoice]
    usage: ChatUsage | None = None
    model: str | None = None

    def token_usage(self) -> TokenUsage:
        """Project provider usage onto the waterfall's canonical counters."""
        if self.usage is None:
            return TokenUsage()
        return TokenUsage(
            input_tokens=self.usage.prompt_tokens,
            output_tokens=self.usage.completion_tokens,
        )

    def wire_payload(self) -> JsonObject:
        """Serialize the response back to the OpenAI-compatible pi bridge."""
        return self.model_dump(mode="json", exclude_none=True)


class ChatResult(BaseModel):
    """A structured completion plus waterfall attribution and failover history."""

    response: ChatResponse
    model_used: str
    provider_used: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    attempts: list[Attempt] = Field(default_factory=list)


class EmbeddingResult(BaseModel):
    """Embedding vectors plus the same attribution as `CompletionResult`."""

    vectors: list[list[float]]
    model_used: str
    provider_used: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    attempts: list[Attempt] = Field(default_factory=list)


class VerifyResult(BaseModel):
    """Outcome of one backend's cheap credential/model ping."""

    ok: bool
    provider: str
    model: str
    detail: str = ""


class WaterfallExhausted(RuntimeError):
    """Every backend in every round was capacity-constrained. Carries the full attempt trail."""

    def __init__(self, message: str, attempts: list[Attempt]) -> None:
        super().__init__(message)
        self.attempts = attempts


class EmbeddingsUnsupported(NotImplementedError):
    """Raised by adapters whose provider has no embeddings API; the waterfall skips them."""


class ToolCallingUnsupported(NotImplementedError):
    """Raised by adapters without structured tool-calling support; the waterfall skips them."""


def normalize_messages(messages: Sequence[Message | Mapping[str, str]]) -> list[Message]:
    """Coerce caller messages (typed or raw dicts) into the canonical `Message` list."""
    return [m if isinstance(m, Message) else Message.model_validate(dict(m)) for m in messages]
