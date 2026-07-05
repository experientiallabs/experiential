"""Public value types: backends, messages, per-call results, and errors.

`Backend` is a frozen dataclass (positional-friendly, hashable, safely shared across threads);
results are pydantic models so callers get validation and `.model_dump()` for persistence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]

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


def normalize_messages(messages: Sequence[Message | Mapping[str, str]]) -> list[Message]:
    """Coerce caller messages (typed or raw dicts) into the canonical `Message` list."""
    return [m if isinstance(m, Message) else Message.model_validate(dict(m)) for m in messages]
