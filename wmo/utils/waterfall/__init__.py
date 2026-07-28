"""Stateless LLM failover across an ordered chain of backends.

Capacity errors (throttling, transient 5xx, timeouts) spill to the next backend; real client
errors raise immediately. Every call returns which backend served it, token usage, USD cost, and
the full attempt trail.

Formerly the standalone `waterfall` workspace member; it was never published to PyPI and
`wmo` was its only consumer, so it lives here as a utility module (MIT, see `LICENSE`) rather
than as a separate distribution. It imports nothing from the rest of `wmo` — keep it that way.
"""

from wmo.utils.waterfall.classify import is_capacity_error, outcome_for
from wmo.utils.waterfall.pricing import ModelPrice, cost_usd, price_for
from wmo.utils.waterfall.types import (
    Attempt,
    Backend,
    ChatMaxTokensField,
    ChatRequest,
    ChatResponse,
    ChatResult,
    CompletionResult,
    EmbeddingResult,
    EmbeddingsUnsupported,
    Message,
    RetryPolicy,
    TokenUsage,
    ToolCallingUnsupported,
    VerifyResult,
    WaterfallExhausted,
)
from wmo.utils.waterfall.waterfall import Waterfall

__version__ = "0.1.4"

__all__ = [
    "Attempt",
    "Backend",
    "ChatMaxTokensField",
    "ChatRequest",
    "ChatResponse",
    "ChatResult",
    "CompletionResult",
    "EmbeddingResult",
    "EmbeddingsUnsupported",
    "Message",
    "ModelPrice",
    "RetryPolicy",
    "TokenUsage",
    "ToolCallingUnsupported",
    "VerifyResult",
    "Waterfall",
    "WaterfallExhausted",
    "cost_usd",
    "is_capacity_error",
    "outcome_for",
    "price_for",
]
