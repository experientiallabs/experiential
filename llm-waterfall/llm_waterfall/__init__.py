"""llm-waterfall: stateless LLM failover across an ordered chain of backends.

Capacity errors (throttling, transient 5xx, timeouts) spill to the next backend; real client
errors raise immediately. Every call returns which backend served it, token usage, USD cost, and
the full attempt trail.
"""

from llm_waterfall.classify import is_capacity_error, outcome_for
from llm_waterfall.pricing import ModelPrice, cost_usd, price_for
from llm_waterfall.types import (
    Attempt,
    Backend,
    CompletionResult,
    EmbeddingResult,
    EmbeddingsUnsupported,
    Message,
    RetryPolicy,
    TokenUsage,
    VerifyResult,
    WaterfallExhausted,
)
from llm_waterfall.waterfall import Waterfall

__version__ = "0.1.3"

__all__ = [
    "Attempt",
    "Backend",
    "CompletionResult",
    "EmbeddingResult",
    "EmbeddingsUnsupported",
    "Message",
    "ModelPrice",
    "RetryPolicy",
    "TokenUsage",
    "VerifyResult",
    "Waterfall",
    "WaterfallExhausted",
    "cost_usd",
    "is_capacity_error",
    "outcome_for",
    "price_for",
]
