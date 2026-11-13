"""The public endpoint families one canonical gateway request can belong to.

Kept apart from :mod:`exp.runtime.gateway.contracts` (which re-exports it) so
the enum has no dependencies and every surface module can import it cheaply.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal


class GatewayApiSurface(StrEnum):
    """Public endpoint family used by one canonical request."""

    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    MESSAGES = "messages"
    GENERATE_CONTENT = "generate_content"
    EMBEDDINGS = "embeddings"
    IMAGES = "images"
    DECISIONS = "decisions"
    SPEECH = "speech"
    TRANSCRIPTION = "transcription"
    VIDEO = "video"


ReasoningEffortParameter = Literal[
    "reasoning_effort",
    "reasoning.effort",
    "output_config.effort",
    "generationConfig.thinkingConfig.thinkingLevel",
]
"""Every public field a caller can carry a reasoning effort in, one per surface."""

JSON_OBJECT_OUTPUT_SURFACES = frozenset(
    {
        GatewayApiSurface.CHAT_COMPLETIONS,
        GatewayApiSurface.RESPONSES,
        GatewayApiSurface.GENERATE_CONTENT,
    }
)
"""The surfaces whose wire defines a schema-free JSON output mode."""
