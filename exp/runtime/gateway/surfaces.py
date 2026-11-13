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

OutputLimitParameter = Literal[
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "generationConfig.maxOutputTokens",
]
"""Every public field a caller can carry an output ceiling in."""

_PROMPT_PARAMETERS: dict[GatewayApiSurface, str] = {
    GatewayApiSurface.RESPONSES: "input",
    GatewayApiSurface.GENERATE_CONTENT: "contents",
}
_OUTPUT_LIMIT_PARAMETERS: dict[GatewayApiSurface, OutputLimitParameter] = {
    GatewayApiSurface.RESPONSES: "max_output_tokens",
    GatewayApiSurface.GENERATE_CONTENT: "generationConfig.maxOutputTokens",
}


def prompt_parameter(surface: GatewayApiSurface) -> str:
    """The public field a surface's conversation arrives in, named by prompt rejections."""
    return _PROMPT_PARAMETERS.get(surface, "messages")


def default_output_limit_parameter(surface: GatewayApiSurface) -> OutputLimitParameter:
    """The surface's output-ceiling field, for a request that did not carry one."""
    return _OUTPUT_LIMIT_PARAMETERS.get(surface, "max_tokens")


JSON_OBJECT_OUTPUT_SURFACES = frozenset(
    {
        GatewayApiSurface.CHAT_COMPLETIONS,
        GatewayApiSurface.RESPONSES,
        GatewayApiSurface.GENERATE_CONTENT,
    }
)
"""The surfaces whose wire defines a schema-free JSON output mode."""
