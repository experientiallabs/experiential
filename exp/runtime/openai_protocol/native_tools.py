"""Non-function OpenAI Responses tool declarations carried opaquely."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NativeResponseTool(BaseModel):
    """One non-function Responses tool declaration carried opaquely.

    Codex ships ``custom`` (freeform grammar), ``namespace`` (nested tool
    tree), ``web_search``, and ``tool_search`` declarations whose shapes
    exist on no other wire. Like ``_AdditionalToolsItem``, validation is
    deliberately shallow and the raw declaration forwards byte-for-byte on
    native Responses rungs only (each type captured live from Codex 0.151.0
    and accepted by the provider with a plain API key, 2026-09-01); the
    provider stays the authority on each declaration's internal shape. A
    ``web_search*`` declaration additionally normalizes into the gateway's
    own web-search request (see ``openai_protocol.web_search``).
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(min_length=1, max_length=64)

    @field_validator("type")
    @classmethod
    def _require_non_function(cls, value: str) -> str:
        """Keep typed function declarations on the strict model."""
        if value == "function":
            raise ValueError("function tool declarations use the typed profile")
        return value
