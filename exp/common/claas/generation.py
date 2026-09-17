"""Engine-independent student generation with original rollout evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import AwareDatetime, Field, field_serializer, model_validator

from exp.common.claas.contracts import ExactTokenEvidence, Identifier
from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.models import AssistantAction, ModelMessage
from exp.common.tasks import ToolSchema


class GenerationRequest(ContractModel):
    """One bounded text/function request to the student owned by a learning run."""

    request_id: Identifier
    model: Identifier
    messages: tuple[ModelMessage, ...] = ()
    prompt: str | None = Field(default=None, min_length=1, max_length=1_048_576)
    tools: tuple[ToolSchema, ...] = ()
    maximum_output_tokens: int = Field(default=512, strict=True, ge=1, le=131072)

    @field_serializer("messages")
    def _serialize_messages(self, messages: tuple[ModelMessage, ...]) -> list[JsonObject]:
        """Include exact tool argument strings in durable request and retry identity."""
        values: list[JsonObject] = []
        for message in messages:
            value = message.model_dump(mode="json")
            if message.assistant_action is not None:
                value["assistant_action"] = _action_payload(message.assistant_action)
            values.append(value)
        return values

    @model_validator(mode="after")
    def _validate_input(self) -> GenerationRequest:
        """Choose either raw completion text or a structured conversation."""
        if bool(self.messages) == (self.prompt is not None):
            raise ValueError("provide either nonempty messages or one prompt")
        if self.prompt is not None and self.tools:
            raise ValueError("tool use requires structured messages")
        for message in self.messages:
            if message.content_parts:
                raise ValueError(
                    "learner generation accepts text without media or content metadata"
                )
            if message.assistant_action is not None:
                _validate_action(message.assistant_action)
        return self


def _created_at() -> datetime:
    """Stamp a generated result once before durable response replay is recorded."""
    return datetime.now(UTC)


def _action_payload(action: AssistantAction) -> JsonObject:
    """Preserve immediate tool wire arguments within the generation persistence boundary."""
    calls: list[JsonObject] = []
    for call in action.tool_calls:
        value = call.model_dump(mode="json")
        value["raw_arguments"] = call.raw_arguments
        calls.append(value)
    return {"content": action.content, "tool_calls": calls}


def _validate_action(action: AssistantAction) -> None:
    """Reject provider replay controls that the text/function learning engines cannot honor."""
    for call in action.tool_calls:
        if any(
            value is not None
            for value in (
                call.cache_control,
                call.provider_item_id,
                call.provider_output_index,
                call.provider_status,
                call.provider_namespace,
                call.provider_caller,
            )
        ):
            raise ValueError(
                "learner function calls do not support provider-specific replay metadata"
            )


class GenerationResult(ContractModel):
    """An assistant action and the exact tokens sampled before any later update."""

    response_id: Identifier
    action: AssistantAction
    exact_tokens: ExactTokenEvidence
    raw_text: str
    created_at: AwareDatetime = Field(default_factory=_created_at)
    finish_reason: Literal["stop", "length"] = "stop"
    cached_input_tokens: int | None = Field(default=None, strict=True, ge=0)
    cache_write_input_tokens: int | None = Field(default=None, strict=True, ge=0)
    reasoning_output_tokens: int | None = Field(default=None, strict=True, ge=0)

    @field_serializer("action")
    def _serialize_action(self, action: AssistantAction) -> JsonObject:
        """Keep the original tool arguments stable when replaying a retained response."""
        return _action_payload(action)

    @model_validator(mode="after")
    def _validate_usage_details(self) -> GenerationResult:
        """Keep optional measured usage breakdowns within the original token counts."""
        _validate_action(self.action)
        prompt = len(self.exact_tokens.prompt_token_ids)
        response = len(self.exact_tokens.response_token_ids)
        if (
            any(
                value is not None and value > prompt
                for value in (
                    self.cached_input_tokens,
                    self.cache_write_input_tokens,
                )
            )
            or self.reasoning_output_tokens is not None
            and self.reasoning_output_tokens > response
        ):
            raise ValueError("measured usage details exceed the exact sampled token counts")
        return self
