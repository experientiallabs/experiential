"""Official SDK response envelopes for retained student generations."""

from __future__ import annotations

from typing import cast

from openai.types import Completion
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

from exp.common.claas.generation import GenerationRequest, GenerationResult
from exp.common.core.artifacts import JsonObject
from exp.optimize.claas.service.http.inputs import ProtocolName


def generation_response(
    result: GenerationResult, request: GenerationRequest, protocol: ProtocolName
) -> JsonObject:
    """Return an SDK-valid envelope whose standard ID is the feedback target."""
    prompt_tokens = len(result.exact_tokens.prompt_token_ids)
    output_tokens = len(result.exact_tokens.response_token_ids)
    action = result.action
    created = int(result.created_at.timestamp())
    finished = result.finish_reason != "length"
    usage: JsonObject = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
    }
    if result.cached_input_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": result.cached_input_tokens}
    if result.reasoning_output_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": result.reasoning_output_tokens}
    if protocol == "chat":
        message: JsonObject = {"role": "assistant", "content": action.content, "refusal": None}
        if action.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_json()},
                }
                for call in action.tool_calls
            ]
        envelope = ChatCompletion.model_validate(
            {
                "id": result.response_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "length"
                        if not finished
                        else "tool_calls"
                        if action.tool_calls
                        else "stop",
                        "logprobs": None,
                    }
                ],
                "usage": usage,
            }
        )
    elif protocol == "completions":
        envelope = Completion.model_validate(
            {
                "id": result.response_id,
                "object": "text_completion",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "text": result.raw_text,
                        "finish_reason": result.finish_reason,
                        "logprobs": None,
                    }
                ],
                "usage": usage,
            }
        )
    else:
        output: list[JsonObject] = []
        if action.content is not None:
            output.append(
                {
                    "id": f"msg_{result.response_id}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed" if finished else "incomplete",
                    "content": [{"type": "output_text", "text": action.content, "annotations": []}],
                }
            )
        for call in action.tool_calls:
            output.append(
                {
                    "id": f"fc_{call.call_id}",
                    "type": "function_call",
                    "status": "completed" if finished else "incomplete",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments_json(),
                }
            )
        envelope = Response.model_validate(
            {
                "id": result.response_id,
                "object": "response",
                "created_at": created,
                "model": request.model,
                "status": "completed" if finished else "incomplete",
                "output": output,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                        "strict": False,
                    }
                    for tool in request.tools
                ],
                "max_output_tokens": request.maximum_output_tokens,
                "temperature": 1.0,
                "top_p": 1.0,
                "usage": _response_usage(result),
                "error": None,
                "incomplete_details": None if finished else {"reason": "max_output_tokens"},
            }
        )
    return cast(JsonObject, envelope.model_dump(mode="json", exclude_unset=True))


def _response_usage(result: GenerationResult) -> JsonObject | None:
    """Report Responses usage only when every SDK-required breakdown was measured."""
    if any(
        value is None
        for value in (
            result.cached_input_tokens,
            result.cache_write_input_tokens,
            result.reasoning_output_tokens,
        )
    ):
        return None
    prompt = len(result.exact_tokens.prompt_token_ids)
    response = len(result.exact_tokens.response_token_ids)
    return {
        "input_tokens": prompt,
        "output_tokens": response,
        "total_tokens": prompt + response,
        "input_tokens_details": {
            "cached_tokens": result.cached_input_tokens,
            "cache_write_tokens": result.cache_write_input_tokens,
        },
        "output_tokens_details": {"reasoning_tokens": result.reasoning_output_tokens},
    }
