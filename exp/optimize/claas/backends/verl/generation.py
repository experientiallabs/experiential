"""Render model input once and preserve the rollout engine's original output evidence."""

from __future__ import annotations

from typing import Literal, Protocol, cast

from transformers import PreTrainedTokenizerBase

from exp.common.claas import ExactTokenEvidence
from exp.common.claas.generation import GenerationRequest, GenerationResult
from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelMessage
from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.backends.verl.decoding import (
    HermesCompletionDecoder,
    Qwen35CompletionDecoder,
    TextCompletionDecoder,
)
from exp.optimize.claas.training_contracts import ClaasTrainingSpec


class ChatTemplateTokenizer(Protocol):
    """Type the structured-message surface accepted by Hugging Face chat templates."""

    def apply_chat_template(
        self,
        conversation: list[JsonObject],
        *,
        tools: list[JsonObject] | None,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
    ) -> list[int]:
        """Render one conversation to original prompt token IDs."""
        ...


def _message(message: ModelMessage) -> JsonObject:
    """Render tool linkage without changing prior assistant actions."""
    if message.content_parts:
        raise ValueError("resident CLaaS generation supports text messages only")
    result: JsonObject = {"role": message.role, "content": message.content}
    if message.role == "tool":
        result["tool_call_id"] = message.tool_call_id
    if message.assistant_action is not None:
        result["content"] = message.assistant_action.content
        if message.assistant_action.tool_calls:
            result["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.assistant_action.tool_calls
            ]
    return result


def prompt_ids(
    request: GenerationRequest,
    tokenizer: PreTrainedTokenizerBase,
    spec: ClaasTrainingSpec,
    settings: ResidentVerlSettings,
) -> tuple[int, ...]:
    """Tokenize the actual prompt before sampling, never reconstructing completed tokens."""
    if request.model not in {spec.base_model, spec.adapter_id}:
        raise ValueError("generation model must name this run's base model or adapter")
    if request.maximum_output_tokens > settings.maximum_output_tokens:
        raise ValueError("generation exceeds this run's maximum_output_tokens")
    if request.prompt is not None:
        tokens = tokenizer.encode(request.prompt, add_special_tokens=False)
    else:
        tools: list[JsonObject] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in request.tools
        ]
        tokens = cast(ChatTemplateTokenizer, tokenizer).apply_chat_template(
            [_message(message) for message in request.messages],
            tools=tools or None,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int for token in tokens)
    ):
        raise ValueError("tokenizer must produce a nonempty exact list of integer token IDs")
    if len(tokens) + request.maximum_output_tokens > spec.max_sequence_tokens:
        raise ValueError("prompt and requested completion exceed this run's sequence limit")
    return tuple(cast(list[int], tokens))


def generation_result(
    request: GenerationRequest,
    prompt: tuple[int, ...],
    response: tuple[int, ...],
    logprobs: tuple[float, ...],
    tokenizer: PreTrainedTokenizerBase,
    spec: ClaasTrainingSpec,
    settings: ResidentVerlSettings,
    policy_revision: str,
    finish_reason: Literal["stop", "length"] = "stop",
) -> GenerationResult:
    """Attach genuine token evidence before decoding the executable action envelope."""
    if len(response) > request.maximum_output_tokens:
        raise ValueError("rollout exceeded its requested output token limit")
    exact = ExactTokenEvidence(
        model_id=spec.base_model,
        model_revision=spec.model_revision,
        policy_revision=policy_revision,
        tokenizer_id=spec.tokenizer_id,
        tokenizer_revision=spec.tokenizer_revision,
        sampling_temperature=1.0,
        sampling_top_p=1.0,
        sampling_top_k=None,
        prompt_token_ids=prompt,
        response_token_ids=response,
        response_logprobs=logprobs,
    )
    visible = list(response)
    # Preserve semantic special tokens (tool calls, reasoning); remove only terminal controls.
    terminal = {tokenizer.eos_token_id, tokenizer.pad_token_id} - {None}
    while visible and visible[-1] in terminal:
        visible.pop()
    raw_text = tokenizer.decode(visible, skip_special_tokens=False)
    if not isinstance(raw_text, str):
        raise ValueError("tokenizer must decode one response into text")
    decoder = {
        "text": TextCompletionDecoder,
        "hermes": HermesCompletionDecoder,
        "qwen35": Qwen35CompletionDecoder,
    }[settings.decoder]()
    action = decoder.decode(raw_text, request.request_id, request.tools)
    if {call.name for call in action.tool_calls} - {tool.name for tool in request.tools}:
        raise ValueError("student emitted an undeclared tool")
    return GenerationResult(
        response_id=request.request_id,
        action=action,
        exact_tokens=exact,
        raw_text=raw_text,
        finish_reason=finish_reason,
    )
