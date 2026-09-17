"""Original-token provenance and prompt/action rendering with a real local tokenizer."""

from pathlib import Path

import pytest

from exp.common.claas.generation import GenerationRequest
from exp.common.models import AssistantAction, ModelMessage, ToolCall
from exp.common.tasks import ToolSchema
from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.backends.verl.generation import _message, generation_result, prompt_ids
from exp.optimize.claas.backends.verl.native_test import tokenizer
from exp.optimize.claas.training_contracts_test import spec


def test_original_tokens_and_probability_values_survive_text_decoding(tmp_path: Path) -> None:
    """Decoded text does not replace IDs, log probabilities, or the sampled policy identity."""
    settings = ResidentVerlSettings(checkpoint_root=tmp_path, decoder="text")
    request = GenerationRequest(
        request_id="r", model=spec().adapter_id, prompt="a b", maximum_output_tokens=3
    )
    model_tokenizer = tokenizer()
    model_tokenizer.eos_token = "g"
    prompt = prompt_ids(request, model_tokenizer, spec(), settings)
    assert prompt == (1, 2)
    result = generation_result(
        request, prompt, (3, 7), (-1.25, -2.5), model_tokenizer, spec(), settings, "policy-7"
    )
    assert result.raw_text == "c"
    assert result.exact_tokens.response_token_ids == (3, 7)
    assert result.exact_tokens.response_logprobs == (-1.25, -2.5)
    assert result.exact_tokens.policy_revision == "policy-7"
    assert result.exact_tokens.sampling_temperature == 1
    assert result.action.content == "c"


def test_prompt_limits_are_rejected_without_truncation(tmp_path: Path) -> None:
    """The sampled input must fit the exact requested output bound and model context."""
    settings = ResidentVerlSettings(checkpoint_root=tmp_path, maximum_output_tokens=4)
    request = GenerationRequest(
        request_id="r", model=spec().base_model, prompt="a b", maximum_output_tokens=5
    )
    with pytest.raises(ValueError, match="maximum_output_tokens"):
        prompt_ids(request, tokenizer(), spec(), settings)
    with pytest.raises(ValueError, match="sequence limit"):
        prompt_ids(
            request.model_copy(update={"maximum_output_tokens": 4}),
            tokenizer(),
            spec().model_copy(update={"max_sequence_tokens": 5}),
            settings,
        )


def test_chat_template_receives_original_tool_linkage(tmp_path: Path) -> None:
    """The actual tokenizer template consumes tool metadata and assistant calls once."""
    model_tokenizer = tokenizer()
    model_tokenizer.chat_template = (
        "{% for message in messages %}{{message['content']}} {% endfor %}"
        "{% if tools %}{{tools[0]['function']['name']}}{% endif %}"
    )
    tool = ToolSchema(name="c", description="tool", input_schema={"type": "object"})
    action = AssistantAction(tool_calls=(ToolCall(call_id="call-1", name="c", arguments={}),))
    message = ModelMessage(role="assistant", content="", assistant_action=action)
    assert _message(message)["tool_calls"] == [
        {"id": "call-1", "type": "function", "function": {"name": "c", "arguments": {}}}
    ]
    request = GenerationRequest(
        request_id="r",
        model=spec().base_model,
        messages=(ModelMessage(role="user", content="a b"),),
        tools=(tool,),
        maximum_output_tokens=4,
    )
    assert prompt_ids(
        request, model_tokenizer, spec(), ResidentVerlSettings(checkpoint_root=tmp_path)
    ) == (1, 2, 3)


def test_tool_syntax_special_tokens_survive_visible_decoding(tmp_path: Path) -> None:
    """Tool syntax may be special tokens; only trailing EOS/padding may disappear from text."""
    model_tokenizer = tokenizer()
    model_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<tool_call>", "</tool_call>"]}
    )
    text = '<tool_call>{"name":"c","arguments":{}}</tool_call>'
    response = tuple(model_tokenizer.encode(text, add_special_tokens=False))
    # The tiny vocabulary leaves JSON unknown, but preserves declared special tool markers.
    rendered = model_tokenizer.decode(list(response), skip_special_tokens=False)
    assert "<tool_call>" in rendered and "</tool_call>" in rendered
    request = GenerationRequest(
        request_id="r", model=spec().base_model, prompt="a", maximum_output_tokens=64
    )
    result = generation_result(
        request,
        (1,),
        response,
        tuple(-1.0 for _ in response),
        model_tokenizer,
        spec(),
        ResidentVerlSettings(checkpoint_root=tmp_path, decoder="text"),
        "policy-1",
    )
    assert result.raw_text == rendered
    assert result.exact_tokens.response_token_ids == response
